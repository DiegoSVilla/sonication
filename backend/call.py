"""Transport-agnostic voice call.

A VoiceCall owns the call clock, event recorder, DB call row, and the
STT -> LLM -> TTS pipeline. It is the shared core behind every transport: the
browser push-to-talk websocket and the edge audio-in/stream-out HTTP API both
wrap one of these, so sessions, event logging, and timing are identical. What
differs is only how audio arrives and how it is delivered back.
"""
import asyncio
import datetime
import uuid
from typing import Any, Optional

from . import config, db, events
from .pipeline import CallPipeline, SendFn


def _est_tokens(history: list[dict[str, Any]]) -> int:
    """Rough token estimate for budget checks (~4 chars/token + per-message overhead)."""
    return sum(len(m.get("content", "")) for m in history) // 4 + 4 * len(history)


def _call_params() -> dict[str, Any]:
    return {
        "model": config.LLM_MODEL,
        "voice": config.TTS_VOICE,
        "language": config.TTS_LANGUAGE,
        "temperature": config.LLM_TEMPERATURE,
        "seed": config.LLM_SEED,
        "max_tokens": config.LLM_MAX_TOKENS,
        "phrase_min_chars": config.PHRASE_MIN_CHARS,
        "audio_bytes_per_sec": config.AUDIO_BYTES_PER_SEC,
    }


class VoiceCall:
    def __init__(self, session_id: str, send: Optional[SendFn] = None,
                 persist_context: bool = False) -> None:
        self.call_id = uuid.uuid4().hex
        self.session_id = session_id
        self.clock = events.CallClock()
        self.rec = events.EventRecorder(self.call_id, self.clock)
        self.pipeline = CallPipeline(self.call_id, self.rec, send=send)
        # One turn at a time per call (history and audio order are sequential).
        self.lock = asyncio.Lock()
        self._playback_logged: set[int] = set()
        # When True, the conversation history is persisted to session_state and
        # restored on the next call for this session (survives eviction/restart).
        self.persist_context = persist_context

    def load_context(self) -> int:
        """Restore prior conversation history for this session, if any.

        Keeps the current system prompt and appends the stored user/assistant
        turns after it. Returns how many messages were restored.
        """
        if not self.persist_context:
            return 0
        stored = db.load_session_state(self.session_id)
        if not stored:
            return 0
        self.pipeline.history = self.pipeline.history[:1] + stored
        return len(stored)

    def save_context(self) -> None:
        """Persist the conversation history (minus the system prompt)."""
        if not self.persist_context:
            return
        db.save_session_state(self.session_id, self.pipeline.history[1:], self.call_id)

    def override_context(self, system: Optional[str] = None,
                         messages: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
        """Client-managed context override (used per shot by the edge API).

        `system` replaces the system prompt; `messages` replaces the whole
        user/assistant history (a leading 'system' item is honoured if no explicit
        `system` is given). Subject to the same context budget as normal turns:
        if the result exceeds LLM_CONTEXT_FRACTION of the model window, the oldest
        user+assistant pairs are trimmed to comply (the system prompt is kept).
        """
        hist = self.pipeline.history
        sys_content = system
        if messages is not None:
            if sys_content is None:
                for m in messages:
                    if m.get("role") == "system":
                        sys_content = m.get("content")
            base_sys = sys_content if sys_content is not None else (
                hist[0]["content"] if hist and hist[0].get("role") == "system"
                else config.SYSTEM_PROMPT)
            convo = [{"role": m["role"], "content": m["content"]}
                     for m in messages if m.get("role") in ("user", "assistant")]
            self.pipeline.history = [{"role": "system", "content": base_sys}] + convo
        elif sys_content is not None:
            if hist and hist[0].get("role") == "system":
                hist[0] = {"role": "system", "content": sys_content}
            else:
                hist.insert(0, {"role": "system", "content": sys_content})

        budget = int(config.LLM_CONTEXT_TOKENS * config.LLM_CONTEXT_FRACTION) if config.LLM_CONTEXT_TOKENS else 0
        trimmed = 0
        if budget:
            while _est_tokens(self.pipeline.history) > budget and len(self.pipeline.history) >= 3:
                del self.pipeline.history[1:3]
                trimmed += 1
        return {"messages": len(self.pipeline.history), "trimmed_pairs": trimmed,
                "est_tokens": _est_tokens(self.pipeline.history), "budget_tokens": budget}

    def start(self) -> tuple[datetime.datetime, dict[str, Any]]:
        """Create the session/call rows and record call_start (wallclock zero)."""
        now = datetime.datetime.fromtimestamp(
            self.clock.started_epoch_ms / 1000.0, datetime.timezone.utc
        )
        db.create_session(self.session_id)
        params = _call_params()
        db.create_call(self.call_id, self.session_id, now.isoformat(),
                       self.clock.started_epoch_ms, params)
        self.rec.record(
            events.CALL_START,
            {"wallclock_iso": now.isoformat(), "epoch_ms": self.clock.started_epoch_ms,
             "params": params},
            t_ms=0.0,
        )
        return now, params

    def record_channel_playback(
        self, turn_index: int, phrase_index: Optional[int] = None,
        chunk_index: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        """Mark first audio on the channel for a turn and compute shot latency.

        Shot latency is ALWAYS the wall-clock distance from the turn's release
        time to this first-audio moment, never a sum of intermediate stages.
        Logged once per turn; returns None if already logged.
        """
        if turn_index in self._playback_logged:
            return None
        self._playback_logged.add(turn_index)
        t_ms = self.clock.now_ms()
        turn = db.get_turn_by_index(self.call_id, turn_index)
        self.rec.record(
            events.CHANNEL_PLAYBACK_START,
            {"turn_index": turn_index, "phrase_index": phrase_index, "chunk_index": chunk_index},
            turn_id=turn["id"] if turn else None,
            t_ms=t_ms,
        )
        shot = None
        if turn and turn.get("enter_t_ms") is not None:
            shot = t_ms - turn["enter_t_ms"]
            db.update_turn(
                turn["id"],
                first_audio_channel_t_ms=round(t_ms, 3),
                shot_latency_ms=round(shot, 3),
            )
        return {
            "turn_index": turn_index,
            "first_audio_channel_t_ms": round(t_ms, 3),
            "shot_latency_ms": round(shot, 3) if shot is not None else None,
        }

    def end(self) -> None:
        self.save_context()  # evict conversation memory to the DB
        db.end_call(self.call_id, self.clock.now_ms())
