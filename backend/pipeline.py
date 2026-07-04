"""Turn orchestrator: text_in -> LLM stream -> phrase aggregator -> TTS -> audio_out.

TTS runs concurrently with LLM generation: the first phrase is sent to the TTS
as soon as it is ready, so tts_call events overlap the still-running llm_call.
Audio is kept in order by draining phrases through a single sequential TTS
consumer, which matches how the audio must be spoken on the channel anyway.
"""
import asyncio
import json
from typing import Any, Awaitable, Callable, Optional

from . import analysis, clients, config, db, events
from .monitor import monitor

SendFn = Callable[[dict[str, Any]], Awaitable[None]]


def _phrase_ready(buf: str) -> bool:
    """A phrase is ready at >= PHRASE_MIN_CHARS ending on .!? or on a newline."""
    stripped = buf.strip()
    if len(stripped) < config.PHRASE_MIN_CHARS:
        return False
    if stripped[-1] in ".!?":
        return True
    return "\n" in buf


class CallPipeline:
    """Runs turns for one call, keeping conversation history and the call clock."""

    def __init__(
        self,
        call_id: str,
        recorder: events.EventRecorder,
        send: Optional[SendFn] = None,
    ) -> None:
        self.call_id = call_id
        self.rec = recorder
        self.send = send
        self.history: list[dict[str, str]] = [
            {"role": "system", "content": config.SYSTEM_PROMPT}
        ]
        self.turn_index = 0
        # Cumulative audio produced across the whole call (ms of speech).
        self.call_audio_ms = 0.0

    async def _emit(self, msg: dict[str, Any]) -> None:
        if self.send is not None:
            await self.send(msg)

    async def run_turn(self, user_text: str) -> dict[str, Any]:
        """Text-in turn (used by the CLI). Shot start is the text arrival time."""
        self.turn_index += 1
        idx = self.turn_index
        enter_t = self.rec.clock.now_ms()
        chars = len(user_text)
        turn_id = db.create_turn(self.call_id, idx, user_text, chars, enter_t)
        self.rec.record(
            events.TEXT_IN,
            {"turn_index": idx, "chars": chars, "text": user_text},
            turn_id=turn_id,
            t_ms=enter_t,
        )
        await self._emit(
            {"type": "text_in", "turn_index": idx, "chars": chars, "t_ms": round(enter_t, 3)}
        )
        return await self._generate(turn_id, idx, user_text, {"t_release": enter_t})

    async def run_voice_turn(self, audio: bytes, release_t: float) -> dict[str, Any]:
        """Push-to-talk turn. The shot starts at Enter release (release_t), so the
        STT and upload time are included in shot latency. The transcript then
        feeds the same LLM -> phrase -> TTS generation as a text turn.
        """
        self.turn_index += 1
        idx = self.turn_index
        turn_id = db.create_turn(self.call_id, idx, "", 0, release_t)

        stt_start = self.rec.clock.now_ms()
        try:
            result = await clients.transcribe(audio, language=config.STT_LANGUAGE or None)
        except Exception as exc:
            await self._emit({"type": "error", "message": f"STT failed: {exc}"})
            return {"turn_id": turn_id, "assistant_text": "", "metrics": {}}
        stt_done = self.rec.clock.now_ms()
        text = (result.get("text") or "").strip()
        stt_event = {
            "turn_index": idx,
            "text": text,
            "start_t_ms": round(stt_start, 3),
            "done_t_ms": round(stt_done, 3),
            "stt_ms": round(stt_done - stt_start, 3),
            "audio_bytes": len(audio),
            "usage": result.get("usage"),
        }
        self.rec.record(events.STT_FINAL, stt_event, turn_id=turn_id)
        db.update_turn(turn_id, user_text=text, chars=len(text),
                       stt_start_t_ms=stt_event["start_t_ms"],
                       stt_done_t_ms=stt_event["done_t_ms"],
                       stt_ms=stt_event["stt_ms"])
        await self._emit({"type": "stt_final", **stt_event})

        if not text:
            await self._emit({"type": "turn_skipped", "turn_index": idx,
                              "reason": "empty transcript"})
            return {"turn_id": turn_id, "assistant_text": "", "metrics": {}}

        marks = {"t_release": release_t, "t_stt_req": stt_start, "t_stt_resp": stt_done}
        return await self._generate(turn_id, idx, text, marks)

    async def _generate(self, turn_id: int, idx: int, user_text: str,
                        marks: dict[str, Any]) -> dict[str, Any]:
        """Shared LLM -> phrase -> TTS generation for both text and voice turns.

        marks carries the wall-clock stage boundaries collected so far (t_release
        and, for voice turns, the STT marks); this method fills in the LLM/TTS
        marks and stores the timing decomposition.
        """
        self.history.append({"role": "user", "content": user_text})

        # Shared per-turn state between the LLM producer and the TTS consumer.
        state: dict[str, Any] = {
            "phrase_queue": asyncio.Queue(),
            "phrase_index": 0,
            "turn_audio_ms": 0.0,
            "tts_events": [],
            "assistant_text": "",
        }

        tts_task = asyncio.create_task(self._tts_consumer(turn_id, idx, state))
        llm_summary = await self._llm_producer(turn_id, idx, state)
        # Signal end of phrases and wait for audio to finish generating.
        await state["phrase_queue"].put(None)
        await tts_task

        self.history.append({"role": "assistant", "content": state["assistant_text"]})
        await self._enforce_context_budget(llm_summary.get("usage") or {}, idx)

        # --- roll up turn metrics ---
        tts_events = state["tts_events"]
        first_tts = tts_events[0] if tts_events else None
        tts_start_t = first_tts["start_t_ms"] if first_tts else None
        tts_ttfb_ms = first_tts["ttfb_ms"] if first_tts else None
        # Generation TTFB on the backend: when the first audio byte arrived.
        gen_first_audio_t = first_tts["first_audio_t_ms"] if first_tts else None

        metrics = {
            "llm_start_t_ms": llm_summary["start_t_ms"],
            "llm_ttft_ms": llm_summary["ttft_ms"],
            "llm_done_t_ms": llm_summary["done_t_ms"],
            "llm_tokens": llm_summary["token_count"],
            "tts_start_t_ms": tts_start_t,
            "tts_ttfb_ms": tts_ttfb_ms,
            "gen_first_audio_t_ms": gen_first_audio_t,
            "audio_generated_ms": round(state["turn_audio_ms"], 3),
        }
        db.update_turn(
            turn_id,
            llm_start_t_ms=metrics["llm_start_t_ms"],
            llm_ttft_ms=metrics["llm_ttft_ms"],
            llm_done_t_ms=metrics["llm_done_t_ms"],
            llm_tokens=metrics["llm_tokens"],
            tts_start_t_ms=metrics["tts_start_t_ms"],
            tts_ttfb_ms=metrics["tts_ttfb_ms"],
            audio_generated_ms=metrics["audio_generated_ms"],
        )

        # --- stage timing decomposition (all wall-clock marks) ---
        marks.update({
            "t_llm_req": llm_summary["start_t_ms"],
            "t_llm_ttft": llm_summary["ttft_abs_t_ms"],
            "t_phrase1": llm_summary["first_phrase_t_ms"],
            "t_tts_req": tts_start_t,
            "t_tts_audio": gen_first_audio_t,
        })
        # The browser reports the channel playback-start on the first audio chunk,
        # which lands before generation finishes, so it is usually available here.
        turn_row = db.get_turn_by_index(self.call_id, idx)
        if turn_row and turn_row.get("first_audio_channel_t_ms") is not None:
            marks["t_channel"] = turn_row["first_audio_channel_t_ms"]
        timing = {"marks": marks, "pings": monitor.snapshot()}
        db.update_turn(turn_id, timing_json=json.dumps(timing))
        waterfall = analysis.compute_waterfall(timing)
        await self._emit({"type": "timing", "turn_index": idx, "waterfall": waterfall})

        await self._emit({"type": "turn_done", "turn_index": idx, "metrics": metrics})
        return {"turn_id": turn_id, "assistant_text": state["assistant_text"],
                "metrics": metrics, "timing": timing}

    async def _enforce_context_budget(self, usage: dict[str, Any], idx: int) -> None:
        """Keep the running history under LLM_CONTEXT_FRACTION of the model's
        context window. Uses the LLM's own reported total_tokens as the measure;
        when over, drops the oldest user+assistant pair(s), never the system
        prompt. Dropped turns are gone (not saved anywhere), as intended.
        """
        ctx = config.LLM_CONTEXT_TOKENS
        total = usage.get("total_tokens")
        if not ctx or not total:
            return
        budget = int(ctx * config.LLM_CONTEXT_FRACTION)
        if total <= budget:
            return
        approx = total
        removed = 0
        # history[0] is the system prompt; pairs start at index 1.
        while approx > budget and len(self.history) >= 3:
            dropped = self.history[1:3]
            approx -= sum(len(m.get("content", "")) for m in dropped) // 4 + 8
            del self.history[1:3]
            removed += 1
        if removed:
            await self._emit({
                "type": "context_trim", "turn_index": idx, "removed_pairs": removed,
                "total_tokens": total, "budget_tokens": budget,
                "history_messages": len(self.history),
            })

    async def _llm_producer(
        self, turn_id: int, idx: int, state: dict[str, Any]
    ) -> dict[str, Any]:
        """Stream the LLM, log per-token timings, and push phrases to the TTS queue."""
        start_t = self.rec.clock.now_ms()
        await self._emit({"type": "llm_start", "turn_index": idx, "t_ms": round(start_t, 3)})

        content_tokens: list[dict[str, Any]] = []
        reasoning_tokens: list[dict[str, Any]] = []
        ttft_ms: Optional[float] = None            # first content (speakable) token
        ttft_abs: Optional[float] = None           # ... as an absolute wall-clock mark
        first_phrase_t: Optional[float] = None     # when the first phrase was ready
        reasoning_ttft_ms: Optional[float] = None  # first reasoning token
        usage: dict[str, Any] = {}
        finish_reason: Optional[str] = None
        buf = ""
        last_content_t = start_t

        async for chunk in clients.stream_llm(
            messages=self.history,
            seed=config.LLM_SEED,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            enable_thinking=config.LLM_ENABLE_THINKING,
        ):
            if chunk["kind"] == "token":
                now = self.rec.clock.now_ms()
                text = chunk["content"]
                if chunk.get("channel") == "reasoning":
                    if reasoning_ttft_ms is None:
                        reasoning_ttft_ms = now - start_t
                    reasoning_tokens.append({"i": len(reasoning_tokens), "text": text,
                                             "t_ms": round(now, 3)})
                    continue
                # content: the only channel that feeds phrases and the TTS.
                if ttft_ms is None:
                    ttft_ms = now - start_t
                    ttft_abs = now
                content_tokens.append(
                    {
                        "i": len(content_tokens),
                        "text": text,
                        "t_ms": round(now, 3),
                        "dt_ms": round(now - last_content_t, 3),
                    }
                )
                last_content_t = now
                state["assistant_text"] += text
                buf += text
                await self._emit({"type": "llm_token", "turn_index": idx, "text": text})
                if _phrase_ready(buf):
                    if first_phrase_t is None:
                        first_phrase_t = self.rec.clock.now_ms()
                    await self._enqueue_phrase(state, buf)
                    buf = ""
            elif chunk["kind"] == "usage":
                usage = chunk["usage"]
            elif chunk["kind"] == "done":
                finish_reason = chunk["finish_reason"]

        # Flush any trailing text as the final phrase.
        if buf.strip():
            if first_phrase_t is None:
                first_phrase_t = self.rec.clock.now_ms()
            await self._enqueue_phrase(state, buf)

        done_t = self.rec.clock.now_ms()
        summary = {
            "start_t_ms": round(start_t, 3),
            "ttft_ms": round(ttft_ms, 3) if ttft_ms is not None else None,
            "ttft_abs_t_ms": round(ttft_abs, 3) if ttft_abs is not None else None,
            "first_phrase_t_ms": round(first_phrase_t, 3) if first_phrase_t is not None else None,
            "reasoning_ttft_ms": round(reasoning_ttft_ms, 3) if reasoning_ttft_ms is not None else None,
            "done_t_ms": round(done_t, 3),
            "token_count": len(content_tokens),
            "reasoning_token_count": len(reasoning_tokens),
            "finish_reason": finish_reason,
            "usage": usage,
        }
        # One consolidated llm_call event carrying the full token lists.
        self.rec.record(
            events.LLM_CALL,
            {
                **summary,
                "request": {
                    "model": config.LLM_MODEL,
                    "seed": config.LLM_SEED,
                    "temperature": config.LLM_TEMPERATURE,
                    "max_tokens": config.LLM_MAX_TOKENS,
                    "enable_thinking": config.LLM_ENABLE_THINKING,
                },
                "tokens": content_tokens,
                "reasoning_tokens": reasoning_tokens,
            },
            turn_id=turn_id,
        )
        await self._emit({"type": "llm_done", "turn_index": idx, "summary": summary})
        return summary

    async def _enqueue_phrase(self, state: dict[str, Any], text: str) -> None:
        state["phrase_index"] += 1
        phrase = {"phrase_index": state["phrase_index"], "text": text}
        await state["phrase_queue"].put(phrase)

    async def _tts_consumer(
        self, turn_id: int, idx: int, state: dict[str, Any]
    ) -> None:
        """Drain phrases in order, stream each through the TTS, log audio_out chunks."""
        queue: asyncio.Queue = state["phrase_queue"]
        while True:
            phrase = await queue.get()
            if phrase is None:
                break
            await self._tts_one_phrase(turn_id, idx, phrase, state)

    async def _tts_one_phrase(
        self, turn_id: int, idx: int, phrase: dict[str, Any], state: dict[str, Any]
    ) -> None:
        text = phrase["text"].strip()
        pindex = phrase["phrase_index"]
        start_t = self.rec.clock.now_ms()
        await self._emit(
            {
                "type": "tts_start",
                "turn_index": idx,
                "phrase_index": pindex,
                "text": text,
                "t_ms": round(start_t, 3),
            }
        )

        first_audio_t: Optional[float] = None
        chunk_index = 0
        phrase_bytes = 0
        phrase_audio_ms = 0.0
        usage: dict[str, Any] = {}

        async for chunk in clients.stream_tts(text, config.TTS_VOICE, config.TTS_LANGUAGE):
            if chunk["kind"] == "audio":
                now = self.rec.clock.now_ms()
                pcm = chunk["pcm"]
                nbytes = len(pcm)
                dur_ms = config.pcm_bytes_to_ms(nbytes)
                if first_audio_t is None:
                    first_audio_t = now
                phrase_bytes += nbytes
                phrase_audio_ms += dur_ms
                state["turn_audio_ms"] += dur_ms
                self.call_audio_ms += dur_ms
                # Each received PCM delta becomes one audio_out event: the queue
                # of speech to be reproduced on the channel.
                self.rec.record(
                    events.AUDIO_OUT,
                    {
                        "turn_index": idx,
                        "phrase_index": pindex,
                        "chunk_index": chunk_index,
                        "pcm_bytes": nbytes,
                        "duration_ms": round(dur_ms, 3),
                        "turn_audio_ms": round(state["turn_audio_ms"], 3),
                        "call_audio_ms": round(self.call_audio_ms, 3),
                    },
                    turn_id=turn_id,
                    t_ms=now,
                )
                # Forward audio to the channel (browser) for playback.
                await self._emit(
                    {
                        "type": "audio_out",
                        "turn_index": idx,
                        "phrase_index": pindex,
                        "chunk_index": chunk_index,
                        "pcm_bytes": nbytes,
                        "duration_ms": round(dur_ms, 3),
                        "t_ms": round(now, 3),
                        "pcm_b64": _b64(pcm),
                    }
                )
                chunk_index += 1
            elif chunk["kind"] == "usage":
                usage = chunk["usage"]

        done_t = self.rec.clock.now_ms()
        ttfb_ms = (first_audio_t - start_t) if first_audio_t is not None else None
        tts_event = {
            "turn_index": idx,
            "phrase_index": pindex,
            "text": text,
            "chars": len(text),
            "start_t_ms": round(start_t, 3),
            "first_audio_t_ms": round(first_audio_t, 3) if first_audio_t is not None else None,
            "ttfb_ms": round(ttfb_ms, 3) if ttfb_ms is not None else None,
            "done_t_ms": round(done_t, 3),
            "chunks": chunk_index,
            "audio_bytes": phrase_bytes,
            "audio_ms": round(phrase_audio_ms, 3),
            "usage": usage,
        }
        state["tts_events"].append(tts_event)
        self.rec.record(events.TTS_CALL, tts_event, turn_id=turn_id)
        await self._emit({"type": "tts_done", **tts_event})


def _b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")
