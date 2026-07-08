"""Minimal voice call.

A VoiceCall owns the call clock, event recorder, and the STT -> LLM -> TTS
pipeline. It handles text-in -> LLM -> TTS -> audio-out minimally.
"""
import asyncio
import uuid
from datetime import datetime
from typing import Any, Optional

from . import config, events
from .pipeline import CallPipeline, SendFn


class VoiceCall:
    def __init__(self, session_id: str, send: Optional[SendFn] = None) -> None:
        self.call_id = uuid.uuid4().hex
        self.session_id = session_id
        self.clock = events.CallClock()
        self.rec = events.EventRecorder(self.call_id, self.clock)
        self.pipeline = CallPipeline(self.call_id, self.rec, send=send)

    def start(self) -> tuple[datetime, dict[str, Any]]:
        """Record call_start and return call parameters."""
        now = datetime.fromtimestamp(self.clock.started_epoch_ms / 1000.0)
        params = {
            "model": config.LLM_MODEL,
            "voice": config.TTS_VOICE,
            "language": config.TTS_LANGUAGE,
            "temperature": config.LLM_TEMPERATURE,
            "seed": config.LLM_SEED,
            "max_tokens": config.LLM_MAX_TOKENS,
            "phrase_min_chars": config.PHRASE_MIN_CHARS,
            "audio_bytes_per_sec": config.AUDIO_BYTES_PER_SEC,
        }
        self.rec.record(
            events.CALL_START,
            {"wallclock_iso": now.isoformat(), "epoch_ms": self.clock.started_epoch_ms,
             "params": params},
            t_ms=0.0,
        )
        return now, params

    async def end(self) -> None:
        """End the call, record call_end, and close the HTTP client."""
        self.rec.record(events.CALL_END, {}, t_ms=self.clock.now_ms())
        await self.pipeline.close()


