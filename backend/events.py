"""Event clock and recorder.

One monotonic clock per call. call_start is the only event with an absolute
wallclock timestamp; every other event carries t_ms relative to that zero.
All timestamps are stamped on the backend so there is a single clock source.
"""
import time
from typing import Any, Optional

from . import db

# Event type constants. The vad_/stt_ types are reserved for the future
# VAD -> turn-selector -> STT front of the pipeline and are not emitted yet.
CALL_START = "call_start"
TEXT_IN = "text_in"
LLM_CALL = "llm_call"
TTS_CALL = "tts_call"
AUDIO_OUT = "audio_out"
CHANNEL_PLAYBACK_START = "channel_playback_start"
CALL_END = "call_end"

# Reserved for later stages:
VAD_START = "vad_start"
VAD_END = "vad_end"
STT_PARTIAL = "stt_partial"
STT_FINAL = "stt_final"


class CallClock:
    """Monotonic clock anchored at call start."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.started_epoch_ms = time.time() * 1000.0

    def now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0


class EventRecorder:
    """Stamps and persists events for a single call."""

    def __init__(self, call_id: str, clock: CallClock) -> None:
        self.call_id = call_id
        self.clock = clock
        self._seq = 0

    def record(
        self,
        etype: str,
        payload: Optional[dict[str, Any]] = None,
        turn_id: Optional[int] = None,
        t_ms: Optional[float] = None,
    ) -> dict[str, Any]:
        """Persist an event and return the serialisable event dict."""
        if t_ms is None:
            t_ms = self.clock.now_ms()
        self._seq += 1
        event = {
            "seq": self._seq,
            "call_id": self.call_id,
            "turn_id": turn_id,
            "type": etype,
            "t_ms": round(t_ms, 3),
            "payload": payload or {},
        }
        db.insert_event(event)
        return event
