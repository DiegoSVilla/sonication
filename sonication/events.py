"""Event clock and recorder.

One monotonic clock per call. call_start is the only event with an absolute
wallclock timestamp; every other event carries t_ms relative to that zero.
All timestamps are stamped on the backend so there is a single clock source.
"""
import time
from typing import Any, Optional

from . import db
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class PipeEvent:
    type: str
    node_name: str
    turn_id: str
    wallclock_ms: float
    local_offset_ms: float
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_event_id: Optional[str] = None
    stage_id: str = ""
    seq: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict suitable for db.log_pipe_event()."""
        return {
            "event_id": self.event_id,
            "type": self.type,
            "node_name": self.node_name,
            "turn_id": self.turn_id,
            "timestamp_wallclock_ms": self.wallclock_ms,
            "timestamp_local_offset": self.local_offset_ms,
            "payload": self.payload,
            "parent_event_id": self.parent_event_id,
            "stage_id": self.stage_id,
            "seq": self.seq,
        }

    @classmethod
    def new(cls, event_type: str, node_name: str, turn_id: str,
            local_offset_ms: float = 0.0,
            payload: Optional[Dict[str, Any]] = None,
            parent_event_id: Optional[str] = None,
            stage_id: str = "",
            seq: int = 0) -> "PipeEvent":
        """Create a PipeEvent with auto-timestamps."""
        return cls(
            type=event_type,
            node_name=node_name,
            turn_id=turn_id,
            wallclock_ms=time.time() * 1000,
            local_offset_ms=local_offset_ms,
            payload=payload or {},
            parent_event_id=parent_event_id,
            stage_id=stage_id,
            seq=seq,
        )

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


# ================== Typed Event Data Models ==================
import uuid as _uuid
from dataclasses import dataclass as _dataclass, field as _field
from typing import Dict as _Dict, Any as _Any, Optional as _Optional


@_dataclass(frozen=True)
class NodeEvent:
    """A typed event from a node with stage context."""
    stage_id: str
    event_type: str
    wallclock_ms: float
    local_offset_ms: float
    seq: int
    payload: _Dict[str, _Any]
    event_id: str = _field(default_factory=lambda: _uuid.uuid4().hex)
    parent_event_id: _Optional[str] = None


@_dataclass(frozen=True)
class PhaseBoundary:
    """A typed phase boundary anchor (wallclock + local offset)."""
    name: str
    wallclock_ms: float
    local_offset_ms: float
    event_id: str


@_dataclass
class NodeStageRecord:
    """Per-invocation record for a node stage in a turn."""
    stage_id: str
    node_name: str
    node_class: str
    config_label: str
    start_wall_ms: float
    end_wall_ms: _Optional[float] = None
    timing: _Dict[str, float] = _field(default_factory=dict)
    events: list = _field(default_factory=list)
    payload_kind: str = "unknown"


@_dataclass(frozen=True)
class InterStageEvent:
    """HotPipe-synthesized event between two stages."""
    event_type: str
    wallclock_ms: float
    local_offset_ms: float = 0.0
    seq: int = 0
    payload: _Dict[str, _Any] = _field(default_factory=dict)
    event_id: str = _field(default_factory=lambda: _uuid.uuid4().hex)
    parent_event_id: _Optional[str] = None
    from_stage_id: _Optional[str] = None
    to_stage_id: _Optional[str] = None
