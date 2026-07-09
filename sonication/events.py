"""Event clock, recorder, and pipeline event models.

One monotonic clock per call. call_start is the only event with an absolute
wallclock timestamp; every other event carries t_ms relative to that zero.
All timestamps are stamped on the backend so there is a single clock source.

New pipeline architecture:
    - PhaseGate: LLM→TTS phrase accumulation
    - EventStream: Central event queue for pipeline nodes
    - StageBoundaries: Typed phase boundary anchors
    - Turn: Complete pipeline run with timing
"""
import asyncio
import time
import logging
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from . import db
from .node_types import NodeConfigLabel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipeEvent:
    """Unified event from any node or pseudo-node in the pipeline.
    
    Every event has a consistent structure regardless of source:
        event_type: "start", "chunk", "token", "response", "done", "usage", "error"
        emitter_node: "stt", "llm", "tts", "phrase_gate"
        emitter_type: "STT_NON_STREAMING", "LLM_STREAMING", etc.
        turn_id, stage_id, wallclock_ms, local_offset_ms, payload, seq
    """
    event_type: str
    emitter_node: str
    emitter_type: str
    turn_id: str
    wallclock_ms: float
    local_offset_ms: float
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_event_id: Optional[str] = None
    stage_id: str = ""
    seq: int = 0

    # Backward-compatible aliases
    @property
    def type(self) -> str:
        return self.event_type

    @property
    def node_name(self) -> str:
        return self.emitter_node

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict suitable for db.log_pipe_event()."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "node_name": self.emitter_node,
            "emitter_type": self.emitter_type,
            "turn_id": self.turn_id,
            "timestamp_wallclock_ms": self.wallclock_ms,
            "timestamp_local_offset": self.local_offset_ms,
            "payload": self.payload,
            "parent_event_id": self.parent_event_id,
            "stage_id": self.stage_id,
            "seq": self.seq,
        }

    @classmethod
    def new(cls, event_type: str, emitter_node: str, emitter_type: str,
            turn_id: str,
            local_offset_ms: float = 0.0,
            payload: Optional[Dict[str, Any]] = None,
            parent_event_id: Optional[str] = None,
            stage_id: str = "",
            seq: int = 0) -> "PipeEvent":
        """Create a PipeEvent with auto-timestamps."""
        return cls(
            event_type=event_type,
            emitter_node=emitter_node,
            emitter_type=emitter_type,
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
    """HotPipe-synthesized event between two stages.
    
    DEPRECATED: Use unified PipeEvent with emitter_node="phrase_gate" instead.
    Kept for backward compatibility during migration.
    """
    event_type: str
    wallclock_ms: float
    local_offset_ms: float = 0.0
    seq: int = 0
    payload: _Dict[str, _Any] = _field(default_factory=dict)
    event_id: str = _field(default_factory=lambda: _uuid.uuid4().hex)
    parent_event_id: _Optional[str] = None
    from_stage_id: _Optional[str] = None
    to_stage_id: _Optional[str] = None
    turn_id: _Optional[str] = None


# ================== Phrase Gate ==================

# Phrase gate config — sentences needed and chars minimum before gating
PHRASE_MIN_CHARS = 20
PHRASE_END_CHARS = {'.', '!', '?'}


def _phrase_ready(text: str) -> bool:
    """Check if accumulated LLM text is ready for phrase gating."""
    if len(text) < PHRASE_MIN_CHARS:
        return False
    stripped = text.rstrip()
    if stripped and stripped[-1] in PHRASE_END_CHARS:
        return True
    return False


class PhaseGate:
    """Inter-stage phrase accumulator for LLM → TTS_CHUNK_IN_STREAM_OUT connections.
    
    Collects LLM tokens, buffers text, and emits complete phrases to a queue
    as soon as they're ready (ASAP, not ALAP). This enables TTS to start
    synthesizing while LLM is still streaming.
    """
    
    def __init__(self, turn: 'Turn', from_stage_id: str = ""):
        self.turn = turn
        self.from_stage_id = from_stage_id
        self.buffer = ""
        self.queue: asyncio.Queue = asyncio.Queue()
        self.phrases_emitted = 0
        self.total_chars = 0
        self.started = False
        self.start_wall_ms: Optional[float] = None
        self.end_wall_ms: Optional[float] = None
    
    async def feed(self, token_text: str) -> None:
        """Feed an LLM token to the phrase gate."""
        self.buffer += token_text
        self.total_chars += len(token_text)
        
        if not self.started:
            self.started = True
            self.start_wall_ms = time.time() * 1000
        
        if _phrase_ready(self.buffer):
            phrase, self.buffer = self._extract_phrase()
            await self.queue.put(phrase)
            self.phrases_emitted += 1
    
    def _extract_phrase(self):
        """Extract the first complete phrase from the buffer."""
        for i in range(PHRASE_MIN_CHARS - 1, len(self.buffer)):
            if self.buffer[i] in PHRASE_END_CHARS:
                phrase = self.buffer[:i+1]
                remaining = self.buffer[i+1:].lstrip()
                return phrase, remaining
        return "", self.buffer
    
    async def close(self) -> None:
        """Signal that LLM is done — flush remaining buffer as final phrase."""
        self.end_wall_ms = time.time() * 1000
        if self.buffer.strip():
            await self.queue.put(self.buffer.strip())
        await self.queue.put(None)
    
    def get_stats(self) -> dict:
        """Return phrase gate statistics."""
        return {
            "phrases_emitted": self.phrases_emitted,
            "total_chars": self.total_chars,
            "buffer_remaining": len(self.buffer),
            "started": self.started,
            "start_wall_ms": self.start_wall_ms,
            "end_wall_ms": self.end_wall_ms,
        }


# ================== Event Stream ==================

class EventStream:
    """Central event stream for all pipeline nodes.
    
    All nodes push events to this shared queue as they happen.
    The main scheduler pulls from the queue and yields events in order.
    """
    
    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._closed = False
    
    async def put(self, event: dict) -> None:
        """Push an event to the stream."""
        if not self._closed:
            logger.debug(f"[EventStream] event_type={event.get('event_type')} emitter={event.get('emitter_node')}")
            await self._queue.put(event)
    
    async def __aiter__(self):
        """Iterate over events from the stream."""
        while True:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                if self._closed and self._queue.empty():
                    break
                continue
    
    def close(self) -> None:
        """Signal that no more events will be pushed."""
        self._closed = True


# ================== Stage Boundaries ==================

class StageBoundaries:
    """Typed phase boundary anchors for a turn."""

    def __init__(self):
        self.t_stt_req: Optional[PhaseBoundary] = None
        self.t_stt_resp: Optional[PhaseBoundary] = None
        self.t_llm_req: Optional[PhaseBoundary] = None
        self.t_llm_ttft: Optional[PhaseBoundary] = None
        self.t_llm_resp: Optional[PhaseBoundary] = None
        self.t_tts_req: Optional[PhaseBoundary] = None
        self.t_tts_ttfb: Optional[PhaseBoundary] = None
        self.t_tts_resp: Optional[PhaseBoundary] = None


# ================== Turn ==================

class Turn:
    """One complete pipeline run with timing.

    Captures all events, records timing checkpoints, and extracts
    final results (transcript, LLM response, audio).
    """

    def __init__(self, turn_id: str, start_wall_ms: float, start_mono_ms: float,
                 pipeline_type: str = "manual"):
        self.turn_id = turn_id
        self.start_wall = start_wall_ms
        self.start_mono = start_mono_ms
        self.event_stream = EventStream()
        self.boundaries = StageBoundaries()
        self.node_stages: Dict[str, NodeStageRecord] = {}
        self.inter_stage_events: List[InterStageEvent] = []
        self.events: List[PipeEvent] = []
        # Legacy float fields (preserved for compat)
        self.stt_start_ms = 0.0
        self.stt_done_ms = 0.0
        self.llm_start_ms = 0.0
        self.llm_ttft_ms = 0.0
        self.llm_done_ms = 0.0
        self.tts_start_ms = 0.0
        self.tts_ttfb_ms = 0.0
        self.tts_done_ms = 0.0
        self.stt_text = ""
        self.llm_text = ""
        self.tts_audio = b""
        self._pipeline_type = pipeline_type
        self._event_seq = 0
        self._log_manager = None

    def _now(self) -> float:
        return (time.monotonic() - self.start_mono) * 1000.0

    def record_node_stage(self, record: NodeStageRecord) -> None:
        """Register/update a node stage record."""
        self.node_stages[record.stage_id] = record

    def record_inter_stage_event(self, event: InterStageEvent) -> None:
        """Register an inter-stage event."""
        self.inter_stage_events.append(event)

    def boundary(self, name: str) -> Optional[PhaseBoundary]:
        """Return the PhaseBoundary for a named anchor."""
        return getattr(self.boundaries, name, None)

    def interval(self, start_name: str, end_name: str) -> Optional[float]:
        """Compute ms between two phase anchors."""
        start = self.boundary(start_name)
        end = self.boundary(end_name)
        if start and end:
            return round(end.wallclock_ms - start.wallclock_ms, 3)
        return None

    def shot_latency_ms(self) -> float:
        """Pipeline-type-aware shot latency."""
        if self.boundaries.t_stt_req and self.boundaries.t_tts_req:
            return round(self.boundaries.t_tts_req.wallclock_ms - self.boundaries.t_stt_req.wallclock_ms, 3)
        if self.tts_ttfb_ms and self.stt_start_ms:
            return self.tts_ttfb_ms - self.stt_start_ms
        return 0.0

    def analyse(self):
        """Delegate to LatencyAnalyser."""
        from .analysis import LatencyAnalyser
        return LatencyAnalyser.analyse(self)

    def _detect_entry_point(self, pipe) -> str:
        """Auto-detect the entry node from pipeline topology."""
        slots = pipe._pipeline_topology.get("slots", [])
        if slots:
            return slots[0]["slot_name"]
        for conn in pipe._pipeline_topology.get("connections", []):
            return conn[0]
        return "stt"

    def _record_phase_boundary(self, stage_record, raw, config_label):
        """Record phase boundaries as events arrive from streaming nodes."""
        now_ms = time.time() * 1000
        events = self.events
        kind = raw.get("kind")

        if config_label == NodeConfigLabel.STT_NON_STREAMING:
            if kind == "done" and self.boundaries.t_stt_resp is None:
                pb = PhaseBoundary(
                    name="t_stt_resp",
                    wallclock_ms=now_ms,
                    local_offset_ms=self._now(),
                    event_id=events[-1].event_id if events else "",
                )
                self.boundaries.t_stt_resp = pb
                stage_record.timing["t_stt_resp"] = now_ms
                self.stt_start_ms = self._now()

        elif config_label in (NodeConfigLabel.LLM_STREAMING, NodeConfigLabel.LLM_STREAMING_WITH_REASONING):
            if kind == "token" and self.boundaries.t_llm_req is None:
                pb = PhaseBoundary(
                    name="t_llm_req",
                    wallclock_ms=now_ms,
                    local_offset_ms=self._now(),
                    event_id=events[-1].event_id if events else "",
                )
                self.boundaries.t_llm_req = pb
                stage_record.timing["t_llm_req"] = now_ms
            if kind == "token" and self.boundaries.t_llm_ttft is None:
                pb = PhaseBoundary(
                    name="t_llm_ttft",
                    wallclock_ms=now_ms,
                    local_offset_ms=self._now(),
                    event_id=events[-1].event_id if events else "",
                )
                self.boundaries.t_llm_ttft = pb
                stage_record.timing["t_llm_ttft"] = now_ms
                self.llm_ttft_ms = self._now()
            if kind == "done" and self.boundaries.t_llm_resp is None:
                pb = PhaseBoundary(
                    name="t_llm_resp",
                    wallclock_ms=now_ms,
                    local_offset_ms=self._now(),
                    event_id=events[-1].event_id if events else "",
                )
                self.boundaries.t_llm_resp = pb
                stage_record.timing["t_llm_resp"] = now_ms

        elif config_label in (NodeConfigLabel.TTS_CHUNK_IN_STREAM_OUT, NodeConfigLabel.TTS_NON_STREAMING):
            if kind == "audio" and self.boundaries.t_tts_req is None:
                pb = PhaseBoundary(
                    name="t_tts_req",
                    wallclock_ms=now_ms,
                    local_offset_ms=self._now(),
                    event_id=events[-1].event_id if events else "",
                )
                self.boundaries.t_tts_req = pb
                stage_record.timing["t_tts_req"] = now_ms
            if kind == "audio" and self.boundaries.t_tts_ttfb is None:
                pb = PhaseBoundary(
                    name="t_tts_ttfb",
                    wallclock_ms=now_ms,
                    local_offset_ms=self._now(),
                    event_id=events[-1].event_id if events else "",
                )
                self.boundaries.t_tts_ttfb = pb
                stage_record.timing["t_tts_ttfb"] = now_ms
                self.tts_start_ms = self._now()
            if kind == "done" and self.boundaries.t_tts_resp is None:
                pb = PhaseBoundary(
                    name="t_tts_resp",
                    wallclock_ms=now_ms,
                    local_offset_ms=self._now(),
                    event_id=events[-1].event_id if events else "",
                )
                self.boundaries.t_tts_resp = pb
                stage_record.timing["t_tts_resp"] = now_ms

    def _record_end_boundary(self, stage_record, config_label, start_wc):
        """Record end-phase boundaries after node stream completes."""
        now_ms = time.time() * 1000
        if config_label == NodeConfigLabel.STT_NON_STREAMING and self.boundaries.t_stt_resp is None:
            pb = PhaseBoundary(
                name="t_stt_resp",
                wallclock_ms=now_ms,
                local_offset_ms=self._now(),
                event_id=self.events[-1].event_id if self.events else "",
            )
            self.boundaries.t_stt_resp = pb
            stage_record.timing["t_stt_resp"] = now_ms
            self.stt_done_ms = self._now()
        elif config_label in (NodeConfigLabel.LLM_STREAMING, NodeConfigLabel.LLM_STREAMING_WITH_REASONING) and self.boundaries.t_llm_resp is None:
            pb = PhaseBoundary(
                name="t_llm_resp",
                wallclock_ms=now_ms,
                local_offset_ms=self._now(),
                event_id=self.events[-1].event_id if self.events else "",
            )
            self.boundaries.t_llm_resp = pb
            stage_record.timing["t_llm_resp"] = now_ms
            self.llm_done_ms = self._now()
        elif config_label in (NodeConfigLabel.TTS_CHUNK_IN_STREAM_OUT, NodeConfigLabel.TTS_NON_STREAMING) and self.boundaries.t_tts_resp is None:
            pb = PhaseBoundary(
                name="t_tts_resp",
                wallclock_ms=now_ms,
                local_offset_ms=self._now(),
                event_id=self.events[-1].event_id if self.events else "",
            )
            self.boundaries.t_tts_resp = pb
            stage_record.timing["t_tts_resp"] = now_ms
            self.tts_done_ms = self._now()

    def _next_data(self, current, last_event):
        if current == "stt":
            return self.stt_text
        if current == "llm":
            return self.llm_text
        return None

    def _next_seq(self) -> int:
        """Return an incrementing sequence number."""
        self._event_seq += 1
        return self._event_seq

    async def run(self, pipe, data):
        """Execute the pipeline from the entry node (auto-detected from topology)."""
        async for event in self.run_events(pipe, data):
            pass
        return {
            "stt_text": self.stt_text,
            "llm_response": self.llm_text,
            "tts_audio": self.tts_audio,
            "shot_latency_ms": self.shot_latency_ms(),
            "stt_ms": self.stt_done_ms - self.stt_start_ms,
            "llm_ttft_ms": self.llm_ttft_ms,
            "tts_ttfb_ms": self.tts_ttfb_ms,
        }

    async def run_events(self, pipe, data):
        """Execute the pipeline and yield all events with timestamps in real-time."""
        self._log_manager = pipe._log_manager
        from .scheduler import PipelineScheduler
        scheduler = PipelineScheduler(self, pipe, data)
        async for event in scheduler.run():
            yield event