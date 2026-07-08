"""HotPipe: orchestrator for multi-stage voice pipelines.

Manages node lifecycle, keepalive connections, turn scheduling and event
logging. Designed to minimize the wall-clock gap from user utterance to
first TTS audio byte (shot latency P50 < 0.7s).

Pipeline-first architecture:
    HotPipe(pipeline_type=PipelineType.X) defines topology.
    add_node(node) auto-detects slot by node type (STTNode/LLMNode/TTSNode).
    connect() is a validation trigger — wiring is 100% topology-driven.
"""
import asyncio
import httpx
import time
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Any

from .events import (
    PipeEvent, NodeEvent, PhaseBoundary, NodeStageRecord, InterStageEvent,
)
from .db import log_pipe_event
from .node_types import PipelineType, NodeConfigLabel

logger = logging.getLogger(__name__)

PING_INTERVAL = 5.0

# Phrase gate config — sentences needed and chars minimum before gating
PHRASE_MIN_CHARS = 20
PHRASE_END_CHARS = {'.', '!', '?'}


def _phrase_ready(text: str) -> bool:
    """Check if accumulated LLM text is ready for phrase gating."""
    if len(text) < PHRASE_MIN_CHARS:
        return False
    # Check if last character is a phrase-ending character
    stripped = text.rstrip()
    if stripped and stripped[-1] in PHRASE_END_CHARS:
        return True
    return False


class PingLoop:
    """Background keepalive ping loop.

    Periodically sends a lightweight /health request to each node to keep
    pooled HTTP connections warm. Matches minimalVoice's EndpointMonitor.
    """

    def __init__(self, nodes: dict, nodes_health: Optional[dict] = None):
        self._nodes = nodes
        self._nodes_health = nodes_health or {}
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self):
        while True:
            for name, (node, _) in self._nodes.items():
                try:
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(connect=5.0, read=10.0)
                    ) as c:
                        r = await c.get(f"{node.base_url}/health")
                        if self._nodes_health:
                            rtt_ms = r.elapsed.total_seconds() * 1000
                            self._nodes_health.setdefault(name, []).append(rtt_ms)
                except (Exception, asyncio.TimeoutError):
                    pass
            await asyncio.sleep(PING_INTERVAL)


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


class Turn:
    """One complete pipeline run with timing.

    Captures all events, records timing checkpoints, and extracts
    final results (transcript, LLM response, audio).
    
    New architecture:
        Turn.boundaries — typed PhaseBoundary anchors
        Turn.node_stages — per-invocation stage records
        Turn.inter_stage_events — HotPipe-synthesized events (phrase gate)
        Turn.analyse() — delegates to LatencyAnalyser
    """

    def __init__(self, turn_id: str, start_wall_ms: float, start_mono_ms: float,
                 pipeline_type: str = "manual"):
        self.turn_id = turn_id
        self.start_wall = start_wall_ms
        self.start_mono = start_mono_ms
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
        # Use PhaseBoundary with legacy fallback
        if self.boundaries.t_stt_req and self.boundaries.t_tts_req:
            return round(self.boundaries.t_tts_req.wallclock_ms - self.boundaries.t_stt_req.wallclock_ms, 3)
        # Legacy fallback
        if self.tts_ttfb_ms and self.stt_start_ms:
            return self.tts_ttfb_ms - self.stt_start_ms
        return 0.0

    def analyse(self):
        """Delegate to LatencyAnalyser."""
        from .analysis import LatencyAnalyser
        return LatencyAnalyser.analyse(self)

    async def run(self, pipe, data):
        """Execute the pipeline from the entry node (auto-detected from topology)."""
        # Store log_manager reference for _enqueue_event
        self._log_manager = pipe._log_manager
        
        # Auto-detect entry point from topology
        entry_point = self._detect_entry_point(pipe)
        
        self.events.append(PipeEvent.new(
            "turn_start", entry_point, self.turn_id, local_offset_ms=0.0))

        # Also record pipeline_start as inter-stage event
        self.inter_stage_events.append(InterStageEvent(
            event_type="turn_start",
            wallclock_ms=time.time() * 1000,
            local_offset_ms=self._now(),
            payload={"pipeline_type": self._pipeline_type, "entry_node": entry_point}
        ))
        if self._log_manager:
            self._log_manager.enqueue({
                "_log_kind": "inter_stage",
                **self.inter_stage_events[-1].__dict__,
            })
        
        try:
            await self._walk(pipe, entry_point, data)
        except Exception as e:
            self.events.append(PipeEvent.new(
                "turn_error", entry_point, self.turn_id,
                payload={"error": str(e)}))
            raise

        return {
            "stt_text": self.stt_text,
            "llm_response": self.llm_text,
            "tts_audio": self.tts_audio,
            "shot_latency_ms": self.shot_latency_ms(),
            "stt_ms": self.stt_done_ms - self.stt_start_ms,
            "llm_ttft_ms": self.llm_ttft_ms,
            "tts_ttfb_ms": self.tts_ttfb_ms,
        }

    def _detect_entry_point(self, pipe) -> str:
        """Auto-detect the entry node from pipeline topology."""
        slots = pipe._pipeline_topology.get("slots", [])
        if slots:
            return slots[0]["slot_name"]
        # Fallback: first node in topology connections
        for conn in pipe._pipeline_topology.get("connections", []):
            return conn[0]
        return "stt"  # Default fallback

    async def _walk(self, pipe, node_name, data, parent_id=None):
        """Run a node, collect events, follow downstream connections,
        record phase boundaries, and handle phrase gating for LLM->TTS."""
        node, category = pipe.nodes[node_name]
        config_label = node.config_label if hasattr(node, 'config_label') else category
        
        # Generate stage_id and assign to node
        stage_id = pipe._generate_stage_id(node, node_name)
        node._stage_id = stage_id  # assign for later reference

        # Create NodeStageRecord
        stage_record = NodeStageRecord(
            stage_id=stage_id,
            node_name=node_name,
            node_class=node.node_class,
            config_label=config_label,
            start_wall_ms=time.time() * 1000,
            end_wall_ms=None,
            timing={},
            events=[],
            payload_kind="unknown",
        )
        self.record_node_stage(stage_record)

        # Track streaming state for LLM->TTS phrase gate
        llm_text_buffer = ""
        
        # Track TTS audio chunks
        tts_audio_chunks = []

        # Record node start event
        seq = self._next_seq()
        start_event = PipeEvent.new(
            f"{node_name}_start", node_name, self.turn_id,
            local_offset_ms=self._now(),
            payload={"category": category, "config_label": config_label},
            stage_id=stage_id,
            seq=seq
        )
        self.events.append(start_event)
        self._enqueue_event(start_event.to_dict())
        last_event_event_id = start_event.event_id

        # Record phase boundary for start (t_start = t_first_event)
        node_start_wc = time.time() * 1000

        last_event_type = None
        last_event_data = None
        try:
            async for raw in node.stream(data):
                # Log event with stage context
                await self._log_event(raw, node_name, parent_id, stage_id=stage_id, seq=seq)
                self._event_seq += 1
                last_event_event_id = self.events[-1].event_id
                last_event_type = raw.get("kind")
                last_event_data = raw

                # Record phase boundaries
                self._record_phase_boundary(stage_record, raw, node_name, config_label)

                # Accumulate LLM text for phrase gate
                if "content" in raw and node_name == "llm":
                    llm_text_buffer += raw.get("content", "")

                # Capture TTS audio chunks
                if node_name == "tts" and raw.get("kind") == "audio":
                    pcm_data = raw.get("pcm", b"")
                    if pcm_data:
                        tts_audio_chunks.append(pcm_data)

                # Capture STT transcript while iterating (last_event_type
                # will be "done" after the loop, not "transcript")
                if node_name == "stt" and raw.get("kind") == "transcript":
                    self.stt_text = raw.get("text", "")

                # Check phrase gate readiness
                if node_name == "llm" and llm_text_buffer and config_label in (NodeConfigLabel.LLM_STREAMING, NodeConfigLabel.LLM_STREAMING_WITH_REASONING):
                    downstream_tts = any(
                        conn[0] == node_name
                        and pipe.nodes.get(conn[1], ())[1] == NodeConfigLabel.TTS_CHUNK_IN_STREAM_OUT
                        for conn in pipe._pipeline_topology.get("connections", [])
                    )
                    if downstream_tts and _phrase_ready(llm_text_buffer):
                        self._emit_phrase_gate_event(node_name, stage_id, llm_text_buffer)

            # After stream completes — record end boundaries
            self._record_end_boundary(stage_record, node_name, node_start_wc)

            # Store accumulated LLM text for downstream stages
            if node_name == "llm":
                self.llm_text = llm_text_buffer
                # Complete the turn in LLMNode's internal history
                stt_input = self.stt_text
                if hasattr(node, 'complete_turn') and stt_input:
                    node.complete_turn(stt_input, llm_text_buffer)

            # Store TTS audio
            if node_name == "tts" and tts_audio_chunks:
                self.tts_audio = b"".join(tts_audio_chunks)

            # Update node data for legacy compatibility
            last_stored_event = last_event_data

        except Exception as e:
            self.events.append(PipeEvent.new(
                f"{node_name}_error", node_name, self.turn_id,
                payload={"error": str(e)}, parent_event_id=parent_id))
            raise
        finally:
            stage_record.end_wall_ms = time.time() * 1000
            self.record_node_stage(stage_record)

        # Legacy data extraction (preserved for compat)
        if node_name == "tts":
            pass  # TTS audio stored in _node_data by HotPipe

        # Follow downstream connections
        for next_name in pipe.connections.get(node_name, []):
            next_data = self._next_data(node_name, last_event_type)
            if next_data is not None:
                await self._walk(pipe, next_name, next_data,
                                 parent_id=last_event_event_id)

    def _record_phase_boundary(self, stage_record, raw, node_name, config_label):
        """Record phase boundaries as events arrive from streaming nodes."""
        now_ms = time.time() * 1000
        events = self.events
        kind = raw.get("kind")
        seq = self._event_seq

        if node_name == "stt":
            if kind == "transcript" and self.boundaries.t_stt_req is None:
                pb = PhaseBoundary(
                    name="t_stt_req",
                    wallclock_ms=now_ms,
                    local_offset_ms=self._now(),
                    event_id=events[-1].event_id if events else "",
                )
                self.boundaries.t_stt_req = pb
                stage_record.timing["t_stt_req"] = now_ms
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

        elif node_name == "llm":
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

        elif node_name == "tts":
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

    def _record_end_boundary(self, stage_record, node_name, start_wc):
        """Record end-phase boundaries after node stream completes."""
        now_ms = time.time() * 1000
        if node_name == "stt" and self.boundaries.t_stt_resp is None:
            pb = PhaseBoundary(
                name="t_stt_resp",
                wallclock_ms=now_ms,
                local_offset_ms=self._now(),
                event_id=self.events[-1].event_id if self.events else "",
            )
            self.boundaries.t_stt_resp = pb
            stage_record.timing["t_stt_resp"] = now_ms
            self.stt_done_ms = self._now()
        elif node_name == "llm" and self.boundaries.t_llm_resp is None:
            pb = PhaseBoundary(
                name="t_llm_resp",
                wallclock_ms=now_ms,
                local_offset_ms=self._now(),
                event_id=self.events[-1].event_id if self.events else "",
            )
            self.boundaries.t_llm_resp = pb
            stage_record.timing["t_llm_resp"] = now_ms
            self.llm_done_ms = self._now()
        elif node_name == "tts" and self.boundaries.t_tts_resp is None:
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

    async def _log_event(self, raw, node_name, parent_id=None, stage_id="", seq=0):
        """Log a node event to events list and queue (or skip if no log_manager)."""
        kind = raw.get("kind", "unknown")
        if node_name == "stt":
            etype = {"transcript": "stt_transcript", "error": f"stt_{kind}",
                     "done": "stt_done"}.get(kind, f"stt_{kind}")
        elif node_name == "llm":
            etype = {"token": "llm_token", "reasoning": "llm_reasoning",
                     "error": f"llm_{kind}", "done": "llm_done",
                     "usage": "llm_usage"}.get(kind, f"llm_{kind}")
        elif node_name == "tts":
            etype = {"audio": "tts_audio_chunk", "usage": "tts_usage",
                     "error": f"tts_{kind}", "done": "tts_done"}.get(kind,
                     f"tts_{kind}")
        else:
            etype = f"{node_name}_{kind}"

        pe = PipeEvent.new(etype, node_name, self.turn_id,
                           local_offset_ms=self._now(), payload=raw,
                           parent_event_id=parent_id,
                           stage_id=stage_id, seq=seq)
        self.events.append(pe)
        self._enqueue_event(pe.to_dict())

    def _enqueue_event(self, event_dict: dict) -> None:
        """Enqueue an event to the log_manager, or skip if disabled."""
        # Get log_manager from pipe (set by HotPipe.connect())
        # We store it on the Turn via run() if available
        if hasattr(self, '_log_manager') and self._log_manager:
            try:
                self._log_manager.enqueue(event_dict)
            except RuntimeError:
                pass  # queue full — don't block the pipeline

    def _next_seq(self) -> int:
        """Return an incrementing sequence number."""
        self._event_seq += 1
        return self._event_seq

    def _emit_phrase_gate_event(self, from_node_name, from_stage_id, buffered_text):
        """Emit an InterStageEvent for a phrase gate between LLM and TTS."""
        ts = time.time() * 1000
        now = self._now()

        self.inter_stage_events.append(InterStageEvent(
            event_type="phrase_gate",
            wallclock_ms=ts,
            local_offset_ms=now,
            seq=self._next_seq(),
            payload={
                "warming_up": len(buffered_text) < 100,
                "sentences": max(1, len([c for c in buffered_text if c in ".!?"])),
                "has_phrase_ready": True,
                "buffered_text": buffered_text[:200],
            }
        ))
        # Enqueue via log_manager if available
        if self._log_manager:
            self._log_manager.enqueue({
                "_log_kind": "inter_stage",
                **self.inter_stage_events[-1].__dict__,
            })


def stream_sync(async_iter_factory, *args, **kwargs):
    """Run async generator synchronously, yielding results."""
    import asyncio
    async def _collect():
        results = []
        async for raw in async_iter_factory(*args, **kwargs):
            results.append(raw)
        return results
    return asyncio.run(_collect())


class HotPipe:
    """Orchestrates multi-stage voice pipelines.

    Pipeline-first architecture:
        - pipeline_type REQUIRED at init
        - add_node(node) auto-detects slot by node type
        - connect() validates topology — wiring is 100% topology-driven
    
    Node whitelist: STTNode, LLMNode, TTSNode only.
    
    Logging:
        Pass a LogManager instance to enable event logging.
        If log_manager is None, all logging is disabled.
    """
    
    # Topology definitions per PipelineType
    _TOPOLOGY = {
        PipelineType.SI_SO_THREE_STEP_PIPELINE_CHAT: {
            "slots": [
                {"slot_name": "stt", "node_class": "STTNode",
                 "config_label": NodeConfigLabel.STT_NON_STREAMING},
                {"slot_name": "llm", "node_class": "LLMNode",
                 "config_label": NodeConfigLabel.LLM_STREAMING},
                {"slot_name": "tts", "node_class": "TTSNode",
                 "config_label": NodeConfigLabel.TTS_CHUNK_IN_STREAM_OUT},
            ],
            "connections": [("stt", "llm"), ("llm", "tts")],
        },
        PipelineType.TI_SO_TWO_STEP_PIPELINE_CHAT: {
            "slots": [
                {"slot_name": "llm", "node_class": "LLMNode",
                 "config_label": NodeConfigLabel.LLM_STREAMING},
                {"slot_name": "tts", "node_class": "TTSNode",
                 "config_label": NodeConfigLabel.TTS_CHUNK_IN_STREAM_OUT},
            ],
            "connections": [("llm", "tts")],
        },
        PipelineType.SI_TO_ONE_STEP_PIPELINE_TRANSCRIBE: {
            "slots": [
                {"slot_name": "stt", "node_class": "STTNode",
                 "config_label": NodeConfigLabel.STT_NON_STREAMING},
            ],
            "connections": [],
        },
        PipelineType.TI_SO_ONE_STEP_PIPELINE_SPEAK: {
            "slots": [
                {"slot_name": "tts", "node_class": "TTSNode",
                 "config_label": NodeConfigLabel.TTS_CHUNK_IN_STREAM_OUT},
            ],
            "connections": [],
        },
        PipelineType.SI_SO_THREE_STEP_PIPELINE_TRANSLATE: {
            "slots": [
                {"slot_name": "stt", "node_class": "STTNode",
                 "config_label": NodeConfigLabel.STT_NON_STREAMING},
                {"slot_name": "llm", "node_class": "LLMNode",
                 "config_label": NodeConfigLabel.LLM_STREAMING},
                {"slot_name": "tts", "node_class": "TTSNode",
                 "config_label": NodeConfigLabel.TTS_CHUNK_IN_STREAM_OUT},
            ],
            "connections": [("stt", "llm"), ("llm", "tts")],
            "translation": True,
        },
        PipelineType.SI_SO_THREE_STEP_PIPELINE_AGENT: {
            "slots": [
                {"slot_name": "stt", "node_class": "STTNode",
                 "config_label": NodeConfigLabel.STT_NON_STREAMING},
                {"slot_name": "llm", "node_class": "LLMNode",
                 "config_label": NodeConfigLabel.LLM_STREAMING_WITH_REASONING},
                {"slot_name": "tts", "node_class": "TTSNode",
                 "config_label": NodeConfigLabel.TTS_CHUNK_IN_STREAM_OUT},
            ],
            "connections": [("stt", "llm"), ("llm", "tts")],
        },
    }

    def __init__(self, pipeline_type: PipelineType, log_manager=None):
        """Pipeline type is REQUIRED at init.
        
        Args:
            pipeline_type: The pipeline topology to use.
            log_manager: Optional LogManager instance for event logging.
                        If None, all logging is disabled.
        """
        if not pipeline_type:
            raise ValueError(
                "HotPipe requires pipeline_type at init. Got None.")

        self.pipeline_type = pipeline_type
        self._log_manager = log_manager
        
        # _slots maps slot_name -> Node instance
        self._slots: Dict[str, Any] = {}
        
        # _node_types tracks which node classes have been added
        self._node_types: set = set()
        
        # Backward compat: HotPipe.nodes = name -> (Node, category)
        self.nodes = {}
        
        # Backward compat: HotPipe.connections = name -> [to_name...]
        self.connections = {}
        
        # Track topology-driven data
        self._pipeline_topology = self._TOPOLOGY.get(pipeline_type)
        if not self._pipeline_topology:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        self._unconnected_nodes: List[Any] = []  # actual node objects, resolved on connect()
        self._node_data = {}  # name -> list of raw events
        self._nodes_health = {}  # name -> [rtt_ms ...]
        self._ping_loop = None
        self._last_turn = None  # last Turn object for analysis access
        self._stage_counter: Dict[str, int] = {}  # config_label -> count

    def add_node(self, node: Any) -> None:
        """Auto-detect pipeline slot based on node type.
        
        Strategy:
        - Look up which slot(s) in the current pipeline_type topology
          this node class maps to.
        - If 0 slots: error "No slot for Node type X on pipeline Y".
        - If 1 slot: auto-fill it.
        - If 2+ slots: error "Ambiguous node — fits slots [A, B]".
        
        Only accept STTNode, LLMNode, TTSNode as node types.
        Custom classes/subclasses NOT ALLOWED.
        
        Connections are NOT established here. They are resolved in connect().
        """
        from .nodes import Node
        
        node_class = node.__class__.__name__
        
        # Whitelisted node types only
        allowed_types = ("STTNode", "LLMNode", "TTSNode")
        if node_class not in allowed_types:
            raise ValueError(
                f"Unhandled node type: '{node_class}'. "
                f"Only {allowed_types} allowed."
            )
        
        if not isinstance(node, Node):
            raise ValueError(f"Node type '{node_class}' is not a valid Node subclass.")
        
        self._node_types.add(node_class)
        
        # Look up topology slot(s) for this node class
        candidates = [
            s for s in self._pipeline_topology["slots"]
            if s["node_class"] == node_class and s["slot_name"] not in self._slots
        ]
        
        if not candidates:
            # Check if already added
            if node_class in [
                s["node_class"] for s in self._pipeline_topology["slots"]
                if s["slot_name"] in self._slots
            ]:
                raise ValueError(
                    f"No slot for Node type '{node_class}': "
                    f"already added to pipeline."
                )
            else:
                raise ValueError(
                    f"No slot for Node type '{node_class}' "
                    f"on pipeline '{self.pipeline_type.value}'."
                )
        
        elif len(candidates) > 1:
            slot_names = [c["slot_name"] for c in candidates]
            raise ValueError(
                f"Ambiguous node class: fits slots {slot_names}. "
                f"Pipeline '{self.pipeline_type.value}' has both, "
                f"and user can't decide."
            )
        
        else:
            slot = candidates[0]
            self._slots[slot["slot_name"]] = node
            # Backward compat: add to nodes dict
            slot_name = slot["slot_name"]
            self.nodes[slot_name] = (node, slot["config_label"])
            self.connections[slot_name] = []
            self._unconnected_nodes.append(node)
            config_label = slot["config_label"]
            
            # Assign config_label to node
            if hasattr(node, '_config_label'):
                node._config_label = config_label
            
            logger.debug(f"Added {node_class} to slot {slot_name} (unconnected)")

    def connect(self) -> bool:
        """Validate all nodes are connected and resolve topology connections.
        
        This is where wiring happens — connections are resolved from the
        pipeline topology. Before connect(), nodes are registered but not
        wired together. After connect(), connections are fully established.
        
        Validates:
        - Exactly the right number of nodes were added
        - No extra nodes beyond what the topology needs
        - All topology connections can be resolved
        
        Also passes the log_manager to each node if one is set on the pipe.
        """
        expected_slots = len(self._pipeline_topology["slots"])
        actual_nodes = len(self._slots)
        
        if actual_nodes > expected_slots:
            extra = [n for n in self._unconnected_nodes if n not in self._slots.values()]
            raise ValueError(
                f"Too many nodes: added {actual_nodes}, expected {expected_slots}. "
                f"Extra nodes: {[n.__class__.__name__ for n in extra]}"
            )
        
        if actual_nodes < expected_slots:
            missing = [s["slot_name"] for s in self._pipeline_topology["slots"]
                       if s["slot_name"] not in self._slots]
            raise ValueError(
                f"Not enough nodes: added {actual_nodes}, expected {expected_slots}. "
                f"Missing slots: {missing}"
            )
        
        # Resolve all connections from topology
        for src_name, dst_name in self._pipeline_topology["connections"]:
            if src_name in self.connections and dst_name in self.connections:
                if dst_name not in self.connections[src_name]:
                    self.connections[src_name].append(dst_name)
                    logger.debug(f"Connected {src_name} -> {dst_name}")
            else:
                raise ValueError(
                    f"Cannot connect {src_name} -> {dst_name}: "
                    f"one or both slots missing from topology"
                )
        
        # Clear unconnected nodes list — all resolved
        self._unconnected_nodes.clear()
        
        # Pass log_manager to all nodes
        if self._log_manager is not None:
            for node, _ in self.nodes.values():
                if hasattr(node, 'log_manager'):
                    node.log_manager = self._log_manager
        
        return True

    def _generate_stage_id(self, node: object, node_name: str) -> str:
        """Generate a deterministic stage_id for a node invocation."""
        config = node.config_label
        label = config if hasattr(config, 'value') else config
        
        # Get a counter for this node_name
        if node_name not in self._stage_counter:
            self._stage_counter[node_name] = 0
        self._stage_counter[node_name] += 1
        step = self._stage_counter[node_name]
        
        # Format: NODENAME_LOCAL_CONFIGLABEL_STEP
        node_id = str(uuid.uuid4())[:8]
        return f"{node_name}_{node_id}_{label}_{step}"

    async def start(self):
        """Start the HotPipe and begin background keepalive pings."""
        self._ping_loop = PingLoop(self.nodes, self._nodes_health)
        await self._ping_loop.start()

    async def stop(self):
        """Stop the HotPipe and cancel background pings."""
        if self._ping_loop:
            await self._ping_loop.stop()

    async def warmup(self):
        """Ping all nodes to establish keepalive connections."""
        results = []
        for node, _ in self.nodes.values():
            results.append(await node.warmup())
        warm = sum(1 for r in results if r)
        logger.info(f"Warmup: {warm}/{len(self.nodes)} nodes warm")
        return results

    async def close(self):
        for node, _ in self.nodes.values():
            await node.close()

    def get_stats(self):
        return {
            "nodes": list(self.nodes.keys()),
            "connections": dict(self.connections),
            "node_warming": {n: node.is_warm for n, (node, _) in self.nodes.items()},
            "health_sample": {n: (round(history[-1]) if history else None)
                              for n, history in self._nodes_health.items()},
        }

    async def turn(self, entry_point: str, data, pipeline_type: str = "manual"):
        """Execute one turn. Returns dict with results."""
        turn_id = f"turn_{round(time.time() * 1000)}"
        start_wall = time.time() * 1000
        turn = Turn(turn_id, start_wall, start_wall,
                    pipeline_type=self.pipeline_type.value if self.pipeline_type else pipeline_type)
        try:
            result = await turn.run(self, data)
            self._last_turn = turn
            return result
        except Exception as e:
            logger.error(f"Turn {turn_id} failed: {e}")
            raise