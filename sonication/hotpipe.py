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

from .db import log_keep_warm_ping


def _phrase_ready(text: str) -> bool:
    """Check if accumulated LLM text is ready for phrase gating."""
    if len(text) < PHRASE_MIN_CHARS:
        return False
    # Check if last character is a phrase-ending character
    stripped = text.rstrip()
    if stripped and stripped[-1] in PHRASE_END_CHARS:
        return True
    return False


class PhaseGate:
    """Inter-stage phrase accumulator for LLM → TTS_CHUNK_IN_STREAM_OUT connections.
    
    Collects LLM tokens, buffers text, and emits complete phrases to a queue
    as soon as they're ready (ASAP, not ALAP). This enables TTS to start
    synthesizing while LLM is still streaming.
    
    Treated as a pseudo-node with its own lifecycle events:
        phrase_start: when first phrase is emitted
        phrase_chunk: each phrase emitted to TTS
        phrase_done: when close() is called
    
    Usage:
        gate = PhaseGate(turn)
        await gate.feed("Hello")       # LLM token 1
        await gate.feed(", how are ")   # LLM token 2
        await gate.feed("you?")         # LLM token 3 — phrase ready!
        # Queue now contains: "Hello, how are you?"
        await gate.close()              # LLM done — flush remaining buffer
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
        """Feed an LLM token to the phrase gate.
        
        If the accumulated text forms a complete phrase (≥PHRASE_MIN_CHARS
        and ending with .!?), the phrase is extracted and put in the queue
        for downstream TTS consumption.
        
        Args:
            token_text: The content from an LLM token event
        """
        self.buffer += token_text
        self.total_chars += len(token_text)
        
        if not self.started:
            self.started = True
            self.start_wall_ms = time.time() * 1000
        
        if _phrase_ready(self.buffer):
            # Extract phrase (everything up to and including the ending char)
            phrase, self.buffer = self._extract_phrase()
            await self.queue.put(phrase)
            self.phrases_emitted += 1
    
    def _extract_phrase(self) -> Tuple[str, str]:
        """Extract the first complete phrase from the buffer.
        
        Returns:
            Tuple of (phrase, remaining_buffer)
        """
        # Find the first phrase-ending character after PHRASE_MIN_CHARS
        for i in range(PHRASE_MIN_CHARS - 1, len(self.buffer)):
            if self.buffer[i] in PHRASE_END_CHARS:
                # Include the ending character
                phrase = self.buffer[:i+1]
                remaining = self.buffer[i+1:].lstrip()
                return phrase, remaining
        
        # No complete phrase found — return empty phrase, keep buffer
        return "", self.buffer
    
    async def close(self) -> None:
        """Signal that LLM is done — flush remaining buffer as final phrase.
        
        Must be called when LLM stream completes to ensure any remaining
        text is sent to TTS.
        """
        self.end_wall_ms = time.time() * 1000
        
        # Flush remaining buffer
        if self.buffer.strip():
            await self.queue.put(self.buffer.strip())
        
        # Put sentinel to signal TTS to stop
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


class EventStream:
    """Central event stream for all pipeline nodes.
    
    All nodes push events to this shared queue as they happen.
    The main scheduler pulls from the queue and yields events in order.
    This ensures events are naturally ordered by when they occur,
    with no post-hoc sorting or merging needed.
    
    Usage:
        stream = EventStream()
        # In node task:
        await stream.put(event_dict)
        # In main loop:
        async for event in stream:
            yield event
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


class PipelineNodeHandler:
    """Handles events for any node or pseudo-node in the pipeline.
    
    Unified event emission for all node types:
        start: node initialized
        chunk: partial data (audio chunks, phrase chunks)
        token: single LLM token
        response: full response (transcript, full text)
        done: flow truly over
        usage: usage stats
        error: error occurred
    
    Usage:
        handler = PipelineNodeHandler(turn, pipe, node_name, node, config_label, stage_id)
        await handler.run(data, phase_gate)
    """
    
    def __init__(self, turn: 'Turn', pipe: 'HotPipe', node_name: str, node: Any, 
                 config_label: str, stage_id: str):
        self.turn = turn
        self.pipe = pipe
        self.node_name = node_name
        self.node = node
        self.config_label = config_label
        self.stage_id = stage_id
        self._first_token_pushed = False
        self._first_chunk_pushed = False
        self._response_pushed = False
        self._start_wall_ms = time.time() * 1000
    
    def _make_event_dict(self, event_type: str, payload: Optional[Dict] = None) -> Dict:
        """Create a unified event dict."""
        return {
            "event_type": event_type,
            "emitter_node": self.node_name,
            "emitter_type": self.config_label,
            "turn_id": self.turn.turn_id,
            "stage_id": self.stage_id,
            "wallclock_ms": time.time() * 1000,
            "local_offset_ms": self.turn._now(),
            "payload": payload or {},
            "seq": self.turn._event_seq,
        }
    
    async def emit_start(self) -> None:
        """Emit start event — when node is initialized."""
        self.turn._event_seq += 1
        await self.turn.event_stream.put(self._make_event_dict("start", {
            "category": self.config_label,
        }))
    
    async def emit_token(self, raw: Dict) -> None:
        """Emit token event — single LLM token."""
        self.turn._event_seq += 1
        await self.turn.event_stream.put(self._make_event_dict("token", {
            "channel": raw.get("channel", "content"),
            "content": raw.get("content", ""),
        }))
        if not self._first_token_pushed:
            self._first_token_pushed = True
    
    async def emit_chunk(self, raw: Dict) -> None:
        """Emit chunk event — partial data (audio, phrase)."""
        self.turn._event_seq += 1
        await self.turn.event_stream.put(self._make_event_dict("chunk", {
            "pcm": raw.get("pcm", b""),
            "text": raw.get("text", ""),
        }))
        if not self._first_chunk_pushed:
            self._first_chunk_pushed = True
    
    async def emit_response(self, raw: Dict) -> None:
        """Emit response event — full response received."""
        # Combine usage into response payload
        payload = {}
        if raw.get("text"):
            payload["text"] = raw["text"]
        if raw.get("usage"):
            payload["usage"] = raw["usage"]
        if raw.get("finish_reason"):
            payload["finish_reason"] = raw["finish_reason"]
        
        self.turn._event_seq += 1
        await self.turn.event_stream.put(self._make_event_dict("response", payload))
        self._response_pushed = True
    
    async def emit_usage(self, raw: Dict) -> None:
        """Emit usage event — usage stats."""
        self.turn._event_seq += 1
        await self.turn.event_stream.put(self._make_event_dict("usage", raw.get("usage", {})))
    
    async def emit_done(self) -> None:
        """Emit done event — flow truly over."""
        self.turn._event_seq += 1
        await self.turn.event_stream.put(self._make_event_dict("done", {}))
    
    async def emit_error(self, error: Exception) -> None:
        """Emit error event."""
        self.turn._event_seq += 1
        await self.turn.event_stream.put(self._make_event_dict("error", {
            "error": str(error),
        }))
    
    async def run(self, data: Any, phase_gate: Optional[PhaseGate] = None) -> None:
        """Run a node and emit unified events.
        
        Args:
            data: Input data for the node
            phase_gate: Optional PhaseGate for LLM → TTS streaming
        """
        await self.emit_start()
        
        # Create NodeStageRecord
        stage_record = NodeStageRecord(
            stage_id=self.stage_id,
            node_name=self.node_name,
            node_class=self.node.node_class,
            config_label=self.config_label,
            start_wall_ms=self._start_wall_ms,
            end_wall_ms=None,
            timing={},
            events=[],
            payload_kind="unknown",
        )
        self.turn.record_node_stage(stage_record)
        
        llm_text_buffer = ""
        tts_audio_chunks = []
        
        try:
            async for raw in self.node.stream(data):
                kind = raw.get("kind", "unknown")
                
                # Emit appropriate event based on kind
                if kind == "transcript":
                    await self.emit_response(raw)
                    self.turn.stt_text = raw.get("text", "")
                elif kind == "token":
                    await self.emit_token(raw)
                    if self.node_name == "llm":
                        llm_text_buffer += raw.get("content", "")
                        if phase_gate:
                            await phase_gate.feed(raw.get("content", ""))
                elif kind == "reasoning":
                    await self.emit_token(raw)
                    if self.node_name == "llm":
                        llm_text_buffer += raw.get("content", "")
                        if phase_gate:
                            await phase_gate.feed(raw.get("content", ""))
                elif kind == "audio":
                    await self.emit_chunk(raw)
                    if self.node_name == "tts":
                        pcm_data = raw.get("pcm", b"")
                        if pcm_data:
                            tts_audio_chunks.append(pcm_data)
                elif kind == "done":
                    await self.emit_response(raw)
                elif kind == "usage":
                    await self.emit_usage(raw)
                
                # Record phase boundaries
                self.turn._record_phase_boundary(stage_record, raw, self.node_name, self.config_label)
            
            # After stream completes
            self.turn._record_end_boundary(stage_record, self.node_name, self._start_wall_ms)
            
            # Store accumulated results
            if self.node_name == "llm":
                self.turn.llm_text = llm_text_buffer
                stt_input = self.turn.stt_text
                if hasattr(self.node, 'complete_turn') and stt_input:
                    self.node.complete_turn(stt_input, llm_text_buffer)
            
            if self.node_name == "tts" and tts_audio_chunks:
                self.turn.tts_audio = b"".join(tts_audio_chunks)
        
        except Exception as e:
            await self.emit_error(e)
            raise
        
        await self.emit_done()
        
        # Update stage record
        stage_record.end_wall_ms = time.time() * 1000
        self.turn.record_node_stage(stage_record)
        
        # Log to DB via event stream
        if self.turn._log_manager:
            self.turn._log_manager.enqueue({
                "_log_kind": "node_stage",
                "stage_id": stage_record.stage_id,
                "turn_id": self.turn.turn_id,
                "session_id": None,
                "conversation_id": None,
                "node_name": stage_record.node_name,
                "node_class": stage_record.node_class,
                "config_label": stage_record.config_label,
                "start_wall_ms": stage_record.start_wall_ms,
                "end_wall_ms": stage_record.end_wall_ms,
                "timing": stage_record.timing,
                "summary": {},
            })
    
    async def run_phrase_gate(self, phase_gate: PhaseGate) -> None:
        """Run PhraseGate and emit events for each phrase.
        
        PhraseGate is treated as a pseudo-node with its own lifecycle:
            phrase_start: when first phrase is emitted
            phrase_chunk: each phrase emitted
            phrase_done: when close() is called
        
        Args:
            phase_gate: The PhaseGate instance to monitor
        """
        await self.emit_start()
        
        phrase_num = 0
        first_phrase = True
        
        while True:
            phrase = await phase_gate.queue.get()
            if phrase is None:  # sentinel
                break
            
            if not phrase.strip():
                continue
            
            phrase_num += 1
            
            # Emit phrase_chunk event
            if first_phrase:
                first_phrase = False
            
            self.turn._event_seq += 1
            await self.turn.event_stream.put(self._make_event_dict("chunk", {
                "text": phrase,
                "phrase_number": phrase_num,
                "from_stage_id": phase_gate.from_stage_id,
            }))
        
        await self.emit_done()


class PipelineScheduler:
    """Concurrent execution engine for pipeline nodes.
    
    Manages asyncio tasks for each node, event queues for inter-node
    communication, and coordinates parallel execution based on topology
    and node streaming types.
    
    All events flow through a central EventStream, ensuring natural
    ordering by timestamp without post-hoc sorting.
    
    Execution model:
        - Non-streaming nodes (STT, TTS_NON_STREAMING): run sequentially
          — upstream completes before downstream starts
        - Streaming nodes (LLM, TTS_CHUNK_IN_STREAM_OUT): run in parallel
          — downstream starts as soon as upstream has enough data
        
    Usage:
        scheduler = PipelineScheduler(turn, pipe)
        async for event in scheduler.run():
            yield event
    """
    
    def __init__(self, turn: 'Turn', pipe: 'HotPipe', data):
        self.turn = turn
        self.pipe = pipe
        self.data = data
        self.event_stream = turn.event_stream
        self.node_tasks: Dict[str, asyncio.Task] = {}
        self.phase_gates: Dict[str, PhaseGate] = {}
        self._completed = set()
    
    def _get_node_type(self, node_name: str) -> str:
        """Get the streaming type for a node."""
        node, category = self.pipe.nodes[node_name]
        config_label = node.config_label if hasattr(node, 'config_label') else category
        return config_label
    
    def _is_parallel_downstream(self, source_node: str, target_node: str) -> bool:
        """Check if connection should run in parallel.
        
        Parallel when:
            - Source is LLM_STREAMING and target is TTS_CHUNK_IN_STREAM_OUT
            - This enables phrase-gated streaming
        """
        source_type = self._get_node_type(source_node)
        target_type = self._get_node_type(target_node)
        
        # LLM → TTS_CHUNK_IN_STREAM_OUT is parallel (phrase-gated)
        if source_type in (NodeConfigLabel.LLM_STREAMING, 
                          NodeConfigLabel.LLM_STREAMING_WITH_REASONING):
            if target_type == NodeConfigLabel.TTS_CHUNK_IN_STREAM_OUT:
                return True
        
        return False
    
    async def _run_node(self, node_name: str, data, phase_gate: Optional[PhaseGate] = None):
        """Run a single node using PipelineNodeHandler.
        
        Args:
            node_name: Name of the node to run
            data: Input data for the node
            phase_gate: Optional PhaseGate for LLM → TTS streaming
        """
        logger.info(f"[_run_node] Starting node: {node_name}")
        node, category = self.pipe.nodes[node_name]
        config_label = node.config_label if hasattr(node, 'config_label') else category
        
        stage_id = self.pipe._generate_stage_id(node, node_name)
        node._stage_id = stage_id
        
        handler = PipelineNodeHandler(
            self.turn, self.pipe, node_name, node, config_label, stage_id
        )
        await handler.run(data, phase_gate)
        logger.info(f"[_run_node] Completed node: {node_name}")
    
    def _get_event_type(self, node_name: str, kind: str) -> str:
        """Map node + kind to event type string."""
        if node_name == "stt":
            return {"transcript": "response", "error": f"stt_{kind}",
                    "done": "done"}.get(kind, f"stt_{kind}")
        elif node_name == "llm":
            return {"token": "token", "reasoning": "token",
                    "error": f"llm_{kind}", "done": "response",
                    "usage": "usage"}.get(kind, f"llm_{kind}")
        elif node_name == "tts":
            return {"audio": "chunk", "usage": "usage",
                    "error": f"tts_{kind}", "done": "done"}.get(kind,
                    f"tts_{kind}")
        else:
            return f"{node_name}_{kind}"
    
    async def run_parallel(self):
        """Execute pipeline with concurrent node execution.
        
        All events flow through a central EventStream, ensuring natural
        ordering by timestamp without post-hoc sorting.
        """
        logger.info("[PipelineScheduler] run_parallel() starting")
        entry_point = self.turn._detect_entry_point(self.pipe)
        logger.info(f"[PipelineScheduler] entry_point={entry_point}")
        
        # Push turn_start to event stream
        await self.event_stream.put(self.turn._make_event_dict(
            "turn_start", entry_point, self.turn.turn_id,
            local_offset_ms=0.0,
            payload={"pipeline_type": self.turn._pipeline_type, 
                    "entry_node": entry_point},
            stage_id="", seq=0
        ))
        logger.info("[PipelineScheduler] turn_start event pushed")
        
        try:
            # 1. Run entry node (STT) to get transcript
            logger.info(f"[PipelineScheduler] Running entry node: {entry_point}")
            await self._run_node(entry_point, self.data)
            logger.info(f"[PipelineScheduler] Entry node {entry_point} completed")
            
            # Capture STT transcript from turn state
            stt_transcript = self.turn.stt_text
            logger.info(f"[PipelineScheduler] STT transcript: {stt_transcript[:50] if stt_transcript else '(empty)'}")
            
            # 2. Determine downstream nodes and their execution mode
            downstream = self.pipe.connections.get(entry_point, [])
            logger.info(f"[PipelineScheduler] Downstream of {entry_point}: {downstream}")
            
            if not downstream:
                # Single-node pipeline, we're done
                logger.info("[PipelineScheduler] No downstream nodes")
                pass
            elif len(downstream) == 1:
                next_node = downstream[0]
                
                # Check if this is a multi-step pipeline with parallel downstream
                # e.g., STT→LLM→TTS where LLM→TTS is parallel
                next_downstream = self.pipe.connections.get(next_node, [])
                is_llm_to_tts_parallel = (
                    next_downstream and
                    self._is_parallel_downstream(next_node, next_downstream[0])
                )
                logger.info(f"[PipelineScheduler] next_node={next_node}, is_parallel={is_llm_to_tts_parallel}")
                
                if is_llm_to_tts_parallel:
                    # Multi-step: entry → LLM → TTS (parallel)
                    # Run LLM and TTS in parallel with phase gate
                    logger.info(f"[PipelineScheduler] Running parallel LLM→TTS")
                    await self._run_parallel_with_phase_gate(
                        next_node, next_downstream[0],
                        llm_data=stt_transcript
                    )
                    logger.info("[PipelineScheduler] Parallel LLM→TTS completed")
                else:
                    # Simple sequential: entry → next_node
                    logger.info(f"[PipelineScheduler] Running sequential {entry_point}→{next_node}")
                    next_data = stt_transcript if entry_point == 'stt' else self.data
                    await self._run_sequential(entry_point, next_node, next_data)
                    logger.info(f"[PipelineScheduler] Sequential {entry_point}→{next_node} completed")
            else:
                # Multiple downstream — run all in parallel
                logger.info(f"[PipelineScheduler] Running {len(downstream)} downstream nodes in parallel")
                tasks = []
                for next_node in downstream:
                    tasks.append(asyncio.create_task(
                        self._run_node(next_node, None)
                    ))
                
                await asyncio.gather(*tasks)
        
        except Exception as e:
            error_event = self.turn._make_event_dict(
                "turn_error", entry_point, self.turn.turn_id,
                payload={"error": str(e)}
            )
            await self.event_stream.put(error_event)
            raise
        
        # Build segments for turn_complete
        segments = []
        try:
            analysis = self.turn.analyse()
            segments = [
                {"stage": s.stage_name, "ms": s.ms, "kind": s.kind}
                for s in analysis.segments
            ]
        except Exception:
            pass
        
        # Push turn_complete to event stream
        logger.info("[PipelineScheduler] Pushing turn_complete")
        await self.event_stream.put(self.turn._make_event_dict(
            "turn_complete", entry_point, self.turn.turn_id,
            payload={
                "stt_text": self.turn.stt_text,
                "llm_response": self.turn.llm_text,
                "tts_audio": self.turn.tts_audio,
                "shot_latency_ms": self.turn.shot_latency_ms(),
                "stt_ms": self.turn.stt_done_ms - self.turn.stt_start_ms,
                "llm_ttft_ms": self.turn.llm_ttft_ms,
                "tts_ttfb_ms": self.turn.tts_ttfb_ms,
                "segments": segments,
            }
        ))
        
        # Close the event stream and yield all events
        logger.info("[PipelineScheduler] Closing event stream, yielding events")
        self.event_stream.close()
        event_count = 0
        async for event in self.event_stream:
            event_count += 1
            yield event
        logger.info(f"[PipelineScheduler] Yielded {event_count} events")
    
    async def _run_parallel_with_phase_gate(self, upstream_node: str, downstream_node: str, llm_data=None):
        """Run upstream → downstream with phrase-gated parallel execution.
        
        Upstream (LLM) streams tokens → PhaseGate accumulates → emits phrases
        Downstream (TTS) reads phrases from queue → synthesizes audio
        
        Both run concurrently, pushing events to the shared event_stream.
        
        Args:
            upstream_node: LLM node name
            downstream_node: TTS node name
            llm_data: Input data for LLM (STT transcript)
        """
        logger.info(f"[_run_parallel_with_phase_gate] Starting: {upstream_node}→{downstream_node}")
        
        # Create phase gate
        phase_gate = PhaseGate(self.turn)
        self.phase_gates[downstream_node] = phase_gate
        
        # Start LLM task with STT transcript
        llm_task = asyncio.create_task(
            self._run_node(upstream_node, llm_data, phase_gate)
        )
        logger.info(f"[_run_parallel_with_phase_gate] LLM task started")
        
        # Start TTS task (fed by phase gate)
        tts_task = asyncio.create_task(
            self._run_tts_from_phase_gate(downstream_node, phase_gate)
        )
        logger.info(f"[_run_parallel_with_phase_gate] TTS task started")
        
        # Wait for LLM to complete
        logger.info(f"[_run_parallel_with_phase_gate] Waiting for LLM...")
        await llm_task
        logger.info(f"[_run_parallel_with_phase_gate] LLM completed")
        
        # Close phase gate to signal TTS to finish
        logger.info(f"[_run_parallel_with_phase_gate] Closing phase gate...")
        await phase_gate.close()
        logger.info(f"[_run_parallel_with_phase_gate] Phase gate closed, stats: {phase_gate.get_stats()}")
        
        # Wait for TTS to complete
        logger.info(f"[_run_parallel_with_phase_gate] Waiting for TTS...")
        await tts_task
        logger.info(f"[_run_parallel_with_phase_gate] TTS completed")
    
    async def _run_tts_from_phase_gate(self, tts_node: str, phase_gate: PhaseGate):
        """Run TTS node, consuming phrases from phase gate queue.
        
        Each phrase triggers a TTS synthesis call. When phase_gate is
        closed (LLM done + buffer flushed), TTS completes.
        
        All events are pushed to the shared event_stream.
        
        Fixed: tts_start is only pushed when first phrase arrives and
        synthesis begins, not when task starts.
        """
        logger.info(f"[_run_tts_from_phase_gate] Starting for {tts_node}")
        node, category = self.pipe.nodes[tts_node]
        config_label = node.config_label if hasattr(node, 'config_label') else category
        
        stage_id = self.pipe._generate_stage_id(node, tts_node)
        node._stage_id = stage_id
        
        tts_audio_chunks = []
        
        # Use PipelineNodeHandler for TTS
        tts_handler = PipelineNodeHandler(
            self.turn, self.pipe, tts_node, node, config_label, stage_id
        )
        
        # Create node_stage for TTS
        stage_record = NodeStageRecord(
            stage_id=stage_id,
            node_name=tts_node,
            node_class=node.node_class,
            config_label=config_label,
            start_wall_ms=time.time() * 1000,
            end_wall_ms=None,
            timing={},
            events=[],
            payload_kind="unknown",
        )
        self.turn.record_node_stage(stage_record)
        
        tts_start_wc = time.time() * 1000
        phrase_num = 0
        tts_started = False
        
        try:
            logger.info(f"[_run_tts_from_phase_gate] Waiting for phrases...")
            while True:
                phrase = await phase_gate.queue.get()
                logger.info(f"[_run_tts_from_phase_gate] Got phrase: {phrase[:50] if phrase else 'None'}")
                if phrase is None:  # sentinel
                    logger.info(f"[_run_tts_from_phase_gate] Received sentinel, breaking")
                    break
                
                # Skip empty or whitespace-only phrases
                if not phrase.strip():
                    logger.info(f"[_run_tts_from_phase_gate] Skipping empty phrase")
                    continue
                
                phrase_num += 1
                
                # Emit phrase_chunk event with phrase_gate as emitter
                self.turn._event_seq += 1
                await self.turn.event_stream.put({
                    "event_type": "chunk",
                    "emitter_node": "phrase_gate",
                    "emitter_type": "PHRASE_GATE",
                    "turn_id": self.turn.turn_id,
                    "stage_id": stage_id,
                    "wallclock_ms": time.time() * 1000,
                    "local_offset_ms": self.turn._now(),
                    "payload": {
                        "text": phrase,
                        "phrase_number": phrase_num,
                        "from_stage_id": phase_gate.from_stage_id,
                    },
                    "seq": self.turn._event_seq,
                })
                logger.info(f"[_run_tts_from_phase_gate] Phrase chunk {phrase_num} emitted")
                
                # Fix: Only emit tts_start when first phrase arrives and synthesis begins
                if not tts_started:
                    tts_started = True
                    await tts_handler.emit_start()
                    logger.info(f"[_run_tts_from_phase_gate] TTS start emitted")
                
                # Synthesize this phrase using TTS handler
                async for raw in node.stream(phrase):
                    kind = raw.get("kind", "unknown")
                    
                    # Record phase boundaries
                    self.turn._record_phase_boundary(stage_record, raw, tts_node, config_label)
                    
                    # Capture TTS audio chunks
                    if kind == "audio":
                        pcm_data = raw.get("pcm", b"")
                        if pcm_data:
                            tts_audio_chunks.append(pcm_data)
                    
                    # Emit unified event
                    self.turn._event_seq += 1
                    await self.turn.event_stream.put(tts_handler._make_event_dict("chunk", {
                        "pcm": pcm_data if kind == "audio" else b"",
                    }))
                    logger.debug(f"[_run_tts_from_phase_gate] TTS chunk {len(tts_audio_chunks)} emitted")
            
            logger.info(f"[_run_tts_from_phase_gate] Phrase loop completed, {phrase_num} phrases, {len(tts_audio_chunks)} chunks")
        
        except Exception as e:
            logger.error(f"[_run_tts_from_phase_gate] Error: {e}")
            await tts_handler.emit_error(e)
            raise
        finally:
            # Record end boundary
            self.turn._record_end_boundary(stage_record, tts_node, tts_start_wc)
            
            # Store accumulated audio
            if tts_audio_chunks:
                self.turn.tts_audio = b"".join(tts_audio_chunks)
            
            # Update stage record
            stage_record.end_wall_ms = time.time() * 1000
            self.turn.record_node_stage(stage_record)
            
            if self.turn._log_manager:
                logger.info(f"Enqueueing node_stage: {stage_record.node_name} turn_id={self.turn.turn_id}")
                self.turn._log_manager.enqueue({
                    "_log_kind": "node_stage",
                    "stage_id": stage_record.stage_id,
                    "turn_id": self.turn.turn_id,
                    "session_id": None,
                    "conversation_id": None,
                    "node_name": stage_record.node_name,
                    "node_class": stage_record.node_class,
                    "config_label": stage_record.config_label,
                    "start_wall_ms": stage_record.start_wall_ms,
                    "end_wall_ms": stage_record.end_wall_ms,
                    "timing": stage_record.timing,
                    "summary": {},
                })
            
            # Emit tts_done
            self.turn._event_seq += 1
            await self.turn.event_stream.put(tts_handler._make_event_dict("done", {}))
            logger.info(f"[_run_tts_from_phase_gate] TTS done emitted")
    
    async def _run_sequential(self, upstream_node: str, downstream_node: str, upstream_data=None):
        """Run upstream → downstream sequentially.
        
        Upstream completes completely before downstream starts.
        Used for non-streaming connections.
        
        Args:
            upstream_node: First node to run
            downstream_node: Second node to run
            upstream_data: Input data for upstream node
        """
        # Run upstream node
        await self._run_node(upstream_node, upstream_data)
        
        # Run downstream node
        next_data = self.turn._next_data(upstream_node, None)
        if next_data is not None:
            await self._run_node(downstream_node, next_data)


class PingLoop:
    """Background keepalive ping loop.

    Periodically sends a lightweight /ping request to each node to keep
    pooled HTTP connections warm. Auto-starts on connect()/turn() and
    stops after keep_warm_duration seconds of inactivity.
    """

    def __init__(self, nodes: dict, keep_warm_duration: float = 30.0, ping_interval: float = 5.0):
        self._nodes = nodes
        self._keep_warm_duration = keep_warm_duration
        self._ping_interval = ping_interval
        self._last_turn_time: Optional[float] = None
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self._started = False

    def _ensure_started(self):
        """Start the background task if not already running."""
        if not self._started:
            self._task = asyncio.create_task(self._loop())
            self._started = True

    def touch(self):
        """Reset keep-warm timer and ensure ping loop is running."""
        self._last_turn_time = time.time()
        self._ensure_started()

    async def stop(self):
        """Stop pings immediately."""
        self._stopped = True
        if self._task and self._task != asyncio.current_task():
            await self._task
            self._task = None

    async def _loop(self):
        while not self._stopped:
            # Check if keep-warm duration expired
            if self._last_turn_time and \
               (time.time() - self._last_turn_time) > self._keep_warm_duration:
                await self.stop()
                break
                
            for name, (node, _) in self._nodes.items():
                try:
                    # REUSE node.connection (not ephemeral client)
                    if node.connection:
                        r = await node.connection.get(
                            f"{node.base_url}/ping",
                            headers=node._auth_headers()
                        )
                        rtt_ms = r.elapsed.total_seconds() * 1000
                        # Log to DB (parent_turn_id=None per AGENTS.md 1.10)
                        log_keep_warm_ping(name, time.time()*1000, rtt_ms, parent_turn_id=None)
                except Exception:
                    pass
            await asyncio.sleep(self._ping_interval)


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
        # Collect all events (run_events is a generator)
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
        """Execute the pipeline and yield all events with timestamps in real-time.
        
        Uses PipelineScheduler for concurrent node execution when topology
        supports it (e.g., LLM → TTS_CHUNK_IN_STREAM_OUT with phrase gate).
        
        Yields event dicts with:
            - event_type: unified event type (start, chunk, token, response, done, usage, error)
            - turn_id: unique turn identifier
            - stage_id: node stage identifier
            - emitter_node: which node produced the event (stt, llm, tts, phrase_gate)
            - emitter_type: node behavior type (STT_NON_STREAMING, LLM_STREAMING, etc.)
            - wallclock_ms: absolute Unix timestamp
            - local_offset_ms: milliseconds since turn start
            - payload: node-specific data
            - seq: sequence number
        
        Final event is "turn_complete" with aggregated results:
            - stt_text, llm_response, tts_audio, shot_latency_ms, segments
        """
        self._log_manager = pipe._log_manager
        
        # Use parallel scheduler for cascade execution
        scheduler = PipelineScheduler(self, pipe, data)
        async for event in scheduler.run_parallel():
            yield event

    def _make_event_dict(self, event_type, node_name, turn_id, local_offset_ms=0.0,
                          payload=None, stage_id="", seq=0):
        """Create a standardized event dict for streaming.
        
        DEPRECATED: Use PipelineNodeHandler._make_event_dict() instead.
        Kept for backward compatibility.
        """
        return {
            "event_type": event_type,
            "emitter_node": node_name,
            "emitter_type": event_type,
            "turn_id": turn_id,
            "stage_id": stage_id,
            "wallclock_ms": time.time() * 1000,
            "local_offset_ms": local_offset_ms,
            "payload": payload or {},
            "seq": seq,
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
        record phase boundaries, and handle phrase gating for LLM->TTS.
        
        Uses unified event format: event_type, emitter_node, emitter_type
        """
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

        # Record node start event using unified format
        self._event_seq += 1
        start_event = PipeEvent.new(
            "start", node_name, config_label, self.turn_id,
            local_offset_ms=self._now(),
            payload={"category": category, "config_label": config_label},
            stage_id=stage_id,
            seq=self._event_seq
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
                kind = raw.get("kind")

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

        except Exception as e:
            self.events.append(PipeEvent.new(
                "error", node_name, config_label, self.turn_id,
                payload={"error": str(e)}, parent_event_id=parent_id))
            raise
        finally:
            stage_record.end_wall_ms = time.time() * 1000
            self.record_node_stage(stage_record)
            
            # Enqueue node_stage to log_manager
            if self._log_manager:
                logger.info(f"Enqueueing node_stage: {stage_record.node_name} turn_id={self.turn_id}")
                self._log_manager.enqueue({
                    "_log_kind": "node_stage",
                    "stage_id": stage_record.stage_id,
                    "turn_id": self.turn_id,
                    "session_id": None,
                    "conversation_id": None,
                    "node_name": stage_record.node_name,
                    "node_class": stage_record.node_class,
                    "config_label": stage_record.config_label,
                    "start_wall_ms": stage_record.start_wall_ms,
                    "end_wall_ms": stage_record.end_wall_ms,
                    "timing": stage_record.timing,
                    "summary": {},
                })

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

    async def _walk_events(self, pipe, node_name, data, parent_id=None):
        """Async generator that walks the pipeline and yields events in real-time.
        
        Uses unified event format: event_type, emitter_node, emitter_type
        """
        node, category = pipe.nodes[node_name]
        config_label = node.config_label if hasattr(node, 'config_label') else category
        
        stage_id = pipe._generate_stage_id(node, node_name)
        node._stage_id = stage_id

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

        llm_text_buffer = ""
        tts_audio_chunks = []

        # Yield node_start using unified format
        self._event_seq += 1
        yield {
            "event_type": "start",
            "emitter_node": node_name,
            "emitter_type": config_label,
            "turn_id": self.turn_id,
            "stage_id": stage_id,
            "wallclock_ms": time.time() * 1000,
            "local_offset_ms": self._now(),
            "payload": {"category": category, "config_label": config_label},
            "seq": self._event_seq,
        }
        
        node_start_wc = time.time() * 1000
        last_event_event_id = None
        last_event_type = None
        last_event_data = None
        
        try:
            async for raw in node.stream(data):
                kind = raw.get("kind", "unknown")
                
                # Determine unified event type
                if node_name == "stt":
                    if kind == "transcript":
                        etype = "response"
                    elif kind == "done":
                        etype = "done"
                    else:
                        etype = kind
                elif node_name == "llm":
                    if kind == "token":
                        etype = "token"
                    elif kind == "reasoning":
                        etype = "token"
                    elif kind == "done":
                        etype = "response"
                    elif kind == "usage":
                        etype = "usage"
                    else:
                        etype = kind
                elif node_name == "tts":
                    if kind == "audio":
                        etype = "chunk"
                    elif kind == "done":
                        etype = "done"
                    elif kind == "usage":
                        etype = "usage"
                    else:
                        etype = kind
                else:
                    etype = kind
                
                # Yield the event
                self._event_seq += 1
                yield {
                    "event_type": etype,
                    "emitter_node": node_name,
                    "emitter_type": config_label,
                    "turn_id": self.turn_id,
                    "stage_id": stage_id,
                    "wallclock_ms": time.time() * 1000,
                    "local_offset_ms": self._now(),
                    "payload": raw,
                    "seq": self._event_seq,
                }
                last_event_type = kind
                last_event_data = raw

                # Record phase boundaries
                self._record_phase_boundary(stage_record, raw, node_name, config_label)

                # Accumulate LLM text
                if "content" in raw and node_name == "llm":
                    llm_text_buffer += raw.get("content", "")

                # Capture TTS audio chunks
                if node_name == "tts" and kind == "audio":
                    pcm_data = raw.get("pcm", b"")
                    if pcm_data:
                        tts_audio_chunks.append(pcm_data)

                # Capture STT transcript
                if node_name == "stt" and kind == "transcript":
                    self.stt_text = raw.get("text", "")

            # After stream completes
            self._record_end_boundary(stage_record, node_name, node_start_wc)

            # Yield node_done
            self._event_seq += 1
            yield {
                "event_type": "done",
                "emitter_node": node_name,
                "emitter_type": config_label,
                "turn_id": self.turn_id,
                "stage_id": stage_id,
                "wallclock_ms": time.time() * 1000,
                "local_offset_ms": self._now(),
                "payload": {},
                "seq": self._event_seq,
            }

            # Store accumulated results
            if node_name == "llm":
                self.llm_text = llm_text_buffer
                stt_input = self.stt_text
                if hasattr(node, 'complete_turn') and stt_input:
                    node.complete_turn(stt_input, llm_text_buffer)

            if node_name == "tts" and tts_audio_chunks:
                self.tts_audio = b"".join(tts_audio_chunks)

            # Enqueue node_stage to log_manager
            stage_record.end_wall_ms = time.time() * 1000
            self.record_node_stage(stage_record)
            if self._log_manager:
                logger.info(f"Enqueueing node_stage: {stage_record.node_name} turn_id={self.turn_id}")
                self._log_manager.enqueue({
                    "_log_kind": "node_stage",
                    "stage_id": stage_record.stage_id,
                    "turn_id": self.turn_id,
                    "session_id": None,
                    "conversation_id": None,
                    "node_name": stage_record.node_name,
                    "node_class": stage_record.node_class,
                    "config_label": stage_record.config_label,
                    "start_wall_ms": stage_record.start_wall_ms,
                    "end_wall_ms": stage_record.end_wall_ms,
                    "timing": stage_record.timing,
                    "summary": {},
                })

            # Follow downstream connections
            for next_name in pipe.connections.get(node_name, []):
                next_data = self._next_data(node_name, last_event_type)
                if next_data is not None:
                    async for event in self._walk_events(pipe, next_name, next_data,
                                                         parent_id=last_event_event_id):
                        yield event

        except Exception as e:
            self._event_seq += 1
            yield {
                "event_type": "error",
                "emitter_node": node_name,
                "emitter_type": config_label,
                "turn_id": self.turn_id,
                "stage_id": stage_id,
                "wallclock_ms": time.time() * 1000,
                "local_offset_ms": self._now(),
                "payload": {"error": str(e)},
                "seq": self._event_seq,
            }
            raise
        finally:
            if not stage_record.end_wall_ms:
                stage_record.end_wall_ms = time.time() * 1000
                self.record_node_stage(stage_record)

    def _next_seq(self) -> int:
        """Return an incrementing sequence number."""
        self._event_seq += 1
        return self._event_seq

    def _enqueue_event(self, event_dict: dict) -> None:
        """Enqueue an event to the log_manager, or skip if disabled.
        
        DEPRECATED: Use unified EventStream path instead.
        Kept for backward compatibility with legacy _walk() path.
        """
        if hasattr(self, '_log_manager') and self._log_manager:
            try:
                self._log_manager.enqueue(event_dict)
            except RuntimeError:
                pass  # queue full — don't block the pipeline


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

    def __init__(self, pipeline_type: PipelineType, log_manager=None,
                 keep_warm_duration: float = 30.0, ping_interval: float = 5.0):
        """Pipeline type is REQUIRED at init.
        
        Args:
            pipeline_type: The pipeline topology to use.
            log_manager: Optional LogManager instance for event logging.
                        If None, all logging is disabled.
            keep_warm_duration: Seconds of inactivity after which the
                               keep-alive ping loop stops (default 30s).
            ping_interval: Seconds between pings (default 5s).
        """
        if not pipeline_type:
            raise ValueError(
                "HotPipe requires pipeline_type at init. Got None.")

        self.pipeline_type = pipeline_type
        self._log_manager = log_manager
        self._keep_warm_duration = keep_warm_duration
        self._ping_interval = ping_interval
        
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
        
        # Initialize ping loop and warm up all node connections
        self._ping_loop = PingLoop(self.nodes, self._keep_warm_duration, self._ping_interval)
        for node, _ in self.nodes.values():
            asyncio.create_task(node.warmup())
        self._ping_loop.touch()
        
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

    async def warmup(self):
        """Ping all nodes to establish keepalive connections.
        
        Deprecated: connections are now established in connect().
        This method is kept for backward compatibility.
        """
        results = []
        for node, _ in self.nodes.values():
            results.append(await node.warmup())
        warm = sum(1 for r in results if r)
        logger.info(f"Warmup: {warm}/{len(self.nodes)} nodes warm")
        if self._ping_loop:
            self._ping_loop.touch()
        return results

    async def close(self):
        """Close all node connections and stop ping loop."""
        if self._ping_loop:
            await self._ping_loop.stop()
        for node, _ in self.nodes.values():
            await node.close()

    def get_stats(self):
        return {
            "nodes": list(self.nodes.keys()),
            "connections": dict(self.connections),
            "node_warming": {n: node.is_warm for n, (node, _) in self.nodes.items()},
        }

    async def turn(self, entry_point: str, data, pipeline_type: str = "manual",
                    stream_events: bool = False, verbose: bool = False):
        """Execute one turn.
        
        Always returns an async generator yielding event dicts.
        
        Args:
            entry_point: Node name to start from (e.g., "stt").
            data: Input data for the entry node.
            pipeline_type: Override pipeline type detection.
            stream_events: If True, yields individual events (start, chunk,
                          token, response, done, usage, error) plus turn_complete.
                          If False (default), yields a single dict with aggregated
                          results.
            verbose: If True, logs each event as it's emitted for debugging.
        
        Yields:
            If stream_events=True: event dicts with event_type, emitter_node,
                                   emitter_type, wallclock_ms, local_offset_ms,
                                   payload, seq
            If stream_events=False: single dict with stt_text, llm_response,
                                    tts_audio, shot_latency_ms, etc.
        """
        # Start/refresh ping loop
        if not self._ping_loop:
            self._ping_loop = PingLoop(
                self.nodes, self._keep_warm_duration, self._ping_interval
            )
        self._ping_loop.touch()
        
        turn_id = f"turn_{round(time.time() * 1000)}"
        start_wall = time.time() * 1000
        start_mono = time.monotonic()
        turn = Turn(turn_id, start_wall, start_mono,
                    pipeline_type=self.pipeline_type.value if self.pipeline_type else pipeline_type)
        self._last_turn = turn
        
        if verbose:
            logger.info(f"[VERBOSE] Turn {turn_id} starting at {start_wall:.0f}ms")
        
        try:
            if stream_events:
                async for event in turn.run_events(self, data):
                    if verbose:
                        logger.info(
                            f"[VERBOSE] {event.get('event_type')} "
                            f"({event.get('emitter_node')}) "
                            f"@ {event.get('local_offset_ms', 0):.0f}ms"
                        )
                    yield event
            else:
                result = await turn.run(self, data)
                yield result
        except Exception as e:
            logger.error(f"Turn {turn_id} failed: {e}")
            raise