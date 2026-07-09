"""Scheduler: PipelineScheduler, PipelineNodeHandler, InternalNode.

Concurrent execution engine for pipeline nodes — starts nodes as asyncio.Tasks,
coordinates parallel execution based on topology, handles phrase-gated streaming.
"""
import asyncio
import time
import logging
from typing import Dict, List, Optional, Any

from .events import EventStream, PhaseGate, Turn, NodeStageRecord
from .node_types import NodeConfigLabel

logger = logging.getLogger(__name__)


class InternalNode:
    """Wrapper for internal/pseudo-nodes (PIPELINE, PHRASE_GATE) that don't have real node instances.
    
    Allows PipelineNodeHandler to emit events uniformly for internal coordination.
    """
    node_class = "INTERNAL"
    
    def __init__(self, node_name: str, config_label: str):
        self.node_name = node_name
        self.config_label = config_label


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
                if kind == "token":
                    await self.emit_token(raw)
                    if self.config_label in (NodeConfigLabel.LLM_STREAMING,
                                             NodeConfigLabel.LLM_STREAMING_WITH_REASONING):
                        llm_text_buffer += raw.get("content", "")
                        if phase_gate:
                            await phase_gate.feed(raw.get("content", ""))
                elif kind == "reasoning":
                    await self.emit_token(raw)
                    if self.config_label in (NodeConfigLabel.LLM_STREAMING,
                                             NodeConfigLabel.LLM_STREAMING_WITH_REASONING):
                        llm_text_buffer += raw.get("content", "")
                        if phase_gate:
                            await phase_gate.feed(raw.get("content", ""))
                elif kind == "audio":
                    await self.emit_chunk(raw)
                    if self.config_label in (NodeConfigLabel.TTS_CHUNK_IN_STREAM_OUT,
                                             NodeConfigLabel.TTS_NON_STREAMING):
                        pcm_data = raw.get("pcm", b"")
                        if pcm_data:
                            tts_audio_chunks.append(pcm_data)
                elif kind == "done":
                    await self.emit_response(raw)
                    if self.config_label == NodeConfigLabel.STT_NON_STREAMING:
                        self.turn.stt_text = raw.get("text", "")
                elif kind == "usage":
                    await self.emit_usage(raw)
                
                # Record phase boundaries
                self.turn._record_phase_boundary(stage_record, raw, self.config_label)
            
            # After stream completes
            self.turn._record_end_boundary(stage_record, self.config_label, self._start_wall_ms)
            
            # Store accumulated results
            if self.config_label in (NodeConfigLabel.LLM_STREAMING,
                                     NodeConfigLabel.LLM_STREAMING_WITH_REASONING):
                self.turn.llm_text = llm_text_buffer
                stt_input = self.turn.stt_text
                if hasattr(self.node, 'complete_turn') and stt_input:
                    self.node.complete_turn(stt_input, llm_text_buffer)

            if self.config_label in (NodeConfigLabel.TTS_CHUNK_IN_STREAM_OUT,
                                     NodeConfigLabel.TTS_NON_STREAMING) and tts_audio_chunks:
                self.turn.tts_audio = self.turn.tts_audio + b"".join(tts_audio_chunks)
        
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
        """Run PhraseGate and emit events for each phrase."""
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
    communication, and coordinates parallel execution based on topology.
    
    All events flow through a central EventStream, ensuring natural
    ordering by timestamp without post-hoc sorting.
    
    Execution model:
        - Non-streaming nodes (STT, TTS_NON_STREAMING): run sequentially
        - Streaming nodes (LLM, TTS_CHUNK_IN_STREAM_OUT): run in parallel
    """
    
    def __init__(self, turn: 'Turn', pipe: 'HotPipe', data):
        self.turn = turn
        self.pipe = pipe
        self.data = data
        self.event_stream = turn.event_stream
        self.node_tasks: Dict[str, asyncio.Task] = {}
        self.phase_gates: Dict[str, PhaseGate] = {}
        self._completed = set()
        self._upstream_failed: asyncio.Event = asyncio.Event()
    
    def _get_node_type(self, node_name: str) -> str:
        """Get the streaming type for a node."""
        node, category = self.pipe.nodes[node_name]
        config_label = node.config_label if hasattr(node, 'config_label') else category
        return config_label
    
    def _is_parallel_downstream(self, source_node: str, target_node: str) -> bool:
        """Check if connection should run in parallel."""
        source_type = self._get_node_type(source_node)
        target_type = self._get_node_type(target_node)
        
        if source_type in (NodeConfigLabel.LLM_STREAMING, 
                          NodeConfigLabel.LLM_STREAMING_WITH_REASONING):
            if target_type == NodeConfigLabel.TTS_CHUNK_IN_STREAM_OUT:
                return True
        return False
    
    async def _run_node(self, node_name: str, data, phase_gate: Optional[PhaseGate] = None):
        """Run a single node using PipelineNodeHandler."""
        logger.info(f"[_run_node] Starting node: {node_name}")
        
        # Wait for STT transcript if data is None and this is the LLM node
        if data is None and node_name == "llm":
            data = await self._wait_for_stt_text()
            logger.info(f"[_run_node] Got STT transcript for LLM: {data[:50] if data else '(empty)'}")
        
        node, category = self.pipe.nodes[node_name]
        config_label = node.config_label if hasattr(node, 'config_label') else category
        
        stage_id = self.pipe._generate_stage_id(node, node_name)
        node._stage_id = stage_id
        
        handler = PipelineNodeHandler(
            self.turn, self.pipe, node_name, node, config_label, stage_id
        )
        await handler.run(data, phase_gate)
        logger.info(f"[_run_node] Completed node: {node_name}")
    
    async def _run_node_safe(self, node_name: str, data, phase_gate: Optional[PhaseGate] = None):
        """Run a node with error handling — pushes error event on failure."""
        try:
            await self._run_node(node_name, data, phase_gate)
        except Exception as e:
            logger.error(f"[_run_node_safe] Node {node_name} failed: {e}")
            # Signal downstream nodes that upstream failed
            self._upstream_failed.set()
            node, category = self.pipe.nodes.get(node_name, (None, None))
            config_label = node.config_label if node and hasattr(node, 'config_label') else category
            stage_id = node._stage_id if node else ""
            internal = InternalNode(node_name, config_label if config_label else "UNKNOWN")
            handler = PipelineNodeHandler(
                self.turn, self.pipe, node_name, internal, config_label if config_label else "UNKNOWN", stage_id
            )
            await self.event_stream.put(handler._make_event_dict("error", {
                "error": str(e),
                "node": node_name,
            }))
            raise
    
    async def _wait_for_stt_text(self, timeout: float = 10.0) -> str:
        """Wait for STT transcript to be available, with timeout.
        
        Also checks if upstream failed (e.g., STT error) — returns empty
        string immediately if upstream failed.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._upstream_failed.is_set():
                logger.warning(f"[_wait_for_stt_text] Upstream failed, aborting wait")
                return ""
            if self.turn.stt_text:
                return self.turn.stt_text
            await asyncio.sleep(0.05)
        logger.warning(f"[_wait_for_stt_text] Timed out after {timeout}s waiting for STT")
        return ""
    
    async def run(self):
        """Execute pipeline — start nodes as tasks, yield events as they arrive."""
        logger.info("[PipelineScheduler] run() starting")
        entry_point = self.turn._detect_entry_point(self.pipe)
        logger.info(f"[PipelineScheduler] entry_point={entry_point}")
        
        # Push turn_start to event stream
        pipeline_node = InternalNode("pipeline", "PIPELINE")
        pipeline_handler = PipelineNodeHandler(
            self.turn, self.pipe, "pipeline", pipeline_node, "PIPELINE", ""
        )
        await self.event_stream.put(pipeline_handler._make_event_dict("start", {
            "pipeline_type": self.turn._pipeline_type,
            "entry_node": entry_point,
        }))
        logger.info("[PipelineScheduler] turn_start event pushed")
        
        node_tasks = {}
        downstream_tasks = []
        
        try:
            # 1. Start entry node as task
            logger.info(f"[PipelineScheduler] Starting entry node task: {entry_point}")
            node_tasks[entry_point] = asyncio.create_task(
                self._run_node_safe(entry_point, self.data)
            )
            
            # 2. Determine downstream nodes
            downstream = self.pipe.connections.get(entry_point, [])
            logger.info(f"[PipelineScheduler] Downstream of {entry_point}: {downstream}")
            
            if not downstream:
                logger.info("[PipelineScheduler] No downstream nodes")
            elif len(downstream) == 1:
                next_node = downstream[0]
                next_downstream = self.pipe.connections.get(next_node, [])
                is_llm_to_tts_parallel = (
                    next_downstream and
                    self._is_parallel_downstream(next_node, next_downstream[0])
                )
                logger.info(f"[PipelineScheduler] next_node={next_node}, is_parallel={is_llm_to_tts_parallel}")
                
                if is_llm_to_tts_parallel:
                    logger.info(f"[PipelineScheduler] Starting parallel LLM→TTS task")
                    downstream_tasks.append(asyncio.create_task(
                        self._run_parallel_with_phase_gate(next_node, next_downstream[0], llm_data=None)
                    ))
                else:
                    logger.info(f"[PipelineScheduler] Starting sequential {entry_point}→{next_node} task")
                    node_tasks[next_node] = asyncio.create_task(
                        self._run_node_safe(next_node, None)
                    )
            else:
                logger.info(f"[PipelineScheduler] Starting {len(downstream)} downstream tasks")
                for next_node in downstream:
                    node_tasks[next_node] = asyncio.create_task(
                        self._run_node_safe(next_node, None)
                    )
        
        except Exception as e:
            pipeline_node = InternalNode("pipeline", "PIPELINE")
            pipeline_handler = PipelineNodeHandler(
                self.turn, self.pipe, "pipeline", pipeline_node, "PIPELINE", ""
            )
            await self.event_stream.put(pipeline_handler._make_event_dict("error", {
                "error": str(e)
            }))
            raise
        
        # Start a task to push turn_complete after all node tasks finish
        async def _push_turn_complete():
            try:
                for task in node_tasks.values():
                    await task
                for task in downstream_tasks:
                    await task
                
                # Build segments
                segments = []
                try:
                    analysis = self.turn.analyse()
                    segments = [
                        {"stage": s.stage_name, "ms": s.ms, "kind": s.kind}
                        for s in analysis.segments
                    ]
                except Exception:
                    pass
                
                logger.info("[PipelineScheduler] Pushing turn_complete")
                pipeline_node = InternalNode("pipeline", "PIPELINE")
                pipeline_handler = PipelineNodeHandler(
                    self.turn, self.pipe, "pipeline", pipeline_node, "PIPELINE", ""
                )
                await self.event_stream.put(pipeline_handler._make_event_dict("response", {
                    "stt_text": self.turn.stt_text,
                    "llm_response": self.turn.llm_text,
                    "tts_audio": self.turn.tts_audio,
                    "shot_latency_ms": self.turn.shot_latency_ms(),
                    "stt_ms": self.turn.stt_done_ms - self.turn.stt_start_ms,
                    "llm_ttft_ms": self.turn.llm_ttft_ms,
                    "tts_ttfb_ms": self.turn.tts_ttfb_ms,
                    "segments": segments,
                }))
                await self.event_stream.put(pipeline_handler._make_event_dict("done", {}))
                # Close stream to unblock the main iterator
                self.event_stream.close()
            except Exception as e:
                logger.error(f"[_push_turn_complete] Failed: {e}")
                pipeline_node = InternalNode("pipeline", "PIPELINE")
                pipeline_handler = PipelineNodeHandler(
                    self.turn, self.pipe, "pipeline", pipeline_node, "PIPELINE", ""
                )
                await self.event_stream.put(pipeline_handler._make_event_dict("error", {
                    "error": str(e),
                    "stage": "turn_complete",
                }))
                self.event_stream.close()
        
        asyncio.create_task(_push_turn_complete())
        
        # Yield events as they arrive
        logger.info("[PipelineScheduler] Yielding events")
        event_count = 0
        async for event in self.event_stream:
            event_count += 1
            yield event
        logger.info(f"[PipelineScheduler] Yielded {event_count} events")
        
        # Close stream after all events have been yielded
        self.event_stream.close()
    
    async def _run_parallel_with_phase_gate(self, upstream_node: str, downstream_node: str, llm_data=None):
        """Run upstream → downstream with phrase-gated parallel execution."""
        logger.info(f"[_run_parallel_with_phase_gate] Starting: {upstream_node}→{downstream_node}")
        
        # Check if upstream already failed
        if self._upstream_failed.is_set():
            logger.warning(f"[_run_parallel_with_phase_gate] Upstream already failed, aborting")
            return
        
        if llm_data is None:
            llm_data = await self._wait_for_stt_text()
            logger.info(f"[_run_parallel_with_phase_gate] Got STT transcript: {llm_data[:50] if llm_data else '(empty)'}")
        
        # Check again after waiting for STT
        if self._upstream_failed.is_set():
            logger.warning(f"[_run_parallel_with_phase_gate] Upstream failed while waiting for STT, aborting")
            return
        
        phase_gate = PhaseGate(self.turn)
        self.phase_gates[downstream_node] = phase_gate
        
        llm_task = asyncio.create_task(
            self._run_node(upstream_node, llm_data, phase_gate)
        )
        logger.info(f"[_run_parallel_with_phase_gate] LLM task started")
        
        tts_task = asyncio.create_task(
            self._run_tts_from_phase_gate(downstream_node, phase_gate)
        )
        logger.info(f"[_run_parallel_with_phase_gate] TTS task started")
        
        try:
            logger.info(f"[_run_parallel_with_phase_gate] Waiting for LLM...")
            await llm_task
            logger.info(f"[_run_parallel_with_phase_gate] LLM completed")
            
            logger.info(f"[_run_parallel_with_phase_gate] Closing phase gate...")
            await phase_gate.close()
            logger.info(f"[_run_parallel_with_phase_gate] Phase gate closed, stats: {phase_gate.get_stats()}")
            
            logger.info(f"[_run_parallel_with_phase_gate] Waiting for TTS...")
            await tts_task
            logger.info(f"[_run_parallel_with_phase_gate] TTS completed")
        except Exception as e:
            logger.error(f"[_run_parallel_with_phase_gate] Error in LLM→TTS: {e}")
            # Cancel TTS task if LLM failed
            if not tts_task.done():
                tts_task.cancel()
            raise
    
    async def _run_tts_from_phase_gate(self, tts_node: str, phase_gate: PhaseGate):
        """Run TTS node, consuming phrases from phase gate queue."""
        logger.info(f"[_run_tts_from_phase_gate] Starting for {tts_node}")
        node, category = self.pipe.nodes[tts_node]
        config_label = node.config_label if hasattr(node, 'config_label') else category
        
        stage_id = self.pipe._generate_stage_id(node, tts_node)
        node._stage_id = stage_id
        
        phrase_num = 0
        
        try:
            logger.info(f"[_run_tts_from_phase_gate] Waiting for phrases...")
            while True:
                phrase = await phase_gate.queue.get()
                logger.info(f"[_run_tts_from_phase_gate] Got phrase: {phrase[:50] if phrase else 'None'}")
                if phrase is None:
                    logger.info(f"[_run_tts_from_phase_gate] Received sentinel, breaking")
                    break
                
                if not phrase.strip():
                    logger.info(f"[_run_tts_from_phase_gate] Skipping empty phrase")
                    continue
                
                phrase_num += 1
                
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
                
                tts_handler = PipelineNodeHandler(
                    self.turn, self.pipe, tts_node, node, config_label, stage_id
                )
                await tts_handler.run(phrase)
                logger.info(f"[_run_tts_from_phase_gate] TTS phrase {phrase_num} completed")
            
            logger.info(f"[_run_tts_from_phase_gate] Phrase loop completed, {phrase_num} phrases")
        
        except Exception as e:
            logger.error(f"[_run_tts_from_phase_gate] Error: {e}")
            raise