"""HotPipe: orchestrator for multi-stage voice pipelines.

Manages node lifecycle, keepalive connections, turn scheduling and event
logging. Designed to minimize the wall-clock gap from user utterance to
first TTS audio byte (shot latency P50 < 0.7s).

Pipeline-first architecture:
    HotPipe(pipeline_type=PipelineType.X) defines topology.
    add_node(node) auto-detects slot by node type (STTNode/LLMNode/TTSNode).
    connect() is a validation trigger — wiring is 100% topology-driven.

Module layout:
    events.py — PhaseGate, EventStream, StageBoundaries, Turn
    scheduler.py — PipelineScheduler, PipelineNodeHandler, InternalNode
    hotpipe.py — PingLoop, HotPipe (topology definition)
"""
import asyncio
import httpx
import time
import logging
import uuid
from typing import Dict, List, Optional, Any

from .events import (
    PipeEvent, NodeEvent, PhaseBoundary, NodeStageRecord, InterStageEvent,
    Turn, StageBoundaries,
)
from .db import log_pipe_event, log_keep_warm_ping
from .node_types import PipelineType, NodeConfigLabel

logger = logging.getLogger(__name__)

PING_INTERVAL = 5.0


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


def stream_sync(async_iter_factory, *args, **kwargs):
    """Run async generator synchronously, yielding results."""
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