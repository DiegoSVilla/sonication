# Main Pipeline Classes

This document defines the HotPipe and Turn classes, their architecture, and how they work together in the pipeline-first SDK.

## 1. HotPipe Class

### 1.1 Initialization
```python
class HotPipe:
    def __init__(self, pipeline_type: PipelineType):
        """Pipeline type is REQUIRED at init."""
        if not pipeline_type:
            raise ValueError("HotPipe requires pipeline_type at init. Got None.")
        
        self.pipeline_type = pipeline_type
        self._slots: dict[str, Node] = {}
        self._node_types: set = set()
        self.nodes = {}  # name -> (Node, category) — backward compat
        self.connections = {}  # name -> [to_name...] — backward compat
        self._pipeline_topology = self._TOPOLOGY.get(pipeline_type)
        self._connections: list[tuple[str, str]] = []
        self._stage_counter: dict[str, int] = {}
```

### 1.2 Topology Map
```python
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
    # ... other pipeline types
}
```

### 1.3 add_node(node) — Auto-Slot by Type
```python
def add_node(self, node: Node) -> None:
    """Auto-detect pipeline slot based on node type."""
    node_class = node.__class__.__name__
    
    # Whitelisted node types only
    allowed_types = ("STTNode", "LLMNode", "TTSNode")
    if node_class not in allowed_types:
        raise ValueError(f"Unhandled node type: '{node_class}'. Only {allowed_types} allowed.")
    
    # Look up topology slot(s) for this node class
    candidates = [
        s for s in self._pipeline_topology["slots"]
        if s["node_class"] == node_class and s["slot_name"] not in self._slots
    ]
    
    if not candidates:
        # Check if already added
        if node_class in [s["node_class"] for s in self._pipeline_topology["slots"]
                          if s["slot_name"] in self._slots]:
            raise ValueError(f"No slot for Node type '{node_class}': already added to pipeline.")
        else:
            raise ValueError(f"No slot for Node type '{node_class}' on pipeline '{self.pipeline_type.value}'.")
    
    elif len(candidates) > 1:
        slot_names = [c["slot_name"] for c in candidates]
        raise ValueError(f"Ambiguous node class: fits slots {slot_names}.")
    
    else:
        slot = candidates[0]
        self._slots[slot["slot_name"]] = node
        # Backward compat: add to nodes dict
        self.nodes[slot["slot_name"]] = (node, slot["config_label"])
        self.connections[slot["slot_name"]] = []
```

### 1.4 connect() — Validation Only
```python
def connect(self) -> bool:
    """Validate all nodes are connected and topology is sound."""
    self._validate_topology()
    return True

def _validate_topology(self) -> None:
    """Check all expected slots are filled."""
    for slot in self._pipeline_topology["slots"]:
        if slot["slot_name"] not in self._slots:
            raise ValueError(f"Missing node for slot '{slot['slot_name']}' in pipeline '{self.pipeline_type.value}'.")
```

### 1.5 _generate_stage_id(node, node_name)
```python
def _generate_stage_id(self, node: Node, node_name: str) -> str:
    """Generate a deterministic stage_id for a node invocation."""
    config = node.config_label
    label = config if hasattr(config, 'value') else config
    
    if node_name not in self._stage_counter:
        self._stage_counter[node_name] = 0
    self._stage_counter[node_name] += 1
    step = self._stage_counter[node_name]
    
    node_id = str(uuid.uuid4())[:8]
    return f"{node_name}_{node_id}_{label}_{step}"
```

### 1.6 turn(entry_point, data)
```python
async def turn(self, entry_point: str, data, pipeline_type: str = "manual"):
    """Execute one turn. Returns dict with results."""
    turn_id = f"turn_{round(time.time() * 1000)}"
    start_wall = time.time() * 1000
    turn = Turn(turn_id, start_wall, start_wall,
                pipeline_type=self.pipeline_type.value if self.pipeline_type else pipeline_type)
    try:
        return await turn.turn(self, entry_point, data)
    except Exception as e:
        logger.error(f"Turn {turn_id} failed: {e}")
        raise
```

## 2. Turn Class

### 2.1 Initialization
```python
class Turn:
    def __init__(self, turn_id: str, start_wall_ms: float, start_mono_ms: float,
                 pipeline_type: str = "manual"):
        self.turn_id = turn_id
        self.start_wall = start_wall_ms
        self.start_mono = start_mono_ms
        self.boundaries = StageBoundaries()
        self.node_stages: dict[str, NodeStageRecord] = {}
        self.inter_stage_events: list[InterStageEvent] = []
        self.events: list[PipeEvent] = []
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
```

### 2.2 Methods
```python
def record_node_stage(self, record: NodeStageRecord) -> None
def record_inter_stage_event(self, event: InterStageEvent) -> None
def boundary(self, name: str) -> Optional[PhaseBoundary]
def interval(self, start_name: str, end_name: str) -> Optional[float]
def shot_latency_ms(self) -> float
def analyse(self) -> PipelineAnalysis
```

### 2.3 shot_latency_ms()
```python
def shot_latency_ms(self) -> float:
    """Pipeline-type-aware shot latency."""
    if self.boundaries.t_stt_req and self.boundaries.t_tts_req:
        return round(self.boundaries.t_tts_req.wallclock_ms - self.boundaries.t_stt_req.wallclock_ms, 3)
    # Legacy fallback
    if self.tts_ttfb_ms and self.stt_start_ms:
        return self.tts_ttfb_ms - self.stt_start_ms
    return 0.0
```

## 3. _walk() — Phase Boundary Recording

### 3.1 Overview
`_walk()` is the core recursive method that executes nodes in pipeline order. During execution:
1. Generate `stage_id` and assign to node
2. Create `NodeStageRecord` and register it
3. For each streaming event, record `PhaseBoundary` if first occurrence
4. For LLM→TTS connections, check `_phrase_ready()` and emit `InterStageEvent`
5. After stream completes, update `NodeStageRecord.end_wall_ms`
6. Follow downstream connections

### 3.2 Phase Boundary Recording
```python
def _record_phase_boundary(self, stage_record, raw, node_name, config_label):
    """Record phase boundaries as events arrive from streaming nodes."""
    now_ms = time.time() * 1000
    events = self.events
    kind = raw.get("kind")
    
    if node_name == "stt":
        if kind == "transcript" and self.boundaries.t_stt_req is None:
            pb = PhaseBoundary(name="t_stt_req", wallclock_ms=now_ms,
                             local_offset_ms=self._now(), event_id=events[-1].event_id)
            self.boundaries.t_stt_req = pb
            stage_record.timing["t_stt_req"] = now_ms
        # ... similar for t_stt_resp
```

### 3.3 Phrase Gate
```python
if node_name == "llm" and llm_text_buffer and config_label == NodeConfigLabel.LLM_STREAMING:
    downstream_tts = any(
        pipe.nodes.get(n, ())[1] == NodeConfigLabel.TTS_CHUNK_IN_STREAM_OUT
        for n in pipe.connections.get(node_name, [])
    )
    if downstream_tts and _phrase_ready(llm_text_buffer):
        self._emit_phrase_gate_event(node_name, stage_id, llm_text_buffer)

def _emit_phrase_gate_event(self, from_node_name, from_stage_id, buffered_text):
    """Emit an InterStageEvent for a phrase gate between LLM and TTS."""
    self.inter_stage_events.append(InterStageEvent(
        event_type="phrase_gate",
        wallclock_ms=time.time() * 1000,
        local_offset_ms=self._now(),
        seq=self._next_seq(),
        payload={
            "warming_up": len(buffered_text) < 100,
            "sentences": max(1, len(buffered_text.split("."))),
            "has_phrase_ready": True,
            "buffered_text": buffered_text[:200],
        }
    ))
```

## 4. Example Usage

### 4.1 Three-Step Chat Pipeline
```python
# User creates nodes with their own vLLM config
stt = STTNode("http://my-vllm:8092")
llm = LLMNode("http://my-vllm:8093", system_prompt="You are a helpful assistant")
tts = TTSNode("http://my-vllm:8094", voice="Marco", language="English")

# Pipeline-first: set template, then add nodes
pipe = HotPipe(pipeline_type=PipelineType.SI_SO_THREE_STEP_PIPELINE_CHAT)
pipe.add_node(stt)   # Auto-fills "stt" slot (STTNode type)
pipe.add_node(llm)   # Auto-fills "llm" slot (LLMNode type)
pipe.add_node(tts)   # Auto-fills "tts" slot (TTSNode type)
pipe.connect()       # Validates: all 3 slots filled, topology is sound

# Turn execution — fully automated chain
turn = await pipe.turn("stt", audio_bytes)
turn.boundaries.t_llm_ttft   # PhaseBoundary(42.3ms, ...)
turn.boundaries.t_tts_ttfb   # PhaseBoundary(200.1ms, ...)
turn.shot_latency_ms()  # 168.9 (wallclock interval)
turn.analyse()  # PipelineAnalysis(segments=[...], shot=168.9)
```

### 4.2 Translation Pipeline
```python
stt = STTNode("http://my-vllm:8092")
llm = LLMNode("http://my-vllm:8093", system_prompt="Translate to French", model="MyCustomModel")
tts = TTSNode("http://my-vllm:8094", voice="Marco", language="French")

pipe = HotPipe(pipeline_type=PipelineType.SI_SO_THREE_STEP_PIPELINE_TRANSLATE)
pipe.add_node(stt)
pipe.add_node(llm)
pipe.add_node(tts)
pipe.connect()

turn = await pipe.turn("stt", audio_bytes)
```

## 5. Backward Compatibility

### 5.1 HotPipe.nodes and HotPipe.connections
These are preserved as backward-compat dicts:
- `self.nodes[name] = (node, category)` — same format as before
- `self.connections[from_name] = [to_name...]` — same format as before

### 5.2 Turn legacy float fields
- `self.stt_start_ms`, `self.llm_ttft_ms`, `self.tts_ttfb_ms`, etc. — preserved
- New code should use `Turn.boundaries` exclusively