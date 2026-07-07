# Event Stream and Logging Architecture

This document defines the event stream architecture for the Sonication pipeline-first SDK. It covers event types, data models, database schema, and analysis.

## 1. Event Types

### 1.1 Node Events
Events emitted by individual nodes during streaming:
- `stt_transcript`: STT node produced a transcript
- `stt_done`: STT stream completed
- `llm_token`: LLM produced a token
- `llm_reasoning`: LLM produced a reasoning token
- `llm_done`: LLM stream completed
- `llm_usage`: LLM provided usage statistics
- `tts_audio_chunk`: TTS produced an audio chunk
- `tts_done`: TTS stream completed
- `tts_usage`: TTS provided usage statistics

### 1.2 Inter-Stage Events
Events synthesized by HotPipe between stages:
- `phrase_gate`: LLM text reached phrase boundary, ready for TTS
- `pipeline_start`: Turn began execution
- `turn_start`: Turn lifecycle event

### 1.3 Turn Events
Events at the turn level:
- `turn_start`: Turn execution began
- `turn_error`: Turn failed with exception

## 2. Event Data Model

### 2.1 PipeEvent (Extended)
```python
@dataclass(frozen=True)
class PipeEvent:
    type: str
    node_name: str
    turn_id: str
    wallclock_ms: float
    local_offset_ms: float
    payload: dict[str, Any]
    event_id: str
    parent_event_id: Optional[str] = None
    stage_id: str = ""
    seq: int = 0
```

**Changes from original:**
- `node` → `node_name` (consistency with DB schema)
- Added `stage_id: str` (which node stage this event belongs to)
- Added `seq: int` (sequence number within stage)

### 2.2 NodeEvent
```python
@dataclass(frozen=True)
class NodeEvent:
    stage_id: str
    event_type: str
    wallclock_ms: float
    local_offset_ms: float
    seq: int
    payload: dict[str, Any]
    event_id: str
    parent_event_id: Optional[str] = None
```

### 2.3 PhaseBoundary
```python
@dataclass(frozen=True)
class PhaseBoundary:
    name: str
    wallclock_ms: float
    local_offset_ms: float
    event_id: str
```

**Named anchors:**
- `t_stt_req`: STT request started
- `t_stt_resp`: STT response completed
- `t_llm_req`: LLM request started
- `t_llm_ttft`: LLM time-to-first-token
- `t_llm_resp`: LLM response completed
- `t_tts_req`: TTS request started
- `t_tts_ttfb`: TTS time-to-first-byte
- `t_tts_resp`: TTS response completed

### 2.4 StageBoundaries
```python
class StageBoundaries:
    t_stt_req: Optional[PhaseBoundary] = None
    t_stt_resp: Optional[PhaseBoundary] = None
    t_llm_req: Optional[PhaseBoundary] = None
    t_llm_ttft: Optional[PhaseBoundary] = None
    t_llm_resp: Optional[PhaseBoundary] = None
    t_tts_req: Optional[PhaseBoundary] = None
    t_tts_ttfb: Optional[PhaseBoundary] = None
    t_tts_resp: Optional[PhaseBoundary] = None
```

### 2.5 NodeStageRecord
```python
@dataclass(frozen=True)
class NodeStageRecord:
    stage_id: str
    node_name: str
    node_class: str
    config_label: str
    start_wall_ms: float
    end_wall_ms: Optional[float] = None
    timing: dict[str, float] = field(default_factory=dict)
    events: list = field(default_factory=list)
    payload_kind: str = "unknown"
```

### 2.6 InterStageEvent
```python
@dataclass(frozen=True)
class InterStageEvent:
    event_type: str
    wallclock_ms: float
    local_offset_ms: float = 0.0
    seq: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str
    parent_event_id: Optional[str] = None
    from_stage_id: Optional[str] = None
    to_stage_id: Optional[str] = None
```

## 3. Phrase Gate Semantics

### 3.1 How It Works
1. During `_walk()`, HotPipe accumulates LLM tokens in `llm_text_buffer`
2. After each token, checks `_phrase_ready(llm_text_buffer)`
3. When phrase is ready, emits `InterStageEvent: phrase_gate`
4. Buffered text chunk is pushed to TTS via `TTS_CHUNK_IN_STREAM_OUT` API

### 3.2 Phrase Readiness Criteria
- `len(text) >= PHRASE_MIN_CHARS` (default 20)
- `text.rstrip()[-1] in PHRASE_END_CHARS` (`.`, `!`, `?`, `\n`)

### 3.3 Inter-Stage Event Payload
```python
{
    "warming_up": bool,      # True if first phrase
    "sentences": int,        # Number of sentences in buffer
    "has_phrase_ready": bool,
    "buffered_text": str,    # First 200 chars of buffer
}
```

## 4. Database Schema

### 4.1 node_stages
```sql
CREATE TABLE IF NOT EXISTS node_stages (
    stage_id            TEXT PRIMARY KEY,
    turn_id             TEXT NOT NULL,
    session_id          TEXT,
    conversation_id     TEXT,
    node_name           TEXT NOT NULL,
    node_class          TEXT NOT NULL,
    config_label        TEXT NOT NULL,
    start_wall_ms       REAL NOT NULL,
    end_wall_ms         REAL,
    timing_json         TEXT,
    summary_json        TEXT,
    FOREIGN KEY(turn_id) REFERENCES turns(id)
);
```

### 4.2 inter_stage_events
```sql
CREATE TABLE IF NOT EXISTS inter_stage_events (
    event_id            TEXT PRIMARY KEY,
    turn_id             TEXT NOT NULL,
    session_id          TEXT,
    conversation_id     TEXT,
    event_type          TEXT NOT NULL,
    wallclock_ms        REAL NOT NULL,
    local_offset_ms     REAL,
    seq                 INTEGER,
    from_stage_id       TEXT,
    to_stage_id         TEXT,
    payload_json        TEXT NOT NULL,
    UNIQUE(turn_id, seq)
);
```

### 4.3 keep_warm_pings
```sql
CREATE TABLE IF NOT EXISTS keep_warm_pings (
    ping_id             TEXT PRIMARY KEY,
    node_name           TEXT NOT NULL,
    wallclock_ms        REAL NOT NULL,
    rtt_ms              REAL NOT NULL,
    parent_turn_id      TEXT,
    FOREIGN KEY(parent_turn_id) REFERENCES turns(id)
);
```

### 4.4 Extended pipe_events
```sql
ALTER TABLE pipe_events ADD COLUMN IF NOT EXISTS stage_id TEXT;
ALTER TABLE pipe_events ADD COLUMN IF NOT EXISTS seq INTEGER;
```

## 5. Turn Recording

### 5.1 Turn.__init__
```python
self.boundaries = StageBoundaries()
self.node_stages: dict[str, NodeStageRecord] = {}
self.inter_stage_events: list[InterStageEvent] = []
self.turn_events: list[PipeEvent] = []
self._pipeline_type = pipeline_type or "manual"
self._event_seq = 0
```

### 5.2 Turn Methods
```python
def record_node_stage(self, record: NodeStageRecord) -> None
def record_inter_stage_event(self, event: InterStageEvent) -> None
def boundary(self, name: str) -> Optional[PhaseBoundary]
def interval(self, start_name: str, end_name: str) -> Optional[float]
def shot_latency_ms(self) -> float
def analyse(self) -> PipelineAnalysis
```

### 5.3 Phase Boundary Recording in _walk()
During `_walk()`, as events stream from nodes:
1. Generate `stage_id` via `HotPipe._generate_stage_id(node, node_name)`
2. Create `NodeStageRecord` with `stage_id`, `node_name`, `node_class`, `config_label`
3. Record `NodeStageRecord` via `self.record_node_stage(stage_record)`
4. For each event, record `PhaseBoundary` if first occurrence of that type
5. For LLM→TTS connections, check `_phrase_ready()` and emit `InterStageEvent`

## 6. Analysis Layer

### 6.1 AnalysisSegment
```python
@dataclass
class AnalysisSegment:
    stage_name: str
    ms: float
    kind: str  # "transport" | "service" | "pipeline"
    ping_ms: Optional[float] = None
    internal_ms: Optional[float] = None
```

### 6.2 PipelineAnalysis
```python
@dataclass
class PipelineAnalysis:
    turn_id: str
    segments: list[AnalysisSegment]
    shot_latency_ms: float
    stt_ms: float
    llm_ttft_ms: float
    tts_ttfb_ms: float
```

### 6.3 LatencyAnalyser
Static helper with methods:
- `analyse(turn)` → PipelineAnalysis
- `segments(turn)` → list[AnalysisSegment]
- `shot_latency(turn, pipeline_type)` → float
- `segment_mismatch(turn, template)` → str
- `statistical_summary(turns)` → dict with p50/p90/p99

### 6.4 Waterfall Segments
```python
add("pre_stt", "t_release", "t_stt_req", "transport")
add("stt", "t_stt_req", "t_stt_resp", "service")
add("pre_llm", llm_from, "t_llm_req", "pipeline")
add("llm_ttft", "t_llm_req", "t_llm_ttft", "service")
add("phrase_gate", "t_llm_ttft", "t_tts_req", "pipeline")
add("tts_ttfb", "t_tts_req", "t_tts_ttfb", "service")
```

## 7. CRUD Operations

```python
def log_node_event(event: dict) -> None
def insert_inter_stage_event(event: dict) -> None
def log_keep_warm_ping(node_name, wallclock_ms, rtt_ms, parent_turn_id=None) -> None
def log_node_stage(record: dict) -> None
def get_node_stages(turn_id: str) -> list[dict]
def get_inter_stage_events(turn_id: str) -> list[dict]
def get_keep_warm_pings(parent_turn_id=None) -> list[dict]
```

## 8. Backward Compatibility

Old-style float fields preserved during migration:
- `Turn.stt_start_ms` → alias to `Turn.boundaries.t_stt_req.local_offset_ms`
- `Turn.llm_ttft_ms` → alias to `Turn.boundaries.t_llm_ttft.local_offset_ms`
- `Turn.tts_ttfb_ms` → alias to `Turn.boundaries.t_tts_ttfb.local_offset_ms`

New code should use `Turn.boundaries` exclusively.