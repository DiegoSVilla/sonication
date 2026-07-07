# Sonication SDK — Agent Contract

This document defines the rules, conventions, and architecture that ALL agents working on the Sonication SDK must follow. It is the single source of truth.

## 1. Architecture Principles

### 1.1 Pipeline-First
`HotPipe(pipeline_type=...)` is REQUIRED at `__init__`. No exceptions. This defines the topology — which nodes exist, which slots they fill, which edges connect them.

### 1.2 Auto-Wiring
`add_node(node)` auto-detects which slot to fill based on node type. NO `connect()` calls ever. Wiring is 100% topology-driven from `pipeline_type`.

### 1.3 Node Whitelist
Only `STTNode`, `LLMNode`, `TTSNode` accepted. No custom subclasses. Custom classes/subclasses NOT ALLOWED.

### 1.4 Ambiguity Detection
If a node type fits 2+ slots → error "Ambiguous node — fits slots [A, B]".
If a node type fits 0 slots → error "No slot for node type X on pipeline Y".

### 1.5 No base_url in SDK
Inference endpoints are configured by the user at node creation (`STTNode("http://...")`). The SDK does not define or manage base URLs.

### 1.6 Stage + Step
- `stage_id` repeats for same node-type across a turn
- `step_id` increments per invocation
- DB `node_stages` table uses both

### 1.7 Dual-Anchored Timestamps
Every event has `wallclock_ms` (absolute Unix) and `local_offset_ms` (relative to turn start). No raw floats without context.

### 1.8 Phase Boundaries Over Raw Floats
`Turn.boundaries.t_llm_ttft` replaces `Turn.llm_ttft_ms` (float). Phase boundaries are typed dataclasses with wallclock + local offset + event_id.

### 1.9 Phrase Gate
Between `LLM_STREAMING` and `TTS_CHUNK_IN_STREAM_OUT`, HotPipe synthesizes an `InterStageEvent: phrase_gate` when accumulated LLM text reaches `PHRASE_MIN_CHARS` (default 20) and ends with `PHRASE_END_CHARS` (`.!?`).

### 1.10 Keep Alive Pings Separate
`keep_warm_pings` table has `parent_turn_id` = null. Pings are NOT turn-aligned.

### 1.11 Backward Compatibility
Old-style float fields (`stt_start_ms`, `llm_ttft_ms`, etc.) preserved as aliases during migration period. New code uses `Turn.boundaries`.

## 2. Data Model Rules

### 2.1 Every dataclass has both:
- Pydantic model for validation (schemas.py)
- Frozen dataclass for immutability (events.py)

### 2.2 DB schema alignment
PhaseBoundary fields match DB columns exactly: `wallclock_ms`, `local_offset_ms`, `event_id`, `name`.

### 2.3 Pipeline topology = topology definition only
`_TOPOLOGY_MAP` maps pipelines to slots/connections. NO auto-instantiation of nodes. User provides actual node instances.

## 3. New Types (from node_types.py)

```python
class NodeConfigLabel(StrEnum):
    STT_NON_STREAMING = "STT_NON_STREAMING"
    LLM_STREAMING = "LLM_STREAMING"
    LLM_STREAMING_WITH_REASONING = "LLM_STREAMING_WITH_REASONING"
    TTS_CHUNK_IN_STREAM_OUT = "TTS_CHUNK_IN_STREAM_OUT"
    TTS_NON_STREAMING = "TTS_NON_STREAMING"

class PipelineType(StrEnum):
    SI_SO_THREE_STEP_PIPELINE_CHAT = "SI_SO_THREE_STEP_PIPELINE_CHAT"
    TI_SO_TWO_STEP_PIPELINE_CHAT = "TI_SO_TWO_STEP_PIPELINE_CHAT"
    SI_TO_ONE_STEP_PIPELINE_TRANSCRIBE = "SI_TO_ONE_STEP_PIPELINE_TRANSCRIBE"
    TI_SO_ONE_STEP_PIPELINE_SPEAK = "TI_SO_ONE_STEP_PIPELINE_SPEAK"
    SI_SO_THREE_STEP_PIPELINE_TRANSLATE = "SI_SO_THREE_STEP_PIPELINE_TRANSLATE"
    SI_SO_THREE_STEP_PIPELINE_AGENT = "SI_SO_THREE_STEP_PIPELINE_AGENT"
```

## 4. Connectivity Rules

```python
connectivity_rules: dict[tuple[str, str], Optional[str]] = {
    ("STT_NON_STREAMING", "LLM_STREAMING"): None,
    ("LLM_STREAMING", "TTS_CHUNK_IN_STREAM_OUT"): "phrase_gate",
    ("LLM_STREAMING_WITH_REASONING", "TTS_CHUNK_IN_STREAM_OUT"): "phrase_gate",
    ("LLM_STREAMING", "TTS_NON_STREAMING"): None,
}
```

## 5. New Data Models (from events.py)

### 5.1 NodeEvent
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

### 5.2 PhaseBoundary
```python
@dataclass(frozen=True)
class PhaseBoundary:
    name: str
    wallclock_ms: float
    local_offset_ms: float
    event_id: str
```

### 5.3 StageBoundaries
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

### 5.4 NodeStageRecord
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

### 5.5 InterStageEvent
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

## 6. New Database Tables (from db.py)

### 6.1 node_stages
```sql
CREATE TABLE IF NOT EXISTS node_stages (
    stage_id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL,
    session_id TEXT,
    conversation_id TEXT,
    node_name TEXT NOT NULL,
    node_class TEXT NOT NULL,
    config_label TEXT NOT NULL,
    start_wall_ms REAL NOT NULL,
    end_wall_ms REAL,
    timing_json TEXT,
    summary_json TEXT,
    FOREIGN KEY(turn_id) REFERENCES turns(id)
);
```

### 6.2 inter_stage_events
```sql
CREATE TABLE IF NOT EXISTS inter_stage_events (
    event_id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL,
    session_id TEXT,
    conversation_id TEXT,
    event_type TEXT NOT NULL,
    wallclock_ms REAL NOT NULL,
    local_offset_ms REAL,
    seq INTEGER,
    from_stage_id TEXT,
    to_stage_id TEXT,
    payload_json TEXT NOT NULL,
    UNIQUE(turn_id, seq)
);
```

### 6.3 keep_warm_pings
```sql
CREATE TABLE IF NOT EXISTS keep_warm_pings (
    ping_id TEXT PRIMARY KEY,
    node_name TEXT NOT NULL,
    wallclock_ms REAL NOT NULL,
    rtt_ms REAL NOT NULL,
    parent_turn_id TEXT,
    FOREIGN KEY(parent_turn_id) REFERENCES turns(id)
);
```

## 7. Analysis Layer (from analysis.py)

### 7.1 AnalysisSegment
```python
@dataclass
class AnalysisSegment:
    stage_name: str
    ms: float
    kind: str  # "transport" | "service" | "pipeline"
    ping_ms: Optional[float] = None
    internal_ms: Optional[float] = None
```

### 7.2 PipelineAnalysis
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

### 7.3 LatencyAnalyser
Static helper with methods:
- `analyse(turn)` → PipelineAnalysis
- `segments(turn)` → list[AnalysisSegment]
- `shot_latency(turn, pipeline_type)` → float
- `segment_mismatch(turn, template)` → str
- `statistical_summary(turns)` → dict with p50/p90/p99

## 8. New Exports (from __init__.py)

```python
from .node_types import NodeConfigLabel, PipelineType, get_inter_stage_event
from .events import NodeEvent, InterStageEvent, PhaseBoundary, NodeStageRecord, StageBoundaries
from .hotpipe import HotPipe, Turn
from .analysis import LatencyAnalyser, PipelineAnalysis, AnalysisSegment
from .db import init_db, log_node_event, insert_inter_stage_event, log_keep_warm_ping, log_node_stage
```

## 9. QA Strategy (Mute/Deaf QA Agent)

The QA Agent cannot read source code. It only sees:
- Audio artifacts (PCM/WAV files)
- DB records (node_stages, inter_stage_events, pipe_events)
- Analysis output (PipelineAnalysis segments)
- Test pass/fail

### QA Test Matrix
| # | Test | Input | Pipeline | Expected Evidence |
|---|------|-------|----------|-------------------|
| QA-1 | TTS→STT Roundtrip | Text: "This is a test phrase." | TI_SO_ONE_STEP_PIPELINE_SPEAK | audio file → STT transcript matches ±2 chars |
| QA-2 | STT transcription | PCM from QA-1 audio | SI_TO_ONE_STEP_PIPELINE_TRANSCRIBE | node_stages has 1 row, pipe_events has stt_transcript + stt_done |
| QA-3 | LLM speech-to-text → TTS | Text: "What is the capital of France?" | TI_SO_TWO_STEP_PIPELINE_CHAT | phrase_gate in inter_stage_events[t_llm_ttft → t_tts_req] |
| QA-4 | Full 3-step | PCM (external audio) | SI_SO_THREE_STEP_PIPELINE_CHAT | full chain: STT→LLM→TTS, all boundaries populated |
| QA-5 | Translation app echo | Text: "Hello, how are you?" | SI_SO_THREE_STEP_PIPELINE_TRANSLATE | LLM translated correctly, TTS produces audio |
| QA-6 | Grammar segment check | Any input | SI_SO_THREE_STEP_PIPELINE_CHAT | phrase_gate present, node_stages has 3 rows |

### QA Gate Criteria (per test)
1. Audio file exists in `data/audio/qa_<test_id>_<timestamp>.pcm`
2. STT transcript matches source text (±2 characters tolerance)
3. `node_stages` table has correct number of rows
4. `inter_stage_events` table has `phrase_gate` event when applicable
5. `pipe_events` table has events with `stage_id` and `seq` set

## 10. Execution Order & Dependencies

```
Phase 0 (SEED — no deps):
  [0-A] Fix node_registry.py import
  [0-B] Fix app.py attribute
  Gate: `python -c "import backend"` → works

Phase 1 (Foundation Types):
  [1] Create node_types.py (NodeConfigLabel, PipelineType, connectivity rules)
  Gate: `import backend.node_types` works

Phase 2 (Event Data Models):
  [2] Extend events.py (NodeEvent, PhaseBoundary, StageBoundaries, etc.)
  Depends on: 1 (uses NodeConfigLabel strings)
  Gate: `import backend.events` works

Phase 3 (Database Layer):
  [3] Extend db.py (new tables, migration, CRUD ops)
  Depends on: 2 (uses field names from events)
  Gate: DB schema creates all tables + CRUD works

Phase 4 (Node Layer):
  [4] Enhance nodes.py (config_label, stage_id properties)
  Depends on: 1 (uses NodeConfigLabel enum)
  Gate: node.config_label works on all subclasses

Phase 5 (HotPipe Core — THE BIG ONE):
  [5] Refactor hotpipe.py (topology, phase boundaries, phrase gate, Turn.recording)
  Depends on: 2, 3, 4
  Gate: `HotPipe(pipeline_type=SI_SO_THREE_STEP_PIPELINE_CHAT)` auto-wires

Phase 6 (Analysis Layer):
  [6] Create analysis.py (LatencyAnalyser, PipelineAnalysis, segments)
  Depends on: 2, 5
  Gate: `LatencyAnalyser.analyse(turn)` returns PipelineAnalysis with valid segments

Phase 7 (Polish):
  [7] Extend schemas.py (Pydantic models), update __init__.py (exports)
  Depends on: 2, 6
  Gate: `import backend` imports all new types

Phase 8 (Regression):
  [8] Verify all existing tests still pass
  Gate: No regression
```

Total estimated effort: ~10-12 hours depending on thoroughness.

---

This plan is complete. Ready for Coder to execute.