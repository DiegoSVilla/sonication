# Node Types and Connectivity

This document defines the node configuration labels, pipeline types, and connectivity rules for the Sonication pipeline-first SDK.

## 1. NodeConfigLabel Enum

```python
class NodeConfigLabel(StrEnum):
    STT_NON_STREAMING = "STT_NON_STREAMING"
    LLM_STREAMING = "LLM_STREAMING"
    LLM_STREAMING_WITH_REASONING = "LLM_STREAMING_WITH_REASONING"
    TTS_CHUNK_IN_STREAM_OUT = "TTS_CHUNK_IN_STREAM_OUT"
    TTS_NON_STREAMING = "TTS_NON_STREAMING"
```

### 1.1 STT_NON_STREAMING
- Used by: `STTNode`
- Behavior: Accepts PCM bytes, yields full transcript at once
- Stream contract: `yield {"kind": "transcript", "text": text}`, `yield {"kind": "done"}`

### 1.2 LLM_STREAMING
- Used by: `LLMNode` (standard chat)
- Behavior: Accepts messages, yields tokens one at a time
- Stream contract: `yield {"kind": "token", "content": "..."}`, `yield {"kind": "done"}`

### 1.3 LLM_STREAMING_WITH_REASONING
- Used by: `LLMNode` (with enable_thinking=True)
- Behavior: Accepts messages, yields reasoning tokens then content tokens
- Stream contract: `yield {"kind": "reasoning", "content": "..."}`, `yield {"kind": "token", "content": "..."}`, `yield {"kind": "done"}`

### 1.4 TTS_CHUNK_IN_STREAM_OUT
- Used by: `TTSNode`
- Behavior: Accepts text, yields PCM audio chunks
- Stream contract: `yield {"kind": "audio", "pcm": b"..."}`, `yield {"kind": "done"}`

### 1.5 TTS_NON_STREAMING
- Used by: `TTSNode` (future non-streaming variant)
- Behavior: Accepts text, yields full audio at once
- Stream contract: `yield {"kind": "audio", "pcm": b"..."}`, `yield {"kind": "done"}`

## 2. Node Properties

### 2.1 config_label
```python
@property
def config_label(self) -> str:
    """The streaming configuration label for this node."""
    return self._config_label  # set by subclass
```

Each subclass declares its label:
- `STTNode`: `_config_label = "STT_NON_STREAMING"`
- `LLMNode`: `_config_label = "LLM_STREAMING"` (or `LLM_STREAMING_WITH_REASONING` for agent mode)
- `TTSNode`: `_config_label = "TTS_CHUNK_IN_STREAM_OUT"`

### 2.2 stage_id
```python
@property
def stage_id(self) -> Optional[str]:
    """Unique ID for this node's stage in a pipeline."""
    return self._stage_id  # set by HotPipe during execution
```

### 2.3 node_class
```python
@property
def node_class(self) -> str:
    """Class name for observability."""
    return self.__class__.__name__  # e.g. "LLMNode"
```

## 3. PipelineType Enum

```python
class PipelineType(StrEnum):
    SI_SO_THREE_STEP_PIPELINE_CHAT = "SI_SO_THREE_STEP_PIPELINE_CHAT"
    TI_SO_TWO_STEP_PIPELINE_CHAT = "TI_SO_TWO_STEP_PIPELINE_CHAT"
    SI_TO_ONE_STEP_PIPELINE_TRANSCRIBE = "SI_TO_ONE_STEP_PIPELINE_TRANSCRIBE"
    TI_SO_ONE_STEP_PIPELINE_SPEAK = "TI_SO_ONE_STEP_PIPELINE_SPEAK"
    SI_SO_THREE_STEP_PIPELINE_TRANSLATE = "SI_SO_THREE_STEP_PIPELINE_TRANSLATE"
    SI_SO_THREE_STEP_PIPELINE_AGENT = "SI_SO_THREE_STEP_PIPELINE_AGENT"
```

### 3.1 Naming Convention
- `SI_` = Speech Input (STT first)
- `TI_` = Text Input (LLM first, no STT)
- `SO_` = Speech Output (TTS last)
- `TO_` = Text Output (no TTS)
- `THREE_STEP` = STT→LLM→TTS
- `TWO_STEP` = LLM→TTS
- `ONE_STEP` = Single stage only

## 4. Connectivity Rules

```python
connectivity_rules: dict[tuple[str, str], Optional[str]] = {
    ("STT_NON_STREAMING", "LLM_STREAMING"): None,
    ("LLM_STREAMING", "TTS_CHUNK_IN_STREAM_OUT"): "phrase_gate",
    ("LLM_STREAMING_WITH_REASONING", "TTS_CHUNK_IN_STREAM_OUT"): "phrase_gate",
    ("LLM_STREAMING", "TTS_NON_STREAMING"): None,
}
```

### 4.1 Rules Explained
- `STT → LLM`: No special event needed, transcript flows directly
- `LLM → TTS_CHUNK_IN_STREAM_OUT`: **Phrase gate** enabled — LLM text buffered until phrase boundary
- `LLM → TTS_NON_STREAMING`: No phrase gate, full text sent to TTS at once
- `LLM_WITH_REASONING → TTS`: Phrase gate enabled for reasoning tokens too

### 4.2 Helper Function
```python
def get_inter_stage_event(from_label: str, to_label: str) -> Optional[str]:
    """Return the inter-stage event type between two node config labels."""
    return connectivity_rules.get((from_label, to_label))
```

## 5. Topology Slot Mapping

Each `PipelineType` defines which slots exist and which node class fills each:

### 5.1 SI_SO_THREE_STEP_PIPELINE_CHAT
- `stt` slot → `STTNode` (STT_NON_STREAMING)
- `llm` slot → `LLMNode` (LLM_STREAMING)
- `tts` slot → `TTSNode` (TTS_CHUNK_IN_STREAM_OUT)
- Connections: `stt→llm`, `llm→tts`

### 5.2 TI_SO_TWO_STEP_PIPELINE_CHAT
- `llm` slot → `LLMNode` (LLM_STREAMING)
- `tts` slot → `TTSNode` (TTS_CHUNK_IN_STREAM_OUT)
- Connections: `llm→tts`

### 5.3 SI_TO_ONE_STEP_PIPELINE_TRANSCRIBE
- `stt` slot → `STTNode` (STT_NON_STREAMING)
- Connections: none

### 5.4 TI_SO_ONE_STEP_PIPELINE_SPEAK
- `tts` slot → `TTSNode` (TTS_CHUNK_IN_STREAM_OUT)
- Connections: none

### 5.5 SI_SO_THREE_STEP_PIPELINE_TRANSLATE
Same as CHAT but LLM prompt is "Translate to {language}"

### 5.6 SI_SO_THREE_STEP_PIPELINE_AGENT
- `stt` slot → `STTNode` (STT_NON_STREAMING)
- `llm` slot → `LLMNode` (LLM_STREAMING_WITH_REASONING)
- `tts` slot → `TTSNode` (TTS_CHUNK_IN_STREAM_OUT)
- Connections: `stt→llm`, `llm→tts`

## 6. Node Whitelist

Only these node types are accepted:
- `STTNode`
- `LLMNode`
- `TTSNode`

Custom classes/subclasses NOT ALLOWED. If a node type doesn't match any slot:
- Error: `"No slot for Node type '{node_class}' on pipeline '{pipeline_type.value}'."`

If a node type matches 2+ slots:
- Error: `"Ambiguous node class: fits slots [{slot_names}]."`