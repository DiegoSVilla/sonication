"""Pydantic validation models for pipeline-first event types."""
from typing import Dict, Any, Optional, Literal
from pydantic import BaseModel


class NodeEventModel(BaseModel):
    """Validation for NodeEvent from a node with stage context."""
    stage_id: str
    event_type: str
    wallclock_ms: float
    local_offset_ms: float
    seq: int
    payload: Dict[str, Any]
    event_id: str
    parent_event_id: Optional[str] = None


class PhaseBoundaryModel(BaseModel):
    """Validation for PhaseBoundary anchor."""
    name: str
    wallclock_ms: float
    local_offset_ms: float
    event_id: str


class StageBoundariesModel(BaseModel):
    """Validation for StageBoundaries with all typed anchors."""
    stage_id: str
    t_start: Optional[PhaseBoundaryModel] = None
    t_end: Optional[PhaseBoundaryModel] = None
    t_stt_req: Optional[PhaseBoundaryModel] = None
    t_stt_resp: Optional[PhaseBoundaryModel] = None
    t_llm_req: Optional[PhaseBoundaryModel] = None
    t_llm_ttft: Optional[PhaseBoundaryModel] = None
    t_llm_resp: Optional[PhaseBoundaryModel] = None
    t_tts_req: Optional[PhaseBoundaryModel] = None
    t_tts_ttfb: Optional[PhaseBoundaryModel] = None
    t_tts_resp: Optional[PhaseBoundaryModel] = None


class LlmTokenPayloadModel(BaseModel):
    """Validation for LLM token data."""
    content: str
    channel: Literal["content", "reasoning"] = "content"
    token_count: int
    delta_offset_ms: float = 0.0


class TtsChunkPayloadModel(BaseModel):
    """Validation for TTS audio payload."""
    pcm_bytes: int
    cumulative_ms: float
    phrase_source: str
    chunk_index: int = 0


class PhraseGatePayloadModel(BaseModel):
    """Validation for phrase gate payload."""
    warming_up: bool
    sentences: int
    has_phrase_ready: bool
    buffered_text: str


class NodeStageRecordModel(BaseModel):
    """Validation for NodeStageRecord."""
    stage_id: str
    node_name: str
    node_class: str
    config_label: str
    start_wall_ms: float
    end_wall_ms: Optional[float] = None
    timing: Dict[str, float] = {}
    events: list[NodeEventModel] = []
    payload_kind: str