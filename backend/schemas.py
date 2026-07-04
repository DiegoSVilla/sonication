"""Border objects: the request/response schemas that cross the API boundary.

These exist so /docs (OpenAPI) is meaningful. The HTTP handlers still return
JSONResponse for speed, so these models document the shapes without adding
response validation on the hot path.
"""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class PingSnapshot(BaseModel):
    """Median health-route round-trip (ms) per backend service, over the last few
    5-second checks. A field is null until the first successful sample."""
    llm: Optional[int] = Field(None, description="RTT to the LLM host, ms")
    tts: Optional[int] = Field(None, description="RTT to the TTS host, ms")
    stt: Optional[int] = Field(None, description="RTT to the STT host, ms")


class StatBlock(BaseModel):
    """Percentile summary of one metric across turns, in milliseconds.
    An `n=0` block (all other fields absent) means no data yet."""
    n: int
    min: Optional[float] = None
    p50: Optional[float] = None
    p90: Optional[float] = None
    p99: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None


class AudioFormat(BaseModel):
    """PCM format the TTS emits (duration = pcm_bytes / bytes_per_sec)."""
    sample_rate: int
    channels: int
    bytes_per_sec: int


class WaterfallSegment(BaseModel):
    """One stage of the shot decomposition."""
    stage: str = Field(description="pre_stt | stt | pre_llm | llm_ttft | phrase_gate | tts_ttfb | channel_out")
    ms: float = Field(description="wall-clock duration of this stage")
    kind: Literal["transport", "service", "pipeline"]
    ping_ms: Optional[float] = Field(None, description="network floor for service stages")
    internal_ms: Optional[float] = Field(
        None, description="service-only estimate = observed - ping (clamped >= 0)")


class Waterfall(BaseModel):
    """Per-turn stage breakdown. `shot_ms` is ALWAYS the wall-clock distance
    release -> first audio on channel; it is never derived from the segment sum."""
    segments: list[WaterfallSegment]
    shot_ms: Optional[float] = None
    sum_ms: float = Field(description="sum of segments; diagnostic only")
    residual_ms: Optional[float] = Field(None, description="shot_ms - sum_ms (should be ~0)")
    pings: PingSnapshot


class AudioStats(BaseModel):
    """Audio-lead / underrun analysis for a turn."""
    chunks: int
    underrun: bool = Field(description="true if generated audio fell behind the playback clock")
    max_lateness_ms: Optional[float] = None
    lead_ms: Optional[float] = Field(None, description="audio buffered ahead when generation finished")
    audio_generated_ms: Optional[float] = None


class TurnAnalysis(BaseModel):
    """Per-turn (shot) metrics."""
    turn_index: int
    chars_in: Optional[int] = None
    stt_ms: Optional[float] = None
    waterfall: Optional[Waterfall] = None
    llm_ttft_ms: Optional[float] = None
    llm_tokens: Optional[int] = None
    tts_ttfb_ms: Optional[float] = None
    shot_latency_ms: Optional[float] = Field(None, description="wall-clock release -> first audio on channel")
    first_audio_channel_t_ms: Optional[float] = None
    audio_generated_ms: Optional[float] = None
    audio: AudioStats


class CallAnalysis(BaseModel):
    """Per-call timing breakdown (GET /api/analysis/{call_id})."""
    call_id: str
    started_iso: Optional[str] = None
    model: Optional[str] = None
    voice: Optional[str] = None
    turns: list[TurnAnalysis]
    shot_latency_ms: StatBlock
    stt_ms: StatBlock
    llm_ttft_ms: StatBlock
    tts_ttfb_ms: StatBlock


class AggregateAnalysis(BaseModel):
    """Cross-call rollup (GET /api/analysis)."""
    calls: int
    turns: int
    audio_format: AudioFormat
    shot_latency_ms: StatBlock
    stt_ms: StatBlock
    llm_ttft_ms: StatBlock
    tts_ttfb_ms: StatBlock
    audio_generated_ms: StatBlock


class CallInfo(BaseModel):
    """A call row (GET /api/calls). call_start is wallclock zero for the call."""
    id: str
    session_id: Optional[str] = None
    started_iso: Optional[str] = None
    started_epoch_ms: Optional[float] = None
    model: Optional[str] = None
    voice: Optional[str] = None
    language: Optional[str] = None
    params_json: Optional[str] = None
    ended_t_ms: Optional[float] = None


class Event(BaseModel):
    """One timed event. `t_ms` is relative to the call's call_start (wallclock zero)."""
    seq: int
    call_id: str
    turn_id: Optional[int] = None
    type: str = Field(description="call_start | text_in | llm_call | tts_call | audio_out | stt_final | channel_playback_start | call_end")
    t_ms: float
    payload: dict[str, Any]


class CallEventsExport(BaseModel):
    """Full event dump for one call (GET /api/calls/{call_id}/events)."""
    call: Optional[CallInfo] = None
    turns: list[dict[str, Any]]
    events: list[Event]


class Message(BaseModel):
    """One conversation message for the optional per-shot context override."""
    role: Literal["system", "user", "assistant"]
    content: str


class EdgeSessionEnded(BaseModel):
    ended: str = Field(description="session id that was ended")
    call_id: str


class ErrorResponse(BaseModel):
    error: str
