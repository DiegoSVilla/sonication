"""Node types — configuration labels, pipeline types, and connectivity rules.

Defines:
  - NodeConfigLabel: the streaming configuration label for each node
  - PipelineType: the pipeline template types (CHAT, TRANSCRIBE, SPEAK, etc.)
  - connectivity_rules: maps (from_label, to_label) -> inter_stage_event_name
  - get_inter_stage_event(): lookup helper
"""
from enum import StrEnum
from typing import Optional


class NodeConfigLabel(StrEnum):
    """Streaming configuration label for each node type."""
    STT_NON_STREAMING = "STT_NON_STREAMING"
    LLM_STREAMING = "LLM_STREAMING"
    LLM_STREAMING_WITH_REASONING = "LLM_STREAMING_WITH_REASONING"
    TTS_CHUNK_IN_STREAM_OUT = "TTS_CHUNK_IN_STREAM_OUT"
    TTS_NON_STREAMING = "TTS_NON_STREAMING"


class PipelineType(StrEnum):
    """Pipeline template type — defines topology (slots + connections)."""
    SI_SO_THREE_STEP_PIPELINE_CHAT = "SI_SO_THREE_STEP_PIPELINE_CHAT"
    TI_SO_TWO_STEP_PIPELINE_CHAT = "TI_SO_TWO_STEP_PIPELINE_CHAT"
    SI_TO_ONE_STEP_PIPELINE_TRANSCRIBE = "SI_TO_ONE_STEP_PIPELINE_TRANSCRIBE"
    TI_SO_ONE_STEP_PIPELINE_SPEAK = "TI_SO_ONE_STEP_PIPELINE_SPEAK"
    SI_SO_THREE_STEP_PIPELINE_TRANSLATE = "SI_SO_THREE_STEP_PIPELINE_TRANSLATE"
    SI_SO_THREE_STEP_PIPELINE_AGENT = "SI_SO_THREE_STEP_PIPELINE_AGENT"


# Connectivity rules: maps each possible pair of labels to the inter-stage event name
connectivity_rules: dict[tuple[str, str], Optional[str]] = {
    ("STT_NON_STREAMING", "LLM_STREAMING"): None,
    ("LLM_STREAMING", "TTS_CHUNK_IN_STREAM_OUT"): "phrase_gate",
    ("LLM_STREAMING_WITH_REASONING", "TTS_CHUNK_IN_STREAM_OUT"): "phrase_gate",
    ("LLM_STREAMING", "TTS_NON_STREAMING"): None,
}


def get_inter_stage_event(from_label: str, to_label: str) -> Optional[str]:
    """Return the inter-stage event type between two node config labels.
    
    Returns None if no special event is needed.
    """
    return connectivity_rules.get((from_label, to_label))