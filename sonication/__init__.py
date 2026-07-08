"""Sonication backend package.

A HotPipe architecture for building voice agent pipelines.

This SDK provides a node-based graph architecture where data flows
asynchronously through STT, LLM, and TTS nodes. Each node operates
on vLLM-compatible routes with explicit URL configuration.

Key components:
    - Node: Abstract base class for streaming transformers
    - STTNode, LLMNode, TTSNode: Concrete node implementations
    - PipeEvent: Events with wall-clock timestamps
    - CallPipeline: Orchestrates node streaming with explicit connections
    - EventRecorder: Logs events to SQLite database
    - CallClock: Monotonic timing infrastructure
"""

# Corporate networks here terminate TLS with an internal root CA. truststore
# makes Python's ssl use the OS trust store (where that root CA is installed),
# so httpx trusts the intercepted chain without disabling verification.
try:
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - fall back to certifi if unavailable
    pass

# Version
VERSION = "0.2.0"
__version__ = VERSION

# Export key components
from .events import (
    PipeEvent, EventRecorder, CallClock, CALL_START, CALL_END,
    NodeEvent, InterStageEvent, PhaseBoundary, NodeStageRecord,
)
from .nodes import Node, STTNode, LLMNode, TTSNode
from .hotpipe import HotPipe, Turn, StageBoundaries
from .pipeline import CallPipeline
from .node_types import NodeConfigLabel, PipelineType, get_inter_stage_event
from .analysis import (
    LatencyAnalyser, PipelineAnalysis, AnalysisSegment, _PIPELINE_SEGMENTS,
)
from .log_manager import LogManager
from .db import (
    init_db, log_pipe_event, log_node_event,
    insert_inter_stage_event, log_keep_warm_ping, log_node_stage,
    get_node_stages, get_inter_stage_events, get_keep_warm_pings,
)
from .config import (
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_SEED,
    LLM_MAX_TOKENS, TTS_BASE_URL, TTS_VOICE, TTS_LANGUAGE, TTS_MODEL,
    STT_BASE_URL, STT_MODEL, STT_LANGUAGE, STT_API_KEY,
    AUDIO_SAMPLE_RATE, SYSTEM_PROMPT,
)

__all__ = [
    'Node', 'STTNode', 'LLMNode', 'TTSNode', 'PipeEvent',
    'HotPipe', 'Turn', 'StageBoundaries',
    'EventRecorder', 'CallClock', 'CallPipeline',
    'NodeConfigLabel', 'PipelineType', 'get_inter_stage_event',
    'NodeEvent', 'InterStageEvent', 'PhaseBoundary', 'NodeStageRecord',
    'LatencyAnalyser', 'PipelineAnalysis', 'AnalysisSegment', '_PIPELINE_SEGMENTS',
    'CALL_START', 'CALL_END',
    'LogManager',
    'init_db', 'log_pipe_event', 'log_node_event',
    'insert_inter_stage_event', 'log_keep_warm_ping', 'log_node_stage',
    'get_node_stages', 'get_inter_stage_events', 'get_keep_warm_pings',
    'LLM_BASE_URL', 'LLM_API_KEY', 'LLM_MODEL', 'LLM_TEMPERATURE', 'LLM_SEED',
    'LLM_MAX_TOKENS', 'TTS_BASE_URL', 'TTS_VOICE', 'TTS_LANGUAGE', 'TTS_MODEL',
    'STT_BASE_URL', 'STT_MODEL', 'STT_LANGUAGE', 'STT_API_KEY',
    'AUDIO_SAMPLE_RATE', 'SYSTEM_PROMPT',
]