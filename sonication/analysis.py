"""Analysis layer — latency computation, waterfall decomposition.

Provides:
    - AnalysisSegment: a named fragment of the waterfall with timing
    - PipelineAnalysis: aggregate view of all segments + summary stats
    - LatencyAnalyser: static helpers for segment extraction + percentile stats
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from .events import PhaseBoundary


@dataclass
class AnalysisSegment:
    """A named fragment of the pipeline waterfall.
    
    Fields:
        stage_name: Human-readable stage name (e.g., "stt", "llm_ttft")
        ms: Duration in milliseconds
        kind: One of "transport", "service", "pipeline"
        ping_ms: Optional RTT to the node (only for transport segments)
        internal_ms: Optional internal processing time (only for node-level segments)
    """
    stage_name: str
    ms: float
    kind: str  # "transport" | "service" | "pipeline"
    ping_ms: Optional[float] = None
    internal_ms: Optional[float] = None


@dataclass
class PipelineAnalysis:
    """Complete analysis of a single turn's latency waterfall.
    
    Fields:
        turn_id: The turn this analysis covers
        segments: List of named waterfall segments
        shot_latency_ms: Total wall-clock delta from input to first TTS byte
        stt_ms: STT transcription duration
        llm_ttft_ms: LLM time-to-first-token
        tts_ttfb_ms: TTS time-to-first-byte
    """
    turn_id: str
    segments: List[AnalysisSegment]
    shot_latency_ms: float
    stt_ms: float
    llm_ttft_ms: float
    tts_ttfb_ms: float


# Template → expected segments (for validation)
_PIPELINE_SEGMENTS: Dict[str, set] = {
    "SI_SO_THREE_STEP_PIPELINE_CHAT": {
        "t_stt_req → t_stt_resp",  # STT
        "t_llm_req → t_llm_ttft",  # LLM TTFT
        "t_tts_req → t_tts_ttfb",  # TTS TTFB
    },
    "TI_SO_TWO_STEP_PIPELINE_CHAT": {
        "t_llm_req → t_llm_ttft",  # LLM TTFT
        "t_tts_req → t_tts_ttfb",  # TTS TTFB
    },
    "SI_TO_ONE_STEP_PIPELINE_TRANSCRIBE": {
        "t_stt_req → t_stt_resp",  # STT
    },
    "TI_SO_ONE_STEP_PIPELINE_SPEAK": {
        "t_tts_req → t_tts_ttfb",  # TTS TTFB
    },
}


class LatencyAnalyser:
    """Static helper to analyse latency from a turn.
    
    Methods:
        analyse(turn): Full PipelineAnalysis of a single turn
        segments(turn): List of waterfall segments
        shot_latency(turn, pipeline_type): Total shot latency
        segment_mismatch(turn, template): What segments are missing vs expected
        statistical_summary(turns): Percentile stats across multiple turns
    """
    
    @staticmethod
    def analyse(turn) -> PipelineAnalysis:
        """Compute full PipelineAnalysis for a turn."""
        segments = LatencyAnalyser.segments(turn)
        return PipelineAnalysis(
            turn_id=turn.turn_id,
            segments=segments if segments else [],
            shot_latency_ms=turn.shot_latency_ms(),
            stt_ms=turn.interval("t_stt_req", "t_stt_resp") or 0,
            llm_ttft_ms=turn.interval("t_llm_req", "t_llm_ttft") or 0,
            tts_ttfb_ms=turn.interval("t_tts_req", "t_tts_ttfb") or 0,
        )
    
    @staticmethod
    def segments(turn) -> List[AnalysisSegment]:
        """Extract waterfall segments from a turn.
        
        Border edges:
            "t_release" → the press-to-talk release (or pipeline start for text input)
            "t_stt_req" → STT start
            "t_stt_resp" → STT done
            "t_llm_req" → LLM start
            "t_llm_ttft" → LLM first token
            "t_tts_req" → TTS start
            "t_tts_ttfb" → TTS first byte
        """
        segs: List[AnalysisSegment] = []
        
        def _add(stage: str, a: str, b: str, kind: str,
                 ping: float = None, internal: float = None) -> None:
            """Helper to add a segment if both endpoints exist."""
            a_b = turn.interval(a, b)
            if a_b is not None:
                segs.append(AnalysisSegment(
                    stage_name=stage,
                    ms=a_b,
                    kind=kind,
                    ping_ms=ping,
                    internal_ms=internal,
                ))
        
        # Pre-STT: from press-to-talk release to STT start
        _add("pre_stt", "t_release", "t_stt_req", "transport")
        
        # STT transcription
        _add("stt", "t_stt_req", "t_stt_resp", "service")
        
        # Pre-LLM: from STT done to LLM start
        llm_from = "t_stt_resp" if turn.boundary("t_stt_resp") else "t_release"
        _add("pre_llm", llm_from, "t_llm_req", "pipeline")
        
        # LLM TTFT
        _add("llm_ttft", "t_llm_req", "t_llm_ttft", "service")
        
        # Phrase gate gap: from LLM completion to TTS start
        _add("phrase_gate", "t_llm_resp", "t_tts_req", "pipeline")
        
        # TTS TTFB
        _add("tts_ttfb", "t_tts_req", "t_tts_ttfb", "service")
        
        return segs
    
    @staticmethod
    def shot_latency(turn, pipeline_type: str) -> float:
        """Compute pipeline-type-aware shot latency.
        
        TI_ pipelines start from LLM request, not STT request.
        """
        t_start = turn.boundary("t_stt_req")
        t_end = turn.boundary("t_tts_ttfb")
        if pipeline_type.startswith("TI_"):
            t_start = turn.boundary("t_llm_req") or t_start
        if t_start and t_end:
            return t_end.wallclock_ms - t_start.wallclock_ms
        return 0.0
    
    @staticmethod
    def segment_mismatch(turn, template: str) -> str:
        """Check what segments are present vs what pipeline template expects."""
        present = set(
            turn.interval(k, v) is not None
            for k, v in [
                ("t_stt_req", "t_stt_resp"),
                ("t_llm_req", "t_llm_ttft"),
                ("t_tts_req", "t_tts_ttfb"),
            ]
        )
        expected = _PIPELINE_SEGMENTS.get(template, set())
        missing = expected - present
        return ", ".join(f"{s} missing" for s in missing) if missing else ""
    
    @staticmethod
    def statistical_summary(turns: List) -> Dict[str, Optional[float]]:
        """Compute percentile stats across multiple turns."""
        shot_latencies = [t.shot_latency_ms() for t in turns if t.shot_latency_ms()]
        
        def _percentile(values: List[float], p: float) -> Optional[float]:
            good = sorted(v for v in values if v is not None)
            if not good:
                return None
            rank = (p / 100.0) * (len(good) - 1)
            lo = int(rank)
            hi = min(lo + 1, len(good) - 1)
            frac = rank - lo
            return round(good[lo] + (good[hi] - good[lo]) * frac, 3)
        
        return {
            "shot_latency_ms_p50": _percentile(shot_latencies, 50),
            "p90": _percentile(shot_latencies, 90),
            "p99": _percentile(shot_latencies, 99),
        }