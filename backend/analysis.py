"""Call timing analysis space.

Computes per-turn shot metrics and cross-turn percentiles from stored events.
The headline metric is shot latency: Enter pressed to the first audio reproduced
on the channel. When VAD/STT land later, the shot start simply moves from the
text_in stamp to the vad_end stamp; the rest of this code does not change.
"""
import json
from typing import Any, Optional

from . import config, db

# Playback is considered late (a gap / underrun risk) beyond this slack.
UNDERRUN_SLACK_MS = 10.0


def compute_waterfall(timing: Optional[dict]) -> Optional[dict]:
    """Decompose a shot into stages from wall-clock marks.

    timing = {"marks": {t_release, t_stt_req, ...}, "pings": {stt, llm, tts}}.
    Each service stage also reports an internal estimate = observed - ping (the
    network floor), clamped at 0. Transport/pipeline stages are our own overhead.
    """
    if not timing:
        return None
    m = timing.get("marks") or {}
    p = timing.get("pings") or {}
    segs: list[dict] = []

    def add(stage: str, a: str, b: str, kind: str, ping: Optional[float] = None) -> None:
        if m.get(a) is None or m.get(b) is None:
            return
        span = round(m[b] - m[a], 3)
        seg = {"stage": stage, "ms": span, "kind": kind}
        if kind == "service" and ping is not None:
            seg["ping_ms"] = ping
            seg["internal_ms"] = round(max(0.0, span - ping), 3)
        segs.append(seg)

    add("pre_stt", "t_release", "t_stt_req", "transport")
    add("stt", "t_stt_req", "t_stt_resp", "service", p.get("stt"))
    llm_from = "t_stt_resp" if m.get("t_stt_resp") is not None else "t_release"
    add("pre_llm", llm_from, "t_llm_req", "pipeline")
    add("llm_ttft", "t_llm_req", "t_llm_ttft", "service", p.get("llm"))
    add("phrase_gate", "t_llm_ttft", "t_tts_req", "pipeline")
    add("tts_ttfb", "t_tts_req", "t_tts_audio", "service", p.get("tts"))
    add("channel_out", "t_tts_audio", "t_channel", "transport")

    # shot_ms is ALWAYS the wall-clock distance release -> first audio on channel.
    # It is never derived from the segment sum. sum_ms is a diagnostic only; any
    # gap between it and shot_ms shows up as residual_ms (should be near zero when
    # the marks tile the timeline cleanly).
    shot = None
    if m.get("t_channel") is not None and m.get("t_release") is not None:
        shot = round(m["t_channel"] - m["t_release"], 3)
    sum_ms = round(sum(s["ms"] for s in segs), 3)
    return {
        "segments": segs,
        "shot_ms": shot,
        "sum_ms": sum_ms,
        "residual_ms": round(shot - sum_ms, 3) if shot is not None else None,
        "pings": p,
    }


def percentile(values: list[float], p: float) -> Optional[float]:
    """Linear-interpolation percentile. p in [0, 100]."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return round(vals[0], 3)
    rank = (p / 100.0) * (len(vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return round(vals[lo] + (vals[hi] - vals[lo]) * frac, 3)


def _stat_block(values: list[float]) -> dict[str, Any]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "min": round(min(vals), 3),
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "p99": percentile(vals, 99),
        "max": round(max(vals), 3),
        "mean": round(sum(vals) / len(vals), 3),
    }


def _audio_underrun(events: list[dict[str, Any]], turn_index: int,
                    channel_start_t: Optional[float]) -> dict[str, Any]:
    """Check whether generated audio kept ahead of the channel playback clock."""
    chunks = [
        e for e in events
        if e["type"] == "audio_out" and e["payload"].get("turn_index") == turn_index
    ]
    if not chunks:
        return {"chunks": 0, "underrun": False, "max_lateness_ms": None, "lead_ms": None}
    # If the channel never reported a playback start, use the first arriving byte.
    base = channel_start_t if channel_start_t is not None else chunks[0]["t_ms"]
    max_lateness = 0.0
    for e in chunks:
        cum_after = e["payload"]["turn_audio_ms"]
        dur = e["payload"]["duration_ms"]
        cum_before = cum_after - dur
        scheduled_play_start = base + cum_before
        lateness = e["t_ms"] - scheduled_play_start
        if lateness > max_lateness:
            max_lateness = lateness
    last = chunks[-1]
    total_audio_ms = last["payload"]["turn_audio_ms"]
    # Lead when generation finished: buffered audio minus elapsed playback time.
    lead_ms = total_audio_ms - (last["t_ms"] - base)
    return {
        "chunks": len(chunks),
        "underrun": max_lateness > UNDERRUN_SLACK_MS,
        "max_lateness_ms": round(max_lateness, 3),
        "lead_ms": round(lead_ms, 3),
        "audio_generated_ms": round(total_audio_ms, 3),
    }


def summarize_call(call_id: str) -> dict[str, Any]:
    call = db.get_call(call_id)
    if not call:
        return {"error": "call not found", "call_id": call_id}
    turns = db.get_turns(call_id)
    events = db.get_events(call_id)

    channel_starts: dict[int, float] = {}
    for e in events:
        if e["type"] == "channel_playback_start":
            channel_starts[e["payload"].get("turn_index")] = e["t_ms"]

    turn_rows = []
    for t in turns:
        idx = t["turn_index"]
        audio = _audio_underrun(events, idx, channel_starts.get(idx))
        # Merge the separately-stored channel-playback mark into the timing so
        # the waterfall includes the final channel_out transport segment.
        timing = json.loads(t["timing_json"]) if t.get("timing_json") else None
        if timing is not None and t.get("first_audio_channel_t_ms") is not None:
            timing.setdefault("marks", {})["t_channel"] = t["first_audio_channel_t_ms"]
        turn_rows.append(
            {
                "turn_index": idx,
                "chars_in": t.get("chars"),
                "stt_ms": t.get("stt_ms"),
                "waterfall": compute_waterfall(timing),
                "llm_ttft_ms": t.get("llm_ttft_ms"),
                "llm_tokens": t.get("llm_tokens"),
                "tts_ttfb_ms": t.get("tts_ttfb_ms"),
                "shot_latency_ms": t.get("shot_latency_ms"),
                "first_audio_channel_t_ms": t.get("first_audio_channel_t_ms"),
                "audio_generated_ms": t.get("audio_generated_ms"),
                "audio": audio,
            }
        )

    return {
        "call_id": call_id,
        "started_iso": call.get("started_iso"),
        "model": call.get("model"),
        "voice": call.get("voice"),
        "turns": turn_rows,
        "shot_latency_ms": _stat_block([r["shot_latency_ms"] for r in turn_rows]),
        "stt_ms": _stat_block([r["stt_ms"] for r in turn_rows]),
        "llm_ttft_ms": _stat_block([r["llm_ttft_ms"] for r in turn_rows]),
        "tts_ttfb_ms": _stat_block([r["tts_ttfb_ms"] for r in turn_rows]),
    }


def summarize_all() -> dict[str, Any]:
    turns = db.all_turns()
    return {
        "calls": len(db.list_calls()),
        "turns": len(turns),
        "audio_format": {
            "sample_rate": config.AUDIO_SAMPLE_RATE,
            "channels": config.AUDIO_CHANNELS,
            "bytes_per_sec": config.AUDIO_BYTES_PER_SEC,
        },
        "shot_latency_ms": _stat_block([t.get("shot_latency_ms") for t in turns]),
        "stt_ms": _stat_block([t.get("stt_ms") for t in turns]),
        "llm_ttft_ms": _stat_block([t.get("llm_ttft_ms") for t in turns]),
        "tts_ttfb_ms": _stat_block([t.get("tts_ttfb_ms") for t in turns]),
        "audio_generated_ms": _stat_block([t.get("audio_generated_ms") for t in turns]),
    }
