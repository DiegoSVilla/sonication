"""Headless pipeline runner: text in -> LLM -> TTS, stores events, prints metrics.

No browser, so there is no channel playback stamp; the analysis falls back to
the first-audio arrival time as the playback baseline. Useful for batch latency
runs and for smoke-testing against the live services.

    python -m backend.cli "Hello, who are you?" "What is the capital of France?"
"""
import asyncio
import datetime
import json
import sys
import uuid

from . import analysis, clients, config, db, events
from .pipeline import CallPipeline


async def run(texts: list[str]) -> None:
    db.init_db()
    call_id = uuid.uuid4().hex
    clock = events.CallClock()
    rec = events.EventRecorder(call_id, clock)
    now = datetime.datetime.fromtimestamp(
        clock.started_epoch_ms / 1000.0, datetime.timezone.utc
    )
    params = {
        "model": config.LLM_MODEL, "voice": config.TTS_VOICE,
        "language": config.TTS_LANGUAGE, "temperature": config.LLM_TEMPERATURE,
        "seed": config.LLM_SEED, "max_tokens": config.LLM_MAX_TOKENS,
    }
    db.create_session("cli")
    db.create_call(call_id, "cli", now.isoformat(), clock.started_epoch_ms, params)
    rec.record(events.CALL_START,
               {"wallclock_iso": now.isoformat(), "epoch_ms": clock.started_epoch_ms}, t_ms=0.0)

    pipe = CallPipeline(call_id, rec, send=None)
    for text in texts:
        print(f"\n>>> {text}")
        res = await pipe.run_turn(text)
        m = res["metrics"]
        print(f"    reply: {res['assistant_text'][:120]!r}")
        print(f"    LLM TTFT={m['llm_ttft_ms']} ms  tokens={m['llm_tokens']}  "
              f"TTS TTFB={m['tts_ttfb_ms']} ms  audio_gen={m['audio_generated_ms']} ms")
    db.end_call(call_id, clock.now_ms())

    print("\n=== analysis (channel baseline = first audio byte) ===")
    print(json.dumps(analysis.summarize_call(call_id), indent=2))
    print(f"\ncall_id={call_id}")
    await clients.aclose()


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--new-connection" in argv:
        argv = [a for a in argv if a != "--new-connection"]
        clients.use_new_connections(True)
        print(">>> NEW-CONNECTION MODE: fresh connection per request/ping.")
    args = argv or ["Hi there, can you tell me a fun fact about the ocean?"]
    asyncio.run(run(args))
