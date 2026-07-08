#!/usr/bin/env python3
"""Event timing analysis — 5 turns, filtered events, aggregated stats.

Runs 5 pipeline turns the normal way (pipe.turn with stream_events=True),
collects all events, filters to key events, and outputs a timing table.

Usage:
    python examples/test_event_timing.py
"""
import asyncio
import io
import statistics
import sys
import wave
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import sonication
from sonication import PipelineType

# Silence httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx_pool").setLevel(logging.WARNING)


def create_silence_wav(duration_secs=1.0, sample_rate=24000):
    """Create a WAV file with silence."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = int(sample_rate * duration_secs)
        wf.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


def filter_events(all_events):
    """Filter events to key types only.
    
    Keeps:
      - start: all start events
      - response: all response events
      - done: all done events
      - first token: first token event per emitter_node (LLM)
      - first chunk: first chunk event per emitter_node (TTS)
      - first phrase_chunk: first chunk from phrase_gate per LLM request
    
    Returns dict: {turn_index: [(event_type, emitter_node, local_offset_ms), ...]}
    """
    results = {}
    
    for turn_idx, events in all_events:
        turn_start = None
        kept = []
        
        # Track first occurrences
        first_token_by_node = {}  # emitter_node -> seen?
        first_chunk_by_node = {}  # emitter_node -> seen?
        first_phrase_chunk = False
        
        for evt in events:
            wallclock = evt["wallclock_ms"]
            
            # Find turn start from first event
            if turn_start is None:
                turn_start = wallclock
            
            etype = evt.get("event_type", "")
            enode = evt.get("emitter_node", "")
            local_ms = wallclock - turn_start
            
            # Keep start, response, done always
            if etype in ("start", "response", "done"):
                kept.append((etype, enode, round(local_ms, 2)))
            
            # Keep first token per node
            elif etype == "token":
                if enode not in first_token_by_node:
                    first_token_by_node[enode] = True
                    kept.append((etype, enode, round(local_ms, 2)))
            
            # Keep first chunk per TTS node
            elif etype == "chunk" and enode == "tts":
                if enode not in first_chunk_by_node:
                    first_chunk_by_node[enode] = True
                    kept.append((etype, enode, round(local_ms, 2)))
            
            # Keep first phrase_gate chunk
            elif etype == "chunk" and enode == "phrase_gate":
                if not first_phrase_chunk:
                    first_phrase_chunk = True
                    kept.append((etype, enode, round(local_ms, 2)))
        
        results[turn_idx] = kept
    
    return results


def print_timing_table(filtered):
    """Print timing table with aggregated stats."""
    # Collect all timings by event_type
    timings_by_type = {}
    
    for turn_idx, events in filtered.items():
        for etype, enode, local_ms in events:
            key = f"{etype} ({enode})"
            if key not in timings_by_type:
                timings_by_type[key] = []
            timings_by_type[key].append((turn_idx, local_ms))
    
    # Print header
    print("\n" + "=" * 95)
    print("EVENT TIMING ANALYSIS (ms, relative to turn start)")
    print("=" * 95)
    
    # Column headers
    header = f"{'Event':<30}"
    for i in range(1, 6):
        header += f"  Turn {i:<6}"
    header += f"  {'Mean':>8}  {'Median':>8}  {'Std':>8}"
    print(header)
    print("-" * 95)
    
    # Sort keys by typical order
    order = [
        "start (stt)", "start (llm)", "start (tts)", "start (phrase_gate)",
        "token (llm)", "chunk (phrase_gate)",
        "chunk (tts)",
        "response (stt)", "response (llm)", "response (tts)",
        "done (stt)", "done (llm)", "done (tts)",
    ]
    
    all_keys = sorted(timings_by_type.keys())
    sorted_keys = []
    for k in order:
        if k in all_keys:
            sorted_keys.append(k)
    for k in all_keys:
        if k not in sorted_keys:
            sorted_keys.append(k)
    
    for key in sorted_keys:
        turn_timings = timings_by_type[key]
        row = f"{key:<30}"
        
        # Build turn columns (5 turns)
        for i in range(1, 6):
            found = False
            for t_idx, ms in turn_timings:
                if t_idx == i:
                    row += f"  {ms:>6.1f}"
                    found = True
                    break
            if not found:
                row += f"  {'—':>6}"
        
        # Aggregated stats
        all_ms = [ms for _, ms in turn_timings]
        if all_ms:
            mean_ms = statistics.mean(all_ms)
            median_ms = statistics.median(all_ms)
            std_ms = statistics.stdev(all_ms) if len(all_ms) > 1 else 0.0
            row += f"  {mean_ms:>8.1f}  {median_ms:>8.1f}  {std_ms:>8.1f}"
        else:
            row += f"  {'—':>8}  {'—':>8}  {'—':>8}"
        
        print(row)
    
    # Aggregated summary row
    print("-" * 95)
    summary = f"{'AGGREGATED':<30}"
    for i in range(1, 6):
        summary += f"  {'—':>6}"
    
    # Overall mean/median/std across all event types
    all_values = []
    for key in sorted_keys:
        all_ms = [ms for _, ms in timings_by_type[key]]
        all_values.extend(all_ms)
    
    if all_values:
        overall_mean = statistics.mean(all_values)
        overall_median = statistics.median(all_values)
        overall_std = statistics.stdev(all_values) if len(all_values) > 1 else 0.0
        summary += f"  {overall_mean:>8.1f}  {overall_median:>8.1f}  {overall_std:>8.1f}"
    else:
        summary += f"  {'—':>8}  {'—':>8}  {'—':>8}"
    
    print(summary)
    print("=" * 95)


async def run_timing_test():
    """Run 5 turns and analyze event timings."""
    print("=" * 60)
    print("Event Timing Analysis — 5 Turns")
    print("=" * 60)
    sys.stdout.flush()
    
    # Create nodes
    print("Creating nodes...")
    sys.stdout.flush()
    stt_node = sonication.STTNode("http://127.0.0.1:8092", input_format="wav")
    llm_node = sonication.LLMNode("http://192.168.15.6:8000", api_key="")
    tts_node = sonication.TTSNode("http://127.0.0.1:8091", voice="Ryan", language="English")
    
    # Create pipeline
    print("Creating pipeline...")
    sys.stdout.flush()
    pipe = sonication.HotPipe(
        pipeline_type=PipelineType.SI_SO_THREE_STEP_PIPELINE_CHAT,
        keep_warm_duration=30.0,
    )
    pipe.add_node(stt_node)
    pipe.add_node(llm_node)
    pipe.add_node(tts_node)
    print("Connecting...")
    sys.stdout.flush()
    pipe.connect()
    print("Pipeline connected.")
    sys.stdout.flush()
    
    audio = create_silence_wav(duration_secs=1.0, sample_rate=24000)
    
    # Run 5 turns, collect events
    all_events = []  # list of (turn_index, [event_dicts])
    
    for turn_idx in range(1, 6):
        print(f"\n[Turn {turn_idx}/5] Running...")
        sys.stdout.flush()
        turn_events = []
        
        async for result in pipe.turn("stt", audio, stream_events=True):
            if isinstance(result, dict) and "event_type" in result:
                result["_turn_index"] = turn_idx
                turn_events.append(result)
        
        print(f"  ✓ Completed ({len(turn_events)} events)")
        sys.stdout.flush()
        all_events.append((turn_idx, turn_events))
    
    # Filter events
    print("\nFiltering events to key types...")
    sys.stdout.flush()
    filtered = filter_events(all_events)
    
    # Print timing table
    print_timing_table(filtered)
    
    # Print event counts per turn
    print("\n" + "=" * 60)
    print("EVENT COUNTS PER TURN")
    print("=" * 60)
    for turn_idx in range(1, 6):
        events = filtered[turn_idx]
        type_counts = {}
        for etype, enode, _ in events:
            key = f"{etype} ({enode})"
            type_counts[key] = type_counts.get(key, 0) + 1
        print(f"  Turn {turn_idx}: {len(events)} filtered events")
        for k, v in sorted(type_counts.items()):
            print(f"    {k}: {v}")
    
    await pipe.close()
    print("\n[+] Test complete")


if __name__ == "__main__":
    asyncio.run(run_timing_test())