"""Test keep-warm behavior of HotPipe.

Test sequence:
1. Instantiate HotPipe (PingLoop auto-starts on connect())
2. Call turn() → measure time (fast, connections warm from connect())
3. Wait 1 minute (PingLoop stops after 30s, connections expire)
4. Call turn() → measure time (slow, connections cold)
5. Call turn() again → measure time (fast, connections warmed up)

Expected:
- Turn 1: ~0.5s (warm from connect())
- Turn 2: ~1.5s (cold after 1 min wait)
- Turn 3: ~0.5s (warm again)

Usage:
    python examples/test_warmth_pipe.py
"""
import asyncio
import io
import struct
import time
import wave

import sonication
from sonication import PipelineType


# Silence httpx logs during test
import logging
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


async def test_keep_warm():
    """Test that HotPipe keeps connections warm for keep_warm_duration."""
    print("=" * 60)
    print("HotPipe Keep-Warm Test")
    print("=" * 60)
    
    # Create nodes with correct URLs
    stt_node = sonication.STTNode("http://127.0.0.1:8092", input_format="wav")
    llm_node = sonication.LLMNode("http://192.168.15.6:8000", api_key="")
    tts_node = sonication.TTSNode("http://127.0.0.1:8091", voice="Ryan", language="English")
    
    # Create pipeline with 30s keep-warm duration
    pipe = sonication.HotPipe(
        pipeline_type=PipelineType.SI_SO_THREE_STEP_PIPELINE_CHAT,
        keep_warm_duration=30.0,  # Stop pings after 30s of inactivity
    )
    pipe.add_node(stt_node)
    pipe.add_node(llm_node)
    pipe.add_node(tts_node)
    pipe.connect()  # Auto-starts PingLoop, warms up connections
    
    print("\n[+] Pipeline connected, PingLoop started")
    print(f"[+] Keep-warm duration: 30s")
    
    # Use silence WAV for testing (no speech needed)
    audio = create_silence_wav(duration_secs=1.0, sample_rate=24000)
    
    # Test 1: Warm connection (just connected)
    print("\n" + "=" * 60)
    print("Test 1: Warm connection (just connected)")
    print("=" * 60)
    start = time.time()
    results = []
    async for result in pipe.turn("stt", audio):
        results.append(result)
    warm_time = (time.time() - start) * 1000
    result = results[0] if results else {}
    print(f"  Turn 1 latency: {warm_time:.0f}ms")
    print(f"  Expected: ~500ms (connections warm from connect())")
    
    # Wait 1 minute (exceeds 30s keep-warm)
    print("\n" + "=" * 60)
    print("Waiting 60 seconds (PingLoop will stop after 30s)")
    print("=" * 60)
    await asyncio.sleep(60)
    
    # Test 2: Cold connection (PingLoop stopped, connections expired)
    print("\n" + "=" * 60)
    print("Test 2: Cold connection (after 60s wait)")
    print("=" * 60)
    start = time.time()
    results = []
    async for result in pipe.turn("stt", audio):
        results.append(result)
    cold_time = (time.time() - start) * 1000
    result = results[0] if results else {}
    print(f"  Turn 2 latency: {cold_time:.0f}ms")
    print(f"  Expected: ~1500ms (connections cold, new handshake)")
    
    # Test 3: Warm again (connection reused from turn 2)
    print("\n" + "=" * 60)
    print("Test 3: Warm connection (just used turn 2)")
    print("=" * 60)
    start = time.time()
    results = []
    async for result in pipe.turn("stt", audio):
        results.append(result)
    warm_time_2 = (time.time() - start) * 1000
    result = results[0] if results else {}
    print(f"  Turn 3 latency: {warm_time_2:.0f}ms")
    print(f"  Expected: ~500ms (connection warm from turn 2)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Turn 1 (warm):  {warm_time:.0f}ms")
    print(f"  Turn 2 (cold):  {cold_time:.0f}ms")
    print(f"  Turn 3 (warm):  {warm_time_2:.0f}ms")
    print()
    if warm_time > 0:
        print(f"  Speedup from keep-warm: {cold_time/warm_time:.1f}x (turn 1 vs 2)")
    if warm_time_2 > 0:
        print(f"  Speedup from reuse:     {cold_time/warm_time_2:.1f}x (turn 2 vs 3)")
    
    # Check results
    print()
    if cold_time > warm_time * 1.2:
        print("[PASS] Keep-warm is working: cold turn is slower than warm turn")
    else:
        print("[WARN] Keep-warm effect not clearly visible (cold/warm ratio < 1.2x)")
    
    if warm_time_2 < cold_time * 0.8:
        print("[PASS] Connection reuse is working: turn 3 is faster than turn 2")
    else:
        print("[WARN] Connection reuse effect not clearly visible")
    
    await pipe.close()
    print("\n[+] Test complete")


if __name__ == "__main__":
    asyncio.run(test_keep_warm())