#!/usr/bin/env python3
"""pizza_agent_test.py — End-to-end pizza agent simulation.

Simulates a complete pizza ordering conversation using the actual SDK pipeline.
Tests the full path: TTS → audio → STTNode → pipeline → LLMNode → pipeline → TTSNode.

Usage:
    python examples/pizza_agent_test.py
"""
import asyncio
import wave
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import sonication
from sonication import client
from sonication import config


# Pizza agent system prompt
PIZZA_SYSTEM_PROMPT = """You are Marco, the friendly pizza ordering assistant.

You work at Marco's Pizza and help customers order pizzas over a phone call.

Rules:
- Ask for the pizza size (small, medium, large)
- Ask about toppings (pepperoni, mushrooms, olives, onions, extra cheese)
- Ask if they want drinks or sides
- Confirm the order before placing it
- Be warm, concise, and speak in short sentences
- Never use lists, markdown, or emojis
- Always confirm the total price

Example interaction:
Customer: "Hi, I'd like to order a pizza"
Marco: "Great choice! What size would you like — small, medium, or large?"
"""


def save_pcm_as_wav(pcm_bytes: bytes, filename: str) -> Path:
    """Save PCM bytes as a WAV file."""
    output_path = Path("data/audio") / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_bytes)
    return output_path


async def test_pipeline_turn(text: str, turn_num: int) -> dict:
    """Test a full pipeline turn: TTS → audio → STTNode → pipeline → LLMNode → pipeline → TTSNode.

    Generates audio via TTS, feeds it through the pipeline, validates results.
    """
    print(f"\n{'='*60}")
    print(f"TURN {turn_num}: '{text}'")
    print(f"{'='*60}")

    # Step 1: Generate audio via TTS node
    print("  [1/5] TTS generating audio...")
    tts_node = sonication.TTSNode(config.TTS_BASE_URL, voice="ryan", language="English")
    await tts_node.warmup()

    pcm_chunks = []
    async for chunk in tts_node.stream(text):
        if chunk.get("kind") == "audio":
            pcm_data = chunk.get("pcm", b"")
            if pcm_data:
                pcm_chunks.append(pcm_data)

    pcm_bytes = b"".join(pcm_chunks)
    wav_path = save_pcm_as_wav(pcm_bytes, f"pipeline_turn{turn_num}.wav")
    print(f"  ✓ TTS generated {len(pcm_bytes)} bytes → {wav_path}")
    await tts_node.close()

    # Step 2: Build a fresh pipeline for this turn WITH LogManager
    print("  [2/5] Building pipeline with LogManager...")
    
    # Create LogManager for this test
    log = sonication.LogManager(mode="console")
    log.start()

    stt_node = sonication.STTNode(config.STT_BASE_URL)
    await stt_node.warmup()

    llm_node = sonication.LLMNode(
        config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        system_prompt=PIZZA_SYSTEM_PROMPT,
    )
    await llm_node.warmup()

    tts_node2 = sonication.TTSNode(config.TTS_BASE_URL, voice="ryan", language="English")
    await tts_node2.warmup()

    pipeline = sonication.HotPipe(
        pipeline_type=sonication.PipelineType.SI_SO_THREE_STEP_PIPELINE_CHAT,
        log_manager=log,
    )
    pipeline.add_node(stt_node)
    pipeline.add_node(llm_node)
    pipeline.add_node(tts_node2)
    pipeline.connect()

    # Step 3: Run the turn through the pipeline
    print("  [3/5] Running pipeline turn...")
    try:
        results = []
        async for result in pipeline.turn("stt", pcm_bytes):
            results.append(result)
        if results:
            result = results[0]
        else:
            result = {}
    except Exception as e:
        print(f"  ✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        await pipeline.close()
        return {
            "success": False,
            "turn": turn_num,
            "input_text": text,
            "error": str(e),
        }

    stt_text = result.get("stt_text", "")
    llm_response = result.get("llm_response", "")
    tts_audio = result.get("tts_audio", b"")

    print(f"  STT text: '{stt_text}'")
    print(f"  LLM response: '{llm_response}'")
    print(f"  TTS audio: {len(tts_audio)} bytes")

    # Step 4: Validate
    print("  [4/5] Validating...")

    # STT should have transcribed the TTS audio
    stt_lower = stt_text.lower().strip()
    text_lower = text.lower().strip()
    keywords = ["hi", "order", "pizza", "large", "pepperoni", "total", "sounds", "great"]
    stt_matched = [w for w in keywords if w in stt_lower]

    # LLM should have responded with relevant content
    llm_lower = llm_response.lower().strip()
    llm_keywords = ["size", "small", "medium", "large", "toppings", "total", "price", "drinks"]
    llm_matched = [w for w in llm_keywords if w in llm_lower]

    # TTS should have produced audio
    tts_ok = len(tts_audio) > 0

    print(f"  STT keywords matched: {stt_matched}/{len(keywords)}")
    print(f"  LLM keywords matched: {llm_matched}/{len(llm_keywords)}")
    print(f"  TTS audio produced: {tts_ok}")

    # Step 5: Cleanup
    print("  [5/5] Cleaning up...")
    await pipeline.close()
    log.shutdown()

    success = len(stt_matched) >= 1 and len(llm_matched) >= 1 and tts_ok

    print(f"  {'✓ PASS' if success else '✗ FAIL'}")

    return {
        "success": success,
        "turn": turn_num,
        "input_text": text,
        "stt_text": stt_text,
        "llm_response": llm_response,
        "tts_audio_size": len(tts_audio),
        "stt_matched": stt_matched,
        "llm_matched": llm_matched,
    }


async def test_stt_node_directly() -> dict:
    """Test STTNode directly — does it yield transcript events correctly?"""
    print(f"\n{'='*60}")
    print(f"TEST: STTNode Direct Yield Check")
    print(f"{'='*60}")

    # Generate short TTS audio
    pcm_chunks = []
    async for chunk in client.stream_tts("hello", "ryan", "English"):
        if chunk.get("kind") == "audio":
            pcm_data = chunk.get("pcm", b"")
            if pcm_data:
                pcm_chunks.append(pcm_data)

    pcm_bytes = b"".join(pcm_chunks)
    print(f"  Generated {len(pcm_bytes)} bytes of PCM")

    # Test STTNode directly
    stt_node = sonication.STTNode(config.STT_BASE_URL)
    await stt_node.warmup()

    events = []
    async for raw in stt_node.stream(pcm_bytes):
        events.append(raw)
        print(f"  STTNode yielded: kind={raw.get('kind')}, keys={list(raw.keys())}")

    await stt_node.close()

    # Check what we got
    transcripts = [e for e in events if e.get("kind") == "transcript"]
    done_events = [e for e in events if e.get("kind") == "done"]

    print(f"  Transcript events: {len(transcripts)}")
    print(f"  Done events: {len(done_events)}")

    if transcripts:
        print(f"  Transcript text: '{transcripts[0].get('text')}'")

    success = len(transcripts) >= 1
    print(f"  {'✓ PASS' if success else '✗ FAIL'}")

    return {
        "success": success,
        "events": events,
        "transcript_count": len(transcripts),
        "done_count": len(done_events),
    }


async def simulate_pizza_conversation() -> dict:
    """Simulate a complete pizza ordering conversation through the pipeline."""
    print("\n" + "=" * 70)
    print("  PIZZA AGENT END-TO-END PIPELINE SIMULATION")
    print("=" * 70)
    print(f"\n  STT: {config.STT_BASE_URL}")
    print(f"  LLM: {config.LLM_BASE_URL}")
    print(f"  TTS: {config.TTS_BASE_URL}")

    results = {
        "turns": [],
        "validations": [],
    }

    # ===== PHASE 0: STTNode Direct Test =====
    print("\n\n" + "=" * 70)
    print("  PHASE 0: STTNode Direct Yield Validation")
    print("=" * 70)

    stt_test = await test_stt_node_directly()
    results["validations"].append({"phase": "stt_direct", **stt_test})

    # ===== PHASE 1: Pipeline Turns =====
    print("\n\n" + "=" * 70)
    print("  PHASE 1: Full Pipeline Turns")
    print("=" * 70)

    conversation_texts = [
        "Hi, I'd like to order a pizza",
        "A large pepperoni pizza please",
        "That sounds great, what's the total?",
    ]

    for i, text in enumerate(conversation_texts, 1):
        result = await test_pipeline_turn(text, i)
        results["turns"].append(result)
        results["validations"].append({"phase": "pipeline_turn", **result})

    # ===== SUMMARY =====
    print("\n\n" + "=" * 70)
    print("  SIMULATION COMPLETE — VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True

    # STTNode direct
    v = results["validations"][0]
    status = "✓ PASS" if v.get("success") else "✗ FAIL"
    if not v.get("success"):
        all_passed = False
    print(f"\n  STTNode Direct: {status}")
    print(f"    Transcript events: {v.get('transcript_count', 0)}")
    print(f"    Done events: {v.get('done_count', 0)}")

    # Pipeline turns
    for v in results["turns"]:
        status = "✓ PASS" if v.get("success") else "✗ FAIL"
        if not v.get("success"):
            all_passed = False
        print(f"\n  Turn {v['turn']}: {status}")
        print(f"    Input:    '{v.get('input_text')}'")
        print(f"    STT:      '{v.get('stt_text', '')}'")
        print(f"    LLM:      '{v.get('llm_response', '')}'")
        print(f"    STT keys: {v.get('stt_matched', [])}")
        print(f"    LLM keys: {v.get('llm_matched', [])}")
        print(f"    TTS bytes: {v.get('tts_audio_size', 0)}")
        if v.get("error"):
            print(f"    Error:    {v['error']}")

    print(f"\n  {'='*60}")
    if all_passed:
        print(f"  ✓ ALL TESTS PASSED")
    else:
        print(f"  ✗ SOME TESTS FAILED")
    print(f"  {'='*60}")

    return results


async def main() -> int:
    """Main entry point."""
    try:
        results = await simulate_pizza_conversation()
        all_passed = all(v.get("success", False) for v in results["validations"])
        return 0 if all_passed else 1
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)