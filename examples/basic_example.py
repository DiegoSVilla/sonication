"""Example: Using the Sonication SDK with pipeline-first architecture.

This example demonstrates:
1. Creating nodes with vLLM endpoints
2. Setting up a pipeline with HotPipe
3. Running a turn
4. Analyzing latency

Usage:
    pip install sonication
    python examples/basic_example.py
"""
import asyncio
import sonication


async def main():
    # 1. Create nodes with your own vLLM endpoints
    stt = sonication.STTNode("http://localhost:8092")
    llm = sonication.LLMNode(
        "http://localhost:8093",
        system_prompt="You are a helpful voice assistant."
    )
    tts = sonication.TTSNode(
        "http://localhost:8094",
        voice="Marco",
        language="English"
    )

    # 2. Initialize the database (creates tables if they don't exist)
    sonication.init_db()

    # 3. Create a pipeline with pipeline-first architecture
    pipe = sonication.HotPipe(
        pipeline_type=sonication.PipelineType.SI_SO_THREE_STEP_PIPELINE_CHAT
    )
    
    # Auto-slot by node type (no connect() needed!)
    pipe.add_node(stt)   # STTNode → fills "stt" slot
    pipe.add_node(llm)   # LLMNode → fills "llm" slot
    pipe.add_node(tts)   # TTSNode → fills "tts" slot
    
    # Validate topology
    pipe.connect()
    
    print(f"Pipeline type: {pipe.pipeline_type}")
    print(f"Nodes: {list(pipe.nodes.keys())}")
    print(f"Connections: {dict(pipe.connections)}")
    
    # 4. Warmup connections
    await pipe.warmup()
    
    # 5. Run a turn (text input for demo — in production, use audio)
    # For text input, use TI_SO_TWO_STEP_PIPELINE_CHAT instead
    print("\nRunning turn...")
    
    # Note: This example uses text input for demonstration
    # In production, you'd pass audio_bytes to the STT node
    try:
        # Simulate a turn with text (TI_SO_TWO_STEP_PIPELINE_CHAT)
        text_pipe = sonication.HotPipe(
            pipeline_type=sonication.PipelineType.TI_SO_TWO_STEP_PIPELINE_CHAT
        )
        text_pipe.add_node(llm)
        text_pipe.add_node(tts)
        text_pipe.connect()
        
        # Run turn with text
        result = await text_pipe.turn("llm", [{"role": "user", "content": "Hello!"}])
        
        print(f"\nResult:")
        print(f"  stt_text: {result['stt_text']}")
        print(f"  llm_response: {result['llm_response'][:100]}...")
        print(f"  shot_latency_ms: {result['shot_latency_ms']}")
        
        # 6. Analyze latency
        analysis = text_pipe.turn.__self__.analyse() if hasattr(text_pipe.turn.__self__, 'analyse') else None
        if analysis:
            print(f"\nLatency Analysis:")
            print(f"  STT: {analysis.stt_ms}ms")
            print(f"  LLM TTFT: {analysis.llm_ttft_ms}ms")
            print(f"  TTS TTFB: {analysis.tts_ttfb_ms}ms")
            
    except Exception as e:
        print(f"Turn failed (expected in demo): {e}")
        print("This is normal — the example doesn't have real vLLM endpoints.")
    
    # 7. Close connections
    await pipe.close()
    
    print("\nExample complete!")


if __name__ == "__main__":
    asyncio.run(main())