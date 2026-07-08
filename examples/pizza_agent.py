"""Pizza Agent — End-to-End Example

Demonstrates the Sonication SDK pipeline-first architecture for a pizza ordering agent.

This example shows:
1. Creating STT/LLM/TTS nodes with vLLM endpoints
2. Setting up HotPipe with SI_SO_THREE_STEP_PIPELINE_CHAT
3. Pizza agent system prompt
4. WebSocket server for live conversation
5. Turn-by-turn pipeline execution with latency analysis

Usage:
    pip install sonication fastapi uvicorn
    python examples/pizza_agent.py

The server will start on http://localhost:8000
Connect via WebSocket at ws://localhost:8000/ws
"""
import asyncio
import json
import logging
from typing import Optional

import sonication
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Pizza Agent Configuration
# ============================================================

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

# ============================================================
# Pipeline Setup
# ============================================================


class PizzaPipeline:
    """Manages the voice pipeline for the pizza ordering agent."""

    def __init__(
        self,
        stt_url: str = "http://localhost:8092",
        llm_url: str = "http://localhost:8093",
        tts_url: str = "http://localhost:8094",
    ):
        self.stt_url = stt_url
        self.llm_url = llm_url
        self.tts_url = tts_url

        # Create nodes
        self.stt_node = sonication.STTNode(stt_url)
        self.llm_node = sonication.LLMNode(
            llm_url,
            system_prompt=PIZZA_SYSTEM_PROMPT,
        )
        self.tts_node = sonication.TTSNode(tts_url, voice="Marco", language="English")

        # Create pipeline
        self.pipeline = sonication.HotPipe(
            pipeline_type=sonication.PipelineType.SI_SO_THREE_STEP_PIPELINE_CHAT
        )
        self.pipeline.add_node(self.stt_node)
        self.pipeline.add_node(self.llm_node)
        self.pipeline.add_node(self.tts_node)
        self.pipeline.connect()

        # Conversation history
        self.history = [{"role": "system", "content": PIZZA_SYSTEM_PROMPT}]

        # Stats
        self.turn_count = 0
        self.total_latency = 0.0

    async def warmup(self):
        """Warm up all node connections."""
        logger.info("Warming up pizza agent nodes...")
        await self.stt_node.warmup()
        await self.llm_node.warmup()
        await self.tts_node.warmup()
        logger.info("All nodes warm.")

    async def process_audio(self, audio_bytes: bytes) -> dict:
        """Process audio input and return the full turn result."""
        self.turn_count += 1
        logger.info(f"Processing turn {self.turn_count}...")

        # Execute the pipeline turn
        result = await self.pipeline.turn("stt", audio_bytes)

        # Update conversation history
        if result.get("stt_text"):
            self.history.append({"role": "user", "content": result["stt_text"]})
        if result.get("llm_response"):
            self.history.append({"role": "assistant", "content": result["llm_response"]})

        # Track latency
        latency = result.get("shot_latency_ms", 0)
        self.total_latency += latency

        # Analyze
        analysis = self.pipeline.turn.__self__.analyse()

        return {
            "turn_index": self.turn_count,
            "stt_text": result.get("stt_text", ""),
            "llm_response": result.get("llm_response", ""),
            "tts_audio": result.get("tts_audio", b""),
            "shot_latency_ms": latency,
            "avg_latency_ms": self.total_latency / self.turn_count,
            "analysis": {
                "stt_ms": analysis.stt_ms,
                "llm_ttft_ms": analysis.llm_ttft_ms,
                "tts_ttfb_ms": analysis.tts_ttfb_ms,
            },
        }

    async def process_text(self, text: str) -> dict:
        """Process text input (for testing without STT)."""
        self.turn_count += 1
        logger.info(f"Processing text turn {self.turn_count}: {text[:50]}...")

        # Use TI_SO_TWO_STEP_PIPELINE_CHAT for text input
        text_pipeline = sonication.HotPipe(
            pipeline_type=sonication.PipelineType.TI_SO_TWO_STEP_PIPELINE_CHAT
        )
        text_pipeline.add_node(self.llm_node)
        text_pipeline.add_node(self.tts_node)
        text_pipeline.connect()

        # Update history
        self.history.append({"role": "user", "content": text})

        # Run turn
        result = await text_pipeline.turn("llm", self.history)

        if result.get("llm_response"):
            self.history.append({"role": "assistant", "content": result["llm_response"]})

        latency = result.get("shot_latency_ms", 0)
        self.total_latency += latency

        return {
            "turn_index": self.turn_count,
            "stt_text": text,
            "llm_response": result.get("llm_response", ""),
            "tts_audio": result.get("tts_audio", b""),
            "shot_latency_ms": latency,
            "avg_latency_ms": self.total_latency / self.turn_count,
        }

    def get_stats(self) -> dict:
        """Return pipeline statistics."""
        return {
            "turn_count": self.turn_count,
            "total_latency_ms": round(self.total_latency, 3),
            "avg_latency_ms": round(self.total_latency / max(self.turn_count, 1), 3),
            "nodes": list(self.pipeline.nodes.keys()),
            "connections": dict(self.pipeline.connections),
            "conversation_turns": len(self.history) - 1,  # minus system prompt
        }


# ============================================================
# WebSocket Server
# ============================================================

app = FastAPI(title="Pizza Agent", version="0.2.0")

pizza_pipeline: Optional[PizzaPipeline] = None


@app.on_event("startup")
async def startup():
    """Initialize the pizza pipeline on startup."""
    global pizza_pipeline
    pizza_pipeline = PizzaPipeline()
    await pizza_pipeline.warmup()
    logger.info("Pizza agent ready!")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for live pizza ordering conversation."""
    await ws.accept()
    logger.info("Client connected.")

    # Send welcome message
    await ws.send_json({
        "type": "system",
        "message": "Welcome to Marco's Pizza! Please speak or type your order.",
    })

    try:
        while True:
            # Receive message from client
            data = await ws.receive_json()

            msg_type = data.get("type", "")
            content = data.get("content", "")

            if msg_type == "text":
                # Text input (for testing)
                result = await pizza_pipeline.process_text(content)
                await ws.send_json({
                    "type": "response",
                    "turn_index": result["turn_index"],
                    "stt_text": result["stt_text"],
                    "llm_response": result["llm_response"],
                    "shot_latency_ms": result["shot_latency_ms"],
                })

            elif msg_type == "audio":
                # Audio input (base64 encoded PCM)
                import base64
                audio_bytes = base64.b64decode(content)
                result = await pizza_pipeline.process_audio(audio_bytes)
                await ws.send_json({
                    "type": "response",
                    "turn_index": result["turn_index"],
                    "stt_text": result["stt_text"],
                    "llm_response": result["llm_response"],
                    "shot_latency_ms": result["shot_latency_ms"],
                    "tts_audio": result.get("tts_audio", b""),
                })

            elif msg_type == "stats":
                # Request stats
                await ws.send_json({
                    "type": "stats",
                    **pizza_pipeline.get_stats(),
                })

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws.send_json({
            "type": "error",
            "message": str(e),
        })


@app.get("/")
async def index():
    """Simple HTML page for testing the WebSocket connection."""
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head>
    <title>Pizza Agent Demo</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }
        .message { padding: 10px; margin: 5px 0; border-radius: 8px; }
        .user { background: #e3f2fd; text-align: right; }
        .system { background: #f3e5f5; text-align: center; }
        .response { background: #e8f5e9; }
        .error { background: #ffebee; color: #c62828; }
        input { width: 70%; padding: 10px; }
        button { padding: 10px 20px; cursor: pointer; }
        #output { max-height: 400px; overflow-y: auto; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>🍕 Marco's Pizza Agent</h1>
    <p>Connect via WebSocket and type your pizza order!</p>
    <div id="output"></div>
    <input id="input" placeholder="Type your message..." />
    <button onclick="send()">Send</button>
    <button onclick="getStats()">Stats</button>

    <script>
        let ws = new WebSocket('ws://localhost:8000/ws');
        const output = document.getElementById('output');
        const input = document.getElementById('input');

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const div = document.createElement('div');
            div.className = 'message ' + data.type;
            div.textContent = JSON.stringify(data, null, 2);
            output.appendChild(div);
            output.scrollTop = output.scrollHeight;
        };

        ws.onerror = (err) => {
            console.error('WebSocket error', err);
        };

        function send() {
            const msg = input.value.trim();
            if (!msg) return;
            ws.send(JSON.stringify({ type: 'text', content: msg }));
            input.value = '';
        }

        function getStats() {
            ws.send(JSON.stringify({ type: 'stats' }));
        }

        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') send();
        });
    </script>
</body>
</html>
        """
    )


@app.get("/stats")
async def get_stats():
    """Return pipeline statistics."""
    if pizza_pipeline:
        return pizza_pipeline.get_stats()
    return {"error": "Pipeline not initialized"}


def main():
    """Run the pizza agent server."""
    import uvicorn
    logger.info("Starting Pizza Agent server on http://localhost:8000")
    uvicorn.run("examples.pizza_agent:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()