"""Pizza Agent — End-to-End Example

Demonstrates the Sonication SDK pipeline-first architecture for a pizza
ordering agent.

Usage:
    pip install sonication fastapi uvicorn
    python examples/pizza_agent.py

The server will start on http://localhost:8000
Connect via WebSocket at ws://localhost:8000/ws
"""
import asyncio
import base64
import json
import logging
from typing import Optional

import sonication
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


class PizzaAgent:
    """Manages the voice pipeline for the pizza ordering agent."""

    def __init__(
        self,
        stt_url: str = "http://localhost:8092",
        llm_url: str = "http://localhost:8093",
        tts_url: str = "http://localhost:8094",
    ):
        # Create nodes
        self.stt_node = sonication.STTNode(stt_url)
        self.llm_node = sonication.LLMNode(
            llm_url,
            system_prompt=PIZZA_SYSTEM_PROMPT,
        )
        self.tts_node = sonication.TTSNode(tts_url, voice="Marco", language="English")

        # Create pipeline — context management is internal to LLMNode
        self.pipeline = sonication.HotPipe(
            pipeline_type=sonication.PipelineType.SI_SO_THREE_STEP_PIPELINE_CHAT
        )
        self.pipeline.add_node(self.stt_node)
        self.pipeline.add_node(self.llm_node)
        self.pipeline.add_node(self.tts_node)
        self.pipeline.connect()

        self.turn_count = 0

    async def warmup(self):
        """Warm up all node connections via the pipeline."""
        logger.info("Warming up pizza agent nodes...")
        await self.pipeline.warmup()
        logger.info("All nodes warm.")

    async def run_turn(self, data: bytes) -> dict:
        """Process one turn (audio input) and return results."""
        self.turn_count += 1
        logger.info(f"Processing turn {self.turn_count}...")

        result = await self.pipeline.turn("stt", data)

        return {
            "turn_index": self.turn_count,
            "stt_text": result.get("stt_text", ""),
            "llm_response": result.get("llm_response", ""),
            "shot_latency_ms": result.get("shot_latency_ms", 0),
        }

    async def run_text_turn(self, text: str) -> dict:
        """Process one text-only turn (skips STT)."""
        self.turn_count += 1
        logger.info(f"Processing text turn {self.turn_count}: {text[:50]}...")

        # For text input, use the two-step pipeline (LLM → TTS)
        text_pipeline = sonication.HotPipe(
            pipeline_type=sonication.PipelineType.TI_SO_TWO_STEP_PIPELINE_CHAT
        )
        text_pipeline.add_node(self.llm_node)
        text_pipeline.add_node(self.tts_node)
        text_pipeline.connect()

        result = await text_pipeline.turn("llm", text)

        return {
            "turn_index": self.turn_count,
            "stt_text": text,
            "llm_response": result.get("llm_response", ""),
            "shot_latency_ms": result.get("shot_latency_ms", 0),
        }

    def get_stats(self) -> dict:
        """Return pipeline statistics."""
        return {
            "turn_count": self.turn_count,
            "nodes": list(self.pipeline.nodes.keys()),
            "connections": dict(self.pipeline.connections),
        }


# ============================================================
# WebSocket Server
# ============================================================

app = FastAPI(title="Pizza Agent", version="0.2.0")
pizza_agent: Optional[PizzaAgent] = None


@app.on_event("startup")
async def startup():
    """Initialize the pizza pipeline on startup."""
    global pizza_agent
    pizza_agent = PizzaAgent()
    await pizza_agent.warmup()
    logger.info("Pizza agent ready!")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for live pizza ordering conversation."""
    await ws.accept()
    logger.info("Client connected.")

    await ws.send_json({
        "type": "system",
        "message": "Welcome to Marco's Pizza! Please speak or type your order.",
    })

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "")
            content = data.get("content", "")

            if msg_type == "text":
                result = await pizza_agent.run_text_turn(content)
                await ws.send_json({
                    "type": "response",
                    "turn_index": result["turn_index"],
                    "stt_text": result["stt_text"],
                    "llm_response": result["llm_response"],
                    "shot_latency_ms": result["shot_latency_ms"],
                })

            elif msg_type == "audio":
                audio_bytes = base64.b64decode(content)
                result = await pizza_agent.run_turn(audio_bytes)
                await ws.send_json({
                    "type": "response",
                    "turn_index": result["turn_index"],
                    "stt_text": result["stt_text"],
                    "llm_response": result["llm_response"],
                    "shot_latency_ms": result["shot_latency_ms"],
                })

            elif msg_type == "stats":
                await ws.send_json({
                    "type": "stats",
                    **pizza_agent.get_stats(),
                })

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws.send_json({"type": "error", "message": str(e)})


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
    <h1>Marco's Pizza Agent</h1>
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
    if pizza_agent:
        return pizza_agent.get_stats()
    return {"error": "Pipeline not initialized"}


def main():
    """Run the pizza agent server."""
    import uvicorn
    logger.info("Starting Pizza Agent server on http://localhost:8000")
    uvicorn.run("examples.pizza_agent:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()