"""Pizza Agent — End-to-End Example

Demonstrates the Sonication SDK pipeline-first architecture for a pizza
ordering agent.

Usage:
    pip install sonication fastapi uvicorn
    python examples/pizza_agent.py

The server will start on http://localhost:8000
Open in a browser and hold Enter to talk.

Curl API:
    curl -X POST http://localhost:8000/turn -F "file=@audio.wav"
"""
import asyncio
import base64
import io
import logging
import wave
from pathlib import Path
from typing import Optional

import sonication
from fastapi import FastAPI, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent / "frontend"

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
        self.stt_url = stt_url
        self.llm_url = llm_url
        self.tts_url = tts_url

        # Create nodes
        self.stt_node = sonication.STTNode(stt_url, sample_rate=16000, input_format="wav")
        self.llm_node = sonication.LLMNode(
            llm_url,
            system_prompt=PIZZA_SYSTEM_PROMPT,
        )
        self.tts_node = sonication.TTSNode(tts_url, voice="ryan", language="English")

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

    async def run_turn(self, audio_bytes: bytes, session_id: str = "") -> dict:
        """Process one turn (audio input) and return results."""
        self.turn_count += 1
        logger.info(f"Processing turn {self.turn_count} [session={session_id}]...")

        result = await self.pipeline.turn("stt", audio_bytes)

        # Build analysis segments from the turn
        try:
            last_turn = self.pipeline._last_turn
            if last_turn:
                analysis = last_turn.analyse()
                segments = [
                    {"stage": s.stage_name, "ms": s.ms, "kind": s.kind}
                    for s in analysis.segments
                ]
        except Exception:
            segments = []

        return {
            "type": "response",
            "turn_index": self.turn_count,
            "stt_text": result.get("stt_text", ""),
            "llm_response": result.get("llm_response", ""),
            "tts_audio_b64": base64.b64encode(result.get("tts_audio", b"")).decode("ascii") if result.get("tts_audio") else "",
            "shot_latency_ms": result.get("shot_latency_ms", 0),
            "segments": segments,
        }

    def get_stats(self) -> dict:
        """Return pipeline statistics."""
        return {
            "turn_count": self.turn_count,
            "nodes": list(self.pipeline.nodes.keys()),
            "connections": dict(self.pipeline.connections),
        }


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(title="Pizza Agent", version="0.2.0")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
pizza_agent: Optional[PizzaAgent] = None


@app.on_event("startup")
async def startup():
    """Initialize the pizza pipeline on startup."""
    global pizza_agent
    pizza_agent = PizzaAgent()
    await pizza_agent.warmup()
    logger.info("Pizza agent ready!")


@app.get("/", response_class=FileResponse, include_in_schema=False)
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for live pizza ordering conversation."""
    import uuid
    await ws.accept()
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    logger.info(f"Client connected. Session: {session_id}")

    await ws.send_json({
        "type": "system",
        "message": f"Welcome to Marco's Pizza! Session: {session_id}",
        "session_id": session_id,
    })

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "user_audio":
                audio_b64 = data.get("audio_b64", "")
                audio_bytes = base64.b64decode(audio_b64)
                try:
                    result = await pizza_agent.run_turn(audio_bytes, session_id)
                    await ws.send_json(result)
                except Exception as e:
                    logger.error(f"Turn error: {e}", exc_info=True)
                    await ws.send_json({"type": "error", "message": str(e)})

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
    finally:
        await ws.close()


@app.get("/stats")
async def get_stats():
    """Return pipeline statistics."""
    if pizza_agent:
        return pizza_agent.get_stats()
    return {"error": "Pipeline not initialized"}


@app.post("/turn")
async def curl_turn(file: UploadFile):
    """Process audio from curl. Accepts WAV file, returns JSON with transcript and response.
    
    Usage:
        curl -X POST http://localhost:8000/turn -F "file=@audio.wav"
    """
    if not pizza_agent:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=503)
    
    try:
        audio_bytes = await file.read()
        logger.info(f"Curl turn: {len(audio_bytes)} bytes received")
        result = await pizza_agent.run_turn(audio_bytes, "curl")
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Turn error: {e}", exc_info=True)
        return JSONResponse({"type": "error", "message": str(e)}, status_code=500)


@app.websocket("/curl_turn")
async def curl_ws_turn(ws: WebSocket):
    """Process audio from curl via WebSocket — same path as browser.
    
    Usage:
        # Encode WAV to base64 and send via WebSocket
        python3 -c "
        import base64, websockets
        audio = open('/tmp/test.wav','rb').read()
        b64 = base64.b64encode(audio).decode()
        ws = await websockets.connect('ws://localhost:8000/curl_turn')
        await ws.send('{\"type\":\"user_audio\",\"audio_b64\":\"' + b64 + '\"}')
        resp = await ws.recv()
        print(resp)
        await ws.close()
        "
    """
    await ws.accept()
    session_id = f"curl_{__import__('uuid').uuid4().hex[:8]}"
    await ws.send_json({
        "type": "system",
        "message": f"Curl WS connected. Session: {session_id}",
    })
    
    try:
        data = await ws.receive_json()
        msg_type = data.get("type", "")
        
        if msg_type == "user_audio":
            audio_b64 = data.get("audio_b64", "")
            audio_bytes = base64.b64decode(audio_b64)
            logger.info(f"Curl WS: {len(audio_bytes)} bytes base64-decoded")
            try:
                result = await pizza_agent.run_turn(audio_bytes, session_id)
                await ws.send_json(result)
            except Exception as e:
                logger.error(f"Turn error: {e}", exc_info=True)
                await ws.send_json({"type": "error", "message": str(e)})
        else:
            await ws.send_json({"type": "error", "message": f"Unknown type: {msg_type}"})
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws.send_json({"type": "error", "message": str(e)})
    finally:
        await ws.close()


def main():
    """Run the pizza agent server."""
    import uvicorn
    logger.info("Starting Pizza Agent server on http://localhost:8000")
    uvicorn.run("examples.pizza_agent:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
