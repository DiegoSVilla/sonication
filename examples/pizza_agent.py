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
        "message": "Welcome to Marco's Pizza! Hold Enter to speak, release to send.",
    })

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "")
            content = data.get("content", "")

            if msg_type == "audio":
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
        content="""<!DOCTYPE html>
<html>
<head>
    <title>Pizza Agent</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #fafafa; }
        h1 { text-align: center; margin-bottom: 5px; }
        .subtitle { text-align: center; color: #888; margin-bottom: 20px; }
        #output { max-height: 50vh; overflow-y: auto; margin-bottom: 20px; }
        .message { padding: 10px 14px; margin: 6px 0; border-radius: 12px; line-height: 1.4; }
        .user { background: #dbeafe; margin-left: 30%; }
        .agent { background: #dcfce7; margin-right: 30%; }
        .system { background: #f3e8ff; text-align: center; font-size: 0.85em; }
        .error { background: #fee2e2; color: #991b1b; }
        .latency { font-size: 0.75em; color: #999; margin-top: 2px; }
        #controls { display: flex; flex-direction: column; align-items: center; gap: 10px; }
        #mic-btn {
            width: 80px; height: 80px; border-radius: 50%; border: none;
            background: #e5e7eb; cursor: pointer; font-size: 28px;
            transition: all 0.15s; outline: none; user-select: none;
            display: flex; align-items: center; justify-content: center;
        }
        #mic-btn.recording { background: #ef4444; transform: scale(1.1); }
        #mic-btn:active { transform: scale(0.95); }
        .hint { color: #666; font-size: 0.85em; }
        #status { font-size: 0.8em; color: #999; height: 18px; }
        button.stats { padding: 6px 16px; border: 1px solid #ccc; background: white; border-radius: 6px; cursor: pointer; font-size: 0.85em; }
    </style>
</head>
<body>
    <h1>Marco's Pizza</h1>
    <p class="subtitle">Hold the mic button to speak, release to send</p>
    <div id="output"></div>
    <div id="controls">
        <div id="status"></div>
        <button id="mic-btn" onmousedown="startRecord()" onmouseup="stopRecord()" ontouchstart="startRecord()" ontouchend="stopRecord()">🎤</button>
        <p class="hint">Hold to record • Release to send</p>
        <button class="stats" onclick="getStats()">Stats</button>
    </div>

    <script>
        let ws = new WebSocket('ws://localhost:8000/ws');
        const output = document.getElementById('output');
        const micBtn = document.getElementById('mic-btn');
        const status = document.getElementById('status');
        let mediaRecorder, chunks = [];
        let isRecording = false;

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'system') {
                addMessage(data.message, 'system');
            } else if (data.type === 'response') {
                addMessage(data.stt_text || '(audio)', 'user');
                addMessage(data.llm_response, 'agent');
                if (data.shot_latency_ms) {
                    const lat = document.createElement('div');
                    lat.className = 'latency';
                    lat.textContent = data.shot_latency_ms + 'ms latency';
                    output.appendChild(lat);
                }
                speak(data.llm_response);
            } else if (data.type === 'stats') {
                addMessage('Turns: ' + data.turn_count + ' | Nodes: ' + data.nodes.join(', '), 'system');
            } else if (data.type === 'error') {
                addMessage(data.message, 'error');
            }
        };

        function addMessage(text, cls) {
            const div = document.createElement('div');
            div.className = 'message ' + cls;
            div.textContent = text;
            output.appendChild(div);
            output.scrollTop = output.scrollHeight;
        }

        function speak(text) {
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                const u = new SpeechSynthesisUtterance(text);
                u.rate = 1.0;
                u.pitch = 1.0;
                window.speechSynthesis.speak(u);
            }
        }

        async function startRecord() {
            if (isRecording) return;
            micBtn.classList.add('recording');
            status.textContent = 'Recording...';

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                chunks = [];
                mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
                mediaRecorder.onstop = () => sendAudio(stream);
                mediaRecorder.start(100);
            } catch (err) {
                status.textContent = 'Mic error: ' + err.message;
                micBtn.classList.remove('recording');
            }
            isRecording = true;
        }

        function stopRecord() {
            if (!isRecording) return;
            isRecording = false;
            micBtn.classList.remove('recording');
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
        }

        async function sendAudio(stream) {
            status.textContent = 'Processing...';
            const blob = new Blob(chunks, { type: 'audio/webm' });
            try {
                const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                const decoded = await audioCtx.decodeAudioData(await blob.arrayBuffer());
                const wav = audioBufferToWav(decoded);
                ws.send(JSON.stringify({ type: 'audio', content: arrayBufferToBase64(wav) }));
                await audioCtx.close();
            } catch (err) {
                status.textContent = 'Audio error: ' + err.message;
                return;
            }
            status.textContent = 'Listening...';
            stream.getTracks().forEach(t => t.stop());
        }

        function audioBufferToWav(buffer) {
            const numChannels = buffer.numberOfChannels;
            const sampleRate = buffer.sampleRate;
            const format = 1;
            const bitDepth = 16;
            const bytesPerSample = bitDepth / 8;
            const blockAlign = numChannels * bytesPerSample;
            const data = [];
            for (let c = 0; c < numChannels; c++) data.push(buffer.getChannelData(c));
            const outputLength = 44 + data[0].length * blockAlign;
            const output = new Uint8Array(outputLength);
            const view = new DataView(output);
            writeString(view, 0, 'RIFF');
            view.setUint32(4, outputLength - 8, true);
            writeString(view, 8, 'WAVE');
            writeString(view, 12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, format, true);
            view.setUint16(22, numChannels, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * blockAlign, true);
            view.setUint16(32, blockAlign, true);
            view.setUint16(34, bitDepth, true);
            writeString(view, 36, 'data');
            view.setUint32(40, data[0].length * blockAlign, true);
            let offset = 44;
            for (let i = 0; i < data[0].length; i++) {
                for (let c = 0; c < numChannels; c++) {
                    const sample = Math.max(-1, Math.min(1, data[c][i]));
                    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
                    offset += 2;
                }
            }
            return output;
        }

        function writeString(view, offset, str) {
            for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
        }

        function arrayBufferToBase64(buffer) {
            let binary = '';
            const bytes = new Uint8Array(buffer);
            const chunkSize = 8192;
            for (let i = 0; i < bytes.length; i += chunkSize) {
                binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
            }
            return btoa(binary);
        }

        function getStats() {
            ws.send(JSON.stringify({ type: 'stats' }));
        }
    </script>
</body>
</html>"""
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