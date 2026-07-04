"""
minimal HTTP server examples for the sonication voice agent library.

This module provides simple transport adapters that use the core library.
Build your own using the library directly; the core functions are in call.py,
pipeline.py, clients.py, and config.py.
"""
import asyncio
import uuid
from typing import Any, Optional

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse

from .call import VoiceCall

# Minimal websocket server example:


async def _send_to_client(ws: WebSocket, msg: dict[str, Any]) -> None:
    """Send JSON to a websocket client."""
    try:
        await ws.send_json(msg)
    except Exception:
        pass


class VoiceChannel:
    """Handles one WebSocket connection for a single call."""

    def __init__(self, ws: WebSocket, session_id: str):
        self.ws = ws
        self.call = VoiceCall(session_id, send=lambda msg: _send_to_client(ws, msg))

    async def start(self):
        """Send call_start to client."""
        now, params = self.call.start()
        await _send_to_client(self.ws, {
            "type": "call_start",
            "call_id": self.call.call_id,
            "wallclock_iso": now.isoformat(),
            "t_ms": 0.0,
            "params": params,
        })

    async def handle_message(self, msg: dict[str, Any]) -> str:
        """Process client message and return result summary."""
        from .pipeline import CallPipeline  # import here to avoid circular
        mtype = msg.get("type")
        if mtype == "text":
            text = msg.get("text", "").strip()
            if text:
                # Queue turn processing: run in background task.
                await self.call.pipeline.run_turn(text)
                return "text_in processed"
        return "ignored"

    async def run(self):
        """Main loop: start call, process messages."""
        await self.start()
        try:
            while True:
                msg = await self.ws.receive_json()
                await self.handle_message(msg)
        except Exception:
            pass  # client disconnected
        finally:
            self.call.end()


app = FastAPI(
    title="sonication",
    version="0.1.0",
    description="Minimal voice agent library. See code for pure async usage.",
)


@app.post("/api/start_call")
async def api_start_call():
    """Start a new call with a generated UUID."""
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


@app.get("/api/health", tags=["Health"])
async def api_health():
    """Basic health check."""
    return {"status": "ok"}


def main():
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()

