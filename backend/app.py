"""FastAPI app: websocket call channel, static chat UI, and analysis endpoints."""
import asyncio
import contextlib
import uuid
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import analysis, clients, config, db, edge, models, schemas
from .call import VoiceCall
from .monitor import monitor

API_DESCRIPTION = """
Voice-to-voice latency experiment. A **call** is one conversation; `call_start`
is wallclock zero and every other event's `t_ms` is relative to it. A **turn**
(a "shot") is one Enter-to-answer cycle. The headline metric is **shot latency**:
release -> first audio reproduced on the channel.

Pipeline: **text/voice in -> STT -> LLM -> phrase aggregator -> TTS -> audio out**,
all timed. Two transports share the same core (`VoiceCall`):

* **Browser push-to-talk** over the `/ws` websocket (documented below).
* **Edge audio API** under `/v1/edge/*` (audio in, streamed audio out) for devices.

### Realtime websocket protocol (`/ws`)
Not part of OpenAPI, so documented here. Connect with an optional `?session=<id>`.

**Client -> server** (JSON frames):
* `{"type":"user_text","text":"..."}` — text-in turn.
* `{"type":"ptt_release"}` — stamps the shot start the instant Enter is released.
* `{"type":"user_audio","audio_b64":"<wav>"}` — push-to-talk audio for a voice turn.
* `{"type":"channel_playback_start","turn_index":N,"phrase_index":..,"chunk_index":..}`
  — the browser telling the server audio actually started on the channel.

**Server -> client** (JSON frames):
* `call_start` — call id + wallclock zero.
* `text_in`, `stt_final` — input recognised.
* `llm_start`, `llm_token`, `llm_done` — LLM stream.
* `tts_start`, `audio_out` (base64 PCM chunk), `tts_done` — speech synthesis.
* `timing` — per-turn stage waterfall (see the Waterfall schema).
* `turn_done` — rolled-up turn metrics.
* `shot_latency` — wall-clock release -> first audio on channel.
* `ping` — endpoint RTT snapshot, every 5s (see PingSnapshot).
* `error`, `turn_skipped`.
"""

TAGS_METADATA = [
    {"name": "UI", "description": "Served HTML pages (chat + analysis dashboard)."},
    {"name": "Analysis", "description": "Stored call/turn timings and percentiles."},
    {"name": "Monitoring", "description": "Live backend health / network floor."},
    {"name": "Edge", "description": "Audio-in / stream-out API for edge devices."},
    {"name": "Realtime", "description": "Browser push-to-talk websocket (see the description above)."},
]


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    info = await models.discover()  # adopt each service's first model + LLM context
    print(">>> model discovery:", info)
    await monitor.start()  # background endpoint pinger (also keeps connections warm)
    try:
        yield
    finally:
        await monitor.stop()
        await clients.aclose()


app = FastAPI(
    title="minimalVoice",
    version="0.1.0",
    summary="Text/voice -> LLM -> TTS latency experiment with per-stage timing.",
    description=API_DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
)
db.init_db()
app.mount("/static", StaticFiles(directory=str(config.FRONTEND_DIR)), name="static")


@app.get("/", tags=["UI"], summary="Chat UI",
         description="The push-to-talk browser client (hold Enter to talk).",
         response_class=FileResponse, include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(config.FRONTEND_DIR / "index.html")


@app.get("/analysis", tags=["UI"], summary="Analysis dashboard",
         description="p50/p90 shot latency, STT/LLM/TTS timings, audio lead/underrun.",
         response_class=FileResponse, include_in_schema=False)
async def analysis_page() -> FileResponse:
    return FileResponse(config.FRONTEND_DIR / "analysis.html")


@app.get("/api/calls", tags=["Analysis"], summary="List calls",
         description="Every recorded call, oldest first.",
         response_model=list[schemas.CallInfo])
async def api_calls() -> JSONResponse:
    return JSONResponse(db.list_calls())


@app.get("/api/calls/{call_id}/events", tags=["Analysis"],
         summary="Export a call's events",
         description="Full timeline: the call row, its turns, and every timed event "
                     "(t_ms relative to call_start).",
         response_model=schemas.CallEventsExport)
async def api_events(call_id: str) -> JSONResponse:
    return JSONResponse(
        {
            "call": db.get_call(call_id),
            "turns": db.get_turns(call_id),
            "events": db.get_events(call_id),
        }
    )


@app.get("/api/ping", tags=["Monitoring"], summary="Endpoint ping snapshot",
         description="Median health-route RTT (ms) to LLM/TTS/STT over the last few 5s checks.",
         response_model=schemas.PingSnapshot)
async def api_ping() -> JSONResponse:
    return JSONResponse(monitor.snapshot())


@app.get("/api/analysis", tags=["Analysis"], summary="Aggregate analysis",
         description="Cross-call rollup: p50/p90/p99 of shot latency, STT, LLM TTFT, TTS TTFB.",
         response_model=schemas.AggregateAnalysis)
async def api_analysis() -> JSONResponse:
    return JSONResponse(analysis.summarize_all())


@app.get("/api/analysis/{call_id}", tags=["Analysis"], summary="Per-call analysis",
         description="Per-turn stage waterfall and shot latency for one call.",
         response_model=schemas.CallAnalysis)
async def api_analysis_call(call_id: str) -> JSONResponse:
    return JSONResponse(analysis.summarize_call(call_id))


app.include_router(edge.router)  # audio-in / stream-out API for edge devices


class CallChannel:
    """Browser transport: one websocket == one VoiceCall. Serialises turns and
    handles the browser's playback-start callback."""

    def __init__(self, ws: WebSocket, session_id: str) -> None:
        self.ws = ws
        self._send_lock = asyncio.Lock()
        self.call = VoiceCall(session_id, send=self._send)
        self._turn_queue: asyncio.Queue = asyncio.Queue()
        # Enter-release time (shot start) captured before the audio is uploaded.
        self._pending_release_t: Optional[float] = None

    async def _send(self, msg: dict[str, Any]) -> None:
        async with self._send_lock:
            await self.ws.send_json(msg)

    async def start(self) -> None:
        now, _ = self.call.start()
        await self._send(
            {"type": "call_start", "call_id": self.call.call_id,
             "wallclock_iso": now.isoformat(), "t_ms": 0.0}
        )

    async def ping_pusher(self) -> None:
        """Push the current endpoint ping snapshot to this browser every 5s."""
        try:
            while True:
                await self._send({"type": "ping", **monitor.snapshot()})
                await asyncio.sleep(5.0)
        except (WebSocketDisconnect, RuntimeError):
            pass  # socket closed; the pusher exits with the connection

    async def turn_worker(self) -> None:
        while True:
            item = await self._turn_queue.get()
            if item is None:
                break
            try:
                if item["kind"] == "text":
                    await self.call.pipeline.run_turn(item["text"])
                elif item["kind"] == "voice":
                    await self.call.pipeline.run_voice_turn(item["audio"], item["release_t"])
            except Exception as exc:  # surface pipeline errors to the channel
                await self._send({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    async def handle_message(self, msg: dict[str, Any]) -> None:
        mtype = msg.get("type")
        if mtype == "user_text":
            text = (msg.get("text") or "").strip()
            if text:
                await self._turn_queue.put({"kind": "text", "text": text})
        elif mtype == "ptt_release":
            # Stamp the shot start the instant Enter is released, before the
            # audio is encoded and uploaded, so that time counts toward latency.
            self._pending_release_t = self.call.clock.now_ms()
        elif mtype == "user_audio":
            import base64
            audio = base64.b64decode(msg.get("audio_b64", ""))
            release_t = self._pending_release_t
            if release_t is None:
                release_t = self.call.clock.now_ms()
            self._pending_release_t = None
            await self._turn_queue.put(
                {"kind": "voice", "audio": audio, "release_t": release_t}
            )
        elif mtype == "channel_playback_start":
            res = self.call.record_channel_playback(
                int(msg.get("turn_index", 0)), msg.get("phrase_index"), msg.get("chunk_index"))
            if res is not None:
                await self._send({"type": "shot_latency", **res})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session_id = ws.query_params.get("session") or uuid.uuid4().hex
    channel = CallChannel(ws, session_id)
    await channel.start()
    worker = asyncio.create_task(channel.turn_worker())
    pinger = asyncio.create_task(channel.ping_pusher())
    try:
        while True:
            msg = await ws.receive_json()
            await channel.handle_message(msg)
    except WebSocketDisconnect:
        pass
    finally:
        pinger.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pinger
        await channel._turn_queue.put(None)
        await worker
        channel.call.end()


def main() -> None:
    import sys
    import uvicorn
    if "--new-connection" in sys.argv:
        clients.use_new_connections(True)
        print(">>> NEW-CONNECTION MODE: every request and ping opens a fresh "
              "connection (no keepalive). Expect higher latencies.")
    # wsproto is the pure-Python websocket backend (the binary `websockets`
    # wheels are blocked in the private index).
    uvicorn.run("backend.app:app", host=config.HOST, port=config.PORT,
                reload=False, ws="wsproto")


if __name__ == "__main__":
    main()
