"""Edge audio API: POST audio in, stream response audio out.

Same VoiceCall core as the browser websocket (same session, event logging, and
STT -> LLM -> TTS pipeline); only the transport differs. Intended for a
constrained device (e.g. an Arduino assistant) that records a clip and plays the
streamed reply.

  POST /v1/edge/talk?session=<id>
    body     : input audio (WAV bytes)
    response : streamed raw PCM (24kHz mono 16-bit) as it is generated
    ?session : keep a persistent conversation across requests; omit for one-off
  DELETE /v1/edge/session/<id>   end and drop a session
"""
import asyncio
import base64
import contextlib
import json
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import config, schemas
from .call import VoiceCall

router = APIRouter()

# Active edge sessions, keyed by session id, so a device keeps its conversation
# history and one call timeline across requests. Sessions are kept hot for
# config.EDGE_SESSION_TTL_S since last use, then ended (lazy sweep per request).
_sessions: dict[str, VoiceCall] = {}
_last_used: dict[str, float] = {}


def _sweep_expired() -> None:
    now = time.monotonic()
    for sid in list(_sessions):
        call = _sessions[sid]
        # Never evict a session that is mid-request (its lock is held).
        if now - _last_used.get(sid, now) > config.EDGE_SESSION_TTL_S and not call.lock.locked():
            _sessions.pop(sid, None)
            _last_used.pop(sid, None)
            call.end()


def _get_or_create(session_id: str) -> VoiceCall:
    _sweep_expired()
    call = _sessions.get(session_id)
    if call is None:
        # New hot call for this session, but restore any prior conversation so
        # context carries across eviction / restart.
        call = VoiceCall(session_id, persist_context=True)
        call.start()
        call.load_context()
        _sessions[session_id] = call
    _last_used[session_id] = time.monotonic()
    return call


@router.post(
    "/v1/edge/talk",
    tags=["Edge"],
    summary="One spoken turn: audio in, streamed audio out",
    description=(
        "Two request shapes:\n\n"
        "* **Raw body** (`audio/wav`): the WAV bytes, no override. Simplest for a device.\n"
        "* **multipart/form-data**: an `audio` file part plus optional per-shot overrides "
        "so the client can manage the session on its side and update it every shot:\n"
        "  * `system` — replace the system prompt.\n"
        "  * `messages` — a JSON array of `{role, content}` (roles: system/user/assistant) "
        "that replaces the whole conversation.\n\n"
        "Overrides are subject to the same context budget as normal turns "
        "(LLM_CONTEXT_FRACTION of the model window); if they exceed it, the oldest "
        "user+assistant pairs are trimmed to comply (the system prompt is kept).\n\n"
        f"The response **streams raw PCM** ({config.AUDIO_SAMPLE_RATE} Hz mono 16-bit, "
        "`audio/L16`) as it is generated. Use `?session=<id>` to keep a persistent "
        "conversation; omit it for a one-off call. Response headers: `X-Session`, "
        "`X-Call-Id`, `X-Audio-Format`, `X-Sample-Rate`, `X-Channels`."
    ),
    responses={
        200: {"description": "Streamed raw PCM audio",
              "content": {"audio/L16": {"schema": {"type": "string", "format": "binary"}}}},
        400: {"model": schemas.ErrorResponse, "description": "Empty/invalid audio or bad messages JSON"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                "multipart/form-data": {"schema": {
                    "type": "object",
                    "required": ["audio"],
                    "properties": {
                        "audio": {"type": "string", "format": "binary",
                                  "description": "WAV audio for this shot"},
                        "system": {"type": "string",
                                   "description": "Optional: override the system prompt."},
                        "messages": {"type": "string",
                                     "description": "Optional: JSON array of {role, content} "
                                                    "replacing the conversation."},
                    },
                }},
            },
        }
    },
)
async def edge_talk(
    request: Request,
    session: Optional[str] = Query(
        None, description="Conversation id; reuse to continue, omit for a fresh one-off call."),
):
    session_id = session or uuid.uuid4().hex
    system: Optional[str] = None
    messages: Optional[list] = None
    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        form = await request.form()
        f = form.get("audio")
        audio = await f.read() if hasattr(f, "read") else b""
        system = (form.get("system") or None)
        raw = form.get("messages")
        if raw:
            try:
                messages = json.loads(raw)
                if not isinstance(messages, list):
                    raise ValueError("not a list")
            except Exception:
                return JSONResponse(
                    {"error": "'messages' must be a JSON array of {role, content}"}, status_code=400)
    else:
        audio = await request.body()
    if not audio:
        return JSONResponse({"error": "empty audio body"}, status_code=400)

    call = _get_or_create(session_id)
    # One turn at a time per session; hold the lock across the whole stream so a
    # second request for the same session waits its turn.
    await call.lock.acquire()
    try:
        # Client-managed per-shot override (system prompt / full conversation).
        if system is not None or messages is not None:
            call.override_context(system=system, messages=messages)
        queue: asyncio.Queue = asyncio.Queue()

        async def send(msg: dict[str, Any]) -> None:
            if msg.get("type") != "audio_out":
                return  # this transport only forwards audio
            # First audio out is this API's "on channel" moment for shot latency.
            call.record_channel_playback(
                msg["turn_index"], msg.get("phrase_index"), msg.get("chunk_index"))
            await queue.put(base64.b64decode(msg["pcm_b64"]))

        call.pipeline.send = send
        release_t = call.clock.now_ms()

        async def runner() -> None:
            try:
                await call.pipeline.run_voice_turn(audio, release_t)
            finally:
                await queue.put(None)  # end-of-stream sentinel

        task = asyncio.create_task(runner())
    except BaseException:
        call.lock.release()
        raise

    async def body_stream():
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            with contextlib.suppress(Exception):
                await task
            call.pipeline.send = None
            call.save_context()  # persist history now, so it survives a restart while hot
            _last_used[session_id] = time.monotonic()  # idle timer runs from reply end
            call.lock.release()

    headers = {
        "X-Session": session_id,
        "X-Call-Id": call.call_id,
        "X-Audio-Format": "pcm_s16le",
        "X-Sample-Rate": str(config.AUDIO_SAMPLE_RATE),
        "X-Channels": str(config.AUDIO_CHANNELS),
    }
    return StreamingResponse(
        body_stream(),
        media_type=f"audio/L16; rate={config.AUDIO_SAMPLE_RATE}; channels={config.AUDIO_CHANNELS}",
        headers=headers,
    )


@router.delete(
    "/v1/edge/session/{session_id}",
    tags=["Edge"],
    summary="End an edge session",
    description="Ends and drops a kept-hot edge session (frees its in-memory VoiceCall).",
    response_model=schemas.EdgeSessionEnded,
    responses={404: {"model": schemas.ErrorResponse, "description": "Unknown session"}},
)
async def edge_end(session_id: str):
    call = _sessions.pop(session_id, None)
    _last_used.pop(session_id, None)
    if call is None:
        return JSONResponse({"error": "unknown session"}, status_code=404)
    call.end()
    return JSONResponse({"ended": session_id, "call_id": call.call_id})
