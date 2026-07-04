"""Async streaming clients for the LLM (vLLM, OpenAI-compatible) and the TTS.

Both endpoints stream Server-Sent Events. These clients stay timing-agnostic:
they yield each item as soon as it arrives, and the pipeline stamps arrival
times on the call clock. That keeps a single clock source.
"""
import base64
import json
from typing import Any, AsyncIterator, Optional

import httpx

from . import config

# One shared client for the whole process. Connections are pooled and kept warm
# (keepalive_expiry) so real requests skip the TCP+TLS handshake. The background
# monitor pings the same hosts through this client every 5s, which keeps those
# connections alive between turns. Generous read timeout for long generations.
_timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
_limits = httpx.Limits(max_keepalive_connections=20, keepalive_expiry=60.0)
_client: Optional[httpx.AsyncClient] = None

# When True, every request delegates to client_new_connection, which opens a
# fresh connection per call (set by the --new-connection flag). Used to measure
# the handshake cost against the pooled default.
_new_conn = False


def use_new_connections(flag: bool = True) -> None:
    global _new_conn
    _new_conn = flag


def new_connection_mode() -> bool:
    return _new_conn


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=_timeout, limits=_limits)
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def ping(url: str, headers: Optional[dict[str, str]] = None) -> int:
    if _new_conn:
        from . import client_new_connection
        return await client_new_connection.ping(url, headers)
    resp = await get_client().get(url, headers=headers or {})
    return resp.status_code


async def stream_llm(
    messages: list[dict[str, str]],
    seed: int,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """Yield chunks from /v1/chat/completions.

    Each yielded dict is one of:
      {"kind": "token", "channel": "content"|"reasoning", "content": str}
      {"kind": "usage", "usage": {...}}
      {"kind": "done", "finish_reason": str|None}

    The model separates its reasoning trace (delta.reasoning) from the spoken
    answer (delta.content); only content is meant for the TTS.
    """
    if _new_conn:
        from . import client_new_connection
        async for item in client_new_connection.stream_llm(
            messages, seed, temperature, max_tokens, enable_thinking):
            yield item
        return
    body = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": seed,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{config.LLM_BASE_URL}/v1/chat/completions"
    async with get_client().stream("POST", url, json=body, headers=headers) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if choices:
                choice = choices[0]
                delta = choice.get("delta") or {}
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning:
                    yield {"kind": "token", "channel": "reasoning", "content": reasoning}
                content = delta.get("content")
                if content:
                    yield {"kind": "token", "channel": "content", "content": content}
                if choice.get("finish_reason"):
                    yield {"kind": "done", "finish_reason": choice["finish_reason"]}
            # vLLM sends a final chunk with usage when include_usage is set.
            if obj.get("usage"):
                yield {"kind": "usage", "usage": obj["usage"]}


async def stream_tts(
    text: str, voice: str, language: str
) -> AsyncIterator[dict[str, Any]]:
    """Yield audio chunks from /v1/audio/speech (response_format=pcm, streamed).

    Each yielded dict is one of:
      {"kind": "audio", "pcm": bytes}          raw 24kHz mono 16-bit PCM
      {"kind": "usage", "usage": {...}}
      {"kind": "done"}
    """
    if _new_conn:
        from . import client_new_connection
        async for item in client_new_connection.stream_tts(text, voice, language):
            yield item
        return
    body = {
        "input": text,
        "voice": voice,
        "language": language,
        "stream": True,
        "response_format": "pcm",
    }
    if config.TTS_MODEL:
        body["model"] = config.TTS_MODEL
    url = f"{config.TTS_BASE_URL}/v1/audio/speech"
    async with get_client().stream("POST", url, json=body,
                                   headers=config.bearer(config.TTS_API_KEY)) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data:
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            otype = obj.get("type")
            if otype == "speech.audio.delta":
                b64 = obj.get("audio")
                if b64:
                    yield {"kind": "audio", "pcm": base64.b64decode(b64)}
            elif otype == "speech.audio.done":
                if obj.get("usage"):
                    yield {"kind": "usage", "usage": obj["usage"]}
                yield {"kind": "done"}


async def synthesize_wav(
    text: str, voice: str, language: str, response_format: str = "wav"
) -> bytes:
    """Non-streaming TTS: return the whole encoded audio in one response.

    Handy for producing a fixed audio file (e.g. to feed the STT).
    """
    if _new_conn:
        from . import client_new_connection
        return await client_new_connection.synthesize_wav(text, voice, language, response_format)
    body = {
        "input": text,
        "voice": voice,
        "language": language,
        "stream": False,
        "response_format": response_format,
    }
    if config.TTS_MODEL:
        body["model"] = config.TTS_MODEL
    url = f"{config.TTS_BASE_URL}/v1/audio/speech"
    resp = await get_client().post(url, json=body, headers=config.bearer(config.TTS_API_KEY))
    resp.raise_for_status()
    return resp.content


async def transcribe(
    audio: bytes,
    filename: str = "audio.wav",
    content_type: str = "audio/wav",
    language: str | None = None,
) -> dict[str, Any]:
    """STT: transcribe audio via /v1/audio/transcriptions (multipart).

    language must be an ISO-639-1 code (en, pt, ...) or None for auto-detect.
    Returns the parsed JSON, e.g. {"text": "...", "usage": {...}}.
    """
    if _new_conn:
        from . import client_new_connection
        return await client_new_connection.transcribe(audio, filename, content_type, language)
    files = {"file": (filename, audio, content_type)}
    data: dict[str, str] = {"model": config.STT_MODEL, "response_format": "json"}
    if language:
        data["language"] = language
    url = f"{config.STT_BASE_URL}/v1/audio/transcriptions"
    resp = await get_client().post(url, files=files, data=data,
                                   headers=config.bearer(config.STT_API_KEY))
    resp.raise_for_status()
    return resp.json()
