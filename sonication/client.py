"""HTTP client interface for LLM, TTS, and STT services.

Stateless module — all functions accept an optional `client` parameter.
When `client` is None, a temporary httpx.AsyncClient is created and closed
after the call. This lets callers (Node, CallPipeline, etc.) manage their
own connection pools.

Usage:
    # With your own client (recommended):
    client = httpx.AsyncClient(timeout=..., limits=...)
    async for chunk in stream_llm(messages, client=client):
        ...
    await client.aclose()

    # Ephemeral client (convenient for one-off calls):
    async for chunk in stream_llm(messages):  # client=None → creates temp client
        ...
"""
import json
from typing import Any, AsyncIterator, Optional

import httpx

from . import config

# Standard timeout/limits — shared constants, no state.
_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
_LIMITS = httpx.Limits(max_keepalive_connections=20, keepalive_expiry=60.0)


def _make_client(client: Optional[httpx.AsyncClient] = None) -> tuple[httpx.AsyncClient, bool]:
    """Return a client to use.

    Returns (client, was_created) where was_created is True if we created
    a temporary client that the caller should NOT close.
    """
    if client is not None:
        return client, False
    return httpx.AsyncClient(timeout=_TIMEOUT, limits=_LIMITS), True


async def stream_llm(
    messages: list[dict[str, str]],
    seed: int,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool = False,
    client: Optional[httpx.AsyncClient] = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield chunks from /v1/chat/completions.

    Args:
        messages: Conversation messages.
        seed: Random seed.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        enable_thinking: Whether to enable thinking/reasoning.
        client: Optional httpx.AsyncClient. If None, a temporary client
               is created and closed after this call.
    """
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
    client, close_after = _make_client(client)
    try:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
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
                if obj.get("usage"):
                    yield {"kind": "usage", "usage": obj["usage"]}
    finally:
        if close_after:
            await client.aclose()


async def stream_tts(
    text: str, voice: str, language: str,
    client: Optional[httpx.AsyncClient] = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield audio chunks from /v1/audio/speech (response_format=pcm, streamed).

    Uses stream_format='audio' to get raw PCM bytes instead of SSE events.

    Args:
        text: Text to synthesize.
        voice: Voice name.
        language: Language code.
        client: Optional httpx.AsyncClient. If None, a temporary client
               is created and closed after this call.
    """
    body = {
        "input": text,
        "voice": voice,
        "language": language,
        "stream": True,
        "response_format": "pcm",
        "stream_format": "audio",
    }
    if config.TTS_MODEL:
        body["model"] = config.TTS_MODEL
    url = f"{config.TTS_BASE_URL}/v1/audio/speech"
    client, close_after = _make_client(client)
    try:
        async with client.stream("POST", url, json=body,
                                 headers=config.bearer(config.TTS_API_KEY)) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes(chunk_size=8192):
                if chunk:
                    yield {"kind": "audio", "pcm": chunk}
            yield {"kind": "done"}
    finally:
        if close_after:
            await client.aclose()


async def transcribe(
    audio: bytes,
    filename: str = "audio.wav",
    content_type: str = "audio/wav",
    language: str | None = None,
    client: Optional[httpx.AsyncClient] = None,
) -> dict[str, Any]:
    """STT: transcribe audio via /v1/audio/transcriptions (multipart).

    Expects audio already in a format the STT endpoint accepts (typically WAV).

    Args:
        audio: Audio bytes.
        filename: Filename for the multipart upload.
        content_type: MIME type of the audio.
        language: Optional language code.
        client: Optional httpx.AsyncClient. If None, a temporary client
               is created and closed after this call.
    """
    files = {"file": (filename, audio, content_type)}
    data: dict[str, str] = {"model": config.STT_MODEL}
    if language:
        data["language"] = language
    url = f"{config.STT_BASE_URL}/v1/audio/transcriptions"
    client, close_after = _make_client(client)
    try:
        resp = await client.post(url, files=files, data=data,
                                 headers=config.bearer(config.STT_API_KEY))
        resp.raise_for_status()
        return resp.json()
    finally:
        if close_after:
            await client.aclose()