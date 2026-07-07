"""Async streaming clients for LLM, TTS, and STT."""
import base64
import json
from typing import Any, AsyncIterator, Optional

import httpx

from . import config

_timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
_limits = httpx.Limits(max_keepalive_connections=20, keepalive_expiry=60.0)
_client: Optional[httpx.AsyncClient] = None


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


async def stream_llm(
    messages: list[dict[str, str]],
    seed: int,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """Yield chunks from /v1/chat/completions."""
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
            if obj.get("usage"):
                yield {"kind": "usage", "usage": obj["usage"]}


async def stream_tts(
    text: str, voice: str, language: str
) -> AsyncIterator[dict[str, Any]]:
    """Yield audio chunks from /v1/audio/speech (response_format=pcm, streamed)."""
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


async def transcribe(
    audio: bytes,
    filename: str = "audio.wav",
    content_type: str = "audio/wav",
    language: str | None = None,
) -> dict[str, Any]:
    """STT: transcribe audio via /v1/audio/transcriptions (multipart)."""
    files = {"file": (filename, audio, content_type)}
    data: dict[str, str] = {"model": config.STT_MODEL, "response_format": "json"}
    if language:
        data["language"] = language
    url = f"{config.STT_BASE_URL}/v1/audio/transcriptions"
    resp = await get_client().post(url, files=files, data=data,
                                    headers=config.bearer(config.STT_API_KEY))
    resp.raise_for_status()
    return resp.json()

