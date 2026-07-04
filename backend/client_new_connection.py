"""Alternative HTTP client that opens a FRESH connection for every request.

This is the deliberately un-optimised counterpart to clients.py. Each call
creates and closes its own httpx.AsyncClient, so every LLM/TTS/STT request and
every health ping pays a new TCP + TLS handshake. Select it with the
--new-connection flag to measure the handshake cost against the pooled client.
"""
import base64
import json
from typing import Any, AsyncIterator, Optional

import httpx

from . import config

_timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)


async def stream_llm(
    messages: list[dict[str, str]],
    seed: int,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool = False,
) -> AsyncIterator[dict[str, Any]]:
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
    async with httpx.AsyncClient(timeout=_timeout) as client:
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


async def stream_tts(
    text: str, voice: str, language: str
) -> AsyncIterator[dict[str, Any]]:
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
    async with httpx.AsyncClient(timeout=_timeout) as client:
        async with client.stream("POST", url, json=body,
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
    body = {
        "input": text, "voice": voice, "language": language,
        "stream": False, "response_format": response_format,
    }
    if config.TTS_MODEL:
        body["model"] = config.TTS_MODEL
    url = f"{config.TTS_BASE_URL}/v1/audio/speech"
    async with httpx.AsyncClient(timeout=_timeout) as client:
        resp = await client.post(url, json=body, headers=config.bearer(config.TTS_API_KEY))
        resp.raise_for_status()
        return resp.content


async def transcribe(
    audio: bytes,
    filename: str = "audio.wav",
    content_type: str = "audio/wav",
    language: Optional[str] = None,
) -> dict[str, Any]:
    files = {"file": (filename, audio, content_type)}
    data: dict[str, str] = {"model": config.STT_MODEL, "response_format": "json"}
    if language:
        data["language"] = language
    url = f"{config.STT_BASE_URL}/v1/audio/transcriptions"
    async with httpx.AsyncClient(timeout=_timeout) as client:
        resp = await client.post(url, files=files, data=data,
                                 headers=config.bearer(config.STT_API_KEY))
        resp.raise_for_status()
        return resp.json()


async def ping(url: str, headers: Optional[dict[str, str]] = None) -> int:
    async with httpx.AsyncClient(timeout=httpx.Timeout(4.0)) as client:
        resp = await client.get(url, headers=headers or {})
        return resp.status_code


async def aclose() -> None:
    pass  # nothing pooled to close in new-connection mode
