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
    """Yield audio chunks from /v1/audio/speech (response_format=pcm, streamed).

    Uses stream_format='audio' to get raw PCM bytes instead of SSE events.
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
    async with get_client().stream("POST", url, json=body,
                                    headers=config.bearer(config.TTS_API_KEY)) as resp:
        resp.raise_for_status()
        # Read raw PCM bytes directly
        async for chunk in resp.aiter_bytes(chunk_size=8192):
            if chunk:
                yield {"kind": "audio", "pcm": chunk}
        yield {"kind": "done"}


async def transcribe(
    audio: bytes,
    filename: str = "audio.wav",
    content_type: str = "audio/wav",
    language: str | None = None,
    input_format: str | None = None,
) -> dict[str, Any]:
    """STT: transcribe audio via /v1/audio/transcriptions (multipart).

    Args:
        audio: Raw audio bytes.
        filename: Filename sent with the multipart upload.
        content_type: MIME type of the audio file.
        language: Optional language code.
        input_format: Explicit input format hint (e.g. "pcm", "wav", "flac") sent
                      as the ``input_format`` query/body parameter.  When set to
                      ``"pcm"`` the raw PCM bytes are automatically wrapped in a
                      minimal WAV container so the STT endpoint can parse them.
                      Defaults to ``None`` — the endpoint decides based on the file
                      extension and content_type.

    The STT endpoint behaviour is general — it does not assume TTS output.
    Use ``input_format="pcm"`` only when you know the bytes are raw PCM.
    """
    if input_format == "pcm":
        import io
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio)
        audio = buf.getvalue()
        content_type = "audio/wav"
        filename = "audio.wav"

    files = {"file": (filename, audio, content_type)}
    data: dict[str, str] = {"model": config.STT_MODEL}
    if language:
        data["language"] = language
    if input_format:
        data["input_format"] = input_format
    url = f"{config.STT_BASE_URL}/v1/audio/transcriptions"
    resp = await get_client().post(url, files=files, data=data,
                                    headers=config.bearer(config.STT_API_KEY))
    resp.raise_for_status()
    return resp.json()

