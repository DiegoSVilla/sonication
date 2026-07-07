"""Nodes — streaming transformers with keepalive connections.

Each Node manages a single HTTP keepalive session to its endpoint.
Subclasses implement stream() which calls a backend client function
and yields data dicts with a "kind" key.

The HotPipe wraps stream() yields into PipeEvents for logging
and timing.

Keepalive:
    Node.warmup() - establishes one HTTP connection via /ping
    Node.close()  - closes the connection on shutdown
    HotPipe periodically pings to maintain keepalive sessions
"""
import asyncio
import io
import logging
import wave
from typing import AsyncIterator, Optional

import httpx

from . import clients
from . import config
from .node_types import NodeConfigLabel

logger = logging.getLogger(__name__)


class Node:
    """Base node with keepalive connection management.

    Usage:
        node = STTNode("http://127.0.0.1:8092")
        assert await node.warmup()  # ONE connection per turn
        async for event in node.stream(data):
            print(event)
        await node.close()
    """

    _config_label: str = "UNKNOWN"

    def __init__(self, base_url: str, api_key: str = "", db=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.db = db
        self.connection: httpx.AsyncClient = None
        self.is_warm: bool = False
        self._stage_id: Optional[str] = None

    def _auth_headers(self) -> dict:
        """Build auth headers from api_key."""
        return clients.config.bearer(self.api_key) if self.api_key else {}

    async def warmup(self, timeout: float = 5.0) -> bool:
        """Establish keepalive connection via /ping."""
        if not self.connection:
            self.connection = httpx.AsyncClient(timeout=clients._timeout, limits=clients._limits)
        try:
            resp = await asyncio.wait_for(
                self.connection.get(f"{self.base_url}/ping"),
                timeout=timeout,
            )
            if resp.status_code == 200:
                self.is_warm = True
                return True
        except Exception as e:
            logger.warning(f"Node({self.base_url}/ping) failed: {e}")
        return self.is_warm

    async def close(self):
        """Close the keepalive connection."""
        if self.connection:
            await self.connection.aclose()
            self.connection = None
        self.is_warm = False

    async def collect_status(self) -> dict:
        """Health check."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/ping")
                return {
                    "status": resp.status_code,
                    "route": self.route(),
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def route(self) -> str:
        raise NotImplementedError

    async def stream(self, *args, **kwargs) -> AsyncIterator[dict]:
        raise NotImplementedError

    @property
    def config_label(self) -> str:
        """The streaming configuration label for this node."""
        return self._config_label

    @property
    def stage_id(self) -> Optional[str]:
        """Unique ID for this node's stage in a pipeline."""
        return self._stage_id

    @property
    def node_class(self) -> str:
        """Class name for observability."""
        return self.__class__.__name__


class STTNode(Node):
    """Speech-to-Text. Accepts PCM bytes, yields transcript."""

    _config_label = NodeConfigLabel.STT_NON_STREAMING

    def __init__(self, base_url: str, api_key: str = "", db=None):
        super().__init__(base_url, api_key, db)

    def route(self) -> str:
        return "/v1/audio/transcriptions"

    async def stream(self, audio_bytes: bytes, language: str = None) -> AsyncIterator[dict]:
        """Transcribe PCM audio. Converts to WAV internally."""
        try:
            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(config.AUDIO_SAMPLE_RATE)
                wf.writeframes(audio_bytes)
            wav_bytes = wav_buf.getvalue()

            result = await clients.transcribe(wav_bytes, filename="audio.wav", language=language)
            text = result.get("text", "")
            yield {"kind": "transcript", "text": text, "usage": result.get("usage", {})}
            yield {"kind": "done", "text": text}

        except Exception as e:
            logger.error(f"STT failed: {e}")
            yield {"kind": "error", "message": str(e)}


class LLMNode(Node):
    """LLM. Handles messages + system prompt, yields tokens."""

    _config_label = NodeConfigLabel.LLM_STREAMING

    def __init__(self, base_url: str, api_key: str = "", db=None, system_prompt: str = ""):
        super().__init__(base_url, api_key, db)
        self.system_prompt: str = system_prompt

    def route(self) -> str:
        return "/v1/chat/completions"

    async def stream(self, messages: list, **kwargs) -> AsyncIterator[dict]:
        """Stream tokens with system prompt injected."""
        all_messages = []
        if self.system_prompt:
            all_messages.append({"role": "system", "content": self.system_prompt})
        all_messages.extend(messages)

        async for chunk in clients.stream_llm(
            all_messages,
            seed=kwargs.get("seed", config.LLM_SEED),
            temperature=kwargs.get("temperature", config.LLM_TEMPERATURE),
            max_tokens=kwargs.get("max_tokens", config.LLM_MAX_TOKENS),
            enable_thinking=kwargs.get("enable_thinking", config.LLM_ENABLE_THINKING),
        ):
            yield chunk


class TTSNode(Node):
    """Text-to-Speech. Accepts text, yields PCM audio chunks."""

    _config_label = NodeConfigLabel.TTS_CHUNK_IN_STREAM_OUT

    def __init__(self, base_url: str, api_key: str = "", db=None,
                 voice: str = None, language: str = None):
        super().__init__(base_url, api_key, db)
        self.voice = voice or config.TTS_VOICE
        self.language = language or config.TTS_LANGUAGE

    def route(self) -> str:
        return "/v1/audio/speech"

    async def stream(self, text: str, voice: str = None, language: str = None) -> AsyncIterator[dict]:
        """Stream audio chunks."""
        voice = voice or self.voice
        language = language or self.language
        async for chunk in clients.stream_tts(text, voice, language):
            yield chunk