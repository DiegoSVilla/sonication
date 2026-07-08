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
        self.log_manager = None  # set by HotPipe.connect()
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
    """LLM. Handles messages + system prompt, yields tokens.

    Context management:
        The node maintains an internal conversation history starting with
        the system prompt (if provided). Each call to stream() appends the
        user message to history, sends the full history to the LLM, and
        yields tokens. After the stream completes, call complete_turn()
        to append the assistant response to history.

    Usage:
        node = LLMNode(url, system_prompt="You are helpful.")
        await node.warmup()
        full_response = ""
        async for chunk in node.stream("Hello!"):
            full_response += chunk.get("content", "")
        node.complete_turn("Hello!", full_response)
        # History now contains: system prompt + user message + assistant response
    """

    _config_label = NodeConfigLabel.LLM_STREAMING

    def __init__(self, base_url: str, api_key: str = "", db=None,
                 system_prompt: str = "", max_history: int = 50):
        super().__init__(base_url, api_key, db)
        self.system_prompt = system_prompt
        self.max_history = max_history
        self._history: list[dict] = []
        if system_prompt:
            self._history.append({"role": "system", "content": system_prompt})

    def route(self) -> str:
        return "/v1/chat/completions"

    def complete_turn(self, user_message: str, assistant_response: str) -> None:
        """Append the assistant response to conversation history.

        This is called after stream() completes. The user message was
        already appended by stream(). Only the assistant response needs
        to be added here.
        """
        self._history.append({"role": "assistant", "content": assistant_response})
        # Trim history to max_history to avoid unbounded growth
        if len(self._history) > self.max_history:
            # Keep system prompt + last max_history-1 messages
            self._history = [self._history[0]] + self._history[-(self.max_history - 1):]

    def get_history(self) -> list[dict]:
        """Return the current conversation history."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear conversation history, keeping only the system prompt."""
        if self.system_prompt:
            self._history = [{"role": "system", "content": self.system_prompt}]
        else:
            self._history = []

    async def stream(self, user_message: str, **kwargs) -> AsyncIterator[dict]:
        """Stream tokens for a single user message with context management.

        Appends user_message to history, sends full history to LLM,
        yields tokens. Call complete_turn() after streaming to persist
        the assistant response.
        """
        self._history.append({"role": "user", "content": user_message})

        # Trim history if needed (in case stream was called without complete_turn)
        if len(self._history) > self.max_history:
            self._history = [self._history[0]] + self._history[-(self.max_history - 1):]

        async for chunk in clients.stream_llm(
            self._history,
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