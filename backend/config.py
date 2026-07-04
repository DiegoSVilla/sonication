"""Configuration and audio constants for the minimal voice latency experiment."""
import os
from pathlib import Path

# Load .env if present (tiny loader, no external dependency needed for this).
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if _ENV_PATH.exists():
    for _line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


# LLM
LLM_BASE_URL = _get("LLM_BASE_URL", "https://inferenced2.beestorm.ai").rstrip("/")
LLM_API_KEY = _get("LLM_API_KEY", "")
LLM_MODEL = _get("LLM_MODEL", "LeaderboardModel1/Agents-A1-AutoRound-W4A16-RTN")
LLM_TEMPERATURE = float(_get("LLM_TEMPERATURE", "0.7"))
LLM_SEED = int(_get("LLM_SEED", "42"))
LLM_MAX_TOKENS = int(_get("LLM_MAX_TOKENS", "512"))
# Context window (tokens). Discovered from /v1/models (max_model_len) at startup;
# 0 until then, which disables trimming. History is kept under this fraction of it.
LLM_CONTEXT_TOKENS = int(_get("LLM_CONTEXT_TOKENS", "0"))
LLM_CONTEXT_FRACTION = float(_get("LLM_CONTEXT_FRACTION", "0.7"))
# This model is a Qwen3 reasoning model. Thinking emits a long reasoning trace
# before any speakable content, which would dominate time-to-speech, so it is
# off by default for the voice experiment. Turn on to study reasoning latency.
LLM_ENABLE_THINKING = _get("LLM_ENABLE_THINKING", "false").lower() in ("1", "true", "yes")

# TTS
TTS_BASE_URL = _get("TTS_BASE_URL", "https://tts.beestorm.ai").rstrip("/")
TTS_VOICE = _get("TTS_VOICE", "Ryan")
TTS_LANGUAGE = _get("TTS_LANGUAGE", "English")
TTS_API_KEY = _get("TTS_API_KEY", "")  # optional; empty means no auth header
TTS_MODEL = _get("TTS_MODEL", "")  # discovered from /v1/models at startup; sent if set

# STT (Qwen3-ASR, OpenAI-compatible transcription). Note the language convention
# differs from the TTS: STT wants an ISO-639-1 code (en, pt, ...) and rejects
# full names like "English"; leave it empty for auto-detect.
STT_BASE_URL = _get("STT_BASE_URL", "https://stt.beestorm.ai").rstrip("/")
STT_MODEL = _get("STT_MODEL", "Qwen/Qwen3-ASR-0.6B")
STT_LANGUAGE = _get("STT_LANGUAGE", "en")
STT_API_KEY = _get("STT_API_KEY", "")  # optional; empty means no auth header


def bearer(key: str) -> dict[str, str]:
    """Authorization header for a key, or empty dict when the key is unset."""
    return {"Authorization": f"Bearer {key}"} if key else {}

# Phrase aggregation
PHRASE_MIN_CHARS = int(_get("PHRASE_MIN_CHARS", "20"))
PHRASE_END_CHARS = set(".!?\n")

# Edge sessions are kept hot for this long since last use; after that the call
# is ended and the next request starts a fresh one.
EDGE_SESSION_TTL_S = float(_get("EDGE_SESSION_TTL_S", "300"))

# Audio format confirmed from the TTS service (WAV header + PCM stream):
# 24000 Hz, mono, 16-bit signed PCM => 48000 bytes per second of audio.
AUDIO_SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1
AUDIO_BYTES_PER_SAMPLE = 2
AUDIO_BYTES_PER_SEC = AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_BYTES_PER_SAMPLE  # 48000

# System prompt kept short and conversational so replies speak well through TTS.
SYSTEM_PROMPT = (
    "You are a concise, friendly voice assistant on a phone call. "
    "Reply in short spoken sentences. Avoid lists, markdown, and emojis."
)

# Server
HOST = _get("HOST", "127.0.0.1")
PORT = int(_get("PORT", "8000"))

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "calls.db"
FRONTEND_DIR = ROOT_DIR / "frontend"


def pcm_bytes_to_ms(num_bytes: int) -> float:
    """Duration in milliseconds for a count of raw PCM bytes at the service format."""
    return (num_bytes / AUDIO_BYTES_PER_SEC) * 1000.0
