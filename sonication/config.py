"""Configuration and audio constants for the minimal voice agent library."""
import os
from pathlib import Path

# Load .env if present (tiny loader, no external dependency needed).
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
LLM_CONTEXT_TOKENS = int(_get("LLM_CONTEXT_TOKENS", "0"))
LLM_CONTEXT_FRACTION = float(_get("LLM_CONTEXT_FRACTION", "0.7"))
LLM_ENABLE_THINKING = _get("LLM_ENABLE_THINKING", "false").lower() in ("1", "true", "yes")

# TTS
TTS_BASE_URL = _get("TTS_BASE_URL", "https://tts.beestorm.ai").rstrip("/")
TTS_VOICE = _get("TTS_VOICE", "Ryan")
TTS_LANGUAGE = _get("TTS_LANGUAGE", "English")
TTS_API_KEY = _get("TTS_API_KEY", "")  # optional; empty means no auth header
TTS_MODEL = _get("TTS_MODEL", "")  # discovered from /v1/models at startup; sent if set

# STT
STT_BASE_URL = _get("STT_BASE_URL", "https://stt.beestorm.ai").rstrip("/")
STT_MODEL = _get("STT_MODEL", "Qwen/Qwen3-ASR-0.6B")
STT_LANGUAGE = _get("STT_LANGUAGE", "en")
STT_API_KEY = _get("STT_API_KEY", "")  # optional; empty means no auth header

# Phrase aggregation
PHRASE_MIN_CHARS = int(_get("PHRASE_MIN_CHARS", "20"))
PHRASE_END_CHARS = set(".!?\n")

# Audio format: 24000 Hz, mono, 16-bit signed PCM => 48000 bytes/sec
AUDIO_SAMPLE_RATE = 24000
AUDIO_CHANNELS = 1
AUDIO_BYTES_PER_SAMPLE = 2
AUDIO_BYTES_PER_SEC = AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_BYTES_PER_SAMPLE  # 48000

# System prompt
SYSTEM_PROMPT = (
    "You are a concise, friendly voice assistant on a phone call. "
    "Reply in short spoken sentences. Avoid lists, markdown, and emojis."
)

# Server (example usage only)
HOST = _get("HOST", "127.0.0.1")
PORT = int(_get("PORT", "8000"))

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "calls.db"


def bearer(key: str) -> dict[str, str]:
    """Authorization header for a key, or empty dict when the key is unset."""
    return {"Authorization": f"Bearer {key}"} if key else {}


def pcm_bytes_to_ms(num_bytes: int) -> float:
    """Duration in milliseconds for a count of raw PCM bytes at the service format."""
    return (num_bytes / AUDIO_BYTES_PER_SEC) * 1000.0

