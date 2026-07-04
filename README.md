# sonication

A minimal voice agent pipeline: text in → LLM → TTS → audio out.

This is the core intelligence layer, focused on latency optimization and clean
architecture. It's designed to be used as a library or backend service.

## Pipeline

1. **Input** - receive text or audio
2. **STT** - transcribe audio to text (optional)
3. **LLM** - stream text responses from language model
4. **TTS** - stream audio from text-to-speech
5. **Output** - stream PCM audio to playback

The pipeline runs asynchronously, with TTS starting as soon as the first
phrase from the LLM is ready.

## Events

Core events:
- `call_start` - call begins, wallclock zero
- `text_in` / `audio_in` - input received
- `llm_call` - LLM streaming (tokens, usage, done)
- `tts_call` - TTS streaming (audio chunks, usage, done)
- `audio_out` - audio chunks queued for playback
- `call_end` - call finished

## Configuration

All in `.env` (see `.env.example`):

**LLM:**
- `LLM_BASE_URL` - API endpoint
- `LLM_API_KEY` - authentication key
- `LLM_MODEL` - model name
- `LLM_TEMPERATURE`, `LLM_SEED`, `LLM_MAX_TOKENS` - sampling

**TTS:**
- `TTS_BASE_URL` - API endpoint
- `TTS_VOICE` - voice selection
- `TTS_LANGUAGE` - language (full names, e.g., `English`)

**STT:**
- `STT_BASE_URL` - API endpoint
- `STT_MODEL` - model name
- `STT_LANGUAGE` - language (ISO-639-1, e.g., `en`)

**Behavior:**
- `PHRASE_MIN_CHARS` - minimum chars before sending to TTS (default 20)
- `LLM_CONTEXT_TOKENS` - if >0, trim history to this fraction of context window

## Setup

```bash
uv venv .venv
uv pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your API keys.

## Usage

### Library

```python
from backend.call import VoiceCall
from backend.pipeline import CallPipeline

# Create a call with text input
call = VoiceCall(session_id="session123")
now, params = call.start()
result = await call.pipeline.run_turn("Hello, how are you?")
print(result)
call.end()
```

### API

The minimal version is library-focused; there is no built-in HTTP server.

## Storage

SQLite at `data/calls.db` records all events and turn metrics (if persistence
is desired). Can be disabled for purely in-memory operation.

## Architecture

- `call.py` - `VoiceCall` class: one call, one clock, one pipeline
- `pipeline.py` - orchestrates: STT → LLM → phrase buffer → TTS → audio
- `clients.py` - streaming clients for LLM, TTS, STT APIs
- `config.py` - configuration and audio constants
- `events.py` - event recording with monotonic clock
- `db.py` - SQLite persistence (optional)
- `models.py` - Pydantic models

## Design Principles

1. **Minimal**: Only essential voice agent logic
2. **Optimized**: Async streaming, phrasing, warm connections
3. **Clean**: Clear separation between transport (voiced in) and core (pipeline)
4. **Generic**: Works with any OpenAI-compatible API endpoints

All services pinged every 5s for keepalive and network latency measurement.

