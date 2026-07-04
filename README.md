# minimalVoice

A minimal latency experiment for a voice agent. Today it covers the
**text in -> LLM -> TTS -> audio on channel** part of the pipeline and logs
fully timed events so latencies are reproducible and analysable. The front of
the pipeline (VAD -> turn selector -> STT) is not built yet, but the event and
timing model already reserves slots for it.

## Concepts

- **call**: one conversation. `call_start` is wallclock zero and is the only
  event with an absolute timestamp. Every other event carries `t_ms` relative
  to that zero, stamped on the backend so there is a single clock source.
- **turn** (a "shot"): one Enter-to-answer cycle inside a call. The headline
  metric is **shot latency**: Enter pressed to the first audio reproduced on the
  channel. When VAD/STT arrive, the shot start moves from `text_in` to `vad_end`
  and nothing else changes.
- **phrase aggregation**: LLM tokens are condensed into phrases of at least
  `PHRASE_MIN_CHARS` (default 20) that end on `.`, `!`, `?`, or a newline. Each
  phrase is sent to the TTS as soon as it is ready, so `tts_call` overlaps the
  still-running `llm_call`. The time spent aggregating phrases is the `phrase_gate`
  stage.
- **audio_out**: each PCM delta received from the TTS is one `audio_out` event,
  the queue of speech to be reproduced on the channel. Audio is 24000 Hz mono
  16-bit PCM, so duration is `pcm_bytes / 48000` seconds. This is how we check
  that audio is generated faster than it is spoken (lead / underrun).

## Stage timing (internal vs transport)

Shot latency stays on the wall clock (release to first audio on the channel),
which keeps it honest. On top of that, each turn stores wall-clock marks that
decompose the shot into stages:

Shot latency is measured only when channel playback is present (e.g., browser mode);
headless CLI runs do not record it. The decomposition is:

```
shot = pre_stt + stt + pre_llm + llm_ttft + phrase_gate + tts_ttfb + channel_out
        transport  service  glue     service      pipeline     service    transport
```

Service stages (stt, llm_ttft, tts_ttfb) also report an `internal_ms` estimate =
observed minus the network floor (the ping RTT to that host), clamped at 0. The
services do not expose their true internal time in the API, so this is a ping
adjusted estimate. Transport and pipeline stages (upload, phrase accumulation)
are our own overhead. The live breakdown shows in the call panel; the full
decomposition per turn is in `GET /api/analysis/{call_id}`.

A background monitor pings each host's `/health` every 5s. It serves two jobs:
it is the network-floor reference shown next to each metric (in yellow), and
because it shares the one pooled HTTP client with the pipeline, those pings keep
the TLS connections warm so real requests skip the handshake (this cut TTS TTFB
from ~395ms to ~155ms and avoids LLM cold-connection stalls).

## Events

Emitted now: `call_start`, `text_in`, `llm_call` (start, ttft, full token list),
`tts_call` (per phrase: start, ttfb, usage), `audio_out` (per PCM chunk),
`channel_playback_start` (reported by the browser when audio actually plays),
`call_end`, and `stt_final` (when a voice turn is made).

Reserved for later stages: `vad_start`, `vad_end`, `stt_partial`.

## Model and context management

- **Model discovery**: on startup the server queries each service's `/v1/models`
  and adopts the first model id (LLM, STT, TTS). The configured `*_MODEL` values
  are fallbacks used only if discovery fails. It also reads the LLM's
  `max_model_len` as the context window.
- **Context trimming**: history is kept under `LLM_CONTEXT_FRACTION` (default
  0.7) of that window. After each turn the LLM's own `usage.total_tokens` is the
  measure; if it exceeds the budget, the oldest user+assistant pair(s) are
  dropped (never the system prompt). Dropped turns are gone: not saved anywhere,
  and the session_state upsert simply persists the trimmed history.

## Transports

Both transports wrap the same `VoiceCall` (backend/call.py): same session, event
logging, timing, and STT -> LLM -> TTS pipeline. Only how audio arrives and
leaves differs.

- **Browser push-to-talk** (`/ws`, backend/app.py): hold Enter to record, release
  to send; audio plays back in the page. For computer use.
- **Edge audio API** (`/v1/edge/*`, backend/edge.py): audio in, streamed audio
  out. For a constrained device (e.g. an Arduino assistant).

  ```
  POST /v1/edge/talk?session=<id>
      body     : input audio, one of:
                 - raw WAV bytes (Content-Type audio/wav): simplest, no override
                 - multipart/form-data: an `audio` file part + optional overrides
      response : streamed raw PCM, 24kHz mono 16-bit (Content-Type
                 audio/L16; rate=24000; channels=1), as it is generated
      headers  : X-Session, X-Call-Id, X-Audio-Format, X-Sample-Rate, X-Channels
      ?session : reuse to keep conversation history and one call timeline;
                 omit for a one-off call
  DELETE /v1/edge/session/<id>    end and drop a session
  ```

  Sessions are kept hot for `EDGE_SESSION_TTL_S` (default 300s) since last use;
  after that the call is ended and the next request with that id starts fresh.

  **Client-managed context (per shot).** With a multipart request the device can
  own the session state and update it on every shot via two optional form fields:

  * `system` — replace the system prompt.
  * `messages` — a JSON array of `{role, content}` (roles: system/user/assistant)
    that replaces the whole conversation.

  Overrides obey the same context budget as normal turns (`LLM_CONTEXT_FRACTION`
  of the model window); if an override exceeds it, the oldest user+assistant
  pairs are trimmed to comply (the system prompt is always kept).

  Examples:
  ```bash
  # simple: raw WAV body, server-managed session
  curl -N --ssl-no-revoke -X POST "http://127.0.0.1:8000/v1/edge/talk?session=arduino1" \
       --data-binary @question.wav -o reply.pcm

  # client-managed: override system + conversation on this shot
  curl -N --ssl-no-revoke -X POST "http://127.0.0.1:8000/v1/edge/talk?session=arduino1" \
       -F "audio=@question.wav;type=audio/wav" \
       -F 'system=You are a terse assistant. Answer in one short sentence.' \
       -F 'messages=[{"role":"user","content":"My name is Zephyr."},
                     {"role":"assistant","content":"Nice to meet you, Zephyr."}]' \
       -o reply.pcm
  ```
  Shot latency for the edge path is release (request received) to the first audio
  byte leaving the server, recorded the same way as the browser channel.

## Storage

SQLite at `data/calls.db`: `sessions -> calls -> turns -> events`. Per-turn
rolled-up metrics live on the `turns` row so the analysis space can query
percentiles without replaying every event.

## Setup

This machine installs Python packages from a private JFrog index via `uv`
(configured in `~/.config/uv/uv.toml`). The binary `websockets` wheels are
blocked there, so the app uses the pure-Python `wsproto` backend.

```powershell
uv venv .venv
$env:UV_HTTP_TIMEOUT = "300"   # the index is slow on cold-cache downloads
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the LLM API key (a `.env` with the
key from the dev is already present locally and is gitignored).

## Run

```powershell
# Web call UI with audio playback:
.\.venv\Scripts\python.exe -m backend.app
# open http://127.0.0.1:8000

# Headless batch run (no browser, no audio playback):
.\.venv\Scripts\python.exe -m backend.cli "Hello, who are you?" "Tell me a fun fact."
# Note: shot latency not recorded; requires channel playback

# Benchmark: open a fresh connection per request (measures handshake costs):
.\.venv\Scripts\python.exe -m backend.cli --new-connection "Hello, who are you?"

# TTS -> STT round-trip probe (verifies both endpoints and the audio format):
.\.venv\Scripts\python.exe -m backend.probe_stt "Sentence to speak and transcribe."
```

- Chat UI: `http://127.0.0.1:8000/`
- Analysis dashboard (p50/p90 shot latency, TTFT, TTS TTFB, audio lead/underrun):
  `http://127.0.0.1:8000/analysis`
- JSON export of a call: `GET /api/calls/{call_id}/events`

## Config

All in `.env` (see `.env.example`): LLM endpoint/key/model, TTS voice/language,
STT endpoint/model/language, sampling (`LLM_TEMPERATURE`, `LLM_SEED`,
`LLM_MAX_TOKENS`), and `PHRASE_MIN_CHARS`.

Note the language conventions differ per service: TTS takes full names
(`English`), STT takes ISO-639-1 codes (`en`) or blank for auto-detect.
