// Marco's Pizza push-to-talk client. Hold Enter to record the mic, release to
// send. Captures raw PCM and encodes a WAV in the browser so the STT gets a
// format it decodes reliably.

let ws = null;
let audioCtx = null;
let sessionId = null;  // unique session ID for debugging

// Recording state.
let recording = false;
let micStream = null;
let srcNode = null;
let procNode = null;
let recSamples = [];

const $ = (id) => document.getElementById(id);
const transcript = $("transcript");
const shotTimes = [];

function connect() {
  sessionId = "sess_" + Date.now() + "_" + Math.random().toString(36).slice(2, 8);
  $("session-id").textContent = sessionId;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => $("dot").classList.add("on");
  ws.onclose = () => $("dot").classList.remove("on");
  ws.onmessage = (ev) => handle(JSON.parse(ev.data));
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function ensureAudio() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === "suspended") audioCtx.resume();
}

function setMic(state) {
  const ptt = $("ptt");
  ptt.classList.toggle("recording", state === "recording");
  ptt.classList.toggle("processing", state === "processing");
  const label = {
    idle: "Hold <kbd>Enter</kbd> to talk",
    recording: "Listening... release <kbd>Enter</kbd> to send",
    processing: "Transcribing and thinking..."
  }[state];
  $("pttlabel").innerHTML = label +
    '<small>Release to send. The shot clock runs from release to first audio.</small>';
}

function bubble(role, text) {
  const el = document.createElement("div");
  el.className = "msg " + role;
  el.textContent = text;
  transcript.appendChild(el);
  transcript.scrollTop = transcript.scrollHeight;
  return el;
}

function bubbleSystem(text) {
  const el = document.createElement("div");
  el.className = "msg system";
  el.style.cssText = "background:#21262d;border:1px solid #30363d;font-size:11px;color:#7d8590;font-family:ui-monospace;";
  el.textContent = text;
  transcript.appendChild(el);
  transcript.scrollTop = transcript.scrollHeight;
}

// ---- recording ----

async function startRecording() {
  ensureAudio();
  if (!micStream) {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  }
  recSamples = [];
  srcNode = audioCtx.createMediaStreamSource(micStream);
  procNode = audioCtx.createScriptProcessor(4096, 1, 1);
  procNode.onaudioprocess = (e) => {
    recSamples.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  srcNode.connect(procNode);
  procNode.connect(audioCtx.destination);
  recording = true;
  setMic("recording");
}

function stopRecordingAndSend() {
  recording = false;
  if (procNode) { procNode.onaudioprocess = null; procNode.disconnect(); procNode = null; }
  if (srcNode) { srcNode.disconnect(); srcNode = null; }

  const total = recSamples.reduce((a, c) => a + c.length, 0);
  const flat = new Float32Array(total);
  let o = 0;
  for (const c of recSamples) { flat.set(c, o); o += c.length; }
  recSamples = [];

  if (total < audioCtx.sampleRate * 0.2) {  // ignore < 200ms taps
    setMic("idle");
    return;
  }
  setMic("processing");

  // Resample from AudioContext rate (48kHz) to 16kHz (STT default)
  const targetRate = 16000;
  const ratio = audioCtx.sampleRate / targetRate;
  const resampled = new Float32Array(Math.floor(flat.length / ratio));
  for (let i = 0; i < resampled.length; i++) {
    resampled[i] = flat[Math.floor(i * ratio)];
  }

  // Convert Float32 to raw 16-bit PCM (no WAV header — STTNode wraps it)
  const pcm = new ArrayBuffer(resampled.length * 2);
  const view = new DataView(pcm);
  for (let i = 0; i < resampled.length; i++) {
    const s = Math.max(-1, Math.min(1, resampled[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  send({ type: "user_audio", audio_b64: abToB64(pcm) });
}

function abToB64(ab) {
  let bin = "";
  const bytes = new Uint8Array(ab);
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(bin);
}

// ---- playback ----

const TTS_SAMPLE_RATE = 24000;
let ttsPlayCursor = 0;
const ttsPlaybackReported = new Set();

function pcmFromB64(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  const n = bytes.length >> 1;
  const f32 = new Float32Array(n);
  for (let i = 0; i < n; i++) f32[i] = view.getInt16(i * 2, true) / 32768;
  return f32;
}

function playTtsChunk(msg) {
  ensureAudio();
  if (!msg.pcm_b64) return;
  const f32 = pcmFromB64(msg.pcm_b64);
  const buf = audioCtx.createBuffer(1, f32.length, TTS_SAMPLE_RATE);
  buf.getChannelData(0).set(f32);
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);
  const startAt = Math.max(audioCtx.currentTime, ttsPlayCursor);
  src.start(startAt);
  ttsPlayCursor = startAt + buf.duration;

  if (!ttsPlaybackReported.has(msg.turn_index)) {
    ttsPlaybackReported.add(msg.turn_index);
    setMic("idle");
    const delayMs = Math.max(0, (startAt - audioCtx.currentTime) * 1000);
    setTimeout(() => {
      send({ type: "channel_playback_start", turn_index: msg.turn_index,
             phrase_index: msg.phrase_index, chunk_index: msg.chunk_index });
    }, delayMs);
  }
}

const fmt = (v) => (v == null ? "-" : `${Math.round(v)} ms`);

function renderWaterfall(segments) {
  const el = $("waterfall");
  if (!segments || !segments.length) { el.innerHTML = ""; return; }
  const max = Math.max(...segments.map((s) => s.ms), 1);
  el.innerHTML = segments.map((s) => {
    const width = Math.max(2, Math.round((s.ms / max) * 80));
    return `<div class="wf ${s.kind}"><span class="bar" style="width:${width}px"></span>` +
           `<span class="lbl">${s.stage}</span>` +
           `<span class="val">${Math.round(s.ms)}</span></div>`;
  }).join("");
}

function handle(msg) {
  switch (msg.type) {
    case "system":
      bubbleSystem(msg.message);
      break;
    case "response":
      bubble("user", msg.stt_text || "(no speech detected)");
      bubble("assistant", msg.llm_response || "");
      if (msg.tts_audio_b64) {
        playTtsChunk({ pcm_b64: msg.tts_audio_b64, turn_index: msg.turn_index });
      }
      if (msg.shot_latency_ms) {
        $("latency").textContent = msg.shot_latency_ms + "ms";
        shotTimes.push(msg.shot_latency_ms);
        $("m_shot").textContent = fmt(msg.shot_latency_ms);
        const avg = shotTimes.reduce((a, b) => a + b, 0) / shotTimes.length;
        $("m_shot_avg").textContent = fmt(avg);
        $("m_shot_n").textContent = `(n=${shotTimes.length})`;
      }
      if (msg.segments) {
        renderWaterfall(msg.segments);
      }
      break;
    case "audio_out":
      playTtsChunk(msg);
      break;
    case "channel_playback_start":
      setMic("idle");
      break;
    case "error":
      bubble("assistant", `[error] ${msg.message}`);
      setMic("idle");
      break;
  }
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.repeat && !recording) {
    e.preventDefault();
    startRecording().catch((err) => bubble("assistant", `[mic error] ${err.message}`));
  }
});
document.addEventListener("keyup", (e) => {
  if (e.key === "Enter" && recording) {
    e.preventDefault();
    stopRecordingAndSend();
  }
});

connect();