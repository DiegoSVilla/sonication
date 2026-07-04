// minimalVoice push-to-talk client. Hold Enter to record the mic, release to
// send. The shot clock runs from Enter-release to the first audio played on the
// channel. Captures raw PCM and encodes a WAV in the browser so the STT gets a
// format it decodes reliably.

const SAMPLE_RATE = 24000; // TTS playback format (24kHz mono 16-bit)

let ws = null;
let audioCtx = null;
let playCursor = 0;
const playbackReported = new Set();
const assistantEls = {};

// Recording state.
let recording = false;
let micStream = null;
let srcNode = null;
let procNode = null;
let recSamples = [];

const $ = (id) => document.getElementById(id);
const transcript = $("transcript");
const phrasesEl = $("phrases");

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => $("dot").classList.add("on");
  ws.onclose = () => { $("dot").classList.remove("on"); $("callId").textContent = "disconnected"; };
  ws.onmessage = (ev) => handle(JSON.parse(ev.data));
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function ensureAudio() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    playCursor = audioCtx.currentTime;
  }
  if (audioCtx.state === "suspended") audioCtx.resume();
}

function setMic(state) {
  const ptt = $("ptt");
  ptt.classList.toggle("recording", state === "recording");
  ptt.classList.toggle("processing", state === "processing");
  const label = { idle: "Hold <kbd>Enter</kbd> to talk", recording: "Listening... release <kbd>Enter</kbd> to send",
                  processing: "Transcribing and thinking..." }[state];
  $("pttlabel").innerHTML = label +
    '<small>Release to send. The shot clock runs from release to first audio.</small>';
}

function bubble(role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  transcript.appendChild(el);
  transcript.scrollTop = transcript.scrollHeight;
  return el;
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
  // Connect through to destination so the processor runs; we never write the
  // output buffer, so nothing is played back (no echo).
  srcNode.connect(procNode);
  procNode.connect(audioCtx.destination);
  recording = true;
  setMic("recording");
}

function stopRecordingAndSend() {
  recording = false;
  // Stamp Enter-release immediately, before encoding/upload, so that time
  // counts toward shot latency.
  send({ type: "ptt_release" });
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
  const wav = encodeWav(flat, audioCtx.sampleRate);
  send({ type: "user_audio", audio_b64: abToB64(wav) });
}

function encodeWav(samples, sampleRate) {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const dv = new DataView(buf);
  const ws = (off, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
  ws(0, "RIFF"); dv.setUint32(4, 36 + n * 2, true); ws(8, "WAVE");
  ws(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, sampleRate, true); dv.setUint32(28, sampleRate * 2, true);
  dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
  ws(36, "data"); dv.setUint32(40, n * 2, true);
  let off = 44;
  for (let i = 0; i < n; i++) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    dv.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    off += 2;
  }
  return buf;
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

function playChunk(msg) {
  ensureAudio();
  const f32 = pcmFromB64(msg.pcm_b64);
  const buf = audioCtx.createBuffer(1, f32.length, SAMPLE_RATE);
  buf.getChannelData(0).set(f32);
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);
  const startAt = Math.max(audioCtx.currentTime, playCursor);
  src.start(startAt);
  playCursor = startAt + buf.duration;

  if (!playbackReported.has(msg.turn_index)) {
    playbackReported.add(msg.turn_index);
    setMic("idle");
    const delayMs = Math.max(0, (startAt - audioCtx.currentTime) * 1000);
    setTimeout(() => {
      send({ type: "channel_playback_start", turn_index: msg.turn_index,
             phrase_index: msg.phrase_index, chunk_index: msg.chunk_index });
    }, delayMs);
  }
}

const fmt = (v) => (v == null ? "-" : `${Math.round(v)} ms`);

// Per-stage shot breakdown. Service stages also show the ping-adjusted internal
// time in yellow (observed minus the network floor).
function renderWaterfall(w) {
  const el = $("waterfall");
  if (!w || !w.segments || !w.segments.length) { el.innerHTML = ""; return; }
  const max = Math.max(...w.segments.map((s) => s.ms), 1);
  el.innerHTML = w.segments.map((s) => {
    const width = Math.max(2, Math.round((s.ms / max) * 80));
    const ping = s.ping_ms != null ? `<span class="ping">(${Math.round(s.ping_ms)})</span>` : "";
    return `<div class="wf ${s.kind}"><span class="bar" style="width:${width}px"></span>` +
           `<span class="lbl">${s.stage}</span>` +
           `<span class="val">${Math.round(s.ms)}</span>${ping}</div>`;
  }).join("");
}

// Network-floor ping per service (median of the last few 5s health checks),
// shown live in the header.
function updatePings(p) {
  const parts = ["llm", "tts", "stt"].map((k) => `${k} ${p[k] == null ? "?" : p[k]}`);
  const net = $("net");
  net.textContent = parts.join(" · ") + " ms";
  net.classList.remove("stale");
}

// Running average of shot latency for this call.
const shotTimes = [];

function handle(msg) {
  switch (msg.type) {
    case "call_start":
      $("callId").textContent = msg.call_id;
      break;
    case "ping":
      updatePings(msg);
      break;
    case "timing":
      renderWaterfall(msg.waterfall);
      break;
    case "stt_final":
      bubble("user", msg.text || "(no speech detected)");
      break;
    case "turn_skipped":
      setMic("idle");
      break;
    case "llm_token": {
      let el = assistantEls[msg.turn_index];
      if (!el) { el = bubble("assistant", ""); assistantEls[msg.turn_index] = el; }
      el.textContent += msg.text;
      transcript.scrollTop = transcript.scrollHeight;
      break;
    }
    case "tts_start": {
      const p = document.createElement("div");
      p.className = "phrase";
      p.textContent = `#${msg.phrase_index} "${msg.text}"`;
      phrasesEl.prepend(p);
      break;
    }
    case "tts_done":
      break;
    case "audio_out":
      playChunk(msg);
      break;
    case "shot_latency":
      // Shot latency is the wall-clock release -> first audio on channel.
      $("m_shot").textContent = fmt(msg.shot_latency_ms);
      if (msg.shot_latency_ms != null) {
        shotTimes.push(msg.shot_latency_ms);
        const avg = shotTimes.reduce((a, b) => a + b, 0) / shotTimes.length;
        $("m_shot_avg").textContent = fmt(avg);
        $("m_shot_n").textContent = `(n=${shotTimes.length})`;
      }
      break;
    case "turn_done":
      $("m_audio").textContent = fmt(msg.metrics.audio_generated_ms);
      $("m_tokens").textContent = msg.metrics.llm_tokens ?? "-";
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
