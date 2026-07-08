// Marco's Pizza push-to-talk client. Hold Enter to record the mic, release to
// send. Captures raw PCM and encodes a WAV in the browser so the STT gets a
// format it decodes reliably.

let ws = null;
let audioCtx = null;

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

  const wav = encodeWav(resampled, targetRate);
  send({ type: "user_audio", audio_b64: abToB64(wav) });
}
  setMic("processing");

  // Resample from AudioContext rate (48kHz) to 16kHz for STT
  const targetRate = 16000;
  const ratio = audioCtx.sampleRate / targetRate;
  const resampled = new Float32Array(Math.floor(flat.length / ratio));
  for (let i = 0; i < resampled.length; i++) {
    resampled[i] = flat[Math.floor(i * ratio)];
  }

  const wav = encodeWav(resampled, targetRate);
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

function speak(text) {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.0;
    u.pitch = 1.0;
    window.speechSynthesis.speak(u);
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
    case "response":
      bubble("user", msg.stt_text || "(no speech detected)");
      bubble("assistant", msg.llm_response || "");
      speak(msg.llm_response);
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