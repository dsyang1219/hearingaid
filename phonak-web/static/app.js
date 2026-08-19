"use strict";

const LANGUAGES = [
  ["English", "English"], ["Spanish", "Spanish"], ["French", "French"],
  ["German", "German"], ["Japanese", "Japanese"], ["Korean", "Korean"],
  ["Mandarin Chinese", "Mandarin Chinese"], ["Portuguese", "Portuguese"],
  ["Arabic", "Arabic"], ["Hindi", "Hindi"], ["Italian", "Italian"],
  ["Vietnamese", "Vietnamese"],
];
const VOICES = ["autumn", "diana", "hannah", "austin", "daniel", "troy"];
const MAX_HISTORY = 4;
const COLD_START_HINT_MS = 4000;

const talkBtn = document.getElementById("talk-btn");
const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const targetSelect = document.getElementById("target-select");
const voiceSelect = document.getElementById("voice-select");

for (const [value, label] of LANGUAGES) {
  const opt = document.createElement("option");
  opt.value = value;
  opt.textContent = label;
  targetSelect.appendChild(opt);
}
for (const voice of VOICES) {
  const opt = document.createElement("option");
  opt.value = voice;
  opt.textContent = voice[0].toUpperCase() + voice.slice(1);
  voiceSelect.appendChild(opt);
}

let history = [];
let mediaRecorder = null;
let chunks = [];
let recording = false;
let unlockAudio = null; // primed <audio> element for iOS autoplay-gesture chaining

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

function setButtonState(state) {
  talkBtn.classList.remove("recording", "processing", "speaking");
  if (state) talkBtn.classList.add(state);
  talkBtn.disabled = state === "processing" || state === "speaking";
}

function pickMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
  for (const type of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(type)) return type;
  }
  return ""; // let the browser pick
}

function addTurn(heard, translated) {
  document.querySelector(".empty-state")?.remove();
  const turn = document.createElement("div");
  turn.className = "turn";
  turn.innerHTML = `
    <div class="bubble-label">Heard</div>
    <div class="bubble heard"></div>
    <div class="bubble-label">Translated</div>
    <div class="bubble translated"></div>
  `;
  turn.querySelector(".heard").textContent = heard;
  turn.querySelector(".translated").textContent = translated;
  transcriptEl.appendChild(turn);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

async function startRecording() {
  if (recording) return;
  // Prime an <audio> element inside this user gesture so a later
  // programmatic play() (after the async translate/speak round trip)
  // is still allowed by iOS Safari's autoplay policy.
  if (!unlockAudio) {
    unlockAudio = new Audio();
    unlockAudio.muted = true;
    unlockAudio.play().catch(() => {});
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    setStatus("Microphone access denied or unavailable.", true);
    return;
  }

  const mimeType = pickMimeType();
  mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  chunks = [];
  mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
  mediaRecorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    handleRecordingStopped(mediaRecorder.mimeType || mimeType || "audio/webm");
  };
  mediaRecorder.start();
  recording = true;
  setButtonState("recording");
  setStatus("Listening...");
}

function stopRecording() {
  if (!recording || !mediaRecorder) return;
  recording = false;
  mediaRecorder.stop();
}

async function handleRecordingStopped(mimeType) {
  if (chunks.length === 0) {
    setStatus("");
    setButtonState(null);
    return;
  }
  const blob = new Blob(chunks, { type: mimeType });
  setButtonState("processing");

  const coldStartTimer = setTimeout(() => {
    setStatus("Still working — the server may be waking up from idle (can take up to a minute)...");
  }, COLD_START_HINT_MS);
  setStatus("Transcribing & translating...");

  const form = new FormData();
  const ext = mimeType.includes("mp4") ? "m4a" : "webm";
  form.append("audio", blob, `utterance.${ext}`);
  form.append("target", targetSelect.value);
  form.append("history", JSON.stringify(history.slice(-MAX_HISTORY)));

  let result;
  try {
    const resp = await fetch("/api/translate", { method: "POST", body: form });
    clearTimeout(coldStartTimer);
    if (!resp.ok) throw new Error((await resp.json().catch(() => ({}))).detail || resp.statusText);
    result = await resp.json();
  } catch (err) {
    clearTimeout(coldStartTimer);
    setStatus(`Error: ${err.message}`, true);
    setButtonState(null);
    return;
  }

  if (!result.heard) {
    setStatus("Didn't catch anything — try again.");
    setButtonState(null);
    return;
  }

  addTurn(result.heard, result.translated);
  history.push({ heard: result.heard, translated: result.translated });

  setButtonState("speaking");
  setStatus("Speaking...");
  try {
    const speakForm = new FormData();
    speakForm.append("text", result.translated);
    speakForm.append("voice", voiceSelect.value);
    const speakResp = await fetch("/api/speak", { method: "POST", body: speakForm });
    if (!speakResp.ok) throw new Error((await speakResp.json().catch(() => ({}))).detail || speakResp.statusText);
    const audioBlob = await speakResp.blob();
    const url = URL.createObjectURL(audioBlob);
    const audio = unlockAudio || new Audio();
    audio.src = url;
    audio.muted = false;
    await audio.play();
    audio.onended = () => URL.revokeObjectURL(url);
  } catch (err) {
    setStatus(`Playback error: ${err.message}`, true);
  } finally {
    setButtonState(null);
    setStatus("");
  }
}

talkBtn.addEventListener("pointerdown", (e) => { e.preventDefault(); startRecording(); });
talkBtn.addEventListener("pointerup", stopRecording);
talkBtn.addEventListener("pointerleave", stopRecording);
talkBtn.addEventListener("pointercancel", stopRecording);
