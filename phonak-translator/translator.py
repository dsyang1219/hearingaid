#!/usr/bin/env python3
"""
Phonak Live Translation Earpiece
================================

Listens to the microphone, translates speech, and speaks the result into your
default audio output -- which is your Phonak hearing aid when it is paired as a
Bluetooth audio device. No hearing-aid hacking required.

Pipeline (per utterance):
    mic --> segmentation (VAD or push-to-talk) --> speech-to-text
        --> LLM translate (with conversation context) --> streaming TTS
        --> default output (your Phonak)

Default configuration: auto-detect any spoken language -> English. Change
TARGET_LANG below, or pass --target, to flip the direction.

Two capture modes:
  * VAD (default): translates automatically after each pause in speech.
  * Push-to-talk (--ptt): hold SPACE to capture, release to translate. Best in
    noisy rooms where automatic detection over-triggers. Needs `pynput`.

Usage:
    python translator.py --list-devices          # find your Phonak / mic
    python translator.py                          # start translating (VAD)
    python translator.py --ptt                    # hold SPACE to talk
    python translator.py --target es              # translate into Spanish
    python translator.py --source ja              # skip language auto-detect
    python translator.py --input "Microphone Array"   # devices by name
    python translator.py --ptt --key f8            # different hold-to-talk key
    python translator.py --log session.jsonl       # keep a transcript

Requires: GROQ_API_KEY in the environment (free at console.groq.com). Uses
open-source models -- Whisper (STT), an open LLM (translation), and Orpheus
(TTS) -- hosted on Groq's OpenAI-compatible API.
"""

import argparse
import io
import json
import os
import queue
import sys
import threading
import time
import wave
from collections import deque
from datetime import datetime, timezone

import numpy as np
import sounddevice as sd
import webrtcvad

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Missing dependency: pip install openai (see requirements.txt)")


# ----------------------------------------------------------------------------
# Config -- tweak these (most are also CLI flags)
# ----------------------------------------------------------------------------
TARGET_LANG = "en"          # ISO code of the language you want to HEAR
SOURCE_LANG = None          # ISO code being spoken; None = auto-detect
SAMPLE_RATE = 16000         # webrtcvad requires 8k/16k/32k/48k; 16k is ideal
FRAME_MS = 30               # VAD frame size (10, 20, or 30 ms)
VAD_AGGRESSIVENESS = 2      # 0 (permissive) .. 3 (aggressive noise filtering)
SILENCE_TAIL_MS = 400       # trailing silence that ends an utterance
MIN_UTTERANCE_MS = 400      # ignore blips shorter than this
MAX_UTTERANCE_MS = 15000    # hard cap so one long talker still gets flushed

BASE_URL = "https://api.groq.com/openai/v1"
STT_MODEL = "whisper-large-v3-turbo"  # or "whisper-large-v3" for higher accuracy
LLM_MODEL = "openai/gpt-oss-120b"     # does the translating
TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "troy"          # autumn, diana, hannah, austin, daniel, troy
CONTEXT_TURNS = 4           # prior exchanges given to the translator

FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 16-bit mono

SYSTEM_PROMPT = (
    "You are a live interpreter for someone wearing a hearing aid. Translate "
    "each user message into {target}. Earlier turns are provided for context -- "
    "pronouns and references may point back to them -- but translate ONLY the "
    "latest message. Reply with the translation alone: no notes, no quotes, no "
    "romanization, no explanation. The text comes from speech recognition and "
    "may contain errors; translate what the speaker most plausibly said. Keep "
    "the register and the length of the original, because it is read aloud "
    "immediately."
)


# ----------------------------------------------------------------------------
# Audio helpers
# ----------------------------------------------------------------------------
def pcm_to_wav_buffer(pcm: bytes, rate: int = SAMPLE_RATE) -> io.BytesIO:
    """Wrap raw 16-bit mono PCM in a WAV container for the transcription API."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    buf.seek(0)
    buf.name = "utterance.wav"  # the SDK infers format from the filename
    return buf


def _device_output_rate(device):
    try:
        return int(round(sd.query_devices(device, "output")["default_samplerate"]))
    except Exception:
        return None


def _device_output_channels(device):
    try:
        return max(1, int(sd.query_devices(device, "output")["max_output_channels"]))
    except Exception:
        return 1


class _StreamResampler:
    """Linear-interpolation resampler that keeps phase continuous across
    chunks, so streamed PCM doesn't click/warble at chunk boundaries."""

    def __init__(self, orig_rate: int, target_rate: int):
        self.step = orig_rate / target_rate  # source samples per output sample
        self.pos = 0.0                        # fractional read position, in source samples
        self.buffer = np.array([], dtype=np.float32)

    def process(self, pcm: bytes) -> bytes:
        new_samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        self.buffer = np.concatenate([self.buffer, new_samples])
        out = []
        while self.pos + 1 < len(self.buffer):
            i0 = int(self.pos)
            frac = self.pos - i0
            out.append(self.buffer[i0] * (1 - frac) + self.buffer[i0 + 1] * frac)
            self.pos += self.step
        consumed = int(self.pos)
        if consumed:
            self.buffer = self.buffer[consumed:]
            self.pos -= consumed
        if not out:
            return b""
        return np.clip(np.array(out, dtype=np.float32), -32768, 32767).astype(np.int16).tobytes()


def list_devices():
    print(sd.query_devices())
    print("\nPick the INPUT for your mic and the OUTPUT for your Phonak, then "
          "pass them with --input / --output.\n"
          "Both accept an index (3) or part of a name (\"Microphone Array\").\n"
          "Tip: on most systems you can instead just set the Phonak as your "
          "system default output and leave --output unset.\n"
          "On Windows each device appears once per host API (MME, WASAPI, ...); "
          "pick one API and use it for both directions.")


def resolve_device(spec, kind: str):
    """Accept an index, a case-insensitive name fragment, or None (= default)."""
    if spec is None:
        return None
    spec = str(spec).strip()
    if spec.lstrip("-").isdigit():
        return int(spec)

    channels = "max_input_channels" if kind == "input" else "max_output_channels"
    matches = [(i, d) for i, d in enumerate(sd.query_devices())
               if d[channels] > 0 and spec.lower() in d["name"].lower()]
    if not matches:
        sys.exit(f"No {kind} device matching {spec!r}. Run --list-devices.")
    if len(matches) > 1:
        names = ", ".join(f"#{i} {d['name'].strip()}" for i, d in matches[:4])
        print(f"[warn] {len(matches)} {kind} devices match {spec!r} ({names}); "
              f"using #{matches[0][0]}.", file=sys.stderr)
    return matches[0][0]


class Microphone:
    """Mic input as a queue of fixed-size frames, with a gate and a mute.

    `enabled` decides whether frames are collected at all (push-to-talk holds
    it low between presses). `muted` is raised while our own TTS is playing so
    the translator never hears -- and re-translates -- its own voice.
    """

    def __init__(self, device):
        self.frames: "queue.Queue[bytes]" = queue.Queue()
        self.enabled = threading.Event()
        self.muted = threading.Event()
        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE, blocksize=FRAME_BYTES // 2,
            dtype="int16", channels=1, device=device, callback=self._on_audio,
        )

    def _on_audio(self, indata, _frames, _time, status):
        if status:
            print(status, file=sys.stderr)
        if self.enabled.is_set() and not self.muted.is_set():
            self.frames.put(bytes(indata))

    def __enter__(self):
        self._stream.start()
        return self

    def __exit__(self, *exc):
        self._stream.stop()
        self._stream.close()

    def flush(self):
        """Discard buffered audio so the next capture starts clean."""
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                return

    def take_all(self) -> bytes:
        chunks = []
        while True:
            try:
                chunks.append(self.frames.get_nowait())
            except queue.Empty:
                return b"".join(chunks)


# ----------------------------------------------------------------------------
# Translation engine (cloud). Swap these methods for local models
# (faster-whisper / Argos Translate / Piper) to go fully offline.
# ----------------------------------------------------------------------------
class Translator:
    def __init__(self, client: OpenAI, target: str, source=None,
                 stt_model=STT_MODEL, llm_model=LLM_MODEL, vocab=None,
                 context_turns=CONTEXT_TURNS):
        self.client = client
        self.target = target
        self.source = source
        self.stt_model = stt_model
        self.llm_model = llm_model
        self.vocab = vocab
        # Alternating user/assistant messages: two entries per exchange.
        self.history = deque(maxlen=max(0, context_turns) * 2)

    def transcribe(self, wav_buf) -> str:
        """Speech -> text in whatever language was spoken."""
        kwargs = {"model": self.stt_model, "file": wav_buf}
        if self.source:
            # Telling it the language is both faster and more accurate than
            # letting it guess from a two-second clip.
            kwargs["language"] = self.source
        if self.vocab:
            # Names and jargon it would otherwise mangle.
            kwargs["prompt"] = self.vocab
        return self.client.audio.transcriptions.create(**kwargs).text.strip()

    def translate(self, text: str) -> str:
        """Text -> text in the target language, using recent turns as context."""
        messages = [{"role": "system",
                     "content": SYSTEM_PROMPT.format(target=self.target)}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": text})

        chat = self.client.chat.completions.create(
            model=self.llm_model, temperature=0, messages=messages,
        )
        out = (chat.choices[0].message.content or "").strip()

        if self.history.maxlen:
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": out})
        return out


class Speaker:
    """Text-to-speech into the output device.

    Groq's TTS endpoint only returns a complete WAV file per request (no
    incremental PCM streaming like OpenAI's), so this fetches the whole clip,
    decodes it, then hands it to a PortAudio callback for playback -- the same
    ring-buffer/device-fallback approach as before, just filled once instead
    of incrementally.
    """

    def __init__(self, client: OpenAI, device, model=TTS_MODEL, voice=TTS_VOICE,
                 timing=False):
        self.client = client
        self.device = device
        self.model = model
        self.voice = voice
        self.timing = timing
        self.playing = threading.Event()
        self.cancel = threading.Event()

    def speak(self, text: str, mic: "Microphone | None" = None):
        self.cancel.clear()
        self.playing.set()
        if mic is not None:
            mic.muted.set()

        buf = bytearray()
        buf_lock = threading.Lock()
        source_done = threading.Event()
        playback_done = threading.Event()

        def callback(outdata, _frames, _time_info, _status):
            needed = len(outdata)
            with buf_lock:
                chunk = bytes(buf[:needed])
                del buf[:needed]
            if len(chunk) < needed:
                chunk += b"\x00" * (needed - len(chunk))
                outdata[:] = chunk
                if source_done.is_set() or self.cancel.is_set():
                    playback_done.set()
                    raise sd.CallbackStop
            else:
                outdata[:] = chunk

        try:
            t_speak0 = time.monotonic()
            resp = self.client.audio.speech.create(
                model=self.model, voice=self.voice, input=text,
                response_format="wav",
            )
            with wave.open(io.BytesIO(resp.read()), "rb") as wf:
                source_rate = wf.getframerate()
                pcm = wf.readframes(wf.getnframes())
            if self.timing:
                print(f"  [timing] tts-download={time.monotonic()-t_speak0:.2f}s",
                      file=sys.stderr)

            # Query the device's real channel count up front rather than only
            # on a PortAudioError: a channel-count mismatch alone (without a
            # samplerate mismatch) can open "successfully" and just play
            # silence, since WASAPI's rejection is specifically rate-based.
            device_channels = _device_output_channels(self.device)
            try:
                stream = sd.RawOutputStream(
                    samplerate=source_rate, channels=device_channels, dtype="int16",
                    device=self.device, callback=callback,
                )
                stream_rate, stream_channels = source_rate, device_channels
            except sd.PortAudioError:
                # Some WASAPI endpoints also reject the stream outright if the
                # rate isn't one they advertise, instead of resampling for us --
                # fall back to the device's own rate and adapt the PCM ourselves.
                stream_rate = _device_output_rate(self.device) or source_rate
                stream_channels = device_channels
                stream = sd.RawOutputStream(
                    samplerate=stream_rate, channels=stream_channels, dtype="int16",
                    device=self.device, callback=callback,
                )

            if stream_rate != source_rate:
                pcm = _StreamResampler(source_rate, stream_rate).process(pcm)
            if stream_channels > 1:
                mono = np.frombuffer(pcm, dtype=np.int16)
                pcm = np.repeat(mono, stream_channels).tobytes()
            with buf_lock:
                buf.extend(pcm)

            with stream:
                if self.cancel.is_set():
                    stream.abort()
                else:
                    source_done.set()
                    while not playback_done.is_set() and not self.cancel.is_set():
                        time.sleep(0.02)
                    if self.cancel.is_set() and not playback_done.is_set():
                        stream.abort()
        finally:
            self.playing.clear()
            if mic is not None:
                mic.muted.clear()
                mic.flush()  # drop whatever leaked in around playback


class TranscriptLog:
    """Append-only JSONL record of everything heard and said."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def write(self, heard: str, translated: str):
        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "heard": heard,
            "translation": translated,
        }
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


class Worker:
    """Single background thread that owns the slow network round trips.

    Capture threads hand utterances over and never block. On shutdown the
    queue is drained rather than abandoned, so quitting mid-translation does
    not silently swallow the last thing you said.
    """

    def __init__(self, handler):
        self._handler = handler
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="translate",
                                        daemon=True)
        self._thread.start()

    def submit(self, item):
        self._queue.put(item)

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._handler(item)
            except Exception as exc:  # one bad utterance must not end the session
                print(f"[error] {exc}", file=sys.stderr)

    def close(self, timeout=30):
        if not self._queue.empty() or self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout)
            if self._thread.is_alive():
                print("[warn] a translation was still in flight; exiting anyway.",
                      file=sys.stderr)


def make_handler(engine: Translator, speaker: Speaker, mic: Microphone,
                 log: "TranscriptLog | None", timing: bool = False):
    def handle(pcm: bytes):
        t0 = time.monotonic()
        try:
            heard = engine.transcribe(pcm_to_wav_buffer(pcm))
        except Exception as exc:
            print(f"[speech-to-text error] {exc}", file=sys.stderr)
            return
        if not heard:
            return
        print(f"  heard: {heard}")
        t1 = time.monotonic()

        try:
            text = engine.translate(heard)
        except Exception as exc:
            print(f"[translate error] {exc}", file=sys.stderr)
            return
        if not text:
            return
        print(f"  ↪ {text}")
        t2 = time.monotonic()

        if log is not None:
            log.write(heard, text)
        try:
            speaker.speak(text, mic)
        except Exception as exc:
            print(f"[tts error] {exc}", file=sys.stderr)
        if timing:
            t3 = time.monotonic()
            print(f"  [timing] stt={t1-t0:.2f}s translate={t2-t1:.2f}s "
                  f"speak={t3-t2:.2f}s total={t3-t0:.2f}s", file=sys.stderr)

    return handle


# ----------------------------------------------------------------------------
# Capture modes
# ----------------------------------------------------------------------------
def run_vad(mic: Microphone, worker: Worker, args):
    vad = webrtcvad.Vad(args.vad_aggressiveness)

    silence_tail_frames = args.silence_ms // FRAME_MS
    min_frames = MIN_UTTERANCE_MS // FRAME_MS
    max_frames = MAX_UTTERANCE_MS // FRAME_MS

    print(f"Listening -> translating to '{args.target}'. Ctrl-C to stop.\n"
          "Make sure your Phonak is the selected output device.")

    mic.enabled.set()
    buffered, trailing_silence, speaking = [], 0, False

    while True:
        frame = mic.frames.get()
        if len(frame) != FRAME_BYTES:
            continue  # skip partial frames near stream start/stop

        is_speech = vad.is_speech(frame, SAMPLE_RATE)

        if is_speech:
            speaking = True
            buffered.append(frame)
            trailing_silence = 0
        elif speaking:
            buffered.append(frame)
            trailing_silence += 1

        end_of_utterance = speaking and trailing_silence >= silence_tail_frames
        over_length = speaking and len(buffered) >= max_frames

        if end_of_utterance or over_length:
            if len(buffered) >= min_frames:
                worker.submit(b"".join(buffered))
            buffered, speaking, trailing_silence = [], False, 0


def resolve_key(name: str):
    """Turn a key name ('space', 'f8', 'a') into something pynput compares."""
    from pynput import keyboard

    key = name.strip().lower()
    if hasattr(keyboard.Key, key):
        return getattr(keyboard.Key, key)
    if len(key) == 1:
        return keyboard.KeyCode.from_char(key)
    sys.exit(f"Unknown key {name!r}. Use a single character or a name like "
             "'space', 'f8', 'ctrl_r', 'alt_r'.")


def run_ptt(mic: Microphone, worker: Worker, speaker: Speaker, args):
    """Push-to-talk: capture only while the hold key is down; ESC quits."""
    try:
        from pynput import keyboard
    except ImportError:
        sys.exit("Push-to-talk needs pynput: pip install pynput")

    hold_key = resolve_key(args.key)
    min_bytes = MIN_UTTERANCE_MS / 1000 * SAMPLE_RATE * 2
    recording = threading.Event()

    def on_press(key):
        if key != hold_key or recording.is_set():
            return
        if speaker.playing.is_set():
            speaker.cancel.set()  # barge in: stop talking, start listening
        recording.set()
        mic.flush()
        mic.enabled.set()
        print(f"● recording… (release {args.key.upper()} to translate)")

    def on_release(key):
        if key == keyboard.Key.esc:
            return False  # stops the listener
        if key != hold_key or not recording.is_set():
            return
        mic.enabled.clear()
        recording.clear()
        pcm = mic.take_all()
        if len(pcm) >= min_bytes:
            worker.submit(pcm)
        else:
            print("  (too short, ignored)")

    print(f"Push-to-talk -> translating to '{args.target}'.\n"
          f"Hold {args.key.upper()} to speak, release to hear it. "
          f"Press {args.key.upper()} while it is talking to cut it off. "
          "ESC to quit.\n"
          "Make sure your Phonak is the selected output device.")

    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as lis:
            lis.join()
    except Exception as exc:
        raise SystemExit(
            f"Could not capture the keyboard ({exc}).\n"
            "pynput needs a desktop session: X11/Wayland on Linux, or "
            "Accessibility permission for your terminal on macOS. "
            "Run without --ptt to use VAD mode instead."
        ) from exc
    finally:
        mic.enabled.clear()
        recording.clear()


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        description="Phonak live translation earpiece",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--list-devices", action="store_true",
                    help="list audio devices and exit")
    ap.add_argument("--input", default=None,
                    help="mic: device index or part of its name")
    ap.add_argument("--output", default=None,
                    help="output: device index or part of its name (default: "
                         "system default, i.e. your Phonak)")
    ap.add_argument("--target", default=TARGET_LANG,
                    help="ISO code of the language to hear")
    ap.add_argument("--source", default=SOURCE_LANG,
                    help="ISO code being spoken; omit to auto-detect")
    ap.add_argument("--vocab", default=None,
                    help="names/jargon to prime speech recognition with, "
                         "e.g. \"Phonak, Audeo, Dani\"")
    ap.add_argument("--context", type=int, default=CONTEXT_TURNS,
                    help="prior exchanges kept as context; 0 disables")
    ap.add_argument("--ptt", action="store_true",
                    help="push-to-talk: hold a key to capture (needs pynput)")
    ap.add_argument("--key", default="space",
                    help="hold-to-talk key: a character, or 'space', 'f8', "
                         "'ctrl_r', 'alt_r'")
    ap.add_argument("--log", default=None,
                    help="append a JSONL transcript to this file")
    ap.add_argument("--vad-aggressiveness", type=int, default=VAD_AGGRESSIVENESS,
                    choices=range(4),
                    help="0-3; higher rejects more background noise (VAD mode)")
    ap.add_argument("--silence-ms", type=int, default=SILENCE_TAIL_MS,
                    help="pause that ends an utterance (VAD mode)")
    ap.add_argument("--stt-model", default=STT_MODEL,
                    help="speech-to-text model")
    ap.add_argument("--llm-model", default=LLM_MODEL,
                    help="translation model")
    ap.add_argument("--tts-model", default=TTS_MODEL, help="text-to-speech model")
    ap.add_argument("--voice", default=TTS_VOICE,
                    help="autumn, diana, hannah, austin, daniel, troy")
    ap.add_argument("--timing", action="store_true",
                    help="print a per-stage latency breakdown to stderr")
    return ap


def main():
    args = build_parser().parse_args()

    if args.list_devices:
        list_devices()
        return

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("Set GROQ_API_KEY in your environment first "
                  "(free at console.groq.com).")
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    input_device = resolve_device(args.input, "input")
    output_device = resolve_device(args.output, "output")

    engine = Translator(client, args.target, args.source, args.stt_model,
                        args.llm_model, args.vocab, args.context)
    speaker = Speaker(client, output_device, args.tts_model, args.voice, args.timing)
    log = TranscriptLog(args.log) if args.log else None

    with Microphone(input_device) as mic:
        worker = Worker(make_handler(engine, speaker, mic, log, args.timing))
        try:
            if args.ptt:
                run_ptt(mic, worker, speaker, args)
            else:
                run_vad(mic, worker, args)
        except KeyboardInterrupt:
            print("\nStopping…")
            speaker.cancel.set()  # Ctrl-C means now, not after this sentence
        finally:
            worker.close()

    print("Done.")


if __name__ == "__main__":
    main()
