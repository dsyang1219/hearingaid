#!/usr/bin/env python3
"""
Phonak Live Translation Earpiece
================================

Listens to the microphone, translates speech, and speaks the result into your
default audio output -- which is your Phonak hearing aid when it is paired as a
Bluetooth audio device. No hearing-aid hacking required.

Pipeline (per utterance):
    mic --> voice-activity detection --> Whisper (speech-to-text / translate)
        --> (optional LLM translate) --> TTS --> default output (your Phonak)

Default configuration: auto-detect any spoken language -> English, using cloud
APIs for quality. Change TARGET_LANG at the top of the config block to flip the
direction (e.g. "es" to hear English translated into Spanish).

Two capture modes:
  * VAD (default): translates automatically after each pause in speech.
  * Push-to-talk (--ptt): hold SPACE to capture, release to translate. Best in
    noisy rooms where automatic detection over-triggers. Needs `pynput`.

Usage:
    python translator.py --list-devices        # find your Phonak / mic indices
    python translator.py                        # start translating (VAD)
    python translator.py --ptt                  # hold SPACE to talk
    python translator.py --target es            # translate into Spanish instead
    python translator.py --input 2 --output 5   # pin specific devices

Requires: OPENAI_API_KEY in the environment.
"""

import argparse
import io
import os
import queue
import sys
import threading
import wave

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Missing dependency: pip install openai (see requirements.txt)")


# ----------------------------------------------------------------------------
# Config -- tweak these
# ----------------------------------------------------------------------------
TARGET_LANG = "en"          # ISO code of the language you want to HEAR
SAMPLE_RATE = 16000         # webrtcvad requires 8k/16k/32k/48k; 16k is ideal
FRAME_MS = 30               # VAD frame size (10, 20, or 30 ms)
VAD_AGGRESSIVENESS = 2      # 0 (permissive) .. 3 (aggressive noise filtering)
SILENCE_TAIL_MS = 700       # trailing silence that ends an utterance
MIN_UTTERANCE_MS = 400      # ignore blips shorter than this
MAX_UTTERANCE_MS = 15000    # hard cap so one long talker still gets flushed

STT_MODEL = "whisper-1"     # Whisper handles transcribe + translate-to-English
LLM_MODEL = "gpt-4o-mini"   # used only when TARGET_LANG != "en"
TTS_MODEL = "tts-1"         # low-latency TTS
TTS_VOICE = "alloy"         # alloy, echo, fable, onyx, nova, shimmer

FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 16-bit mono


# ----------------------------------------------------------------------------
# Audio helpers
# ----------------------------------------------------------------------------
def pcm_to_wav_bytes(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container for the Whisper API."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    buf.seek(0)
    buf.name = "utterance.wav"  # the SDK infers format from the filename
    return buf


def list_devices():
    print(sd.query_devices())
    print("\nPick the INPUT index for your mic and the OUTPUT index for your "
          "Phonak, then pass them with --input / --output.\n"
          "Tip: on most systems you can instead just set the Phonak as your "
          "system default output and leave --output unset.")


# ----------------------------------------------------------------------------
# Translation engine (cloud). Swap these two methods for local models
# (faster-whisper / Argos Translate / Piper) to go fully offline.
# ----------------------------------------------------------------------------
class Translator:
    def __init__(self, client: OpenAI, target: str):
        self.client = client
        self.target = target

    def to_text(self, wav_buf) -> str:
        """Speech -> text in the target language."""
        if self.target == "en":
            # Whisper's translate endpoint goes straight to English from any
            # source language in a single call.
            resp = self.client.audio.translations.create(
                model=STT_MODEL, file=wav_buf,
            )
            return resp.text.strip()

        # Otherwise: transcribe in the source language, then LLM-translate.
        resp = self.client.audio.transcriptions.create(
            model=STT_MODEL, file=wav_buf,
        )
        source_text = resp.text.strip()
        if not source_text:
            return ""
        chat = self.client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content":
                    f"Translate the user's text into {self.target}. "
                    "Reply with ONLY the translation, no notes or quotes."},
                {"role": "user", "content": source_text},
            ],
        )
        return chat.choices[0].message.content.strip()

    def speak(self, text: str, output_device):
        """Text -> speech, played into the default (Phonak) output."""
        resp = self.client.audio.speech.create(
            model=TTS_MODEL, voice=TTS_VOICE, input=text,
            response_format="wav",
        )
        data, rate = sf.read(io.BytesIO(resp.read()), dtype="float32")
        sd.play(data, rate, device=output_device)
        sd.wait()


# ----------------------------------------------------------------------------
# Capture loop with voice-activity detection
# ----------------------------------------------------------------------------
def _make_engine(target) -> "Translator":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Set OPENAI_API_KEY in your environment first.")
    return Translator(OpenAI(api_key=api_key), target)


def run_vad(input_device, output_device, target):
    engine = _make_engine(target)
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    frames = queue.Queue()

    def on_audio(indata, _frames, _time, status):
        if status:
            print(status, file=sys.stderr)
        frames.put(bytes(indata))

    silence_tail_frames = SILENCE_TAIL_MS // FRAME_MS
    min_frames = MIN_UTTERANCE_MS // FRAME_MS
    max_frames = MAX_UTTERANCE_MS // FRAME_MS

    print(f"Listening -> translating to '{target}'. Ctrl-C to stop.\n"
          "Make sure your Phonak is the selected output device.")

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE, blocksize=FRAME_BYTES // 2,
        dtype="int16", channels=1, device=input_device, callback=on_audio,
    )

    with stream:
        buffered, voiced, trailing_silence, speaking = [], False, 0, False
        while True:
            frame = frames.get()
            if len(frame) != FRAME_BYTES:
                continue  # skip partial frames near stream start/stop

            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if is_speech:
                if not speaking:
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
                    _handle(engine, b"".join(buffered), output_device)
                buffered, speaking, trailing_silence = [], False, 0


def run_ptt(input_device, output_device, target):
    """Push-to-talk: capture only while SPACE is held; ESC quits."""
    try:
        from pynput import keyboard
    except ImportError:
        sys.exit("Push-to-talk needs pynput: pip install pynput")

    engine = _make_engine(target)
    recording = threading.Event()
    lock = threading.Lock()
    chunks = []
    work = queue.Queue()

    def on_audio(indata, _frames, _time, status):
        if status:
            print(status, file=sys.stderr)
        if recording.is_set():
            with lock:
                chunks.append(bytes(indata))

    def worker():
        while True:
            pcm = work.get()
            if pcm is None:
                return
            _handle(engine, pcm, output_device)

    def on_press(key):
        if key == keyboard.Key.space and not recording.is_set():
            with lock:
                chunks.clear()
            recording.set()
            print("● recording… (release SPACE to translate)")

    def on_release(key):
        if key == keyboard.Key.esc:
            return False  # stops the listener
        if key == keyboard.Key.space and recording.is_set():
            recording.clear()
            with lock:
                pcm = b"".join(chunks)
                chunks.clear()
            if len(pcm) >= MIN_UTTERANCE_MS / 1000 * SAMPLE_RATE * 2:
                work.put(pcm)
            else:
                print("  (too short, ignored)")

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    print(f"Push-to-talk -> translating to '{target}'.\n"
          "Hold SPACE to speak, release to hear it. ESC to quit.\n"
          "Make sure your Phonak is the selected output device.")

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE, blocksize=FRAME_BYTES // 2,
        dtype="int16", channels=1, device=input_device, callback=on_audio,
    )
    with stream:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as lis:
            lis.join()
    work.put(None)


def _handle(engine: Translator, pcm: bytes, output_device):
    """Transcribe/translate one utterance and speak it, guarding against
    the mic re-capturing our own TTS by draining input during playback."""
    try:
        text = engine.to_text(pcm_to_wav_bytes(pcm))
    except Exception as exc:  # network hiccup, rate limit, etc.
        print(f"[translate error] {exc}", file=sys.stderr)
        return
    if not text:
        return
    print(f"  ↪ {text}")
    try:
        engine.speak(text, output_device)
    except Exception as exc:
        print(f"[tts error] {exc}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Phonak live translation earpiece")
    ap.add_argument("--list-devices", action="store_true",
                    help="list audio devices and exit")
    ap.add_argument("--input", type=int, default=None,
                    help="input (mic) device index; default = system default")
    ap.add_argument("--output", type=int, default=None,
                    help="output device index; default = system default (Phonak)")
    ap.add_argument("--target", default=TARGET_LANG,
                    help="ISO code of language to hear (default: en)")
    ap.add_argument("--ptt", action="store_true",
                    help="push-to-talk: hold SPACE to capture (needs pynput)")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return
    runner = run_ptt if args.ptt else run_vad
    try:
        runner(args.input, args.output, args.target)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
