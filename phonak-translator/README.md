# Phonak Live Translation Earpiece

Turn any Phonak hearing aid that pairs as a Bluetooth audio device (Audéo
Marvel and newer) into a real-time translation earpiece. Your laptop listens,
translates, and speaks the result **into your ear** — because to the OS your
Phonak is just the default speaker. Zero firmware hacking.

```
mic → segmentation (VAD or push-to-talk) → speech-to-text
    → LLM translate (with conversation context) → streaming TTS
    → default output (your Phonak)
```

Default: **auto-detect any language → English**, cloud APIs for quality and low
latency. One flag flips the direction.

## Setup

```bash
cd phonak-translator
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # required
```

<details>
<summary>Windows (PowerShell)</summary>

```powershell
cd phonak-translator
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY = "sk-..."
```

- Use Python **3.12** — the newest releases often have no prebuilt wheels yet.
- If `webrtcvad` fails to build, `pip install webrtcvad-wheels` instead (same
  module name, no C compiler needed).
- If activation is blocked, run
  `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once.
</details>

On Linux/macOS **webrtcvad** needs a C compiler: `sudo apt install
build-essential python3-dev`, or `xcode-select --install`.

## Route audio to your Phonak

1. Pair the Phonak with your computer over Bluetooth (it shows up as a headset).
2. Set it as the **system default output** — then you can leave `--output`
   unset. Or find it and pin it by name:

```bash
python translator.py --list-devices
python translator.py --output "Phonak"
```

`--input` and `--output` both accept an index (`3`) or part of a device name
(`"Microphone Array"`), so you don't have to chase indices that shift whenever
a Bluetooth device connects.

> **Bluetooth quality trap.** Selecting a headset's *microphone* forces the
> whole device into hands-free mode — mono, 8–16 kHz, in both directions. Pin
> `--input` to a different mic (built-in or USB) to keep your aid in
> full-quality stereo. On Windows the two profiles appear as separate devices:
> prefer `Headphones (... Stereo)` over `Headset (... Hands-Free)`.

## Run

```bash
python translator.py                       # auto-detect → English (VAD mode)
python translator.py --ptt                 # hold SPACE to talk
python translator.py --target es           # hear everything in Spanish
python translator.py --source ja           # skip language auto-detect
python translator.py --input "USB Audio" --output "Phonak"
python translator.py --log session.jsonl   # keep a transcript
```

Speak (or point it at a TV / another person). You'll hear the translation in
your aid and see the text in the terminal.

### Two capture modes

- **VAD (default):** translates automatically after each pause in speech.
  Great hands-free for TV or a chat across a table.
- **Push-to-talk (`--ptt`):** hold **SPACE** to capture, release to hear the
  translation; **ESC** quits. Best in noisy rooms where automatic detection
  over-triggers, or when you want to pick exactly what gets translated. Tap the
  hold key while it is speaking to **cut it off and start listening again**.
  Needs `pynput` (in requirements.txt). Use `--key f8` if SPACE collides with
  your typing. On macOS you may have to grant your terminal **Accessibility**
  permission (System Settings → Privacy & Security → Accessibility).

## Options

| Flag | What it does |
|---|---|
| `--target` | Language you want to hear (`en`, `es`, `fr`, …). |
| `--source` | Language being spoken. Omit to auto-detect; setting it is faster *and* more accurate on short clips. |
| `--vocab` | Names and jargon to prime recognition with, e.g. `--vocab "Phonak, Audéo, Dani"`. Fixes words it reliably mangles. |
| `--context` | Prior exchanges kept as context so pronouns and references resolve (default 4, `0` disables). |
| `--input` / `--output` | Device index or name fragment. |
| `--ptt` / `--key` | Push-to-talk, and which key to hold. |
| `--log` | Append a JSONL transcript of everything heard and said. |
| `--vad-aggressiveness` | 0–3; raise it in noisy rooms (VAD mode). |
| `--silence-ms` | Pause length that ends an utterance. Lower = snappier, more fragments. |
| `--stt-model` | Speech-to-text model. Try `gpt-4o-mini-transcribe` for lower latency. |
| `--llm-model` / `--tts-model` / `--voice` | Model and voice selection. |

## Notes & limits

- **Latency** is utterance-based: it waits for a pause (or your key release),
  then runs speech-to-text → translate → speech. Playback starts on the first
  audio chunk rather than after the whole clip is synthesised, which takes a
  noticeable bite out of the wait. Still not word-by-word simultaneous
  interpretation.
- **Self-hearing:** the mic is muted for the duration of playback and its
  buffer flushed afterwards, so the translator never re-translates its own
  voice. Closed earpieces (including the aid) avoid acoustic leakage too.
- **Quitting mid-sentence:** ESC finishes the translation already in flight
  instead of dropping it. Ctrl-C stops immediately.
- **Cost:** roughly a few cents per active minute.
- **Privacy:** audio leaves your machine. See below to avoid that.

## Going fully offline / private

Swap the methods in the `Translator` and `Speaker` classes for local models —
no API key, works on a plane:

- **STT:** [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)
- **Translate:** [Argos Translate](https://github.com/argosopentech/argos-translate)
- **TTS:** [Piper](https://github.com/rhasspy/piper)

The capture, segmentation, and playback plumbing stays identical.
