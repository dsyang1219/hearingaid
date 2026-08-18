# Phonak Live Translation Earpiece

Turn any Phonak hearing aid that pairs as a Bluetooth audio device (Audéo
Marvel and newer) into a real-time translation earpiece. Your laptop listens,
translates, and speaks the result **into your ear** — because to the OS your
Phonak is just the default speaker. Zero firmware hacking.

```
mic → voice-activity detection → Whisper (STT / translate)
    → (optional LLM translate) → TTS → default output (your Phonak)
```

Default: **auto-detect any language → English**, cloud APIs for quality and low
latency. One config line flips the direction.

## Setup

```bash
cd phonak-translator
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...        # required
```

**webrtcvad** needs a C compiler. On Debian/Ubuntu: `sudo apt install
build-essential python3-dev`. On macOS: install Xcode command-line tools
(`xcode-select --install`).

## Route audio to your Phonak

1. Pair the Phonak with your computer over Bluetooth (it shows up as a headset).
2. Set it as the **system default output** — then you can leave `--output`
   unset. Or discover the index and pin it:

```bash
python translator.py --list-devices
```

## Run

```bash
python translator.py                     # auto-detect → English (VAD mode)
python translator.py --ptt               # hold SPACE to talk (push-to-talk)
python translator.py --target es         # hear everything in Spanish
python translator.py --input 2 --output 5  # pin specific devices
```

Speak (or point it at a TV / another person). After each pause you'll hear the
translation in your aid, and see the text in the terminal.

### Two capture modes

- **VAD (default):** translates automatically after each pause in speech.
  Great hands-free for TV or a chat across a table.
- **Push-to-talk (`--ptt`):** hold **SPACE** to capture, release to hear the
  translation; **ESC** quits. Best in noisy rooms where automatic detection
  over-triggers, or when you want to pick exactly what gets translated. Needs
  `pynput` (in requirements.txt). On macOS you may have to grant your terminal
  **Accessibility** permission for global key capture (System Settings →
  Privacy & Security → Accessibility).

## Tuning (top of `translator.py`)

| Setting | What it does |
|---|---|
| `TARGET_LANG` | Language you want to hear (`en`, `es`, `fr`, `de`, …). |
| `VAD_AGGRESSIVENESS` | 0–3; raise it in noisy rooms to reject background sound. |
| `SILENCE_TAIL_MS` | How long a pause ends an utterance. Lower = snappier, more fragments. |
| `MIN_UTTERANCE_MS` | Ignores blips shorter than this. |
| `TTS_VOICE` | `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`. |

## Notes & limits

- **Latency** is utterance-based: it waits for a pause, then does STT→translate
  →TTS (~1–3 s round trip). Great for conversations and TV; not word-by-word
  simultaneous interpretation.
- **Self-hearing:** capture is paused while the TTS plays, so it won't translate
  its own voice. If you use loudspeakers instead of the aid, echo can still leak
  in — the aid (a closed earpiece) avoids that.
- **Cost:** roughly a few cents per active minute (Whisper + TTS).

## Going fully offline / private

Swap the two methods in the `Translator` class for local models — no API key,
works on a plane:

- **STT:** [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)
- **Translate:** [Argos Translate](https://github.com/argosopentech/argos-translate)
- **TTS:** [Piper](https://github.com/rhasspy/piper)

The VAD/capture loop stays identical.
