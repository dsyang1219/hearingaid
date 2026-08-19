#!/usr/bin/env python3
"""
Phonak Live Translation -- web backend
=======================================

FastAPI backend for the browser version of the translator. Push-to-talk: the
browser records a complete utterance and posts it here, so unlike
translator.py's desktop CLI there's no VAD, silence-tail tuning, or
speculative processing needed -- the browser already knows when you're done
talking. Stateless: the client holds the running conversation transcript and
sends recent turns back as `history` on each request, rather than the server
holding a per-session deque.

Requires: GROQ_API_KEY in the environment.
"""

import json
import os

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

BASE_URL = "https://api.groq.com/openai/v1"
STT_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "openai/gpt-oss-120b"
TTS_MODEL = "canopylabs/orpheus-v1-english"
DEFAULT_VOICE = "troy"
CONTEXT_TURNS = 4  # prior exchanges given to the translator

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

# See translator.py for why: Whisper reliably hallucinates these stock
# YouTube sign-offs when fed near-silent or noisy audio instead of
# transcribing nothing.
HALLUCINATION_PHRASES = frozenset({
    "thank you", "thank you very much", "thanks for watching",
    "thank you for watching", "thanks for watching!",
    "please subscribe", "subscribe to my channel",
    "bye", "bye bye", "goodbye", "see you next time",
})


def _is_hallucination(text: str) -> bool:
    normalized = text.strip().lower().strip(".!?")
    return normalized in HALLUCINATION_PHRASES


def _client() -> OpenAI:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(500, "Server is missing GROQ_API_KEY.")
    return OpenAI(api_key=api_key, base_url=BASE_URL)


app = FastAPI()


@app.post("/api/translate")
async def translate(
    audio: UploadFile,
    target: str = Form("English"),
    source: str = Form(""),
    history: str = Form("[]"),
):
    client = _client()
    data = await audio.read()
    if not data:
        raise HTTPException(400, "Empty audio.")

    kwargs = {
        "model": STT_MODEL,
        "file": (audio.filename or "utterance.webm", data, audio.content_type),
    }
    if source:
        kwargs["language"] = source
    try:
        heard = client.audio.transcriptions.create(**kwargs).text.strip()
    except Exception as exc:
        raise HTTPException(502, f"Speech-to-text failed: {exc}") from None

    if not heard or _is_hallucination(heard):
        return {"heard": None, "translated": None}

    try:
        past = json.loads(history)[-CONTEXT_TURNS:]
    except (json.JSONDecodeError, TypeError):
        past = []

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(target=target)}]
    for turn in past:
        messages.append({"role": "user", "content": turn.get("heard", "")})
        messages.append({"role": "assistant", "content": turn.get("translated", "")})
    messages.append({"role": "user", "content": heard})

    try:
        chat = client.chat.completions.create(
            model=LLM_MODEL, temperature=0, messages=messages,
        )
    except Exception as exc:
        raise HTTPException(502, f"Translation failed: {exc}") from None
    translated = (chat.choices[0].message.content or "").strip()

    return {"heard": heard, "translated": translated}


@app.post("/api/speak")
async def speak(text: str = Form(...), voice: str = Form(DEFAULT_VOICE)):
    if not text.strip():
        raise HTTPException(400, "Empty text.")
    client = _client()
    try:
        resp = client.audio.speech.create(
            model=TTS_MODEL, voice=voice, input=text, response_format="wav",
        )
    except Exception as exc:
        raise HTTPException(502, f"Text-to-speech failed: {exc}") from None
    return Response(content=resp.read(), media_type="audio/wav")


app.mount("/", StaticFiles(directory="static", html=True), name="static")
