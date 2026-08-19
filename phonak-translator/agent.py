#!/usr/bin/env python3
"""
Phonak Live Translation Agent (LiveKit Agents + Groq)
======================================================

Real-time speech translation using open-source models hosted on Groq
(Whisper for STT, an open LLM for translation, Orpheus for TTS), orchestrated
by LiveKit Agents instead of translator.py's discrete per-utterance HTTP calls.

This is the same underlying idea as translator.py, but built on infrastructure
(LiveKit) that has official mobile client SDKs (iOS/Android/React Native) --
so this same agent process is what a future phone app would talk to.

STATUS: written but NOT verified to run. `livekit-agents` depends on the `av`
package, whose native binary was blocked by Windows Smart App Control on the
machine this was developed on ("An Application Control policy has blocked
this file" / DLL load failure). Getting this running needs either WSL2 (Linux
isn't subject to that Windows policy -- blocked here too, since WSL2 itself
needs a Windows feature + virtualization that this machine's setup wouldn't
enable even after two restarts, possibly an enterprise "App Control for
Business" policy) or a different, unrestricted machine. translator.py is the
proven, working fallback in the meantime.

Requires: GROQ_API_KEY in the environment.

Usage:
    python agent.py console                # local test: your terminal's mic/speaker
    python agent.py console --list-devices  # list audio devices
"""

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import groq

TARGET_LANG = "English"

INSTRUCTIONS = (
    "You are a live interpreter for someone wearing a hearing aid. Translate "
    f"everything the user says into {TARGET_LANG}. Reply with the translation "
    "alone: no notes, no quotes, no romanization, no explanation. Translate "
    "only the user's most recent message -- ignore your own past responses in "
    "this conversation as context, they are your own prior output, not "
    "something to translate."
)


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    session = AgentSession(
        stt=groq.STT(model="whisper-large-v3-turbo", detect_language=True),
        llm=groq.LLM(model="openai/gpt-oss-120b"),
        tts=groq.TTS(model="canopylabs/orpheus-v1-english", voice="troy"),
    )

    await session.start(
        agent=Agent(instructions=INSTRUCTIONS),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
