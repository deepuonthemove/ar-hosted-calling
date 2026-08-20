"""Cloned-voice reference clips for Chatterbox TTS.

Each clip is a WAV file under CHATTERBOX_VOICES_DIR; the filename (minus
extension) is the voice's selectable name. Kept as its own module so both
routes/misc.py (upload/list/delete over the API) and tts.py (resolving the
active voice name to a file path at synthesis time) can import it without
a circular dependency.
"""
import os
import re

from .config import CHATTERBOX_VOICES_DIR

_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def sanitize_voice_name(name: str) -> str:
    name = _NAME_RE.sub("-", name.strip()).strip("-")
    return name[:64]


def voice_path(name: str) -> str:
    return os.path.join(CHATTERBOX_VOICES_DIR, f"{sanitize_voice_name(name)}.wav")


def list_voices() -> list[str]:
    if not os.path.isdir(CHATTERBOX_VOICES_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(CHATTERBOX_VOICES_DIR) if f.endswith(".wav")
    )


def voice_exists(name: str) -> bool:
    return bool(name) and os.path.isfile(voice_path(name))


def delete_voice(name: str) -> bool:
    path = voice_path(name)
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False
