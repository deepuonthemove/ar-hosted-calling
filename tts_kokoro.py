"""Kokoro-specific TTS text preprocessing.

Pacing for spelled-out runs is handled at the audio layer (server.py splits
on SPELL_START/SPELL_END and synthesizes character-by-character with
injected silence), so the text transform itself is identical to Piper's.
"""
from tts_common import base_transforms


def tts_text(text: str) -> str:
    """Preprocess text for Kokoro TTS (dates, slow numbers, addresses, spelling)."""
    return base_transforms(text)
