"""Kokoro-specific TTS text preprocessing.

Kokoro pauses LONGEST on ellipsis ("...") sequences and only slightly less on
single periods. It also voices every single letter reliably. So the model's
ellipsis spelling ("N... P... I...") is KEPT as-is, which Kokoro reads the
slowest.
"""
from tts_common import base_transforms


def tts_text(text: str) -> str:
    """Preprocess text for Kokoro TTS (dates, slow numbers, addresses, spelling).

    Kokoro pauses LONGEST on ellipsis, so spelled letters are separated with
    "..." (slower than the periods Piper uses). Ellipses from the model are
    also kept as-is, not normalized.
    """
    return base_transforms(text, spell_sep="... ")
