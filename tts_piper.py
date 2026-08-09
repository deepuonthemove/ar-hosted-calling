"""Piper-specific TTS text preprocessing.

Piper (espeak-ng) reads single periods with the longest pauses but runs
ellipsis ("...") sequences together very fast. So letter-spelling that the
model emitted with ellipses must be rebuilt with periods.
"""
from tts_common import base_transforms, _ELLIPSIS_SPELL_RE, _slow_ellipsis_spell


def tts_text(text: str) -> str:
    """Preprocess text for Piper TTS (dates, slow numbers, addresses, spelling)."""
    # Piper reads single periods with the longest pauses.
    text = base_transforms(text, spell_sep=". ")
    # Piper reads ellipses fast — convert "N... P... I..." to slow periods.
    text = _ELLIPSIS_SPELL_RE.sub(lambda m: _slow_ellipsis_spell(m, ". "), text)
    return text
