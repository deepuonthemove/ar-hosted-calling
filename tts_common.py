"""Shared TTS text preprocessing (constants, regexes, helpers).

Text-only pacing tricks (extra periods/ellipses) turned out to be unreliable
across engines: Piper and Kokoro both compress short "sentences" like "S. A."
in ways that depend on the model and can't be tuned from text alone. So
instead of encoding pacing as punctuation, every place that needs to be
spelled out is wrapped in sentinel markers (SPELL_START/SPELL_END). The
audio layer (server.py) splits on those markers and synthesizes spelled
segments character-by-character at a slower rate with real injected silence
between characters — a guaranteed pause, independent of engine quirks.
"""
import re

from prompts import format_dos

# Sentinels marking a run of text that must be spoken character-by-character,
# slowly, with real silence between characters. SPELL_PAUSE marks a longer
# break at a natural grouping boundary (e.g. between hyphenated digit groups)
# without emitting a character. These are private-use control characters
# that never appear in normal model output, so they round-trip safely
# through prompts/LLM text as long as we only add them here, in tts_common.
SPELL_START = "\x02"
SPELL_END = "\x03"
SPELL_PAUSE = "\x04"

_NUM_ID_RE = re.compile(r"\d{5,}")
_DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?)?)\b")

# US state codes, spelled letter-by-letter (CA → "C A") so they are not read
# as words ("ca"). Only spelled when followed by a 5-digit zip (address
# context) so "OK"/"CO" elsewhere are untouched.
_STATE_CODES = set([
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
])

# Street suffixes expanded to full words — only within a matched street
# address (requires a leading number) so "Dr. Smith" is never "drive Smith".
# These are common, unambiguous words: SAY them plainly, never spell them.
_ADDR_SUFFIXES = {
    "st.": "street", "st": "street", "ave.": "avenue", "ave": "avenue",
    "rd.": "road", "rd": "road", "blvd.": "boulevard", "blvd": "boulevard",
    "dr.": "drive", "dr": "drive", "ln.": "lane", "ln": "lane",
    "ct.": "court", "ct": "court", "pkwy.": "parkway", "pkwy": "parkway",
    "hwy.": "highway", "hwy": "highway", "apt.": "apartment", "apt": "apartment",
    "ste.": "suite", "ste": "suite",
}

_PO_BOX_RE = re.compile(r"\bP\.?\s?O\.?\s+Box\b", re.I)

# The model sometimes spells by writing the literal word "hyphen" / "dash"
# ("T - A - hyphen - J"). Replace those with a plain hyphen so the spelling
# regex can treat the whole sequence as one slow spelling.
_WORD_HYPHEN_RE = re.compile(r"\s*[-–—]?\s*(?:hyphen|dash)\s*[-–—]?\s*", re.I)


def _word_hyphen(m):
    return "-"

# A US street address: "123 Main Street, Suite 500, Los Angeles, CA 90210".
# Split into components (number, name, suffix, suite, city, state, zip).
_ADDR_RE = re.compile(
    r"\b(?P<num>\d{1,6})\s+"
    r"(?P<name>[A-Za-z][A-Za-z.\-]*(?:\s+[A-Za-z][A-Za-z.\-]*)*?)\s+"
    r"(?P<suffix>street|avenue|road|boulevard|drive|lane|court|parkway|highway|apartment|suite|"
    r"st|ave|rd|blvd|dr|ln|ct|pkwy|hwy|apt|ste)\b"
    r"(?P<suite>(?:[,\s]*\s+(?:ste|suite|apt|#)\s*\d+))?"
    r"(?P<city>(?:[,\s]*\s+[A-Z][A-Za-z.\-]*(?:\s+[A-Z][A-Za-z.\-]*)*))?"
    r"(?P<state>(?:[,\s]*\s+[A-Z]{2}))?"
    r"(?P<zip>(?:[,\s]*\s+\d{5}))?",
    re.I,
)

# "CA 90210" or "CA, 90210" (state code next to a zip) → spell the state.
_STATE_ZIP_RE = re.compile(r"\b([A-Z]{2})\s*,?\s*(\d{5})\b")

# "city, CA" (state code after a comma, followed by a comma/zip/end) — only
# spelled when it is a known US state abbreviation.
_STATE_AFTER_COMMA_RE = re.compile(r",\s*([A-Z]{2})(?=\s*,|\s+\d{5}\b|\s*$)")


def _wrap(chars) -> str:
    """Wrap a sequence of single characters as a spelled-out segment."""
    return f"{SPELL_START}{' '.join(chars)}{SPELL_END}"


def _spell_state_comma(m):
    code = m.group(1)
    if code in _STATE_CODES:
        return f", {_wrap(code.upper())}"
    return m.group(0)


# Character-by-character spelling — single letters OR digits separated by
# spaces, dashes, commas, "dash" or "hyphen" ("P E N A", "2-0-5-3",
# "T-A-M-A-R-A", "a - hyphen - n"). Any 3+ separated characters are treated
# as a spelling so it is ALWAYS read slowly. NOTE: periods are NOT
# separators — they are sentence boundaries and must not be crossed.
_SPELL_SEP = r"(?:\s*(?:dash|hyphen)\s*|[\s,;\-]+)"
_CHAR_SPELL_RE = re.compile(r"\b(?:\d|[A-Za-z])(?:%s(?:\d|[A-Za-z])){2,}\b" % _SPELL_SEP)


def _slow_char_spell(m):
    return _wrap(re.findall(r"[A-Za-z0-9]", m.group(0).upper()))


# Ellipsis spelling ("B... A... S..."). The model sometimes emits this form
# instead of hyphens/spaces when asked to spell — same handling, just a
# different separator to strip.
_ELLIPSIS_SPELL_RE = re.compile(r"\b(?:\d|[A-Za-z])(?:\.{2,} *(?:\d|[A-Za-z])){2,}\b")


def _slow_ellipsis_spell(m):
    return _wrap(re.findall(r"[A-Za-z0-9]", m.group(0).upper()))


# Hyphen-separated digit groups ("205-39-3195", "555-123-4567") — read each
# group digit-by-digit with a real pause between groups.
_HYPHEN_GROUP_RE = re.compile(r"\b(\d{1,4}(?:-\d{1,4})+)\b")


def _group_hyphen_digits(m):
    groups = m.group(1).split("-")
    chars = []
    for i, g in enumerate(groups):
        if i:
            chars.append(SPELL_PAUSE)
        chars.extend(g)
    return _wrap(chars)


# Mixed alphanumeric IDs / claim numbers ("2026085EH4375", "A1B2C3D4") —
# neither plain words nor plain numbers, so read every character one-by-one.
_ALNUM_ID_RE = re.compile(
    r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{6,}\b"
)


def _slow_alnum_id(m):
    return _wrap(m.group(0).upper())


def _say_plain(raw):
    """A common, unambiguous address word (street suffix): just say it — never spelled."""
    low = raw.lower().strip(".,")
    return _ADDR_SUFFIXES.get(low, raw)


def _spell_digits(raw):
    """A number-only component (street number, zip, suite/apt number). Reading
    the digits one by one already IS the slow/clear form — no separate 'say'
    pass is needed on top of it."""
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None
    return _wrap(digits)


def _spell_letters(raw):
    """A 2-letter state code: there is no word to say first, just spell it."""
    return _wrap(raw.upper())


def _say_and_spell(raw):
    """An ambiguous proper noun (street/city name): say it fully once, then
    spell it letter-by-letter so it can't be misheard."""
    spelled = _wrap(re.sub(r"\s+", "", raw).upper())
    return f"{raw}. spelled, {spelled}"


def _say_suite(raw):
    low = raw.lower().strip(".,")
    m = re.match(r"^(ste|suite|apt)\.?\s+(\d+)$", low)
    if not m:
        return raw
    label = "suite" if m.group(1).startswith(("ste", "suite")) else "apartment"
    digits = _spell_digits(m.group(2))
    return f"{label}. {digits}" if digits else label


# How each address component should be voiced: digits are spelled digit-by-
# digit (that already is the clear/slow form); the suffix is a common word
# that's just said; only the street/city NAME is ambiguous enough to warrant
# saying it once and then spelling it letter-by-letter.
_ADDR_COMPONENT_KIND = {
    "num": "digits",
    "name": "sayspell",
    "suffix": "plain",
    "suite": "suite",
    "city": "sayspell",
    "state": "letters",
    "zip": "digits",
}


def _slow_address(m):
    out = []
    for key, kind in _ADDR_COMPONENT_KIND.items():
        raw = m.group(key)
        if not raw:
            continue
        raw = raw.strip(" ,").strip()
        if not raw:
            continue
        if kind == "digits":
            s = _spell_digits(raw)
        elif kind == "letters":
            s = _spell_letters(raw)
        elif kind == "plain":
            s = _say_plain(raw)
        elif kind == "suite":
            s = _say_suite(raw)
        else:
            s = _say_and_spell(raw)
        if s:
            out.append(s)
    return ". ".join(out)


def _group_digits(m):
    """Read a long digit run slowly: individual digits in chunks of 4, with a
    longer pause between chunks (claims/phone numbers)."""
    s = m.group(0)
    groups = [s[i:i + 4] for i in range(0, len(s), 4)]
    chars = []
    for i, g in enumerate(groups):
        if i:
            chars.append(SPELL_PAUSE)
        chars.extend(g)
    return _wrap(chars)


def base_transforms(text: str) -> str:
    """Apply the shared transforms common to every TTS engine.

    Spelled-out runs (IDs, addresses, letter-by-letter names) are wrapped in
    SPELL_START/SPELL_END sentinels rather than paced with punctuation — see
    split_spell_segments(), which the audio layer uses to synthesize those
    runs character-by-character with real injected silence.
    """
    text = _DATE_RE.sub(lambda m: format_dos(m.group(1)), text)
    text = _PO_BOX_RE.sub("P O Box", text)
    text = _WORD_HYPHEN_RE.sub(_word_hyphen, text)
    text = _ADDR_RE.sub(lambda m: _slow_address(m), text)

    # The address pass above already wrapped some digit/letter runs (e.g. the
    # zip, the state code) in SPELL_START/SPELL_END. Protect those spans with
    # placeholders so the generic matchers below can't re-match the raw
    # characters inside them and double-wrap.
    protected = []

    def _protect(m):
        protected.append(m.group(0))
        return f"\x05{chr(0x2500 + len(protected) - 1)}\x05"

    text = _SPELL_SEGMENT_RE.sub(_protect, text)

    text = _STATE_ZIP_RE.sub(lambda m: f"{_wrap(m.group(1)) if m.group(1) in _STATE_CODES else m.group(1)} {m.group(2)}", text)
    text = _STATE_AFTER_COMMA_RE.sub(_spell_state_comma, text)
    text = _CHAR_SPELL_RE.sub(_slow_char_spell, text)
    text = _ELLIPSIS_SPELL_RE.sub(_slow_ellipsis_spell, text)
    text = _ALNUM_ID_RE.sub(_slow_alnum_id, text)
    text = _HYPHEN_GROUP_RE.sub(_group_hyphen_digits, text)
    text = _NUM_ID_RE.sub(_group_digits, text)

    return re.sub(r"\x05(.)\x05", lambda m: protected[ord(m.group(1)) - 0x2500], text)


_SPELL_SEGMENT_RE = re.compile(f"{SPELL_START}(.*?){SPELL_END}", re.S)


def split_spell_segments(text: str):
    """Split TTS-ready text into ordered (is_spelled, content) parts.

    Plain parts: content is the text to synthesize normally.
    Spelled parts: content is a list of tokens — single characters to
    synthesize in isolation, or SPELL_PAUSE for an extra-long silent gap at a
    grouping boundary (no character to synthesize).
    """
    parts = []
    pos = 0
    for m in _SPELL_SEGMENT_RE.finditer(text):
        if m.start() > pos:
            plain = text[pos:m.start()]
            if plain.strip():
                parts.append((False, plain))
        tokens = [t for t in m.group(1).split(" ") if t]
        if tokens:
            parts.append((True, tokens))
        pos = m.end()
    if pos < len(text):
        plain = text[pos:]
        if plain.strip():
            parts.append((False, plain))
    return parts


def flatten_spell_segments(segments) -> str:
    """Best-effort text-only fallback for engines that can't be driven
    character-by-character (e.g. the Piper CLI subprocess path): turn each
    spelled segment into plain, period-separated text."""
    parts = []
    for is_spell, content in segments:
        if not is_spell:
            parts.append(content)
        else:
            letters = [t for t in content if t != SPELL_PAUSE]
            parts.append(". ".join(letters) + ". ")
    return "".join(parts)
