"""Shared TTS text preprocessing (constants, regexes, helpers).

The per-engine files tts_piper.py / tts_kokoro.py compose these shared
transforms with their own punctuation handling, because Piper and Kokoro
honor punctuation pauses differently.
"""
import re

from prompts import format_dos

_NUM_ID_RE = re.compile(r"\d{5,}")
_DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?)?)\b")

# US state codes spelled letter-by-letter (CA → "C A") so they are not read
# as words ("ca"). Only spelled when followed by a 5-digit zip (address
# context) so "OK"/"CO" elsewhere are untouched.
_STATE_CODES = {
    "AL": "A L", "AK": "A K", "AZ": "A Z", "AR": "A R", "CA": "C A",
    "CO": "C O", "CT": "C T", "DE": "D E", "FL": "F L", "GA": "G A",
    "HI": "H I", "ID": "I D", "IL": "I L", "IN": "I N", "IA": "I A",
    "KS": "K S", "KY": "K Y", "LA": "L A", "ME": "M E", "MD": "M D",
    "MA": "M A", "MI": "M I", "MN": "M N", "MS": "M S", "MO": "M O",
    "MT": "M T", "NE": "N E", "NV": "N V", "NH": "N H", "NJ": "N J",
    "NM": "N M", "NY": "N Y", "NC": "N C", "ND": "N D", "OH": "O H",
    "OK": "O K", "OR": "O R", "PA": "P A", "RI": "R I", "SC": "S C",
    "SD": "S D", "TN": "T N", "TX": "T X", "UT": "U T", "VT": "V T",
    "VA": "V A", "WA": "W A", "WV": "W V", "WI": "W I", "WY": "W Y",
    "DC": "D C",
}

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


def _spell_state_comma(m):
    code = m.group(1)
    if code in _STATE_CODES:
        return f", {_STATE_CODES[code]}"
    return m.group(0)


# Character-by-character spelling — single letters OR digits separated by
# spaces, dashes, commas, "dash" or "hyphen" ("P E N A", "2-0-5-3",
# "T-A-M-A-R-A", "a - hyphen - n"). Any 3+ separated characters are treated
# as a spelling so it is ALWAYS read slowly. NOTE: periods are NOT
# separators — they are sentence boundaries and must not be crossed.
_SPELL_SEP = r"(?:\s*(?:dash|hyphen)\s*|[\s,;\-]+)"
_CHAR_SPELL_RE = re.compile(r"\b(?:\d|[A-Za-z])(?:%s(?:\d|[A-Za-z])){2,}\b" % _SPELL_SEP)


def _slow_char_spell(m, sep):
    return sep.join(re.findall(r"[A-Za-z0-9]", m.group(0))).upper()


# Ellipsis spelling ("B... A... S...", "2.... 0...."). Piper reads ellipses
# as tiny pauses (fast) so it must be rebuilt; Kokoro pauses LONGEST on
# ellipses so it is left alone. Uses 2+ dots so it never touches the
# single-period separators we emit for addresses.
_ELLIPSIS_SPELL_RE = re.compile(r"\b(?:\d|[A-Za-z])(?:\.{2,} *(?:\d|[A-Za-z])){2,}\b")


def _slow_ellipsis_spell(m, sep):
    return sep.join(re.findall(r"[A-Za-z0-9]", m.group(0))).upper()


# Hyphen-separated digit groups ("205-39-3195", "555-123-4567") — read each
# group digit-by-digit with a real pause at the hyphen instead of "dash".
_HYPHEN_GROUP_RE = re.compile(r"\b(\d{1,4}(?:-\d{1,4})+)\b")


def _group_hyphen_digits(m, sep):
    return ", ".join(sep.join(g) for g in m.group(1).split("-"))


# Mixed alphanumeric IDs / claim numbers ("2026085EH4375", "A1B2C3D4") —
# neither plain words nor plain numbers, so read every character one-by-one.
_ALNUM_ID_RE = re.compile(
    r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{6,}\b"
)


def _slow_alnum_id(m, sep):
    return sep.join(m.group(0))


def _say_plain(raw):
    """A common, unambiguous address word (street suffix): just say it — never spelled."""
    low = raw.lower().strip(".,")
    return _ADDR_SUFFIXES.get(low, raw)


def _spell_digits(raw, sep):
    """A number-only component (street number, zip, suite/apt number). Reading
    the digits one by one already IS the slow/clear form — no separate 'say'
    pass is needed on top of it."""
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None
    return sep.join(digits)


def _spell_letters(raw, sep):
    """A 2-letter state code: there is no word to say first, just spell it."""
    return sep.join(raw.upper())


def _say_and_spell(raw, sep):
    """An ambiguous proper noun (street/city name): say it fully once, then
    spell it letter-by-letter so it can't be misheard."""
    spelled = sep.join(re.sub(r"\s+", "", raw).upper())
    return f"{raw}. spelled, {spelled}"


def _say_suite(raw, sep):
    low = raw.lower().strip(".,")
    m = re.match(r"^(ste|suite|apt)\.?\s+(\d+)$", low)
    if not m:
        return raw
    label = "suite" if m.group(1).startswith(("ste", "suite")) else "apartment"
    digits = _spell_digits(m.group(2), sep)
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


def _slow_address(m, sep="."):
    out = []
    for key, kind in _ADDR_COMPONENT_KIND.items():
        raw = m.group(key)
        if not raw:
            continue
        raw = raw.strip(" ,").strip()
        if not raw:
            continue
        if kind == "digits":
            s = _spell_digits(raw, sep)
        elif kind == "letters":
            s = _spell_letters(raw, sep)
        elif kind == "plain":
            s = _say_plain(raw)
        elif kind == "suite":
            s = _say_suite(raw, sep)
        else:
            s = _say_and_spell(raw, sep)
        if s:
            out.append(s)
    return ". ".join(out)


def _group_digits(m, sep):
    """Read a long digit run slowly: individual digits in chunks of 4,
    comma-separated so TTS pauses between chunks (claims/phone numbers)."""
    s = m.group(0)
    groups = [s[i:i + 4] for i in range(0, len(s), 4)]
    return ", ".join(sep.join(g) for g in groups)


def base_transforms(text: str, spell_sep: str = ". ") -> str:
    """Apply the shared transforms common to every TTS engine.

    `spell_sep` is the character separator used when spelling letters/digits
    one by one. It always includes a trailing space so each spelled
    character reads as its own short sentence — without the space, engines
    tend to parse a run like "S.A.N" as a single abbreviation and rush it.
    Piper reads periods slowest (". "); Kokoro pauses longest on "..." so
    tts_kokoro passes spell_sep="... ".
    """
    text = _DATE_RE.sub(lambda m: format_dos(m.group(1)), text)
    text = _PO_BOX_RE.sub("P O Box", text)
    text = _WORD_HYPHEN_RE.sub(_word_hyphen, text)
    text = _ADDR_RE.sub(lambda m: _slow_address(m, spell_sep), text)
    text = _STATE_ZIP_RE.sub(lambda m: f"{_STATE_CODES.get(m.group(1), m.group(1))} {m.group(2)}", text)
    text = _STATE_AFTER_COMMA_RE.sub(_spell_state_comma, text)
    text = _CHAR_SPELL_RE.sub(lambda m: _slow_char_spell(m, spell_sep), text)
    text = _ALNUM_ID_RE.sub(lambda m: _slow_alnum_id(m, spell_sep), text)
    text = _HYPHEN_GROUP_RE.sub(lambda m: _group_hyphen_digits(m, spell_sep), text)
    return _NUM_ID_RE.sub(lambda m: _group_digits(m, spell_sep), text)
