"""Prompt builders, marker parsing, and [CALL_RESULT] JSON parser.
Ported from src/do.ts — extended with state machine, JSON result, and knowledge base injection.
"""
import json
import re
from pathlib import Path

# ── Browser Test Prompt (no IVR/DTMF markers) ────────────────────────────
BROWSER_PROMPT = """You are a helpful voice assistant responding to short voice queries.
Respond concisely in 1-2 sentences. Keep it natural and brief.
Never end the conversation. Always ask if there is anything else you can help with.
Only say goodbye when the caller explicitly says goodbye."""

# ── Telephony AR Prompt (with IVR/DTMF markers) ─────────────────────────
BASE_PROMPT = """You are an AR specialist calling an insurance company's claims department.

CRITICAL: You are the CALLER, not the recipient. You called THEM. You speak first with the claim details.

Your job:
1. State the claim details immediately (patient name, DOS, amount, claim ID)
2. Ask the payer representative for the current status of this specific claim
3. If denied, ask for the denial reason code
4. If paid, confirm the amount
5. Gather all needed information

Rules:
1. NEVER say "How can I help you?" or "feel free to ask" — you are the caller, not the help desk.
2. NEVER ask for the claim number — you already have it from your records.
3. Respond concisely in 1-2 sentences.
4. When you have all needed information, output [CALL_RESULT] followed by a JSON object."""


# ── Call Flow States ────────────────────────────────────────────────────
STATES = [
    "GREETING",
    "IVR_NAV",
    "CLAIM_VERIFY",
    "STATUS_GATHER",
    "DENIAL_HANDLE",
    "APPROVED_HANDLE",
    "CLOSE",
]

STATE_GOALS = {
    "GREETING": "Introduce yourself and state the purpose of the call. Speak naturally.",
    "IVR_NAV": "Navigate the phone menu. Listen for options and press the correct DTMF.",
    "CLAIM_VERIFY": "Verify the claim with the payer representative. State the claim ID, patient name, DOS, and billed amount.",
    "STATUS_GATHER": "Ask if the claim was paid, denied, or is still pending. Get the specific details.",
    "DENIAL_HANDLE": "The claim was denied. Ask for the denial reason code and appeal process.",
    "APPROVED_HANDLE": "The claim was paid. Confirm the amount paid and expected payment date.",
    "CLOSE": "Summarize the outcome and emit [CALL_RESULT] JSON.",
}

# ── Knowledge Base Loader ───────────────────────────────────────────────
PAYERS_DIR = Path(__file__).parent / "payers"

_payer_cache: dict[str, dict] = {}
_denial_codes: dict[str, str] = {}


def load_denial_codes() -> dict[str, str]:
    global _denial_codes
    if _denial_codes:
        return _denial_codes
    try:
        with open(PAYERS_DIR / "denial_codes.json") as f:
            data = json.load(f)
            _denial_codes = data.get("codes", {})
    except (FileNotFoundError, json.JSONDecodeError):
        _denial_codes = {}
    return _denial_codes


def load_payer(payer_name: str) -> dict | None:
    if not payer_name:
        return None
    if payer_name in _payer_cache:
        return _payer_cache[payer_name]
    slug = payer_name.lower().replace(" ", "_").replace("-", "_")
    path = PAYERS_DIR / f"{slug}.json"
    if not path.exists():
        for fp in PAYERS_DIR.glob("*.json"):
            if fp.stem.lower() == slug:
                path = fp
                break
    if not path.exists() or path.name == "denial_codes.json":
        return None
    try:
        with open(path) as f:
            data = json.load(f)
            _payer_cache[payer_name] = data
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ── Layered Prompt Builder ──────────────────────────────────────────────
def build_call_prompt(state: str, payer_knowledge: dict | None,
                      denial_code_subset: list[str] | None,
                      account: dict | None) -> str:
    base = BASE_PROMPT

    # State-specific rules
    extra_rules = []
    if state in ("IVR_NAV",):
        extra_rules.append("When you hear a menu option, output ONLY: [DTMF:digit]")
    if state in ("IVR_NAV", "STATUS_GATHER", "DENIAL_HANDLE"):
        extra_rules.append("When on hold, output ONLY: [WAITING]")

    if extra_rules:
        base += "\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(extra_rules, 1))

    parts = [base]

    if account:
        parts.append(
            f"[CLAIM CONTEXT]\n"
            f"Patient: {account.get('Patient Name', 'unknown')}\n"
            f"DOS: {account.get('DOS', 'unknown')}\n"
            f"CPT: {account.get('CPT', 'unknown')}\n"
            f"Billed: ${account.get('Billed Amount', '0')}\n"
            f"Payer: {account.get('Responsible Payer', 'unknown')}\n"
            f"Account: {account.get('Account Number', 'unknown')}\n"
            f"Objective: {account.get('AR Final Comments', 'Check claim status')}"
        )

    if payer_knowledge and payer_knowledge.get("ivr_tree"):
        tree = payer_knowledge["ivr_tree"]
        parts.append("[PAYER IVR]")
        for key, nodes in tree.items():
            for node in nodes:
                verify = node.get("verify_phrase", "")
                dtmf_info = (
                    f"DTMF={node['dtmf']}"
                    if node.get("dtmf_mode") == "numpad"
                    else f"DTMF={node.get('dtmf', '?')}"
                )
                parts.append(f"  {node['prompt_phrase']} → {dtmf_info}, verify='{verify}'")

    if denial_code_subset:
        codes = load_denial_codes()
        parts.append("[DENIAL CODES]")
        for code in denial_code_subset:
            desc = codes.get(code, "")
            if desc:
                parts.append(f"  {code}: {desc}")

    return "\n\n".join(parts)


def build_greeting(account: dict | None) -> str:
    if not account:
        return "Hello, I am calling to check the status of a medical claim."
    return (
        f"Hello, this is an AR specialist calling regarding claim "
        f"for patient {account.get('Patient Name', 'unknown')}, "
        f"Date of Service {account.get('DOS', 'unknown')}, "
        f"billed amount ${account.get('Billed Amount', 'unknown')}, "
        f"with payer reference {account.get('Account Number', 'unknown')}. "
        f"I need to check the status of this claim."
    )


# ── Marker Parsing (ported from do.ts) ─────────────────────────────────
DTMF_RE = re.compile(r"(?:\[?DTMF\s*[:=]\s*(\d+)\]?)", re.I)
WAITING_RE = re.compile(r"\[?WAITING\]?", re.I)


def parse_markers(bot_text: str) -> dict:
    dtmf = None
    m = DTMF_RE.search(bot_text)
    if m:
        dtmf = m.group(1)

    waiting = bool(WAITING_RE.search(bot_text))
    spoken = strip_markers(bot_text)

    return {"dtmf": dtmf, "waiting": waiting, "spoken": spoken, "call_result": None}


def strip_markers(text: str) -> str:
    text = DTMF_RE.sub("", text)
    text = WAITING_RE.sub("", text)
    text = text.replace("[", "").replace("]", "")
    return text.strip()


# ── [CALL_RESULT] JSON Parser ───────────────────────────────────────────
CALL_RESULT_RE = re.compile(r'\[CALL_RESULT\]\s*(\{.*?\})(?:\s*\[|\s*$)', re.DOTALL)

CALL_RESULT_SCHEMA = {
    "status": str,
    "payer": str,
    "claim_id": str,
    "next_action": str,
}

OPTIONAL_FIELDS = {
    "paid_amount": float,
    "billed_amount": float,
    "denial_code": str,
    "denial_description": str,
    "appeal_deadline": str,
    "call_summary": str,
    "call_duration_sec": int,
    "satisfaction": str,
}


def parse_call_result(text: str) -> dict | None:
    """Extract and validate [CALL_RESULT] JSON. Returns None on failure."""
    match = CALL_RESULT_RE.search(text)
    if not match:
        return None
    raw = match.group(1)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return None
    for field, field_type in CALL_RESULT_SCHEMA.items():
        if field not in result:
            return None
        if not isinstance(result[field], field_type):
            return None
    return result


def attempt_repair(text: str, llm_client, model: str) -> dict | None:
    """Ask the LLM to fix malformed [CALL_RESULT]. Up to 2 retries."""
    for attempt in range(2):
        repair_prompt = (
            f"The following call result is not valid JSON. "
            f"Please output ONLY the corrected JSON with these required fields: "
            f"status, payer, claim_id, next_action. "
            f"Optional: paid_amount, billed_amount, denial_code, denial_description, appeal_deadline, call_summary.\n\n"
            f"Current output:\n{text}"
        )
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            resp = loop.run_until_complete(
                llm_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": repair_prompt}],
                    max_tokens=300,
                    temperature=0,
                )
            )
            loop.close()
            repaired = resp.choices[0].message.content.strip()
            result = parse_call_result(repaired)
            if result:
                return result
            text = repaired
        except Exception:
            pass
    return None
