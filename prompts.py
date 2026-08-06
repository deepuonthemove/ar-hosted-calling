"""Prompt builders, marker parsing, and [CALL_RESULT] JSON parser.
Ported from src/do.ts — extended with state machine, JSON result, and knowledge base injection.
"""
import datetime
import json
import re
from pathlib import Path


def _ordinal_day(day: int) -> str:
    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def format_dos(dos) -> str:
    """Format a Date of Service value for natural speech, e.g. '4th February 2026'.

    Handles ISO datetimes, date-only strings, and Excel serial numbers.
    Falls back to the raw value if it cannot be parsed.
    """
    if dos is None:
        return "unknown"
    s = str(dos).strip()
    if not s:
        return "unknown"

    # Excel serial number (days since 1899-12-30)
    if s.replace("-", "").replace("+", "").isdigit() and len(s) <= 6:
        try:
            dt = datetime.datetime(1899, 12, 30) + datetime.timedelta(days=int(s))
            return f"{_ordinal_day(dt.day)} {dt.strftime('%B')} {dt.year}"
        except Exception:
            pass

    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M",
                "%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%b %d, %Y", "%B %d, %Y",
                "%d/%m/%Y", "%Y/%m/%d"):
        try:
            dt = datetime.datetime.strptime(s, fmt)
            return f"{_ordinal_day(dt.day)} {dt.strftime('%B')} {dt.year}"
        except ValueError:
            continue
    return s

# ── Browser Test Prompt (no IVR/DTMF markers) ────────────────────────────
BROWSER_PROMPT = """You are a helpful voice assistant responding to short voice queries.
Respond concisely in 1-2 sentences. Keep it natural and brief.
Never end the conversation. Always ask if there is anything else you can help with.
Only say goodbye when the caller explicitly says goodbye."""

# ── Telephony AR Prompt (with IVR/DTMF markers) ─────────────────────────
BASE_PROMPT = """You are an AR (Accounts Receivable) specialist calling an insurance company's claims department.

CRITICAL ROLE — YOU: You are the CALLER. You called the insurance company.
CRITICAL ROLE — THEM: The person you are speaking to is an insurance company claims representative. They handle claims.

You called THEM to get claim status information. You are NOT a support agent or help desk. Do not offer help, advice, or suggestions to the person you're speaking to.

Follow these rules strictly:
1. State claim details immediately (patient name, DOS, amount, claim ID)
2. Ask the representative for the current status — is it paid, denied, or pending
3. If denied: ask for the denial reason code and appeal process
4. If paid: confirm the amount and expected payment date
5. NEVER say "How can I help you?" or "feel free to ask" — you called them for help
6. NEVER ask for the claim number — you already have it from your records
7. NEVER tell the representative to "contact support", "check back later", "wait for updates", or offer to set reminders — you are the one handling this claim
8. NEVER offer advice or assistance to the representative
9. Respond concisely in 1-2 sentences. Be direct and professional.
10. Keep the conversation going — ask follow-up questions until you have the complete, confirmed claim picture. Track what you have already asked and what the representative has already told you: NEVER repeat a question you already asked or re-ask for information that was already provided. Each turn should ask only the next relevant question, not restate your role or the claim details. If the representative answers a question, acknowledge it briefly and move forward.
11. ONLY at the very end of the call, after you have confirmed the final status and all details with the representative, output [CALL_RESULT] followed by a JSON object with EXACTLY these keys:
{
  "status": "paid or denied or pending",
  "payer": "payer name",
  "claim_id": "claim number",
  "next_action": "the concrete next step",
  "denial_code": "e.g. CO-11",
  "denial_description": "short reason",
  "paid_amount": 0,
  "appeal_deadline": "date or null",
  "call_summary": "one line summary"
}
Use the exact key names shown above — especially "status" (not claim_status), "claim_id" (not claimId), "next_action" (not appeal_process). Never output [CALL_RESULT] after only one or two exchanges. Never fabricate the status, codes, amounts, or dates — if you don't have a confirmed answer, ask for it instead of ending the call."""


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
_denial_context: dict[str, dict] = {}


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


def load_denial_context() -> dict[str, dict]:
    global _denial_context
    if _denial_context:
        return _denial_context
    try:
        with open(PAYERS_DIR / "denial_context.json") as f:
            data = json.load(f)
            _denial_context = data.get("denial_context", {})
    except (FileNotFoundError, json.JSONDecodeError):
        _denial_context = {}
    return _denial_context


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
    if not path.exists() or path.name in ("denial_codes.json", "denial_context.json"):
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
        denial_info = ""
        known_denial = account.get("Denial Code") or account.get("denialCode")
        if known_denial:
            denial_info = f"\nKnown Denial Code: {known_denial}"

        notes = account.get("Notes", "")
        notes_info = f"\nPrior Call Notes: {notes}" if notes else ""

        parts.append(
            f"[CLAIM CONTEXT]\n"
            f"Patient: {account.get('Patient Name', account.get('patientName', 'unknown'))}\n"
            f"Date of Service: {format_dos(account.get('DOS', account.get('dos')))}\n"
            f"CPT: {account.get('CPT', account.get('cpt', 'unknown'))}\n"
            f"Billed: ${account.get('Billed Amount', account.get('billedAmount', '0'))}\n"
            f"Payer: {account.get('Responsible Payer', account.get('provider', 'unknown'))}\n"
            f"Account: {account.get('Account Number', account.get('accountNumber', 'unknown'))}\n"
            f"Objective: {account.get('AR Final Comments', 'Check claim status')}{denial_info}{notes_info}"
        )

        # Inject AR Learning Denial Rules if a denial code is known
        if known_denial:
            ctx_dict = load_denial_context()
            match_ctx = ctx_dict.get(known_denial.upper())
            if match_ctx:
                parts.append(
                    f"[DENIAL GUIDANCE FOR {known_denial}]\n"
                    f"Category: {match_ctx.get('category', 'Unknown')}\n"
                    f"Description: {match_ctx.get('description', '')}\n"
                    f"Voice Script Guideline: {match_ctx.get('script_guideline', '')}\n"
                    f"Recommended Action: {match_ctx.get('recommended_action', '')}"
                )

    if payer_knowledge and payer_knowledge.get("ivr_tree"):
        tree = payer_knowledge["ivr_tree"]
        parts.append("[PAYER IVR]")
        for key, nodes in tree.items():
            for node in nodes:
                verify = node.get("verify_phrase", "")
                dtmf_info = f"DTMF={node.get('dtmf', '?')}"
                parts.append(f"  {node['prompt_phrase']} → {dtmf_info}, verify='{verify}'")

    # Universal HIPAA 835 denial codes — always injected (all payers)
    codes = load_denial_codes()
    ctxs = load_denial_context()
    lines = []
    for code in sorted(codes):
        desc = codes[code]
        ctx = ctxs.get(code.upper(), {})
        line = f"  {code}: {desc}"
        if ctx.get("script_guideline"):
            line += f" | Script Advice: {ctx['script_guideline']}"
        lines.append(line)
    parts.append("[DENIAL CODES KNOWLEDGE BASE]")
    parts.append("\n".join(lines))

    # Payer-specific likely codes (hint only, not exhaustive)
    if denial_code_subset:
        present = [c for c in denial_code_subset if c in codes]
        if present:
            parts.append("[LIKELY DENIAL CODES FOR THIS PAYER]")
            parts.append("  " + ", ".join(present))

    return "\n\n".join(parts)


def build_greeting(account: dict | None) -> str:
    if not account:
        return "Hello, I am calling to check the status of a medical claim."
    return (
        f"Hello, this is an AR specialist calling regarding claim "
        f"for patient {account.get('Patient Name', 'unknown')}, "
        f"Date of Service {format_dos(account.get('DOS'))}, "
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
    # LLM may emit [CALL_RESULT], CALL_RESULT:, or CALL_RESULT — strip any and the JSON after it
    text = re.sub(r'\[?CALL_RESULT\]?\s*:?\s*\{.*', '', text, flags=re.DOTALL)
    text = text.replace("[", "").replace("]", "")
    return text.strip()


# ── [CALL_RESULT] JSON Parser ───────────────────────────────────────────
CALL_RESULT_RE = re.compile(r'\[?CALL_RESULT\]?\s*:?\s*(\{.*?\})(?:\s*\[|\s*$)', re.DOTALL)

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

    # Accept alternate key names the LLM tends to emit
    if "status" not in result and "claim_status" in result:
        result["status"] = result["claim_status"]
    if "claim_id" not in result and "claimId" in result:
        result["claim_id"] = result["claimId"]
    if "next_action" not in result and "appeal_process" in result:
        result["next_action"] = result["appeal_process"]

    # Only `status` is strictly required; payer/claim_id/next_action fall back
    # to the call record in _finalize when missing.
    if "status" not in result or not isinstance(result["status"], str):
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
