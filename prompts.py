"""System prompts and marker parsing — ported from src/do.ts."""
import re

BASE_PROMPT = """You are a helpful voice assistant responding to short voice queries.
Respond concisely in 1-2 sentences. Keep it natural and brief.
Never end the conversation. Always ask if there is anything else you can help with.
Only say goodbye when the caller explicitly says goodbye."""


def build_system_prompt(account: dict | None) -> str:
    """Ported from do.ts — injects patient/claim context when available."""
    if not account:
        return BASE_PROMPT
    return f"""You are an AR specialist calling insurance payers to check a medical claim.
Context:
- Patient: {account.get('Patient Name', 'unknown')}
- DOS: {account.get('DOS', 'unknown')}
- Amount: ${account.get('Billed Amount', '0')}
- Payer: {account.get('Responsible Payer', 'unknown')}
- Account: {account.get('Account Number', 'unknown')}
- Objective: {account.get('AR Final Comments', 'Check claim status')}

Rules:
1. Menu detected → output: [DTMF:1]
2. Hold music → output: [WAITING]
3. Call resolved → output: [END:{account.get('Responsible Payer', 'unknown')}:{account.get('Account Number', 'unknown')}:{account.get('Billed Amount', '0')}:Completed]
4. Normal conversation → respond in under 15 words.
5. Never describe actions. Just speak naturally and use markers above when needed."""


def build_greeting(account: dict | None) -> str:
    """First thing the agent says when the call connects."""
    if not account:
        return "Hello, thank you. I am calling to check the status of a medical claim."
    return (
        f"Hello, thank you. I am calling to check the status of a claim for patient "
        f"{account.get('Patient Name', 'unknown')}, Date of Service {account.get('DOS', 'unknown')}, "
        f"with billed amount ${account.get('Billed Amount', 'unknown')}."
    )


# ── Marker regexes (ported from do.ts) ──────────────────────────────────
DTMF_RE = re.compile(r"(?:\[?DTMF\s*[:=]\s*(\d+)\]?)", re.I)
END_RE = re.compile(r"\[?END\s*[:=]\s*([^\]\n]+)\]?", re.I)
WAITING_RE = re.compile(r"\[?WAITING\]?", re.I)


def parse_markers(bot_text: str) -> dict:
    """Extract DTMF digit, end-call payload, waiting flag from LLM output."""
    dtmf = None
    m = DTMF_RE.search(bot_text)
    if m:
        dtmf = m.group(1)

    end = None
    m = END_RE.search(bot_text)
    if m:
        parts = m.group(1).split(":")
        end = {
            "status": parts[0] if len(parts) > 0 else "completed",
            "payer": parts[1] if len(parts) > 1 else "unknown",
            "claim_id": parts[2] if len(parts) > 2 else "unknown",
            "amount": parts[3] if len(parts) > 3 else None,
            "next_action": parts[4] if len(parts) > 4 else "none",
        }

    waiting = bool(WAITING_RE.search(bot_text))
    spoken = strip_markers(bot_text)
    return {"dtmf": dtmf, "end": end, "waiting": waiting, "spoken": spoken}


def strip_markers(text: str) -> str:
    """Remove all control markers so TTS never speaks them."""
    text = DTMF_RE.sub("", text)
    text = END_RE.sub("", text)
    text = WAITING_RE.sub("", text)
    text = text.replace("[", "").replace("]", "")
    return text.strip()
