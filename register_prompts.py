"""Register the AR voice-agent prompts into Opik's Prompt Library.

Run inside the voice-agent container:
    python register_prompts.py
"""
import os

import opik as opik_sdk
from prompts import BASE_PROMPT, build_call_prompt, load_payer, load_denial_codes

OPIK_BASE_URL = os.getenv("OPIK_BASE_URL", "http://172.18.0.1:5173")
PROJECT = "ar-voice-agent"

opik_sdk.configure(api_key="local", use_local=True, url_override=OPIK_BASE_URL,
                   project_name=PROJECT)
client = opik_sdk.Opik()

# Sample account context (used to render the full layered prompt)
account = {
    "Patient Name": "BLANCHER, KRISTI",
    "DOS": "2026-02-04T00:00:00",
    "CPT": "99214",
    "Billed Amount": "276",
    "Responsible Payer": "SUREST [500]",
    "Account Number": "69274",
    "AR Final Comments": "Call and get the current status of the claim.",
}

def _denial_kb():
    from prompts import load_denial_context
    codes = load_denial_codes()
    ctx = load_denial_context()
    lines = ["[DENIAL CODES KNOWLEDGE BASE]"]
    for code in sorted(codes):
        line = f"  {code}: {codes[code]}"
        if ctx.get(code.upper(), {}).get("script_guideline"):
            line += f" | Script Advice: {ctx[code.upper()]['script_guideline']}"
        lines.append(line)
    return "\n".join(lines)


PROMPTS = {
    "ar-base": (BASE_PROMPT, "Base AR specialist role prompt (all states)"),
    "ar-greeting": (build_call_prompt("GREETING", None, None, account), "GREETING state prompt with claim context"),
    "ar-ivr-nav": (build_call_prompt("IVR_NAV", None, None, account), "IVR_NAV state (DTMF/WAITING markers)"),
    "ar-claim-verify": (build_call_prompt("CLAIM_VERIFY", None, None, account), "CLAIM_VERIFY state"),
    "ar-status-gather": (build_call_prompt("STATUS_GATHER", None, None, account), "STATUS_GATHER state"),
    "ar-denial-handle": (build_call_prompt("DENIAL_HANDLE", None, None, account), "DENIAL_HANDLE state (denial codes KB)"),
    "ar-approved-handle": (build_call_prompt("APPROVED_HANDLE", None, None, account), "APPROVED_HANDLE state"),
    "ar-close": (build_call_prompt("CLOSE", None, None, account), "CLOSE state (CALL_RESULT emission)"),
    "ar-denial-codes-kb": (_denial_kb(), "Universal HIPAA 835 denial codes KB"),
}


def main():
    for name, (prompt, desc) in PROMPTS.items():
        try:
            client.create_prompt(name=name, prompt=prompt,
                                 description=desc, tags=["ar-agent"],
                                 project_name=PROJECT)
            print(f"registered: {name}")
        except Exception as e:
            print(f"FAILED {name}: {e}")
    client.flush()
    print("done")


if __name__ == "__main__":
    main()
