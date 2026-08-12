"""Call stats/listing, CSV export, and secrets check."""
import csv
import io
import time

from fastapi import APIRouter, Response

from ..config import (
    LLM_MODEL, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER,
    VLLM_BASE_URL, WHISPER_MODEL_SIZE,
)
from ..state import state

router = APIRouter()

# ════════════════════════════════════════════════════════════════════════
# DATA APIs  (ported: calls list, CSV export, secrets check)
# ════════════════════════════════════════════════════════════════════════

_SUFFIX_EXCLUDE = (":audio", ":review", ":transcript", ":meta", ":live")


def _call_record_keys(keys):
    """Filter Redis keys to actual call record hashes (exclude sub-keys)."""
    return [k for k in keys if not k.endswith(_SUFFIX_EXCLUDE)]


@router.get("/api/stats")
async def stats():
    """True aggregate counts (not capped to the recent-20 list)."""
    keys = _call_record_keys(await state["redis"].keys("call:*"))
    total = len(keys)
    completed = 0
    total_dur = 0
    for key in keys:
        d = await state["redis"].hgetall(key)
        if d.get("status") == "completed":
            completed += 1
        total_dur += int(d.get("duration_ms") or 0)
    projects = len(await state["redis"].keys("project:*:rows"))
    return {
        "total_calls": total,
        "completed": completed,
        "projects": projects,
        "total_duration_ms": total_dur,
    }


@router.get("/api/calls")
async def list_calls(page: int | None = None, limit: int = 15):
    r = state["redis"]
    keys = _call_record_keys(await r.keys("call:*"))
    calls = []
    for key in keys:
        data = await r.hgetall(key)
        if data:
            calls.append({"callSid": key.replace("call:", ""), **data})
    calls.sort(key=lambda c: float(c.get("started_at", 0)), reverse=True)
    if page is None:
        return calls[:20]  # backward-compat for legacy /dashboard
    limit = max(1, min(limit, 100))
    page = max(1, page)
    start = (page - 1) * limit
    total = len(calls)
    return {
        "calls": calls[start:start + limit],
        "total": total,
        "page": page,
        "limit": limit,
        "pages": max(1, -(-total // limit)),
    }


@router.get("/export.csv")
async def export_csv():
    r = state["redis"]
    keys = _call_record_keys(await r.keys("call:*"))
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["call_sid", "timestamp", "payer", "claim_id", "status",
                     "amount", "next_action", "duration_sec"])
    for key in keys:
        d = await r.hgetall(key)
        if d.get("status") in ("completed", "failed"):
            ts = float(d.get("ended_at") or d.get("started_at") or 0)
            writer.writerow([
                key.replace("call:", ""),
                time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)),
                d.get("payer", ""), d.get("claim_id", ""), d.get("status", ""),
                d.get("amount", ""), d.get("next_action", ""),
                round(float(d.get("duration_ms", 0)) / 1000),
            ])
    return Response(
        content=out.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition":
                 f'attachment; filename="ar-calls-{time.strftime("%Y-%m-%d")}.csv"'},
    )


@router.get("/api/check-secrets")
async def check_secrets():
    return {
        "twilio_sid": bool(TWILIO_ACCOUNT_SID),
        "twilio_token": bool(TWILIO_AUTH_TOKEN),
        "twilio_from": bool(TWILIO_FROM_NUMBER),
        "vllm_url": VLLM_BASE_URL,
        "whisper_model": WHISPER_MODEL_SIZE,
        "llm_model": LLM_MODEL,
    }
