"""Telephony endpoints (ported from src/index.ts /voice + /media + /make-call)."""
import json
import time
import uuid
from urllib.parse import urlencode

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import JSONResponse, Response

from call_session import CallSession

from ..app_config import get_config as _get_config
from ..config import (
    PUBLIC_DOMAIN, PUBLIC_SCHEME, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
    TWILIO_FROM_NUMBER, WHISPER_MODEL_SIZE,
)
from ..state import state
from ..tts import get_tts_fn
from .accounts import _resolve_account

router = APIRouter()


def _voice_webhook(payer: str, claim_id: str, account_uid: str, call_sid: str) -> str:
    """Build the Twilio voice webhook URL with properly URL-encoded params."""
    params = urlencode({
        "payer": payer, "claim_id": claim_id,
        "account_uid": account_uid, "local_sid": call_sid,
    })
    return f"{PUBLIC_SCHEME}://{PUBLIC_DOMAIN}/voice?{params}"


@router.post("/make-call")
async def make_call(request: Request):
    """Trigger outbound call via Twilio REST API (ported)."""
    data = await request.json()
    phone = data.get("phone")
    if not phone:
        return JSONResponse({"error": "Phone number is required"}, 400)
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER]):
        return JSONResponse({"error": "Twilio credentials not configured"}, 400)

    # Resolve account context from project_id + row_num if provided
    account_uid = data.get("account_uid", "")
    if data.get("project_id") and data.get("row_num"):
        _uid, acct = await _resolve_account(data["project_id"], int(data["row_num"]))
        if not acct:
            return JSONResponse({"error": f"No account for project {data['project_id']} row {data['row_num']}"}, 404)
        account_uid = _uid or ""
        data.setdefault("payer", acct.get("Responsible Payer", "unknown"))
        data.setdefault("claim_id", acct.get("Claim ID") or acct.get("Account Number") or "unknown")

    payer = data.get("payer", "unknown")
    claim_id = data.get("claim_id", "unknown")
    call_id = f"call-{uuid.uuid4().hex[:12]}"

    # Pre-create the call record keyed by a stable call_id (ported from /voice handler)
    await state["redis"].hset(f"call:{call_id}", mapping={
        "claim_id": claim_id, "payer": payer, "account_uid": account_uid,
        "phone": phone, "status": "dialing", "started_at": str(time.time()),
    })

    from twilio.rest import Client as TwilioClient
    from twilio.base.exceptions import TwilioRestException
    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        webhook = _voice_webhook(payer, claim_id, account_uid, call_id)
        call = client.calls.create(to=phone, from_=TWILIO_FROM_NUMBER, url=webhook)
        # Keep the stable call_id as the record key; store the Twilio SID as a field
        await state["redis"].hset(f"call:{call_id}", mapping={"twilio_sid": call.sid})
        return {"ok": True, "call_id": call_id, "callSid": call.sid}
    except TwilioRestException as e:
        return JSONResponse({"error": f"Twilio API error: {e.msg}"}, 500)


@router.post("/make-call-enriched")
async def make_call_enriched(request: Request):
    """Trigger outbound call after storing enriched claim data into Redis."""
    data = await request.json()
    phone = data.get("phone")
    claim_id = data.get("claim_id") or data.get("claimId")
    if not phone or not claim_id:
        return JSONResponse({"error": "Phone number and claim_id are required"}, 400)

    account_uid = data.get("account_uid") or data.get("accountUid") or f"claim-{claim_id}"
    payer = data.get("payer") or data.get("provider") or "unknown"
    enriched_data = data.get("enrichedData") or {}

    # Map enriched fields into account dictionary
    record = {
        "UID": account_uid,
        "Claim ID": claim_id,
        "Responsible Payer": payer,
        "Patient Name": enriched_data.get("patientName") or data.get("patientName") or "Unknown Patient",
        "DOS": enriched_data.get("dos") or data.get("dos") or "",
        "Billed Amount": str(enriched_data.get("billedAmount") or data.get("billedAmount") or "0.00"),
        "CPT": enriched_data.get("cpt") or data.get("cpt") or "",
        "Account Number": enriched_data.get("accountNumber") or data.get("accountNumber") or "",
        "Denial Code": enriched_data.get("denialCode") or data.get("denialCode") or "",
        "Call Status": "Pending",
    }

    # Store into Redis hash so CallSession can pick it up
    r = state["redis"]
    await r.hset(f"account:{account_uid}", mapping=record)

    # Call standard make_call logic
    call_id = f"call-{uuid.uuid4().hex[:12]}"
    await r.hset(f"call:{call_id}", mapping={
        "claim_id": claim_id, "payer": payer, "account_uid": account_uid,
        "phone": phone, "status": "dialing", "started_at": str(time.time()),
    })

    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER]):
        return JSONResponse({"ok": True, "call_id": call_id, "callSid": call_id, "warning": "Twilio not configured; mock mode active"})

    from twilio.rest import Client as TwilioClient
    from twilio.base.exceptions import TwilioRestException
    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        webhook = _voice_webhook(payer, claim_id, account_uid, call_id)
        call = client.calls.create(to=phone, from_=TWILIO_FROM_NUMBER, url=webhook)
        await r.hset(f"call:{call_id}", mapping={"twilio_sid": call.sid})
        return {"ok": True, "call_id": call_id, "callSid": call.sid}
    except TwilioRestException as e:
        return JSONResponse({"error": f"Twilio API error: {e.msg}"}, 500)


@router.get("/api/enriched-claim/{claim_id}")
async def get_enriched_claim(claim_id: str):
    """Retrieve enriched claim data stored in Redis."""
    r = state["redis"]
    account_keys = await r.keys("account:*")
    for key in account_keys:
        data = await r.hgetall(key)
        if data.get("Claim ID") == claim_id or data.get("claimId") == claim_id:
            return {"claimId": claim_id, "data": data}
    return JSONResponse({"error": "Claim not found"}, 404)


@router.api_route("/voice", methods=["GET", "POST"])
async def voice_webhook(request: Request):
    """Twilio webhook → TwiML with Media Stream (ported)."""
    q = request.query_params
    form = await request.form() if request.method == "POST" else {}
    # Prefer local_sid (our call_id) so the media stream + live transcript key on
    # the id returned by /make-call — the one the UI polls.
    call_sid = q.get("local_sid") or q.get("CallSid") or (form.get("CallSid") if form else None) or "unknown"

    # Ensure call record exists with context
    existing = await state["redis"].hgetall(f"call:{call_sid}")
    if not existing:
        await state["redis"].hset(f"call:{call_sid}", mapping={
            "claim_id": q.get("claim_id", "unknown"),
            "payer": q.get("payer", "unknown"),
            "account_uid": q.get("account_uid", ""),
            "status": "dialing", "started_at": str(time.time()),
        })

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://{PUBLIC_DOMAIN}/media/{call_sid}">
      <Parameter name="payer" value="{q.get('payer', 'unknown')}"/>
      <Parameter name="claim_id" value="{q.get('claim_id', 'unknown')}"/>
      <Parameter name="account_uid" value="{q.get('account_uid', '')}"/>
    </Stream>
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="text/xml")


@router.websocket("/media/{call_sid}")
async def media_stream(ws: WebSocket, call_sid: str):
    """Twilio Media Streams WebSocket → CallSession (ported from DO)."""
    _cfg = await _get_config()
    engine = _cfg["tts_engine"]
    call_record = await state["redis"].hgetall(f"call:{call_sid}")
    if call_record and call_record.get("tts_engine"):
        engine = call_record["tts_engine"]
    session = CallSession(ws, call_sid, {
        "redis": state["redis"],
        "stt_model": state["stt_model"],
        "stt_lock": state["stt_lock"],
        "llm_client": state["llm_client"],
        "llm_model": state["llm_model"],
        "tts_stream_fn": get_tts_fn(engine),
        "tts_engine": engine,
        "vad_mode": _cfg["vad_mode"],
        "stt_model_name": WHISPER_MODEL_SIZE,
        "use_silero": _cfg["vad_mode"] == "silero",
        "opik": state.get("opik"),
    })
    await session.run()


@router.post("/call-result")
async def call_result(request: Request):
    """External call completion hook (ported)."""
    data = await request.json()
    call_sid = data.get("callSid")
    if not call_sid:
        return JSONResponse({"error": "callSid required"}, 400)
    await state["redis"].hset(f"call:{call_sid}", mapping={
        **{k: str(v) for k, v in data.items() if k != "callSid"},
        "ended_at": str(time.time()), "status": "completed",
    })
    await state["redis"].publish("call-updates", json.dumps(data))
    return {"ok": True}


@router.post("/retry/{call_sid}")
async def retry_call(call_sid: str):
    """Re-queue a failed call (ported)."""
    data = await state["redis"].hgetall(f"call:{call_sid}")
    if not data or data.get("status") != "failed":
        return JSONResponse({"error": "Not found or not failed"}, 404)
    await state["redis"].hset(f"call:{call_sid}", mapping={
        "status": "queued",
        "retry_count": str(int(data.get("retry_count", 0)) + 1),
        "started_at": str(time.time()),
    })
    return {"ok": True}
