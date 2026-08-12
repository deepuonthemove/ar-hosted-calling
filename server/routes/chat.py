"""Direct chat with the LLM (bypasses STT/TTS, isolated from calls/review),
plus account/chat history and Excel export."""
import io
import json
import re
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from openpyxl import Workbook

from prompts import strip_markers

from ..state import opik_end_trace, opik_span, opik_start_trace, state
from .accounts import _effective_prompt, _resolve_account
from .stats import _call_record_keys

router = APIRouter()


async def _archive_chat(account_uid: str) -> tuple[str | None, dict | None]:
    """Archive the active chat into an ended session (with config + timing)."""
    chat_key = f"chat:{account_uid}:active"
    raw = await state["redis"].lrange(chat_key, 0, -1)
    if not raw:
        return None, None
    turns = [json.loads(x) for x in raw]
    m = await state["redis"].hgetall(f"chat:{account_uid}:active:meta")
    sid = f"chat-{uuid.uuid4().hex[:12]}"
    turns_key = f"chat:{account_uid}:session:{sid}"
    for t in turns:
        await state["redis"].rpush(turns_key, json.dumps(t))

    llm_count = int(m.get("llm_count", 0))
    llm_avg = int(float(m.get("llm_total_ms", 0)) / max(1, llm_count))
    meta = {
        "account_uid": account_uid,
        "started_at": turns[0].get("ts", time.time()),
        "ended_at": time.time(),
        "count": len(turns),
        "preview": (turns[0].get("text") or "")[:120],
        "llm_model": m.get("llm_model", ""),
        "prompt": m.get("prompt", ""),
        "llm_avg_ms": llm_avg,
        "llm_count": llm_count,
    }
    await state["redis"].set(f"chat:session:{sid}", json.dumps(meta))
    await state["redis"].rpush(f"chat:{account_uid}:history", sid)
    await state["redis"].delete(chat_key)
    await state["redis"].delete(f"chat:{account_uid}:active:meta")
    return sid, meta


@router.post("/api/chat")
async def chat_with_llm(request: Request):
    data = await request.json()
    message = (data.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "message required"}, 400)

    account_uid = data.get("account_uid", "")
    if not account_uid and data.get("project_id") and data.get("row_num"):
        account_uid, _ = await _resolve_account(data["project_id"], int(data["row_num"]))
    if not account_uid:
        return JSONResponse({"error": "account_uid or project_id+row_num required"}, 400)

    account = await state["redis"].hgetall(f"account:{account_uid}") or None
    system_prompt = await _effective_prompt(account)
    chat_key = f"chat:{account_uid}:active"
    meta_key = f"chat:{account_uid}:active:meta"

    raw = await state["redis"].lrange(chat_key, 0, -1)
    history = [json.loads(x) for x in raw] if raw else []

    messages = [{"role": "system", "content": system_prompt}]
    for t in history[-16:]:
        messages.append({"role": t["role"], "content": t["text"]})
    messages.append({"role": "user", "content": message})

    trace = opik_start_trace("chat.turn", f"chat:{account_uid}",
                             {"messages": messages[-2:]},
                             {"type": "chat", "account_uid": account_uid})

    llm_t0 = time.time()
    try:
        resp = await state["llm_client"].chat.completions.create(
            model=state["llm_model"], messages=messages, max_tokens=300, temperature=0)
        reply_raw = resp.choices[0].message.content.strip()
    except Exception as e:
        opik_end_trace(trace, error=str(e))
        return JSONResponse({"error": f"LLM error: {e}"}, 500)
    llm_ms = (time.time() - llm_t0) * 1000

    _chat_usage = getattr(resp, "usage", None)
    opik_span(trace, "chat.llm", "llm", {"messages": messages[-2:]}, reply_raw,
              llm_t0, time.time(), model=state["llm_model"],
              metadata={"usage": {"prompt_tokens": getattr(_chat_usage, "prompt_tokens", None),
                                  "completion_tokens": getattr(_chat_usage, "completion_tokens", None),
                                  "total_tokens": getattr(_chat_usage, "total_tokens", None)}} if _chat_usage else None)
    opik_end_trace(trace, output=reply_raw)

    # Track chat timing/config in the ACTIVE meta (never touches call records)
    await state["redis"].hincrbyfloat(meta_key, "llm_total_ms", llm_ms)
    await state["redis"].hincrby(meta_key, "llm_count", 1)
    await state["redis"].hset(meta_key, mapping={
        "llm_model": state["llm_model"], "prompt": system_prompt, "account_uid": account_uid,
    })

    reply = strip_markers(reply_raw)

    # Detect end-of-chat: CALL_RESULT marker, empty reply, or a farewell phrase
    has_call_result = bool(re.search(r"CALL_RESULT", reply_raw, re.IGNORECASE))
    is_empty = not reply.strip()
    is_farewell = any(p in reply.lower() for p in
                      ["goodbye", "good bye", "ending the chat", "end the chat",
                       "that concludes", "no further questions", "have a great day"])
    should_end = has_call_result or is_empty or is_farewell

    now = time.time()
    await state["redis"].rpush(chat_key, json.dumps({"role": "user", "text": message, "ts": now}))
    if reply:
        await state["redis"].rpush(chat_key, json.dumps({"role": "assistant", "text": reply, "ts": now}))
    if len(history) > 200:
        await state["redis"].ltrim(chat_key, -200, -1)

    # Auto-archive to history when the AI ends the chat
    if should_end:
        sid, meta = await _archive_chat(account_uid)
        return {"ok": True, "reply": reply, "ended": True, "session_id": sid}

    return {"ok": True, "reply": reply, "ended": False}


@router.get("/api/chat/{account_uid}")
async def get_chat(account_uid: str):
    raw = await state["redis"].lrange(f"chat:{account_uid}:active", 0, -1)
    return [json.loads(x) for x in raw] if raw else []


@router.post("/api/chat/{account_uid}/end")
async def end_chat(account_uid: str):
    """End the active chat → archive it to history and clear the active chat."""
    sid, meta = await _archive_chat(account_uid)
    if not sid:
        return {"ok": True, "ended": False}
    return {"ok": True, "ended": True, "session_id": sid, "meta": meta}


@router.get("/api/chat/{account_uid}/history")
async def chat_history(account_uid: str):
    sids = await state["redis"].lrange(f"chat:{account_uid}:history", 0, -1)
    out = []
    for sid in sids:
        raw = await state["redis"].get(f"chat:session:{sid}")
        if raw:
            try:
                m = json.loads(raw)
                m["session_id"] = sid
                out.append(m)
            except json.JSONDecodeError:
                pass
    out.sort(key=lambda m: m.get("ended_at", 0), reverse=True)
    return out


@router.get("/api/chat/session/{session_id}")
async def chat_session_detail(session_id: str):
    raw_meta = await state["redis"].get(f"chat:session:{session_id}")
    if not raw_meta:
        return JSONResponse({"error": "Chat session not found"}, 404)
    meta = json.loads(raw_meta)
    uid = meta.get("account_uid", "")
    raw_turns = await state["redis"].lrange(f"chat:{uid}:session:{session_id}", 0, -1)
    turns = [json.loads(x) for x in raw_turns] if raw_turns else []
    account = await state["redis"].hgetall(f"account:{uid}") or {}
    prompt = meta.get("prompt", "")
    if not prompt:
        # Back-fill for sessions created before prompt tracking existed
        prompt = await _effective_prompt(account)
    return {
        "session_id": session_id,
        "meta": meta,
        "turns": turns,
        "config": {"llm_model": meta.get("llm_model", ""), "prompt": prompt},
        "timing": {
            "llm_avg_ms": meta.get("llm_avg_ms", 0),
            "llm_count": meta.get("llm_count", 0),
            "ttr_avg_ms": meta.get("llm_avg_ms", 0),
        },
        "account": {
            "Patient Name": account.get("Patient Name", ""),
            "Responsible Payer": account.get("Responsible Payer", ""),
            "DOS": account.get("DOS", ""),
            "Claim ID": account.get("Claim ID") or account.get("Account Number", ""),
        },
    }


@router.get("/api/accounts/{account_uid}/calls")
async def account_call_history(account_uid: str):
    """Call history for a given account/claim."""
    keys = _call_record_keys(await state["redis"].keys("call:*"))
    calls = []
    for key in keys:
        data = await state["redis"].hgetall(key)
        if data and data.get("account_uid") == account_uid:
            calls.append({"call_id": key.replace("call:", ""), **data})
    calls.sort(key=lambda c: float(c.get("started_at", 0)), reverse=True)
    return calls


@router.get("/api/accounts")
async def list_accounts():
    r = state["redis"]
    raw = await r.get("accounts-list")
    if not raw:
        return []
    accounts = []
    for uid in json.loads(raw):
        row = await r.hgetall(f"account:{uid}")
        if row:
            accounts.append(row)
    return accounts


@router.get("/api/export-excel")
async def export_excel():
    r = state["redis"]
    raw = await r.get("accounts-list")
    if not raw:
        return Response("No accounts to export", 404)
    uids = json.loads(raw)
    headers_raw = await r.get("accounts-headers")
    headers = json.loads(headers_raw) if headers_raw else []

    wb = Workbook()
    ws = wb.active
    ws.append(headers)
    for uid in uids:
        row = await r.hgetall(f"account:{uid}")
        ws.append([row.get(h, "") for h in headers])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="Calling_Accounts_Updated.xlsx"'},
    )
