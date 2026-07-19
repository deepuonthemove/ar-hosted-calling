"""AR Voice Agent — Azure on-prem edition.

Full port of the Cloudflare Workers app (src/index.ts + src/do.ts):
  STT  Deepgram nova-2-medical  → Faster-Whisper large-v3 (local GPU)
  LLM  Workers AI / Azure OpenAI → vLLM Llama 3.1 8B (local GPU)
  TTS  Cartesia sonic-english   → Piper (local CPU)
  DB   Upstash Redis            → local Redis container
"""
import asyncio
import base64
import csv
import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, Request, UploadFile, File
from fastapi.responses import HTMLResponse, Response, JSONResponse
from faster_whisper import WhisperModel
from openai import AsyncOpenAI
from openpyxl import load_workbook, Workbook

from audio import VAD, twilio_to_whisper, piper_to_twilio, rms
from call_session import CallSession
from prompts import build_call_prompt, parse_markers, parse_call_result, build_greeting, load_payer, load_denial_codes, attempt_repair

# ── Config ───────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "distil-large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "float16")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8001/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
PUBLIC_DOMAIN = os.getenv("PUBLIC_DOMAIN", "localhost:8080")
PUBLIC_SCHEME = os.getenv("PUBLIC_SCHEME", "https")

PIPER_VOICE = os.getenv("PIPER_VOICE", "en_US-lessac-medium")
PIPER_RATE = int(os.getenv("PIPER_SAMPLE_RATE", "22050"))
PIPER_DATA_DIR = os.getenv("PIPER_DATA_DIR", "/models/piper")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ar-voice-agent")

# ── Shared state ─────────────────────────────────────────────────────────
state: dict = {}


def load_models():
    log.info("Loading Faster-Whisper %s ...", WHISPER_MODEL_SIZE)
    state["stt_model"] = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    state["stt_lock"] = asyncio.Lock()
    state["llm_client"] = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
    state["llm_model"] = LLM_MODEL
    state["redis"] = aioredis.from_url(REDIS_URL, decode_responses=True)
    log.info("Models loaded.")


# ── Piper TTS streaming ──────────────────────────────────────────────────
async def tts_stream(text: str):
    # piper-tts pip package: auto-downloads voice to PIPER_DATA_DIR on first run
    proc = await asyncio.create_subprocess_exec(
        "python", "-m", "piper",
        "--model", PIPER_VOICE,
        "--data-dir", PIPER_DATA_DIR,
        "--output-raw",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        proc.stdin.write(text.encode())
        await proc.stdin.drain()
        proc.stdin.close()
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            yield chunk
        await proc.wait()
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass


# ── End-of-call detection ────────────────────────────────────────────────
_END_PHRASES = [
    "bye", "goodbye", "see you", "that's all", "i'm done", "cut the call",
    "end the call", "hang up", "that is all", "no more", "i'm finished",
    "all done", "thank you goodbye",
]


def _is_end_of_call(user_text: str, conversation: list[dict]) -> bool:
    if len(conversation) < 2:
        return False
    lower = user_text.lower().strip()
    if any(p in lower for p in _END_PHRASES):
        return True
    if conversation and any(
        "have a great day" in (m.get("content", "")).lower()
        or "goodbye" in (m.get("content", "")).lower()
        for m in conversation[-3:]
    ):
        if lower in ("okay", "ok", "thanks", "thank you", "bye", "sure", "yes"):
            return True
    return False


# ── App lifecycle ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    await state["redis"].aclose()


app = FastAPI(lifespan=lifespan)


# ════════════════════════════════════════════════════════════════════════
# TELEPHONY  (ported from src/index.ts /voice + /media + /make-call)
# ════════════════════════════════════════════════════════════════════════

@app.post("/make-call")
async def make_call(request: Request):
    """Trigger outbound call via Twilio REST API (ported)."""
    data = await request.json()
    phone = data.get("phone")
    if not phone:
        return JSONResponse({"error": "Phone number is required"}, 400)
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER]):
        return JSONResponse({"error": "Twilio credentials not configured"}, 400)

    payer = data.get("payer", "unknown")
    claim_id = data.get("claim_id", "unknown")
    account_uid = data.get("account_uid", "")
    call_sid = f"local-{uuid.uuid4().hex[:16]}"

    # Pre-create the call record (ported from /voice handler)
    await state["redis"].hset(f"call:{call_sid}", mapping={
        "claim_id": claim_id, "payer": payer, "account_uid": account_uid,
        "phone": phone, "status": "dialing", "started_at": str(time.time()),
    })

    from twilio.rest import Client as TwilioClient
    from twilio.base.exceptions import TwilioRestException
    try:
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        webhook = (f"{PUBLIC_SCHEME}://{PUBLIC_DOMAIN}/voice"
                   f"?payer={payer}&claim_id={claim_id}&account_uid={account_uid}"
                   f"&local_sid={call_sid}")
        call = client.calls.create(to=phone, from_=TWILIO_FROM_NUMBER, url=webhook)
        # Re-key record to the real Twilio SID
        await state["redis"].rename(f"call:{call_sid}", f"call:{call.sid}")
        return {"ok": True, "callSid": call.sid}
    except TwilioRestException as e:
        return JSONResponse({"error": f"Twilio API error: {e.msg}"}, 500)


@app.api_route("/voice", methods=["GET", "POST"])
async def voice_webhook(request: Request):
    """Twilio webhook → TwiML with Media Stream (ported)."""
    q = request.query_params
    form = await request.form() if request.method == "POST" else {}
    call_sid = (form.get("CallSid") if form else None) or q.get("CallSid") or q.get("local_sid", "unknown")

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


@app.websocket("/media/{call_sid}")
async def media_stream(ws: WebSocket, call_sid: str):
    """Twilio Media Streams WebSocket → CallSession (ported from DO)."""
    session = CallSession(ws, call_sid, {
        "redis": state["redis"],
        "stt_model": state["stt_model"],
        "stt_lock": state["stt_lock"],
        "llm_client": state["llm_client"],
        "llm_model": state["llm_model"],
        "tts_stream_fn": tts_stream,
    })
    await session.run()


@app.post("/call-result")
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


@app.post("/retry/{call_sid}")
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


# ════════════════════════════════════════════════════════════════════════
# DATA APIs  (ported: calls list, CSV export, secrets check)
# ════════════════════════════════════════════════════════════════════════

@app.get("/api/calls")
async def list_calls():
    r = state["redis"]
    keys = await r.keys("call:*")
    calls = []
    for key in keys[:50]:
        data = await r.hgetall(key)
        if data:
            calls.append({"callSid": key.replace("call:", ""), **data})
    calls.sort(key=lambda c: float(c.get("started_at", 0)), reverse=True)
    return calls[:20]


@app.get("/export.csv")
async def export_csv():
    r = state["redis"]
    keys = await r.keys("call:*")
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


@app.get("/api/check-secrets")
async def check_secrets():
    return {
        "twilio_sid": bool(TWILIO_ACCOUNT_SID),
        "twilio_token": bool(TWILIO_AUTH_TOKEN),
        "twilio_from": bool(TWILIO_FROM_NUMBER),
        "vllm_url": VLLM_BASE_URL,
        "whisper_model": WHISPER_MODEL_SIZE,
        "llm_model": LLM_MODEL,
    }


# ════════════════════════════════════════════════════════════════════════
# EXCEL  (ported from src/index.ts upload/accounts/export)
# ════════════════════════════════════════════════════════════════════════

@app.post("/api/upload-excel")
async def upload_excel(file: UploadFile = File(...)):
    try:
        content = await file.read()
        wb = load_workbook(io.BytesIO(content), data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return JSONResponse({"error": "Sheet is empty"}, 400)

        headers = [str(h) if h else f"col_{i}" for i, h in enumerate(rows[0])]
        r = state["redis"]

        # Clear old accounts (ported)
        old_uids = await r.get("accounts-list")
        if old_uids:
            for uid in json.loads(old_uids):
                await r.delete(f"account:{uid}")

        uids = []
        for i, row in enumerate(rows[1:]):
            record = {}
            for j, val in enumerate(row):
                if j < len(headers) and val is not None:
                    record[headers[j]] = val.isoformat() if hasattr(val, "isoformat") else str(val)
            uid = record.get("UID") or f"KS-PC-{i}-{int(time.time())}"
            record["UID"] = uid
            record.setdefault("Call Status", "Pending")
            uids.append(uid)
            await r.hset(f"account:{uid}", mapping=record)

        await r.set("accounts-headers", json.dumps(headers))
        await r.set("accounts-list", json.dumps(uids))
        return {"ok": True, "count": len(uids)}
    except Exception as e:
        log.error("Excel parse error: %s", e)
        return JSONResponse({"error": f"Parsing error: {e}"}, 500)


@app.get("/api/accounts")
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


@app.get("/api/export-excel")
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


# ════════════════════════════════════════════════════════════════════════
# HEALTH + DASHBOARD
# ════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "whisper": WHISPER_MODEL_SIZE, "llm": LLM_MODEL}


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.get("/test", response_class=HTMLResponse)
async def voice_test():
    return """<!DOCTYPE html>
<html><head><title>Voice Test</title></head>
<body style="font-family:sans-serif;max-width:600px;margin:2rem auto;">
<h2>Voice Agent Test</h2>
<button id="toggle" onclick="toggle()" style="padding:1rem 2rem;font-size:1.2rem;background:#2563eb;color:white;border:none;border-radius:8px;cursor:pointer">Start Test</button>
<p id="status" style="margin:1rem 0;font-weight:bold;">Disconnected</p>
<div id="log" style="height:400px;overflow-y:auto;background:#1a1a1a;color:#0f0;padding:1rem;border-radius:8px;font-family:monospace;"></div>
<script>
let ws, stream, ctx, src, proc, playCtx, audioQ = [], playing = false, currentSrc = null;
const logDiv = document.getElementById('log'), btn = document.getElementById('toggle'), st = document.getElementById('status');
let audioRate = 22050;
function lg(m,c){const d=document.createElement('div');d.textContent=m;d.style.color=c||'#888';logDiv.appendChild(d);logDiv.scrollTop=logDiv.scrollHeight}
function toggle(){if(ws&&ws.readyState===1){ws.close();return}connect()}
async function connect(){
  playCtx = new AudioContext();
  ws=new WebSocket('wss://'+location.host+'/ws/test_'+Math.random().toString(36).slice(2));
  ws.binaryType='arraybuffer';
  btn.disabled=true;btn.textContent='Connecting...';st.textContent='Connecting';
  ws.onopen=()=>{btn.textContent='Stop';st.textContent='Connected';startMic()};
  ws.onclose=()=>{btn.textContent='Start Test';st.textContent='Disconnected';stopMic();ws=null};
  ws.onmessage=e=>{
    if(typeof e.data==='string'){const m=JSON.parse(e.data);
      if(m.type==='config')audioRate=m.sample_rate;
      else lg(m.text,m.type==='transcript'?'#8cf':'#fc8');
    }else{
      const v=new Uint8Array(e.data);
      if(v[0]===1){audioQ.push(v.slice(1));if(!playing)playNext();}
      else if(v[0]===2){audioQ=[];if(playing&&currentSrc){playing=false;try{currentSrc.stop()}catch(e){}}}
    }
  };
}
function playNext(){
  if(!audioQ.length||!playCtx){playing=false;return}
  playing=true;
  const total=audioQ.reduce((s,c)=>s+c.length,0);
  const pcm=new Int16Array(total/2);let off=0;
  while(audioQ.length){const c=audioQ.shift();pcm.set(new Int16Array(c.buffer,c.byteOffset,c.length/2),off);off+=c.length/2}
  const buf=playCtx.createBuffer(1,pcm.length,audioRate);
  const ch=buf.getChannelData(0);
  for(let i=0;i<pcm.length;i++)ch[i]=pcm[i]/32768;
  const s=playCtx.createBufferSource();currentSrc=s;
  s.buffer=buf;s.connect(playCtx.destination);
  s.onended=()=>{playing=false;currentSrc=null;if(audioQ.length)playNext()};
  s.start();
}
async function startMic(){
  ctx=new AudioContext({sampleRate:16000});
  stream=await navigator.mediaDevices.getUserMedia({audio:true});
  src=ctx.createMediaStreamSource(stream);
  proc=ctx.createScriptProcessor(4096,1,1);
  proc.onaudioprocess=e=>{if(!ws||ws.readyState!==1)return;const inp=e.inputBuffer.getChannelData(0);const b=new Int16Array(inp.length);for(let i=0;i<inp.length;i++)b[i]=Math.max(-32768,Math.min(32767,inp[i]*32768));ws.send(b.buffer)};
  src.connect(proc);proc.connect(ctx.destination);lg('Mic started','#4c4')
}
function stopMic(){if(proc){proc.disconnect();proc=null}if(src){src.disconnect();src=null}if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}if(ctx){ctx.close();ctx=null}}
</script></body></html>"""


@app.websocket("/ws/{session_id}")
async def browser_voice_loop(ws: WebSocket, session_id: str):
    """Direct browser voice test (16kHz PCM) with barge-in."""
    await ws.accept()
    await ws.send_json({"type": "config", "sample_rate": PIPER_RATE})

    # Parse account_uid from WebSocket URL query params
    account_uid = ""
    try:
        qs = str(ws.url).split("?")[1] if "?" in str(ws.url) else ""
        for part in qs.split("&"):
            k, _, v = part.partition("=")
            if k == "account_uid" and v:
                account_uid = v
    except Exception:
        pass

    account = None
    if account_uid:
        account = await state["redis"].hgetall(f"account:{account_uid}") or None

    system_prompt = build_call_prompt("GREETING", None, None, account)
    conversation: list[dict] = [{"role": "system", "content": system_prompt}]

    # Speak greeting with claim context right away
    greeting = build_greeting(account)
    conversation.append({"role": "assistant", "content": greeting})
    await ws.send_json({"type": "llm_text", "text": greeting})
    tts_task = asyncio.create_task(_stream_tts_reply(ws, greeting))

    vad = VAD()
    barge_in = False

    async def cancel_tts():
        nonlocal tts_task
        if tts_task and not tts_task.done():
            tts_task.cancel()
            try:
                await tts_task
            except asyncio.CancelledError:
                pass
            tts_task = None

    try:
        while True:
            raw = await ws.receive_bytes()
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            now = time.time()
            energy = rms(audio)

            # If energy spikes while TTS is playing → possible barge-in
            if tts_task and not tts_task.done() and energy > 0.025:
                await cancel_tts()
                barge_in = True
                await ws.send_bytes(b"\x02")
                # Skip this chunk — it's likely echo from the TTS we just cancelled
                # Reset VAD to avoid processing leftover echo
                vad = VAD()
                continue

            segment = vad.add(audio, now)
            if segment is None:
                continue

            await cancel_tts()
            barge_in = False

            async with state["stt_lock"]:
                loop = asyncio.get_running_loop()
                segs, _ = await loop.run_in_executor(
                    None, lambda: state["stt_model"].transcribe(segment, beam_size=1, vad_filter=True))
                text = " ".join(s.text for s in segs).strip()
            if len(text) < 3:
                continue

            if _is_end_of_call(text, conversation):
                await cancel_tts()
                await ws.send_json({"type": "llm_text", "text": "Call ended. Have a great day!"})
                await asyncio.sleep(1)
                await ws.close()
                return

            conversation.append({"role": "user", "content": text})
            await ws.send_json({"type": "transcript", "text": text})
            resp = await state["llm_client"].chat.completions.create(
                model=LLM_MODEL, messages=conversation[-7:], max_tokens=150, temperature=0)
            reply = resp.choices[0].message.content.strip()
            conversation.append({"role": "assistant", "content": reply})
            await ws.send_json({"type": "llm_text", "text": reply})

            # Stream TTS in background so audio reads continue
            tts_task = asyncio.create_task(_stream_tts_reply(ws, reply))

    except Exception:
        pass
    finally:
        if tts_task and not tts_task.done():
            tts_task.cancel()
            try:
                await tts_task
            except asyncio.CancelledError:
                pass


async def _stream_tts_reply(ws: WebSocket, text: str):
    try:
        async for pcm_bytes in tts_stream(text):
            await ws.send_bytes(b"\x01" + pcm_bytes)
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


DASHBOARD_HTML = """<!DOCTYPE html>
<html><head><title>AR Voice Agent — Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}.live{animation:pulse 2s infinite}</style></head>
<body class="bg-gray-900 text-gray-100 p-6">
<div class="max-w-7xl mx-auto">
  <div class="flex justify-between items-center mb-6">
    <h1 class="text-3xl font-bold">Healthcare AR Voice Agent
      <span class="text-green-400 text-sm live">● ON-PREM</span></h1>
    <div class="flex gap-2">
      <a href="/api/export-excel" class="px-3 py-2 bg-green-700 rounded text-sm font-bold">Export Excel</a>
      <a href="/export.csv" class="px-3 py-2 bg-gray-700 rounded text-sm font-bold">Export CSV</a>
    </div>
  </div>
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
    <div class="space-y-6">
      <div class="bg-gray-800 p-6 rounded-lg">
        <h2 class="text-lg font-bold mb-4">Outbound Call Control
          <span id="badge" class="text-xs text-blue-400 font-normal block">No account loaded</span></h2>
        <form id="call-form" class="space-y-4">
          <input type="hidden" id="account_uid">
          <input id="phone" placeholder="+15551234567" required
            class="w-full bg-gray-900 border border-gray-700 rounded p-2 text-sm">
          <input id="payer" placeholder="Aetna"
            class="w-full bg-gray-900 border border-gray-700 rounded p-2 text-sm">
          <input id="claim_id" placeholder="CLM-90210"
            class="w-full bg-gray-900 border border-gray-700 rounded p-2 text-sm">
          <button class="w-full bg-blue-600 hover:bg-blue-500 font-bold p-2 rounded text-sm">
            Place Outbound Call</button>
        </form>
        <div id="msg" class="mt-3 text-xs hidden"></div>
      </div>

      <!-- Browser Call Button -->
      <button id="browser-call-btn" onclick="toggleBrowserCall()"
        class="w-full bg-green-700 hover:bg-green-600 font-bold p-3 rounded text-sm transition duration-200">
        🎤 Call from Browser
      </button>

      <!-- Browser Call Panel (hidden by default) -->
      <div id="browser-call-panel" class="bg-gray-800 rounded-lg p-4 hidden">
        <div class="flex justify-between items-center mb-3">
          <h2 class="text-lg font-bold">🟢 Live Browser Call</h2>
          <button onclick="endBrowserCall()"
            class="px-3 py-1 bg-red-700 hover:bg-red-600 rounded text-xs font-bold">End Call</button>
        </div>
        <div id="browser-log" class="h-48 overflow-y-auto bg-gray-900 rounded p-3 text-xs font-mono space-y-1"></div>
      </div>

      <div class="grid grid-cols-2 gap-4">
        <div class="bg-gray-800 p-4 rounded-lg"><div class="text-sm text-gray-400">Calls</div>
          <div id="calls-today" class="text-2xl font-bold">0</div></div>
        <div class="bg-gray-800 p-4 rounded-lg"><div class="text-sm text-gray-400">Success</div>
          <div id="success-rate" class="text-2xl font-bold">0%</div></div>
        <div class="bg-gray-800 p-4 rounded-lg"><div class="text-sm text-gray-400">Avg Duration</div>
          <div id="avg-dur" class="text-2xl font-bold">0m</div></div>
        <div class="bg-gray-800 p-4 rounded-lg"><div class="text-sm text-gray-400">Cost/min</div>
          <div class="text-2xl font-bold">$0.00</div></div>
      </div>
      <div class="bg-gray-800 rounded-lg p-4">
        <h2 class="text-lg font-bold mb-3">Live Call Feed</h2>
        <div id="call-rows" class="space-y-3 max-h-[300px] overflow-y-auto"></div>
      </div>
    </div>
    <div class="lg:col-span-2 space-y-6">
      <div class="bg-gray-800 p-6 rounded-lg">
        <h2 class="text-lg font-bold mb-2">Excel Calling Context List</h2>
        <form id="upload-form" class="flex gap-4 items-center mt-4">
          <input type="file" id="excel-file" accept=".xlsx" required class="text-sm">
          <button class="px-4 py-2 bg-blue-600 rounded text-sm font-bold">Upload</button>
        </form>
        <div id="upload-msg" class="mt-3 text-xs hidden"></div>
      </div>
      <div class="bg-gray-800 rounded-lg overflow-hidden">
        <div class="p-4 border-b border-gray-700 flex justify-between">
          <h2 class="text-lg font-bold">Calling Checklist</h2>
          <span id="acct-count" class="text-xs text-gray-400">0 Accounts</span></div>
        <div class="overflow-x-auto max-h-[500px] overflow-y-auto">
          <table class="w-full text-sm">
            <thead class="bg-gray-700 sticky top-0"><tr>
              <th class="text-left p-3">Patient</th><th class="text-left p-3">Payer</th>
              <th class="text-left p-3">DOS</th><th class="text-left p-3">Billed</th>
              <th class="text-left p-3">Objective</th><th class="text-left p-3">Outcome</th>
              <th class="text-left p-3">Status</th><th class="text-left p-3"></th></tr></thead>
            <tbody id="acct-rows" class="divide-y divide-gray-700">
              <tr><td colspan="8" class="p-6 text-center text-gray-500">Upload an Excel file.</td></tr>
            </tbody></table></div></div>
    </div></div></div>
<script>
const $ = id => document.getElementById(id);
let stats = {total: 0, success: 0};

async function fetchCalls() {
  const calls = await (await fetch('/api/calls')).json();
  $('call-rows').innerHTML = '';
  stats = {total: 0, success: 0};
  let durSum = 0, durN = 0;
  calls.forEach(c => {
    stats.total++;
    if (c.status === 'completed') stats.success++;
    if (c.duration_ms) { durSum += +c.duration_ms; durN++; }
    const color = c.status==='completed'?'bg-green-900 text-green-300'
      : c.status==='failed'?'bg-red-900 text-red-300':'bg-yellow-900 text-yellow-300';
    $('call-rows').insertAdjacentHTML('afterbegin', `
      <div class="p-3 bg-gray-900 rounded border border-gray-700">
        <div class="flex justify-between"><b class="text-xs">${c.payer||'Unknown'}</b>
        <span class="px-1.5 py-0.5 rounded text-[10px] ${color}">${c.status}</span></div>
        <div class="text-[11px] text-gray-400 mt-1">Claim: ${c.claim_id||'-'}
          ${c.amount?'<br>Billed: $'+c.amount:''}
          ${c.next_action?'<br>Action: '+c.next_action:''}
          ${c.last_error?'<br><span class="text-red-400">'+c.last_error+'</span>':''}
        </div></div>`);
  });
  $('calls-today').textContent = stats.total;
  $('success-rate').textContent = stats.total ? Math.round(stats.success/stats.total*100)+'%' : '0%';
  $('avg-dur').textContent = durN ? Math.round(durSum/durN/60000*10)/10+'m' : '0m';
}

async function fetchAccounts() {
  const accts = await (await fetch('/api/accounts')).json();
  $('acct-count').textContent = accts.length + ' Accounts';
  if (!accts.length) return;
  $('acct-rows').innerHTML = '';
  accts.forEach(a => {
    const st = a['Call Status']||'Pending';
    const color = st==='Calls Done'?'bg-green-950 text-green-300'
      : (st==='Failed'||st==='Disconnected')?'bg-red-950 text-red-300':'bg-gray-700';
    $('acct-rows').insertAdjacentHTML('beforeend', `<tr class="hover:bg-gray-700">
      <td class="p-3 font-semibold">${a['Patient Name']||'-'}</td>
      <td class="p-3">${a['Responsible Payer']||''}</td>
      <td class="p-3 text-gray-400">${(a['DOS']||'').slice(0,10)}</td>
      <td class="p-3">${a['Billed Amount']?'$'+a['Billed Amount']:'-'}</td>
      <td class="p-3 text-xs text-gray-400 max-w-xs truncate">${a['AR Final Comments']||'-'}</td>
      <td class="p-3 text-xs max-w-xs truncate">${a['Call Comments']||'-'}</td>
      <td class="p-3"><span class="px-2 py-0.5 rounded text-[10px] font-bold ${color}">${st}</span></td>
      <td class="p-3"><button onclick="pick('${a.UID}','${(a['Patient Name']||'').replace(/'/g,"")}','${a['Responsible Payer']||''}','${a['Account Number']||''}')"
        class="px-2 py-1 bg-blue-600 rounded text-xs font-bold">Load</button></td></tr>`);
  });
}

function pick(uid, name, payer, claim) {
  $('account_uid').value = uid; $('payer').value = payer; $('claim_id').value = claim;
  $('badge').textContent = 'Loaded: ' + name;
}

$('upload-form').onsubmit = async e => {
  e.preventDefault();
  const fd = new FormData();
  fd.append('file', $('excel-file').files[0]);
  const res = await fetch('/api/upload-excel', {method: 'POST', body: fd});
  const d = await res.json();
  $('upload-msg').classList.remove('hidden');
  $('upload-msg').textContent = res.ok ? `Loaded ${d.count} accounts` : 'Error: ' + d.error;
  $('upload-msg').className = 'mt-3 text-xs ' + (res.ok ? 'text-green-400' : 'text-red-400');
  fetchAccounts();
};

$('call-form').onsubmit = async e => {
  e.preventDefault();
  const res = await fetch('/make-call', {method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({phone: $('phone').value, payer: $('payer').value,
      claim_id: $('claim_id').value, account_uid: $('account_uid').value})});
  const d = await res.json();
  $('msg').classList.remove('hidden');
  $('msg').textContent = res.ok ? 'Call triggered: ' + d.callSid : 'Error: ' + d.error;
  $('msg').className = 'mt-3 text-xs ' + (res.ok ? 'text-green-400' : 'text-red-400');
};

fetchAccounts(); fetchCalls();
setInterval(fetchCalls, 3000);
setInterval(fetchAccounts, 5000);

// ── Browser Call ──────────────────────────────────────────────────────
let bcWS = null, bcMic = null, bcCtx = null, bcSrc = null, bcProc = null;
let bcPlayCtx = null, bcAudioQ = [], bcPlaying = false, bcCurSrc = null;
let bcLogEl = null;

function bcLog(msg, cls) {
  const d = document.createElement('div');
  d.textContent = msg;
  if (cls) d.className = cls;
  if (bcLogEl) { bcLogEl.appendChild(d); bcLogEl.scrollTop = bcLogEl.scrollHeight; }
}

function toggleBrowserCall() {
  if (bcWS && bcWS.readyState === WebSocket.OPEN) { endBrowserCall(); return; }
  startBrowserCall();
}

function endBrowserCall() {
  if (bcPlayCtx) { bcPlayCtx.close(); bcPlayCtx = null; }
  if (bcProc) { bcProc.disconnect(); bcProc = null; }
  if (bcSrc) { bcSrc.disconnect(); bcSrc = null; }
  if (bcMic) { bcMic.getTracks().forEach(t => t.stop()); bcMic = null; }
  if (bcCtx) { bcCtx.close(); bcCtx = null; }
  if (bcWS) { bcWS.close(); bcWS = null; }
  bcAudioQ = []; bcPlaying = false; bcCurSrc = null;
  $('browser-call-panel').classList.add('hidden');
  $('browser-call-btn').textContent = '🎤 Call from Browser';
  $('browser-call-btn').className = 'w-full bg-green-700 hover:bg-green-600 font-bold p-3 rounded text-sm';
}

async function startBrowserCall() {
  const sid = 'browser_' + Math.random().toString(36).slice(2);
  const uid = ($('account_uid') && $('account_uid').value) || '';
  bcPlayCtx = new AudioContext(); // create inside user gesture — ensures running state
  bcWS = new WebSocket('wss://' + location.host + '/ws/' + sid + (uid ? '?account_uid=' + uid : ''));
  bcWS.binaryType = 'arraybuffer';

  $('browser-call-panel').classList.remove('hidden');
  bcLogEl = $('browser-log');
  bcLogEl.innerHTML = '<div class="text-green-400">Connecting...</div>';
  $('browser-call-btn').textContent = '🔴 End Browser Call';
  $('browser-call-btn').className = 'w-full bg-red-700 hover:bg-red-600 font-bold p-3 rounded text-sm';

  bcWS.onopen = async () => {
    bcLogEl.innerHTML = '<div class="text-green-400">Connected — starting mic...</div>';
    try {
      bcCtx = new AudioContext({ sampleRate: 16000 });
      bcMic = await navigator.mediaDevices.getUserMedia({ audio: true });
      bcSrc = bcCtx.createMediaStreamSource(bcMic);
      bcProc = bcCtx.createScriptProcessor(4096, 1, 1);
      bcProc.onaudioprocess = e => {
        if (!bcWS || bcWS.readyState !== 1) return;
        const inp = e.inputBuffer.getChannelData(0);
        const b = new Int16Array(inp.length);
        for (let i = 0; i < inp.length; i++) b[i] = Math.max(-32768, Math.min(32767, inp[i] * 32768));
        bcWS.send(b.buffer);
      };
      bcSrc.connect(bcProc);
      bcProc.connect(bcCtx.destination);
      bcLogEl.innerHTML = '<div class="text-green-400">✅ Mic active — speak now</div>';
    } catch (e) {
      bcLogEl.innerHTML = '<div class="text-red-400">❌ Mic error: ' + e.message + '</div>';
    }
  };

  bcWS.onmessage = e => {
    if (typeof e.data === 'string') {
      const m = JSON.parse(e.data);
      if (m.type === 'config') { bcPlayCtx = new AudioContext(); return; }
      bcLog(m.text, m.type === 'transcript' ? 'text-blue-300' : 'text-orange-300');
    } else {
      const v = new Uint8Array(e.data);
      if (v[0] === 1) { bcAudioQ.push(v.slice(1)); if (!bcPlaying) bcPlayNext(); }
      else if (v[0] === 2) { bcAudioQ = []; if (bcPlaying && bcCurSrc) { bcPlaying = false; try { bcCurSrc.stop(); } catch (e) {} } }
    }
  };

  bcWS.onclose = () => {
    bcLogEl.innerHTML += '<div class="text-gray-500">Disconnected.</div>';
    endBrowserCall();
  };
}

function bcPlayNext() {
  if (!bcAudioQ.length || !bcPlayCtx) { bcPlaying = false; return; }
  bcPlaying = true;
  const total = bcAudioQ.reduce((s, c) => s + c.length, 0);
  const pcm = new Int16Array(total / 2); let off = 0;
  while (bcAudioQ.length) { const c = bcAudioQ.shift(); pcm.set(new Int16Array(c.buffer, c.byteOffset, c.length / 2), off); off += c.length / 2; }
  const buf = bcPlayCtx.createBuffer(1, pcm.length, 22050);
  const ch = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
  const s = bcPlayCtx.createBufferSource();
  bcCurSrc = s;
  s.buffer = buf; s.connect(bcPlayCtx.destination);
  s.onended = () => { bcPlaying = false; bcCurSrc = null; if (bcAudioQ.length) bcPlayNext(); };
  s.start();
}
</script></body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
