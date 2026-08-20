# AR Voice Agent — Architecture

## Overview

Healthcare AR (Accounts Receivable) voice agent that automates outbound calls to insurance payer call centers. The AI navigates IVR trees, verifies claim status, handles denials, and captures structured results — replacing manual phone follow-up by billing teams.

Two implementations exist: an **original Cloudflare Workers** version (`src/`, TypeScript) and the **active on-prem** version (`deploy/`, Python FastAPI running on an Azure NC4as_T4_v3 VM with local GPU).

This document describes the **on-prem** version.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Azure NC4as_T4_v3 VM                         │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐   ┌──────────────┐ │
│  │  vLLM    │   │ Voice Agent  │   │  Redis   │   │   Caddy      │ │
│  │ (GPU)    │   │ (GPU+CPU)    │   │  (CPU)   │   │  (TLS)       │ │
│  │ :8001    │   │ :8080        │   │ :6379    │   │ :443         │ │
│  └────┬─────┘   └──────┬───────┘   └──────────┘   └──────────────┘ │
│       │                │                                             │
│       └──── HTTP ──────┘                                             │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Shared State (in-memory)                                    │   │
│  │  • WhisperModel (GPU)   • AsyncOpenAI (→ vLLM)               │   │
│  │  • Kokoro pipeline (CPU) • asyncio.Lock (STT mutex)          │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
         │                        │
         │ ngrok                  │ Twilio (not yet active)
         ▼                        ▼
    Browser Test              Phone Call
    (dashboard)               (Media Streams WSS)
```

### Components

| Container | Image | Base | Role |
|-----------|-------|------|------|
| `vllm` | `vllm/vllm-openai:latest` | Ubuntu | LLM inference server (OpenAI-compatible API) |
| `voice-agent` | Custom (`Dockerfile.server`) | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9` | FastAPI app — all call logic, STT, TTS |
| `redis` | `redis:7-alpine` | Alpine | Call records, account data, pub/sub updates |
| `caddy` | `caddy:2-alpine` | Alpine | TLS termination (not active — no domain configured) |

---

## Data Flow

### Browser Call Flow (active test path)

```
Browser                    Voice Agent                    Whisper (GPU)     vLLM (GPU)     Piper/Kokoro (CPU)
   │                            │                             │                │                 │
   │ 1. GET /dashboard          │                             │                │                 │
   │◄──── HTML + JS ────────────│                             │                │                 │
   │                            │                             │                │                 │
   │ 2. WSS /ws/{sid}?tts=&uid= │                             │                │                 │
   │════════════════════════════►│                             │                │                 │
   │       config msg           │                             │                │                 │
   │◄════════════════════════════│                             │                │                 │
   │                            │                             │                │                 │
   │ 3. Greeting TTS audio      │                             │                │                 │
   │◄══════ \x01 + PCM ─────────│◄─────────────── TTS ────────│◄───────────────│─────────────────│
   │                            │                             │                │                 │
   │ 4. Raw PCM (Int16Array)    │                             │                │                 │
   │════════════════════════════►│                             │                │                 │
   │                            │── VAD ──────────────────►   │                │                 │
   │                            │── transcribe (lock) ───────►│                │                 │
   │                            │◄── text ────────────────────│                │                 │
   │                            │                             │                │                 │
   │                            │── chat.completions ─────────────────────────►│                 │
   │                            │◄── response ─────────────────────────────────│                 │
   │                            │                             │                │                 │
   │                            │── parse_markers()           │                │                 │
   │                            │── TTS ────────────────────────────────────────────────────────►│
   │ 5. Response TTS audio      │                             │                │                 │
   │◄══════ \x01 + PCM ─────────│◄──────────────────────────────────────────────────────────────│
   │                            │                             │                │                 │
   │ repeat 4-5 until end       │                             │                │                 │
```

### Twilio Call Flow (implemented but untested)

```
Twilio                      Voice Agent                    Redis
  │                              │                           │
  │ POST /make-call              │                           │
  │◄──── {"callSid": "CA..."} ───│── hset call:{sid} ───────►│
  │                              │                           │
  │ POST /voice                  │                           │
  │◄── TwiML (<Stream>) ─────────│                           │
  │                              │                           │
  │ WSS /media/{call_sid}        │                           │
  │══════════════════════════════►│                           │
  │  event: start                │── hset status=in-progress►│
  │  event: media (μ-law b64)    │                           │
  │  ...                         │                           │
  │                              │── CallSession.run()       │
  │                              │   (same pipeline as       │
  │  event: media (μ-law b64)    │    browser but with        │
  │══════════════════════════════►   Twilio protocol)         │
  │                              │                           │
  │  event: media (μ-law b64)    │                           │
  │◄══════════════════════════════│── TTS → μ-law → media     │
  │                              │                           │
  │  event: stop                 │                           │
  │══════════════════════════════►── _finalize() ────────────►│
```

---

## API Reference

### Telephony

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/make-call` | Trigger outbound call via Twilio REST API. Body: `{phone, payer?, claim_id?, account_uid?}` |
| `GET`/`POST` | `/voice` | Twilio webhook — returns TwiML with `<Stream>` pointing to `/media/{call_sid}` |
| `WS` | `/media/{call_sid}` | Twilio Media Streams WebSocket — delegates to `CallSession` |
| `POST` | `/call-result` | External call completion hook. Body: `{callSid, ...result}` |
| `POST` | `/retry/{call_sid}` | Re-queue a failed call |

### Data

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/calls` | Last 20 calls (sorted by `started_at`) |
| `GET` | `/export.csv` | Download completed/failed calls as CSV |
| `GET` | `/api/accounts` | All uploaded accounts from Excel |
| `POST` | `/api/upload-excel` | Upload XLSX — each row stored as Redis hash `account:{uid}` |
| `GET` | `/api/export-excel` | Download accounts as XLSX |
| `GET` | `/api/check-secrets` | Boolean presence of Twilio credentials + model info |

### Control

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | `{"status":"ok","whisper":"...","llm":"..."}` |
| `GET` | `/api/current-llm` | Current model, available options, switching state |
| `POST` | `/api/switch-llm` | Switch LLM model. Body: `{model:"..."}` — restarts vLLM container in background |

### UI

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/dashboard` | Full dashboard: call form, Excel upload, account table, call feed, TTS/LLM selectors |
| `GET` | `/test` | Minimal browser voice test page |
| `WS` | `/ws/{session_id}` | Browser voice WebSocket (raw PCM 16kHz) |

---

## State Machine

### States

```
GREETING → IVR_NAV → CLAIM_VERIFY → STATUS_GATHER → DENIAL_HANDLE
                                      ↘              APPROVED_HANDLE
                                       → DENIAL_HANDLE / APPROVED_HANDLE
                                    CLOSE
```

### Transitions (in `call_session.py:_advance_state`)

| From | Trigger | To |
|------|---------|----|
| `GREETING` | (automatic after greeting spoken) | `IVR_NAV` |
| `IVR_NAV` | "connected", "agent", "representative" in bot text | `CLAIM_VERIFY` |
| `CLAIM_VERIFY` | "denied" or "denial" in bot text | `DENIAL_HANDLE` |
| `CLAIM_VERIFY` | "paid" or "approved" in bot text | `APPROVED_HANDLE` |
| `CLAIM_VERIFY` | other status keywords | `STATUS_GATHER` |
| `STATUS_GATHER` | "denied" or "denial" | `DENIAL_HANDLE` |
| `STATUS_GATHER` | "paid" or "approved" | `APPROVED_HANDLE` |
| any | `[CALL_RESULT]` JSON detected | `CLOSE` (via `_finalize`) |

### Timers

| Timer | Duration | Action |
|-------|----------|--------|
| Silence nudge | 10s of silence while NOT on hold | Injects synthetic `[Hold music detected]` |
| Max silence | 19 min | Fails call |
| Hold poll | 30s interval while on hold | Waits, no action |
| Max hold | 30 min (configurable via `MAX_HOLD_SEC`) | Ends call with `hold_timeout` |

---

## Key Components

### CallSession (`call_session.py`)

Per-call orchestrator wrapping a Twilio Media Streams `start/media/stop` protocol. Instantiated for each WebSocket connection.

```
CallSession
├── run()              — entry: accept WS, load account, build prompt, start watchdog, event loop
├── _on_start()        — Twilio "start" → extract params, speak greeting, transition to IVR_NAV
├── _on_media()        — Twilio "media" → decode μ-law, VAD, barge-in detect, transcribe, LLM
├── _run_llm()         — state-tag user message, stream LLM, parse markers per-sentence, handle DTMF/WAITING/CALL_RESULT, advance state
├── _transcribe()      — Whisper on speech segment (mutex-protected)
├── _speak/_stream_tts — TTS → chunk into 160-byte μ-law frames → send as Twilio media events
├── _barge_in()        — cancel TTS, send Twilio "clear" event
├── _send_dtmf()       — send Twilio "dtmf" event
├── _silence_watchdog()— background timer: silence, hold, max duration
├── _detect_call_end() — phrase-based end-of-call detection
├── _check_ivr_drift() — verify IVR expected phrases match actual transcription
└── _finalize()        — persist CALL_RESULT to Redis, attempt LLM repair on failure, close WS
```

### Prompt Builder (`prompts.py`)

Layered system prompt construction:

```
BASE_PROMPT (AR specialist role, rules)
  + state-specific rules (DTMF/WAITING markers for IVR_NAV/STATUS_GATHER/DENIAL_HANDLE)
  + [CLAIM CONTEXT] (patient, DOS, CPT, billed amount, payer, account, objective)
  + [PAYER IVR] (menu tree from Aetna.json with DTMF mappings and verify phrases)
  + [DENIAL CODES] (subset of common denials for this payer with descriptions)
```

Markers parsed from LLM output:
- `[DTMF:digit]` — send DTMF tone
- `[WAITING]` — on hold, start hold timer
- `[CALL_RESULT] {...}` — structured call result JSON (replaces old `[END:...]` positional format)

### Audio Pipeline (`audio.py`)

```
Twilio → mulaw → twilio_to_whisper() → resample 8k→16k → float32 → Whisper
Piper  → int16 PCM → piper_to_twilio() → resample → mu-law → Twilio
Kokoro → float32 → resample 24k→22.05k → clip → int16 PCM → (same path as Piper)
```

VAD accumulates float32 chunks, flushes on silence (>700ms) or max speech duration (10s).

### STT: Faster-Whisper

- Model: `distil-large-v3` (on GPU, `float16`)
- Segments via VAD, `beam_size=1`
- Mutex-protected (`asyncio.Lock`) — single concurrent transcription

### LLM: vLLM

- Model: `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (AWQ quantized, GPU)
- Alternatives: `Qwen/Qwen2.5-7B-Instruct-AWQ`
- Memory: `--gpu-memory-utilization=0.55`
- Parameters: `temperature=0`, `max_tokens=150-200`, `stream=True`
- Context: last 7 messages (system + 6 history), user messages tagged with `[INSURANCE REP]` and `[STATE: ...]`

### TTS

| Engine | Method | Sample Rate | Quality | Memory |
|--------|--------|-------------|---------|--------|
| Piper | subprocess (`python -m piper --output-raw`) | 22050 Hz | Robotic | ~100MB (voice data on disk) |
| Kokoro | in-process (`KPipeline` on CPU) | 24000 Hz → resampled to 22050 | Natural | ~82MB (model in memory) |
| Chatterbox (Turbo/Nano) | isolated HTTP service (`/tts`) | 24000 Hz → resampled to 22050 | Natural (voice cloning) | Turbo 3.8GB / Nano 2.8GB (weights) |

**Chatterbox runs as a separate process**, never inside voice-agent: it hard-pins a transformers stack that would break Kokoro. On the VM it's the CUDA container `chatterbox-tts`; on the Mac it's a native Python venv driving MPS (see "Local Mac Development"). The main app calls it over HTTP (`CHATTERBOX_SERVICE_URL`).

---

## Data Model

### Redis Keys

| Key Pattern | Type | Fields |
|-------------|------|--------|
| `call:{call_sid}` | Hash | `claim_id`, `payer`, `account_uid`, `phone`, `status`, `started_at`, `ended_at`, `duration_ms`, `next_action`, `denial_code`, `paid_amount`, `billed_amount`, `appeal_deadline`, `call_summary`, `satisfaction`, `last_error`, `last_llm_response`, `ivr_drift`, `tts_engine` |
| `account:{uid}` | Hash | All columns from uploaded Excel (`Patient Name`, `DOS`, `CPT`, `Billed Amount`, `Responsible Payer`, `Account Number`, `AR Final Comments`, `Call Comments`, `Call Date`, `Call Status`, `Denial Code`, `Amount Paid`, `Next Action`, ...) |
| `accounts-list` | String | JSON array of UIDs |
| `ivr_drift:{payer_name}` | Sorted Set | Score=timestamp, Value=JSON `{ts, call_sid, expected, heard}` |

### Call Status Values

`dialing` → `in-progress` → `completed` / `failed` / `disconnected` / `hold_timeout`

### CALL_RESULT Schema

| Field | Required | Type |
|-------|----------|------|
| `status` | yes | string |
| `payer` | yes | string |
| `claim_id` | yes | string |
| `next_action` | yes | string |
| `paid_amount` | no | float |
| `billed_amount` | no | float |
| `denial_code` | no | string |
| `denial_description` | no | string |
| `appeal_deadline` | no | string |
| `call_summary` | no | string |
| `call_duration_sec` | no | int |
| `satisfaction` | no | string |

---

## Configuration

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `WHISPER_MODEL_SIZE` | `distil-large-v3` | Faster-Whisper model |
| `WHISPER_DEVICE` | `cuda` | STT device |
| `WHISPER_COMPUTE` | `float16` | STT precision |
| `VLLM_BASE_URL` | `http://vllm:8001/v1` | LLM endpoint |
| `LLM_MODEL` | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | Active LLM model |
| `REDIS_URL` | `redis://redis:6379` | Redis connection |
| `TWILIO_ACCOUNT_SID` | `""` | Twilio integration |
| `TWILIO_AUTH_TOKEN` | `""` | Twilio integration |
| `TWILIO_FROM_NUMBER` | `""` | Twilio caller ID |
| `PUBLIC_DOMAIN` | `localhost:8080` | Webhook URL base |
| `PUBLIC_SCHEME` | `https` | Webhook URL scheme |
| `TTS_ENGINE` | `piper` | TTS backend |
| `PIPER_VOICE` | `en_US-lessac-medium` | Piper voice |
| `PIPER_SAMPLE_RATE` | `22050` | Piper output rate |
| `KOKORO_VOICE` | `af_bella` | Kokoro voice |
| `CHATTERBOX_SERVICE_URL` | `http://n-tts:8082` (VM) / `http://host.docker.internal:8084` (Mac) | Chatterbox HTTP endpoint |
| `CHATTERBOX_DEVICE` | `cuda` (VM) / `mps` (Mac) | Chatterbox compute device |
| `CHATTERBOX_MODEL_REPO` | `ResembleAI/chatterbox-turbo` | Chatterbox model |
| `HF_TOKEN` | `""` | HuggingFace download auth |
| `MAX_HOLD_SEC` | `1800` | Max hold timeout |

### LLM Model Options

| UI Label | Model ID |
|----------|----------|
| Qwen 2.5 7B | `Qwen/Qwen2.5-7B-Instruct-AWQ` |
| Llama 3.1 8B | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` |

---

## Local Mac Development

The `docker-compose.local.yml` stack is a CPU/Apple-Silicon mirror of the VM:
no CUDA, no vLLM, no Twilio. LLM comes from host Ollama (`ar-agent`, a
`llama3.1:8b` derived model); Opik runs in full (localhost:5173); Chatterbox
runs natively on MPS.

**Prereqs:** Docker Desktop, Ollama with `llama3.1:8b`.

```bash
# one-time: build the Chatterbox venv (torch arm64 + git-master chatterbox-tts)
./chatterbox_service/setup-chatterbox-mac.sh
# start everything (Ollama model check + native Chatterbox + Docker stack)
./start-local.sh
```

| Service | URL | Notes |
|---------|-----|-------|
| Admin UI | http://localhost:3000 | Next.js frontend |
| Backend | http://localhost:8080 | `/health`, `/api/config`, WS `/ws/{sid}` |
| Redis | redis://localhost:6379 | |
| Opik | http://localhost:5173 | full stack incl. ClickHouse/MySQL/ZooKeeper |
| Opik API | http://localhost:8082 | |
| Chatterbox | http://127.0.0.1:8084 | native `chatterbox_service/app_mac.py` |

### Chatterbox on the Mac: use NANO, not Turbo

On MPS, the Turbo model in fp32 is 12.3 GB of Metal buffers and its generation
time is unpredictable (3–80 s for the same sentence — thermal throttling), which
starves the browser WS and kills calls mid-greeting. **Nano is the local choice:**

| | Turbo (fp32, MPS) | Nano (MPS) |
|---|---|---|
| Full greeting w/ claim number | 13–75 s | ~3 s |
| First audio (browser WS) | 45 s+ | ~2 s |
| Physical footprint | 12.3 GB | ~7.8 GB |

`run-mac.sh` defaults `CHATTERBOX_NANO=1`. fp16 does NOT work on MPS: the
library's s3gen flow decoder mixes fp32 (`torch.zeros`) with fp16 weights and
Metal rejects it (`mps.add` requires matching dtypes) — leave Turbo in fp32.

**Keep Turbo on the VM.** The T4 handles fp32 fine and its quality is better
than Nano; Nano exists only to make local dev usable. `chatterbox_service/app_mac.py`
uses the `ChatterboxTurboTTS` class for both — `nano=True` loads the smaller
`ResembleAI/chatterbox-nano` weights.

### Resource footprint (why local can thrash)

The full local stack is heavy for 18 GB Macs: voice-agent holds Whisper
(~2.4 GB), Opik's ClickHouse ~1.9 GB, and the native Chatterbox ~7.8 GB of
Metal buffers. Together with the Docker VM overhead this routinely exceeds
18 GB → macOS compresses ~9 GB and swaps ~6 GB, which stalls MPS TTS.
If local TTS feels slow: stop the idle Opik stack
(`docker compose -f docker-compose.local.yml stop opik-*`) to reclaim ~2.6 GB.

---

## Infrastructure

### Azure Resources

| Resource | Configuration |
|----------|---------------|
| VM | `Standard_NC4as_T4_v3` — 4 vCPU, 28 GB RAM, 16 GB T4 GPU |
| Location | South India |
| OS | Ubuntu 22.04 LTS |
| OS Disk | 128 GB |
| Public IP | Static, `20.219.70.196` |
| Network | VNet `vnet-voice` (10.0.0.0/16), subnet `subnet-voice` (10.0.1.0/24), NSG with ports 22/80/443/8080 |
| Managed Identity | OID `28648ac4-96c1-4834-b048-b5e02bc991ba` with `Virtual Machine Contributor` role |

### Auto-Deactivation

Cron runs every 10 minutes. Deallocates VM after 1 hour of no real call activity.

```
*/10 * * * * /home/azureuser/ar-voice-agent/auto-deactivate.sh
```

Activity check: counts HTTP requests to call-related endpoints (`/voice`, `/media/`, `/make-call`, `/call-result`, `/ws/`, `/health`) in docker logs. Dashboard auto-refresh (`/api/calls`, `/api/accounts`) and bot scans are ignored.

### Docker

Containers run via `docker compose --profile prod`. GPU access via NVIDIA Container Toolkit.

---

## LLM Switching

The dashboard LLM selector triggers:
1. Writes new model to `/project/.env` (host `.env` mounted at `/project`)
2. Updates `state["llm_model"]` in-memory
3. Restarts `vllm` container via Docker socket (`docker restart vllm`)
4. UI polls `/api/current-llm` until `switching=false`

Requires Docker socket mount and `docker.io` in the voice-agent container.

---

## File Map

| File | Lines | Purpose |
|------|-------|---------|
| `server/` | ~2900 | FastAPI app package: config, state, TTS streaming, routes/ (telephony, accounts, chat, stats, misc, dashboard, browser voice) |
| `call_session.py` | 519 | Per-call state machine, Twilio protocol, STT/LLM/TTS orchestration |
| `prompts.py` | 261 | Prompt builder, marker parser, CALL_RESULT validator + repair |
| `audio.py` | 113 | μ-law codec, resampling, VAD |
| `docker-compose.yml` | 130 | 4-service orchestration with GPU, profiles, volume mounts |
| `Dockerfile.server` | 23 | Prod image with pip deps, Piper voice, docker CLI |
| `Dockerfile.dev` | 16 | Dev image with hot reload |
| `requirements.txt` | 12 | Python dependencies |
| `auto-deactivate.sh` | 24 | VM auto-deactivation cron script |
| `Caddyfile` | 3 | TLS reverse proxy config |
| `payers/Aetna.json` | 54 | Aetna IVR tree (4 levels, 6 menu nodes) |
| `payers/denial_codes.json` | >150 | 58 HIPAA denial codes (CO/PR/OA) |

---

## Comparison: Cloudflare Workers vs On-Prem

| Aspect | Cloudflare (`src/`) | On-Prem (`deploy/`) |
|--------|-------------------|-------------------|
| STT | Deepgram `nova-2-medical` (cloud API) | Faster-Whisper `distil-large-v3` (local GPU) |
| LLM | OpenAI GPT-4o-mini / Workers AI Llama 3.1 | vLLM Llama 3.1 8B / Qwen 2.5 7B (local GPU) |
| TTS | Cartesia `sonic-english` (cloud API) | Piper / Kokoro (local CPU) |
| DB | Upstash Redis (cloud) | Redis container (local) |
| Call result | `[END:status:payer:claim:amount:action]` (positional) | `[CALL_RESULT] {...}` (JSON with validation) |
| IVR | Basic DTMF regex | Full state machine + payer KB + drift detection |
| Hold | Basic wait detection | Polling + timeout + separate $2 credit |
| Concurrency | Durable Objects (isolated per call) | asyncio tasks (single process) |
| Deployment | `wrangler deploy` | Docker compose on Azure VM |
| Audio codec | Handled by Deepgram/Cartesia | Manual μ-law ↔ PCM in `audio.py` |
