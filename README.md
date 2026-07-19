# AR Voice Agent — Azure On-Prem Edition

Full port of the Cloudflare Workers stack to self-hosted Azure GPU VM.

## Feature Parity with Cloudflare Version

| Feature | Cloudflare (src/) | Azure (deploy/) |
|---|---|---|
| STT | Deepgram nova-2-medical | Faster-Whisper distil-large-v3 (GPU) |
| LLM | Workers AI / Azure OpenAI | vLLM Llama 3.1 8B (GPU, batched) |
| TTS | Cartesia sonic-english | Piper en_US-lessac (CPU) |
| State | Upstash Redis | Local Redis container |
| Telephony | Twilio Media Streams | Twilio Media Streams (same) |
| DTMF markers `[DTMF:1]` | ✅ | ✅ |
| Hold detection `[WAITING]` | ✅ | ✅ |
| End marker `[END:...]` | ✅ | ✅ |
| Barge-in (clear buffer) | ✅ | ✅ |
| 19-min silence cap | ✅ | ✅ |
| Account context in prompt | ✅ | ✅ |
| Patient greeting | ✅ | ✅ |
| Excel upload/export | ✅ | ✅ |
| Dashboard + live feed | ✅ | ✅ |
| CSV export | ✅ | ✅ |
| Call retry | ✅ | ✅ |
| TLS termination | Cloudflare edge | Caddy (Let's Encrypt) |

## Architecture

```
Insurance Co ←PSTN→ Twilio ←WSS μ-law→ Caddy (TLS) → voice-agent (FastAPI)
                                                        ├─ CallSession per call
                                                        ├─ Faster-Whisper (GPU)
                                                        ├─ vLLM client ──→ vLLM container (GPU)
                                                        ├─ Piper TTS (CPU)
                                                        └─ Redis ────────→ redis container
```

## Files

```
server.py        FastAPI app: /voice, /media, /make-call, /dashboard, Excel APIs
call_session.py  Per-call pipeline (the Durable Object port)
audio.py         μ-law codec, resampling, VAD
prompts.py       System prompts + marker parsing (from do.ts)
```

## Quick Start

```bash
# 1. Infra (Pulumi)
cd infra && pulumi up

# 2. Configure
cp ../.env.example ../.env   # fill in Twilio creds + PUBLIC_DOMAIN

# 3. Point DNS A record: voice.yourcompany.com → VM public IP

# 4. Deploy
cd .. && ./deploy.sh prod    # or dev for hot-reload

# 5. Twilio console: number voice webhook → https://voice.yourcompany.com/voice
#    (handled automatically by /make-call via REST API)

# 6. Open dashboard
open https://voice.yourcompany.com/dashboard
```

## Dev Mode (hot reload)

```bash
./deploy.sh dev
ssh azureuser@$(pulumi stack output public_ip) \
  'cd /opt/ar-voice-agent/deploy && docker compose --profile dev logs -f voice-agent-dev'
```

## Cost (10 concurrent calls, NC4as_T4_v3)

| Schedule | Monthly |
|---|---|
| 8h weekday | ~$55 |
| 24/7 reserved | ~$158 |
| 24/7 on-demand | ~$217 |

Twilio usage separate (~$0.013/min). For HIPAA: Twilio BAA requires
Enterprise plan — or swap to Azure Communication Services (BAA included
with your Azure BAA).
