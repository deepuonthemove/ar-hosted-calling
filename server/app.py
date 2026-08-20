"""AR Voice Agent — Azure on-prem edition.

Full port of the Cloudflare Workers app (src/index.ts + src/do.ts):
  STT  Deepgram nova-2-medical  → Faster-Whisper large-v3 (local GPU)
  LLM  Workers AI / Azure OpenAI → vLLM Llama 3.1 8B (local GPU)
  TTS  Cartesia sonic-english   → Piper (local CPU)
  DB   Upstash Redis            → local Redis container

This module wires the FastAPI app together from the route modules in
server/routes/ — it holds no route logic itself.
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import accounts, browser_ws, chat, dashboard, misc, stats, telephony
from .state import load_models, state


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    await state["redis"].aclose()


app = FastAPI(lifespan=lifespan)

# Local dev: the admin UI (http://localhost:3000) fetches the API cross-origin.
# In prod Caddy routes everything under one origin, so CORS is a no-op there.
_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router_module in (telephony, stats, accounts, chat, misc, dashboard, browser_ws):
    app.include_router(router_module.router)
