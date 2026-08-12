"""AR Voice Agent — Azure on-prem edition.

Full port of the Cloudflare Workers app (src/index.ts + src/do.ts):
  STT  Deepgram nova-2-medical  → Faster-Whisper large-v3 (local GPU)
  LLM  Workers AI / Azure OpenAI → vLLM Llama 3.1 8B (local GPU)
  TTS  Cartesia sonic-english   → Piper (local CPU)
  DB   Upstash Redis            → local Redis container

This module wires the FastAPI app together from the route modules in
server/routes/ — it holds no route logic itself.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .routes import accounts, browser_ws, chat, dashboard, misc, stats, telephony
from .state import load_models, state


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    await state["redis"].aclose()


app = FastAPI(lifespan=lifespan)

for router_module in (telephony, stats, accounts, chat, misc, dashboard, browser_ws):
    app.include_router(router_module.router)
