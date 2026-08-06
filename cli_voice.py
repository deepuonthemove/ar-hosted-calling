#!/usr/bin/env python3
"""CLI voice-agent client — mic → WebSocket → TTS back to speakers.

Usage:
    pip install websockets sounddevice numpy
    python cli_voice.py --url wss://call.ar-voice.com/ws/cli_test_123?tts=piper&vad=silero
    # or against the VM directly:
    python cli_voice.py --url ws://localhost:8080/ws/cli_test_123?tts=piper&vad=silero
"""
import argparse
import asyncio
import json

import numpy as np
import sounddevice as sd
import websockets

MIC_RATE = 16000          # what the server expects from the browser
SPK_RATE = 22050          # server TTS output rate (matches PIPER_RATE)
CHUNK = 4096


async def play_worker(q: asyncio.Queue, stop: asyncio.Event):
    """Play int16 PCM (SPK_RATE) frames; stop on \x02."""
    def cb(outdata, frames, time, status):
        try:
            pcm = q.get_nowait()
            if pcm.size < outdata.shape[0]:
                outdata[:pcm.size] = pcm.reshape(-1, 1)
                outdata[pcm.size:] = 0
            else:
                outdata[:] = pcm[:outdata.shape[0]].reshape(-1, 1)
        except asyncio.QueueEmpty:
            outdata.fill(0)
    with sd.OutputStream(samplerate=SPK_RATE, channels=1, dtype='int16',
                         callback=cb, blocksize=4096):
        while not stop.is_set():
            await asyncio.sleep(0.05)


async def main(url: str):
    stop = asyncio.Event()
    q = asyncio.Queue()

    async with websockets.connect(url) as ws:
        asyncio.create_task(play_worker(q, stop))

        def mic_cb(indata, frames, time, status):
            asyncio.get_event_loop().call_soon_threadsafe(
                ws.send, indata.astype(np.int16).tobytes())

        with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype='int16',
                            callback=mic_cb, blocksize=CHUNK):
            print("🎤 Connected — speak. Ctrl+C to quit.")
            try:
                async for raw in ws:
                    if isinstance(raw, str):
                        m = json.loads(raw)
                        if m.get("type") in ("llm_text", "transcript"):
                            tag = "🤖" if m["type"] == "llm_text" else "🧑"
                            print(f"{tag} {m.get('text','')}")
                    else:
                        v = np.frombuffer(raw, dtype=np.uint8)
                        if len(v) and v[0] == 1:      # audio frame
                            q.put_nowait(v[1:].view(np.int16).copy())
                        elif len(v) and v[0] == 2:    # stop playback (barge-in)
                            q = asyncio.Queue()
            finally:
                stop.set()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", default="", help="account_uid loaded from Redis (sets claim context)")
    ap.add_argument("--project", default="", help="project id from excel upload")
    ap.add_argument("--row", type=int, default=0, help="row number within the project (1 = first data row)")
    ap.add_argument("--tts", default="piper", choices=["piper", "kokoro"])
    ap.add_argument("--vad", default="silero", choices=["silero", "rms"])
    ap.add_argument("--host", default="call.ar-voice.com", help="server host")
    args = ap.parse_args()
    params = f"tts={args.tts}&vad={args.vad}"
    if args.uid:
        params += f"&account_uid={args.uid}"
    if args.project and args.row:
        params += f"&project_id={args.project}&row_num={args.row}"
    ident = args.uid or f"{args.project}-{args.row}"
    url = f"wss://{args.host}/ws/cli_{abs(hash(ident))%10**10}?{params}"
    asyncio.run(main(url))
