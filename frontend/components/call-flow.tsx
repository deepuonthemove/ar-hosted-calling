"use client";

import { useEffect, useRef, useState } from "react";
import { Phone, MonitorPlay, Square } from "lucide-react";
import { api, wsUrl } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface Turn { role: "user" | "assistant"; text: string }

const MIC_RATE = 16000;
const SPK_RATE = 22050;
const CHUNK = 4096;

export function CallFlow({ projectId, rowNum, accountUid, onActiveChange }: {
  projectId: string; rowNum: number; accountUid: string;
  onActiveChange?: (active: boolean) => void;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [active, setActive] = useState<"browser" | "twilio" | null>(null);
  const [phone, setPhone] = useState("");
  const [callId, setCallId] = useState("");
  const [err, setErr] = useState("");
  const wsRef = useRef<WebSocket | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const playCtxRef = useRef<AudioContext | null>(null);

  const setActiveState = (v: "browser" | "twilio" | null) => {
    setActive(v);
    onActiveChange?.(!!v);
  };

  const stopAll = () => {
    if (wsRef.current) { try { wsRef.current.close(); } catch {} wsRef.current = null; }
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    if (playCtxRef.current) { try { playCtxRef.current.close(); } catch {} playCtxRef.current = null; }
    setActiveState(null);
  };

  // ── Browser call via WebSocket ──
  const startBrowser = async () => {
    stopAll();
    setTurns([]);
    setErr("");
    setActiveState("browser");

    // Create the playback AudioContext synchronously in the click gesture so the
    // browser doesn't suspend it (autoplay policy). No sound otherwise.
    let playCtx: AudioContext | null = null;
    try {
      playCtx = new AudioContext();
      await playCtx.resume();
    } catch (e) {
      setErr("Audio error: " + String(e));
      return;
    }
    playCtxRef.current = playCtx;

    const sid = `browser_${Math.random().toString(36).slice(2, 10)}`;
    const params = new URLSearchParams({ project_id: projectId, row_num: String(rowNum), tts: "piper", vad: "silero" });
    const ws = new WebSocket(wsUrl(`/ws/${sid}?${params}`));
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    let micStream: MediaStream | null = null;
    let src: MediaStreamAudioSourceNode | null = null;
    let proc: ScriptProcessorNode | null = null;

    ws.onopen = async () => {
      try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const ctx = new AudioContext({ sampleRate: MIC_RATE });
        src = ctx.createMediaStreamSource(micStream);
        proc = ctx.createScriptProcessor(CHUNK, 1, 1);
        proc.onaudioprocess = (e) => {
          if (ws.readyState !== WebSocket.OPEN) return;
          const inp = e.inputBuffer.getChannelData(0);
          const b = new Int16Array(inp.length);
          for (let i = 0; i < inp.length; i++) b[i] = Math.max(-32768, Math.min(32767, inp[i] * 32768));
          ws.send(b.buffer);
        };
        src.connect(proc); proc.connect(ctx.destination);
      } catch (e: any) { setErr("Mic error: " + e.message); }
    };

    let audioQ: ArrayBuffer[] = [];
    let playing = false;
    let currentSrc: AudioBufferSourceNode | null = null;

    const playNext = () => {
      if (!audioQ.length || !playCtx) { playing = false; return; }
      playing = true;
      const total = audioQ.reduce((s, c) => s + c.byteLength, 0);
      const pcm = new Int16Array(total / 2); let off = 0;
      while (audioQ.length) { const c = audioQ.shift()!; pcm.set(new Int16Array(c, 0, c.byteLength / 2), off); off += c.byteLength / 2; }
      const buf = playCtx.createBuffer(1, pcm.length, SPK_RATE);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
      const s = playCtx.createBufferSource();
      currentSrc = s;
      s.buffer = buf; s.connect(playCtx.destination);
      s.onended = () => { playing = false; currentSrc = null; if (audioQ.length) playNext(); };
      s.start();
    };

    ws.onmessage = (e) => {
      if (typeof e.data === "string") {
        const m = JSON.parse(e.data);
        if (m.type === "llm_text" && m.text) setTurns((t) => [...t, { role: "assistant", text: m.text }]);
        else if (m.type === "transcript" && m.text) setTurns((t) => [...t, { role: "user", text: m.text }]);
      } else {
        const v = new Uint8Array(e.data);
        if (v[0] === 1) { audioQ.push(v.slice(1).buffer.slice(0)); if (!playing) playNext(); }
        else if (v[0] === 2) { audioQ = []; if (playing && currentSrc) { playing = false; try { currentSrc.stop(); } catch {} } }
      }
    };

    ws.onclose = () => {
      if (micStream) micStream.getTracks().forEach((t) => t.stop());
      if (proc) { try { proc.disconnect(); } catch {} }
      if (src) { try { src.disconnect(); } catch {} }
      setActiveState(null);
    };
  };

  // ── Twilio call ──
  const startTwilio = async () => {
    stopAll();
    setTurns([]);
    setErr("");
    if (!phone) { setErr("Enter a phone number"); return; }
    setActiveState("twilio");
    try {
      const d = await api<{ call_id: string; callSid: string }>("/make-call", {
        method: "POST",
        body: JSON.stringify({ phone, project_id: projectId, row_num: rowNum }),
      });
      setCallId(d.call_id);
      const poll = async () => {
        try {
          const live = await api<{ transcript: Turn[]; active?: boolean }>(`/api/calls/${d.call_id}/live`);
          if (live.transcript?.length) setTurns(live.transcript);
          if (live.active === false) {
            if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
            setActiveState(null);
          }
        } catch {}
      };
      poll();
      pollRef.current = setInterval(poll, 2500);
    } catch (e: any) { setErr("Twilio error: " + String(e)); setActiveState(null); }
  };

  useEffect(() => () => stopAll(), []);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Call</span>
          {active && <span className="text-xs font-normal text-green-600 animate-pulse">● Live — {active}</span>}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-3">
          <Button onClick={startBrowser} disabled={!!active} variant="outline">
            <MonitorPlay className="mr-2 h-4 w-4" /> Call via Browser
          </Button>
          <div className="flex items-center gap-2">
            <input
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              placeholder="+1... (Twilio destination)"
              className="h-9 w-52 rounded-md border border-input bg-background px-3 text-sm"
            />
            <Button onClick={startTwilio} disabled={!!active}>
              <Phone className="mr-2 h-4 w-4" /> Call via Twilio
            </Button>
          </div>
          {active && (
            <Button onClick={stopAll} variant="destructive">
              <Square className="mr-2 h-4 w-4" /> End
            </Button>
          )}
        </div>
        {callId && <div className="text-xs text-muted-foreground font-mono">call_id: {callId}</div>}
        {err && <div className="text-sm text-red-600">{err}</div>}

        <div className="max-h-96 space-y-2 overflow-y-auto rounded-md border p-3">
          {turns.length === 0 ? (
            <p className="text-sm text-muted-foreground">No conversation yet. Start a call.</p>
          ) : (
            turns.map((t, i) => (
              <div key={i} className={`rounded-md px-3 py-2 text-sm ${t.role === "assistant" ? "bg-primary/10" : "bg-muted"}`}>
                <span className={`mr-2 text-xs font-semibold ${t.role === "assistant" ? "text-primary" : "text-muted-foreground"}`}>
                  {t.role === "assistant" ? "BOT" : "REP"}
                </span>
                {t.text}
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
}
