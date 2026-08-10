"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, statusVariant, type CallDetail } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, ChevronDown } from "lucide-react";

export function CallDetail({ callId }: { callId: string }) {
  const [detail, setDetail] = useState<CallDetail | null>(null);
  const [err, setErr] = useState("");
  const [showPrompt, setShowPrompt] = useState(false);

  // Auto-refresh until offline fields (full_audio) finish populating
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const fetchDetail = async () => {
      try {
        const d = await api<CallDetail>(`/api/calls/${callId}/detail`);
        if (cancelled) return;
        setDetail(d);
        const pending =
          d.review && d.review.audio_size_bytes && !d.review.full_audio;
        if (pending) timer = setTimeout(fetchDetail, 5000);
      } catch (e: any) {
        if (!cancelled) setErr(String(e));
      }
    };
    fetchDetail();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, [callId]);

  const pending = (v?: any) =>
    v === undefined || v === null || v === "" || v === 0 ? "⏳ In progress..." : String(v);

  if (err) return <div className="text-sm text-red-600">{err}</div>;
  if (!detail) return <div className="text-sm text-muted-foreground">Loading call...</div>;

  const c = detail.call;

  return (
    <div className="space-y-6">
      <Link href="/review" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="h-4 w-4" /> Back to Review
      </Link>
      <div className="flex items-center gap-3">
        <h1 className="text-2xl font-bold font-mono">{callId}</h1>
        <Badge variant={statusVariant(c.status)}>{c.status}</Badge>
        {detail.twilio_sid && <span className="text-xs text-muted-foreground font-mono">Twilio: {detail.twilio_sid}</span>}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Call Config</CardTitle>
            <CardDescription>STT / TTS / VAD / LLM used for this call</CardDescription>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-2 gap-3 text-sm">
              {Object.entries(detail.config || {}).map(([k, v]) => (
                <div key={k}>
                  <dt className="text-xs text-muted-foreground">{k}</dt>
                  <dd className="font-mono text-xs">{v}</dd>
                </div>
              ))}
              <div>
                <dt className="text-xs text-muted-foreground">Payer</dt>
                <dd>{c.payer || "—"}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Claim ID</dt>
                <dd>{c.claim_id || "—"}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Duration</dt>
                <dd>{Math.round((Number(c.duration_ms) || 0) / 1000)}s</dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Next Action</dt>
                <dd>{c.next_action || "—"}</dd>
              </div>
            </dl>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Outcome</CardTitle>
            <CardDescription>Structured call result</CardDescription>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-2 gap-3 text-sm">
              {["reference_number", "denial_code", "denial_description", "paid_amount", "billed_amount", "appeal_deadline", "call_summary"].map((k) => (
                <div key={k} className={k === "call_summary" ? "col-span-2" : ""}>
                  <dt className="text-xs text-muted-foreground">{k}</dt>
                  <dd>{c[k] || "—"}</dd>
                </div>
              ))}
            </dl>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Timing (Time To Respond)</CardTitle>
          <CardDescription>Component latencies for this call (per-response averages)</CardDescription>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
            <div>
              <dt className="text-xs text-muted-foreground">TTR</dt>
              <dd className="text-lg font-bold">{c.ttr_avg_ms ? `${(Number(c.ttr_avg_ms) / 1000).toFixed(2)}s` : "—"}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted-foreground">STT</dt>
              <dd>{c.stt_avg_ms ? `${Math.round(Number(c.stt_avg_ms))}ms` : "—"}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted-foreground">LLM</dt>
              <dd>{c.llm_avg_ms ? `${Math.round(Number(c.llm_avg_ms))}ms` : "—"}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted-foreground">TTS (first byte)</dt>
              <dd>{c.tts_avg_ms ? `${Math.round(Number(c.tts_avg_ms))}ms` : "—"}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted-foreground">Context (peak / limit)</dt>
              <dd>{c.peak_prompt_tokens ? `${Number(c.peak_prompt_tokens).toLocaleString()} / ${c.context_limit || "4096"}` : "—"}</dd>
            </div>
            <div>
              <dt className="text-xs text-muted-foreground">Completion tokens</dt>
              <dd>{c.total_completion_tokens ? `${Number(c.total_completion_tokens).toLocaleString()}` : "—"}</dd>
            </div>
          </dl>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Transcript</CardTitle>
          <CardDescription>Interleaved conversation (user = rep, assistant = bot)</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          {!detail.transcript || detail.transcript.length === 0 ? (
            <p className="text-sm text-muted-foreground">No transcript recorded.</p>
          ) : (
            detail.transcript.map((t, i) => (
              <div key={i} className={`rounded-md px-3 py-2 text-sm ${t.role === "assistant" ? "bg-primary/10" : "bg-muted"}`}>
                <span className={`mr-2 text-xs font-semibold ${t.role === "assistant" ? "text-primary" : "text-muted-foreground"}`}>
                  {t.role === "assistant" ? "BOT" : "REP"}
                </span>
                {t.text}
              </div>
            ))
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Prompt Used</CardTitle>
          <CardDescription>System prompt injected for this call</CardDescription>
        </CardHeader>
        <CardContent>
          <button onClick={() => setShowPrompt((v) => !v)} className="mb-2 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
            <ChevronDown className={`h-4 w-4 transition-transform ${showPrompt ? "rotate-180" : ""}`} />
            {showPrompt ? "Hide" : "Show"} prompt
          </button>
          {showPrompt && detail.prompt && <pre className="whitespace-pre-wrap rounded-md bg-muted p-3 text-xs">{detail.prompt}</pre>}
        </CardContent>
      </Card>

      {detail.review && (
        <Card>
          <CardHeader>
            <CardTitle>Review</CardTitle>
            <CardDescription>Real-time STT vs full recording STT</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <div className="mb-1 text-xs font-semibold text-blue-600">🔴 Real-Time (with VAD)</div>
                <pre className="whitespace-pre-wrap rounded-md bg-muted p-3 text-xs">
                  {(detail.review.real_time || []).join("\n") || "(no speech detected)"}
                </pre>
              </div>
              <div>
                <div className="mb-1 text-xs font-semibold text-green-600">🟢 Full Recording (no VAD)</div>
                {detail.review.audio_size_bytes && !detail.review.full_audio ? (
                  <div className="rounded-md bg-muted p-3 text-xs text-muted-foreground animate-pulse">
                    ⏳ Offline transcription in progress — refreshing automatically...
                  </div>
                ) : (
                  <pre className="whitespace-pre-wrap rounded-md bg-muted p-3 text-xs">
                    {detail.review.full_audio || "(no speech detected)"}
                  </pre>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
