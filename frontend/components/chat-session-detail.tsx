"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft } from "lucide-react";

interface Turn { role: "user" | "assistant"; text: string; ts?: number }
interface SessionDetail {
  session_id: string;
  meta: { account_uid: string; started_at: number; ended_at: number; count: number; preview: string };
  turns: Turn[];
  account: Record<string, string>;
  config: { llm_model: string; prompt: string };
  timing: { llm_avg_ms: number; llm_count: number; ttr_avg_ms: number };
}

export function ChatSessionDetail({ sessionId }: { sessionId: string }) {
  const [d, setD] = useState<SessionDetail | null>(null);
  const [err, setErr] = useState("");
  const [showPrompt, setShowPrompt] = useState(false);

  useEffect(() => {
    api<SessionDetail>(`/api/chat/session/${sessionId}`).then(setD).catch((e) => setErr(String(e)));
  }, [sessionId]);

  if (err) return <div className="text-sm text-red-600">{err}</div>;
  if (!d) return <div className="text-sm text-muted-foreground">Loading chat...</div>;

  return (
    <div className="space-y-6">
      <Link href="/" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="h-4 w-4" /> Back
      </Link>

      <div className="flex items-center gap-3">
        <h1 className="text-2xl font-bold">Chat Session</h1>
        <Badge variant="secondary">{d.meta.count} messages</Badge>
        <span className="font-mono text-xs text-muted-foreground">{sessionId}</span>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Chat Config</CardTitle>
            <CardDescription>LLM used for this chat (no STT/TTS)</CardDescription>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <dt className="text-xs text-muted-foreground">LLM Model</dt>
                <dd className="font-mono text-xs">{d.config.llm_model || "—"}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Messages</dt>
                <dd>{d.meta.count}</dd>
              </div>
            </dl>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Timing (Time To Respond)</CardTitle>
            <CardDescription>LLM latency for this chat (isolated — not in dashboard averages)</CardDescription>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-3 gap-3 text-sm">
              <div>
                <dt className="text-xs text-muted-foreground">TTR</dt>
                <dd className="text-lg font-bold">
                  {d.timing.ttr_avg_ms ? `${(Number(d.timing.ttr_avg_ms) / 1000).toFixed(2)}s` : "—"}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Avg LLM</dt>
                <dd>{d.timing.llm_avg_ms ? `${Math.round(Number(d.timing.llm_avg_ms))}ms` : "—"}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Turns</dt>
                <dd>{d.timing.llm_count || "—"}</dd>
              </div>
            </dl>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Prompt Used</CardTitle>
          <CardDescription>System prompt injected for this chat</CardDescription>
        </CardHeader>
        <CardContent>
          <button onClick={() => setShowPrompt((v) => !v)} className="mb-2 text-xs text-muted-foreground hover:text-foreground">
            {showPrompt ? "Hide" : "Show"} prompt
          </button>
          {showPrompt && d.config.prompt && (
            <pre className="max-h-96 overflow-auto whitespace-pre-wrap rounded-md bg-muted p-3 text-xs">{d.config.prompt}</pre>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Account</CardTitle>
            <CardDescription>Claim context for this chat</CardDescription>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-2 gap-3 text-sm">
              {Object.entries(d.account).filter(([, v]) => v).map(([k, v]) => (
                <div key={k} className={k.includes("Comments") ? "col-span-2" : ""}>
                  <dt className="text-xs text-muted-foreground">{k}</dt>
                  <dd className="font-medium text-xs">{v}</dd>
                </div>
              ))}
            </dl>
            <div className="mt-3 border-t pt-3 text-xs text-muted-foreground">
              <div>Started: {new Date(d.meta.started_at * 1000).toLocaleString()}</div>
              <div>Ended: {new Date(d.meta.ended_at * 1000).toLocaleString()}</div>
            </div>
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Conversation</CardTitle>
            <CardDescription>Direct LLM chat (no STT/TTS)</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {d.turns.length === 0 ? (
              <p className="text-sm text-muted-foreground">No messages in this session.</p>
            ) : (
              d.turns.map((t, i) => (
                <div key={i} className={`rounded-md px-3 py-2 text-sm ${t.role === "assistant" ? "bg-primary/10" : "bg-muted"}`}>
                  <div className="mb-1 flex items-center gap-2">
                    <span className={`text-xs font-semibold ${t.role === "assistant" ? "text-primary" : "text-muted-foreground"}`}>
                      {t.role === "assistant" ? "AI" : "YOU"}
                    </span>
                    {t.ts && <span className="text-[10px] text-muted-foreground">{new Date(t.ts * 1000).toLocaleTimeString()}</span>}
                  </div>
                  {t.text}
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
