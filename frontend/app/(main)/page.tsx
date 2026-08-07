"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Phone, CheckCircle2, FolderKanban, Clock, Gauge } from "lucide-react";
import { api, type CallRecord, type Project } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function DashboardPage() {
  const [calls, setCalls] = useState<CallRecord[]>([]);
  const [stats, setStats] = useState<{ total_calls: number; completed: number; projects: number; total_duration_ms: number }>({
    total_calls: 0, completed: 0, projects: 0, total_duration_ms: 0,
  });
  const [err, setErr] = useState("");

  useEffect(() => {
    const load = () => {
      api<CallRecord[]>("/api/calls").then(setCalls).catch((e) => setErr(String(e)));
      api("/api/stats").then(setStats).catch(() => {});
    };
    load();
    const t = setInterval(load, 5000); // auto-refresh
    return () => clearInterval(t);
  }, []);

  const completed = stats.completed;
  const totalDur = stats.total_duration_ms;

  // TTR averages (ms) across calls that have timing data
  const avg = (k: string) => {
    const v = calls.map((c) => Number((c as any)[k])).filter((n) => !isNaN(n) && n > 0);
    return v.length ? Math.round(v.reduce((a, b) => a + b, 0) / v.length) : null;
  };
  const sttAvg = avg("stt_avg_ms");
  const llmAvg = avg("llm_avg_ms");
  const ttsAvg = avg("tts_avg_ms");
  const ttrAvg = avg("ttr_avg_ms");

  const stat = (label: string, value: string, icon: React.ReactNode, color: string) => (
    <Card>
      <CardContent className="flex items-center gap-4 p-6">
        <div className={`flex h-11 w-11 items-center justify-center rounded-lg ${color}`}>{icon}</div>
        <div>
          <div className="text-2xl font-bold">{value}</div>
          <div className="text-sm text-muted-foreground">{label}</div>
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <p className="text-sm text-muted-foreground">Overview of calling activity</p>
      </div>
      {err && <div className="text-sm text-red-600">{err}</div>}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {stat("Total Calls", String(stats.total_calls), <Phone className="h-5 w-5 text-primary" />, "bg-primary/10")}
        {stat("Completed", String(completed), <CheckCircle2 className="h-5 w-5 text-green-600" />, "bg-green-600/10")}
        {stat("Projects", String(stats.projects), <FolderKanban className="h-5 w-5 text-amber-600" />, "bg-amber-500/10")}
        {stat("Total Time", `${Math.round(totalDur / 60000)}m`, <Clock className="h-5 w-5 text-purple-600" />, "bg-purple-600/10")}
      </div>

      <div>
        <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-muted-foreground">
          <Gauge className="h-4 w-4" /> Time To Respond (average)
        </h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardContent className="flex items-center gap-4 p-6">
              <div className="flex h-11 w-11 items-center justify-center rounded-lg bg-blue-600/10">
                <Phone className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <div className="text-2xl font-bold">{ttrAvg !== null ? `${(ttrAvg / 1000).toFixed(2)}s` : "—"}</div>
                <div className="text-sm text-muted-foreground">TTR (STT+LLM+TTS)</div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6">
              <div className="text-lg font-bold">{sttAvg !== null ? `${Math.round(sttAvg)}ms` : "—"}</div>
              <div className="text-sm text-muted-foreground">Avg STT</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6">
              <div className="text-lg font-bold">{llmAvg !== null ? `${Math.round(llmAvg)}ms` : "—"}</div>
              <div className="text-sm text-muted-foreground">Avg LLM</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6">
              <div className="text-lg font-bold">{ttsAvg !== null ? `${Math.round(ttsAvg)}ms` : "—"}</div>
              <div className="text-sm text-muted-foreground">Avg TTS (first byte)</div>
            </CardContent>
          </Card>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent Calls</CardTitle>
          <CardDescription>Latest calls across all projects</CardDescription>
        </CardHeader>
        <CardContent>
          {calls.length === 0 ? (
            <p className="text-sm text-muted-foreground">No calls yet. Upload a project to get started.</p>
          ) : (
            <div className="space-y-2">
              {calls.slice(0, 10).map((c) => {
                const id = c.call_id || c.callSid || "";
                const statusOk = c.status === "completed";
                return (
                  <Link
                    key={id}
                    href={`/calls/${id}`}
                    className="flex items-center justify-between rounded-md border px-4 py-2.5 hover:bg-accent"
                  >
                    <div className="flex items-center gap-3">
                      <Badge variant={statusOk ? "success" : c.status === "dialing" ? "warning" : "secondary"}>
                        {c.status}
                      </Badge>
                      <span className="font-mono text-xs text-muted-foreground">{id}</span>
                    </div>
                    <div className="flex items-center gap-4 text-sm">
                      <span>{c.payer || "—"}</span>
                      <span className="text-muted-foreground">{Math.round((Number(c.duration_ms) || 0) / 1000)}s</span>
                    </div>
                  </Link>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
