"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { api, type CallRecord } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { CallFlow } from "@/components/call-flow";
import { ArrowLeft, PlusCircle } from "lucide-react";

export function ProjectRowView({ projectId, rowNum }: { projectId: string; rowNum: number }) {
  const [acct, setAcct] = useState<any>(null);
  const [history, setHistory] = useState<CallRecord[]>([]);
  const [note, setNote] = useState("");
  const [err, setErr] = useState("");
  const [callActive, setCallActive] = useState(false);
  const uidRef = useRef("");

  const load = async () => {
    try {
      const d = await api<{ account_uid: string; account: any }>(`/api/projects/${projectId}/accounts/${rowNum}`);
      setAcct(d.account);
      uidRef.current = d.account_uid;
      api<CallRecord[]>(`/api/accounts/${d.account_uid}/calls`).then(setHistory).catch(() => {});
    } catch (e: any) { setErr(String(e)); }
  };
  useEffect(() => { load(); }, [projectId, rowNum]);

  // While a call is active, poll history so the completed call appears automatically
  useEffect(() => {
    if (!callActive) return;
    const t = setInterval(async () => {
      if (uidRef.current) {
        try {
          const h = await api<CallRecord[]>(`/api/accounts/${uidRef.current}/calls`);
          setHistory(h);
        } catch {}
      }
    }, 4000);
    return () => clearInterval(t);
  }, [callActive]);

  const addNote = async () => {
    if (!note.trim()) return;
    try {
      await api("/api/notes", { method: "POST", body: JSON.stringify({ project_id: projectId, row_num: rowNum, note }) });
      setNote("");
      load();
    } catch (e: any) { setErr(String(e)); }
  };

  if (!acct) return <div className="text-sm text-muted-foreground">{err || "Loading account..."}</div>;

  const fields = [
    ["Patient Name", acct["Patient Name"]],
    ["Date of Service", acct["DOS"]],
    ["CPT", acct["CPT"]],
    ["Billed Amount", acct["Billed Amount"]],
    ["Responsible Payer", acct["Responsible Payer"]],
    ["Account Number", acct["Account Number"]],
    ["Claim ID", acct["Claim ID"]],
    ["AR Final Comments", acct["AR Final Comments"]],
    ["Call Status", acct["Call Status"]],
    ["Denial Code", acct["Denial Code"]],
  ].filter(([, v]) => v !== undefined && v !== null && v !== "");

  return (
    <div className="space-y-6">
      <Link href={`/projects/${projectId}`} className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="h-4 w-4" /> Back to {projectId}
      </Link>

      <div>
        <h1 className="text-2xl font-bold">{acct["Patient Name"] || "Account"}</h1>
        <p className="text-sm text-muted-foreground font-mono">{projectId} · row {rowNum} · UID {acct["UID"]}</p>
      </div>
      {err && <div className="text-sm text-red-600">{err}</div>}

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Account Details</CardTitle>
              <CardDescription>Claim context used for calls</CardDescription>
            </CardHeader>
            <CardContent>
              <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
                {fields.map(([k, v]) => (
                  <div key={String(k)}>
                    <dt className="text-xs text-muted-foreground">{k}</dt>
                    <dd className="font-medium">{String(v)}</dd>
                  </div>
                ))}
              </dl>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Notes</CardTitle>
              <CardDescription>Added to the call context on the next call</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {acct["Notes"] ? (
                <pre className="whitespace-pre-wrap rounded-md bg-muted p-3 text-xs">{acct["Notes"]}</pre>
              ) : (
                <p className="text-sm text-muted-foreground">No notes yet.</p>
              )}
              <div className="space-y-2">
                <Label>Add note</Label>
                <Textarea value={note} onChange={(e) => setNote(e.target.value)} placeholder="e.g. Rep said resubmit with corrected ICD code..." />
                <Button size="sm" onClick={addNote}>
                  <PlusCircle className="mr-1 h-4 w-4" /> Add Note
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <CallFlow projectId={projectId} rowNum={rowNum} accountUid={acct["UID"] || ""} onActiveChange={setCallActive} />

          <Card>
            <CardHeader>
              <CardTitle>Call History</CardTitle>
              <CardDescription>All calls for this claim</CardDescription>
            </CardHeader>
            <CardContent>
              {history.length === 0 ? (
                <p className="text-sm text-muted-foreground">No calls yet.</p>
              ) : (
                <div className="space-y-2">
                  {history.map((c) => {
                    const id = c.call_id || c.callSid || "";
                    return (
                      <Link key={id} href={`/calls/${id}`} className="flex items-center justify-between rounded-md border px-3 py-2 hover:bg-accent">
                        <div className="flex items-center gap-2">
                          <Badge variant={c.status === "completed" ? "success" : "secondary"}>{c.status}</Badge>
                          <span className="font-mono text-xs">{id}</span>
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {c.started_at ? new Date(Number(c.started_at) * 1000).toLocaleString() : ""}
                        </div>
                      </Link>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
