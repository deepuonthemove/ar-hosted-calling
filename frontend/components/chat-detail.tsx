"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { ArrowLeft } from "lucide-react";

interface ChatTurn { role: "user" | "assistant"; text: string; ts?: number }

export function ChatDetail({ accountUid }: { accountUid: string }) {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [err, setErr] = useState("");

  useEffect(() => {
    api<ChatTurn[]>(`/api/chat/${accountUid}`).then(setTurns).catch((e) => setErr(String(e)));
  }, [accountUid]);

  return (
    <div className="space-y-6">
      <Link href="/" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="h-4 w-4" /> Back
      </Link>
      <div>
        <h1 className="text-2xl font-bold">Chat Details</h1>
        <p className="text-sm text-muted-foreground font-mono">Account: {accountUid}</p>
      </div>
      {err && <div className="text-sm text-red-600">{err}</div>}

      <Card>
        <CardHeader>
          <CardTitle>Chat History</CardTitle>
          <CardDescription>{turns.length} messages</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          {turns.length === 0 ? (
            <p className="text-sm text-muted-foreground">No chat messages for this account yet.</p>
          ) : (
            turns.map((t, i) => (
              <div key={i} className={`rounded-md px-3 py-2 text-sm ${t.role === "assistant" ? "bg-primary/10" : "bg-muted"}`}>
                <div className="mb-1 flex items-center gap-2">
                  <span className={`text-xs font-semibold ${t.role === "assistant" ? "text-primary" : "text-muted-foreground"}`}>
                    {t.role === "assistant" ? "AI" : "YOU"}
                  </span>
                  {t.ts && <span className="text-[10px] text-muted-foreground">{new Date(t.ts * 1000).toLocaleString()}</span>}
                </div>
                {t.text}
              </div>
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}
