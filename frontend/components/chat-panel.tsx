"use client";

import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Send, Square } from "lucide-react";

interface ChatTurn { role: "user" | "assistant"; text: string; ts?: number }

export function ChatPanel({ accountUid, onEnded }: { accountUid: string; onEnded?: () => void }) {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const listRef = useRef<HTMLDivElement>(null);

  const load = () => api<ChatTurn[]>(`/api/chat/${accountUid}`).then(setTurns).catch(() => {});
  useEffect(() => { load(); }, [accountUid]);

  // Scroll only the chat container (not the page) when turns change
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns]);

  const send = async () => {
    if (!msg.trim() || busy) return;
    setBusy(true);
    setErr("");
    const text = msg.trim();
    setMsg("");
    setTurns((t) => [...t, { role: "user", text }]);
    try {
      const d = await api<{ reply: string; ended?: boolean }>("/api/chat", {
        method: "POST",
        body: JSON.stringify({ account_uid: accountUid, message: text }),
      });
      if (d.ended) {
        setTurns([]);
        onEnded?.();
      } else if (d.reply) {
        setTurns((t) => [...t, { role: "assistant", text: d.reply }]);
      }
    } catch (e: any) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const endChat = async () => {
    try {
      await api(`/api/chat/${accountUid}/end`, { method: "POST" });
      setTurns([]);
      onEnded?.();
    } catch (e: any) {
      setErr(String(e));
    }
  };

  return (
    <div className="flex flex-col space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-xs text-muted-foreground">
          Direct chat with the LLM — bypasses STT/TTS, isolated from calls & review.
        </div>
        <Button size="sm" variant="destructive" onClick={endChat} disabled={turns.length === 0}>
          <Square className="mr-1 h-3.5 w-3.5" /> End Chat
        </Button>
      </div>

      <div ref={listRef} className="h-72 space-y-2 overflow-y-auto rounded-md border p-3">
        {turns.length === 0 ? (
          <p className="text-sm text-muted-foreground">No active chat. Ask anything about this claim.</p>
        ) : (
          turns.map((t, i) => (
            <div key={i} className={`rounded-md px-3 py-2 text-sm ${t.role === "assistant" ? "bg-primary/10" : "bg-muted"}`}>
              <span className={`mr-2 text-xs font-semibold ${t.role === "assistant" ? "text-primary" : "text-muted-foreground"}`}>
                {t.role === "assistant" ? "AI" : "YOU"}
              </span>
              {t.text}
            </div>
          ))
        )}
      </div>

      {err && <div className="text-sm text-red-600">{err}</div>}

      <div className="flex items-end gap-2">
        <Textarea
          value={msg}
          onChange={(e) => setMsg(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
          placeholder="Type a message to the AI (Enter to send)..."
          className="min-h-[56px]"
        />
        <Button onClick={send} disabled={busy || !msg.trim()}>
          <Send className="mr-1 h-4 w-4" /> {busy ? "..." : "Send"}
        </Button>
      </div>
    </div>
  );
}
