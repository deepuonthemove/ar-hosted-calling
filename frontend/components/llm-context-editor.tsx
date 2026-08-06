"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Eye, Pencil, RotateCcw, ChevronDown } from "lucide-react";

interface Ctx { custom: string; original: string; effective: string }

export function LlmContextEditor({ accountUid }: { accountUid: string }) {
  const [ctx, setCtx] = useState<Ctx | null>(null);
  const [edit, setEdit] = useState(false);
  const [draft, setDraft] = useState("");
  const [show, setShow] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  const load = () =>
    api<Ctx>(`/api/accounts/${accountUid}/llm-context`).then(setCtx).catch(() => {});

  useEffect(() => { load(); }, [accountUid]);

  const save = async () => {
    try {
      await api(`/api/accounts/${accountUid}/llm-context`, {
        method: "POST",
        body: JSON.stringify({ context: draft }),
      });
      setEdit(false);
      setMsg({ ok: true, text: "Context updated — used by subsequent chats & calls." });
      load();
      setTimeout(() => setMsg(null), 3000);
    } catch (e: any) {
      setMsg({ ok: false, text: String(e) });
    }
  };

  const reset = async () => {
    try {
      await api(`/api/accounts/${accountUid}/llm-context`, {
        method: "POST",
        body: JSON.stringify({ reset: true }),
      });
      setEdit(false);
      setDraft("");
      setMsg({ ok: true, text: "Reset to original context." });
      load();
      setTimeout(() => setMsg(null), 3000);
    } catch (e: any) {
      setMsg({ ok: false, text: String(e) });
    }
  };

  const hasCustom = !!ctx?.custom;

  const toggleEdit = () => {
    setEdit((v) => {
      if (!v) setDraft(ctx?.custom || ctx?.original || "");
      return !v;
    });
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" variant="outline" onClick={toggleEdit}>
          <Pencil className="mr-1 h-3.5 w-3.5" /> {edit ? "Cancel" : hasCustom ? "Edit Context" : "Set Custom Context"}
        </Button>
        <Button size="sm" variant="outline" onClick={() => setShow((v) => !v)}>
          {show ? <ChevronDown className="mr-1 h-3.5 w-3.5 rotate-180" /> : <Eye className="mr-1 h-3.5 w-3.5" />}
          {show ? "Hide" : "Show"} Current Context
        </Button>
        {hasCustom && (
          <Button size="sm" variant="destructive" onClick={reset}>
            <RotateCcw className="mr-1 h-3.5 w-3.5" /> Reset to Original
          </Button>
        )}
      </div>

      {hasCustom && !edit && (
        <div className="rounded-md border border-amber-300/40 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          Custom LLM context is active for this account.
        </div>
      )}

      {show && ctx?.effective && (
        <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded-md bg-muted p-3 text-xs">{ctx.effective}</pre>
      )}

      {edit && (
        <div className="space-y-2">
          <Label>Custom LLM Context (system prompt)</Label>
          <Textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Paste the full system prompt to use for this account's chats and calls..."
            className="min-h-[160px] font-mono text-xs"
          />
          <div className="flex gap-2">
            <Button size="sm" onClick={save} disabled={!draft.trim()}>Save</Button>
            <Button size="sm" variant="outline" onClick={reset}>Reset to Original</Button>
          </div>
        </div>
      )}

      {msg && (
        <div className={`text-xs ${msg.ok ? "text-green-600" : "text-red-600"}`}>{msg.text}</div>
      )}
    </div>
  );
}
