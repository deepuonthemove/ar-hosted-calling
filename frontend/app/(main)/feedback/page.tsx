"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";

const OPIK_HOST = process.env.NEXT_PUBLIC_OPIK_HOST || "https://optik.call.ar-voice.com";

export default function FeedbackPage() {
  const [opik, setOpik] = useState<{ enabled: boolean; url: string } | null>(null);

  useEffect(() => {
    api<{ opik_enabled?: boolean; opik_url?: string }>("/api/config")
      .then((c) => setOpik({ enabled: !!c.opik_enabled, url: OPIK_HOST || c.opik_url || "" }))
      .catch(() => setOpik({ enabled: false, url: OPIK_HOST }));
  }, []);

  return (
    <div className="flex h-[calc(100vh-3rem)] flex-col space-y-3">
      <div>
        <h1 className="text-2xl font-bold">Feedback & Evaluation</h1>
        <p className="text-sm text-muted-foreground">
          Opik — live traces, prompt playground, and LLM-as-a-judge evaluations.
        </p>
      </div>
      {opik === null ? (
        <div className="text-sm text-muted-foreground">Loading Opik...</div>
      ) : opik.enabled ? (
        <div className="flex-1 overflow-hidden rounded-lg border">
          <iframe src={opik.url} className="h-full w-full" title="Opik" />
        </div>
      ) : (
        <div className="flex flex-1 items-center justify-center rounded-lg border">
          <div className="max-w-md space-y-2 p-6 text-center">
            <p className="text-sm font-medium">Opik is not enabled in this environment</p>
            <p className="text-sm text-muted-foreground">
              On the production VM, Opik traces are served on the{" "}
              <code className="rounded bg-muted px-1 py-0.5">optik.</code> subdomain. Locally it is
              disabled ({`OPIK_ENABLED=0`}) since the Opik stack only runs on the server.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
