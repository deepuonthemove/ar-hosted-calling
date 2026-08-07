"use client";

const OPIK_HOST = process.env.NEXT_PUBLIC_OPIK_HOST || "https://optik.call.ar-voice.com";

export default function FeedbackPage() {
  return (
    <div className="flex h-[calc(100vh-3rem)] flex-col space-y-3">
      <div>
        <h1 className="text-2xl font-bold">Feedback & Evaluation</h1>
        <p className="text-sm text-muted-foreground">
          Opik — live traces, prompt playground, and LLM-as-a-judge evaluations.
        </p>
      </div>
      <div className="flex-1 overflow-hidden rounded-lg border">
        <iframe src={OPIK_HOST} className="h-full w-full" title="Opik" />
      </div>
    </div>
  );
}
