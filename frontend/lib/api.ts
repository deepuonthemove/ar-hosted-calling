"use client";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "https://call.ar-voice.com";

export async function api<T = any>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...opts,
    headers: { "Content-Type": "application/json", ...(opts?.headers || {}) },
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = await res.json();
      msg = j.error || JSON.stringify(j);
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

export const wsUrl = (path: string) =>
  (API_BASE.startsWith("https") ? "wss" : "ws") + "://" + API_BASE.replace(/^https?:\/\//, "") + path;

// Badge variant for a call/account status: paid/green, failed/red, dialing/amber, else gray.
export const statusVariant = (status?: string) => {
  const s = (status || "").toLowerCase();
  if (s === "paid" || s === "completed") return "success" as const;
  if (s === "failed" || s === "error") return "destructive" as const;
  if (s === "dialing" || s === "in-progress") return "warning" as const;
  return "secondary" as const;
};

export interface Project {
  project_id: string;
  rows: number;
}

export interface CallRecord {
  call_id?: string;
  callSid?: string;
  status?: string;
  payer?: string;
  claim_id?: string;
  account_uid?: string;
  started_at?: string;
  duration_ms?: string;
  next_action?: string;
  denial_code?: string;
  paid_amount?: string;
  last_error?: string;
  call_result?: string;
}

export interface Account {
  UID?: string;
  [k: string]: any;
}

export interface TranscriptTurn {
  role: "user" | "assistant";
  text: string;
}

export interface CallDetail {
  call_id: string;
  twilio_sid?: string;
  call: Record<string, string>;
  config: { stt_model: string; tts_engine: string; vad_mode: string; llm_model: string };
  prompt: string;
  transcript: TranscriptTurn[];
  review: {
    real_time?: string[];
    full_audio?: string;
    ai_responses?: string[];
    duration_sec?: number;
    audio_size_bytes?: number;
  } | null;
}
