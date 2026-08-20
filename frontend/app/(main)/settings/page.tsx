"use client";

import { useEffect, useRef, useState } from "react";
import { api, API_BASE } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { Button } from "@/components/ui/button";

interface Config {
  tts_engine: string;
  vad_mode: string;
  stay_awake: boolean;
  kokoro_voice: string;
  spell_engine: string;
  spell_pause_mode: string;
  spell_silence_s: number;
  spell_punct: string;
  chatterbox_voice: string;
  kokoro_voice_options: string[];
  spell_engine_options: string[];
  spell_pause_mode_options: string[];
  spell_silence_options: number[];
  spell_punct_options: string[];
  chatterbox_voice_options: string[];
  llm_model: string;
  llm_options: Record<string, string>;
}

const SPELL_PUNCT_LABELS: Record<string, string> = {
  period: "Period \".\" (~0.4s)",
  semicolon: "Semicolon \";\" (~0.4s)",
  colon: "Colon \":\" (~0.3s)",
  comma: "Comma \",\" (~0.2s)",
  linebreak: "Line break (~0.4s)",
};

const SPELL_MODE_LABELS: Record<string, string> = {
  processed: "Processed (fade/volume-matched characters)",
  silence: "Vanilla silence (plain character, then flat silence)",
  punctuation: "Punctuation (same stream — no per-character synth)",
  plain: "Plain (single stream — numbers & spell-outs read as whole words)",
};

export default function SettingsPage() {
  const [cfg, setCfg] = useState<Config | null>(null);
  const [saved, setSaved] = useState(false);

  const [recording, setRecording] = useState(false);
  const [recordSecs, setRecordSecs] = useState(0);
  const [hasClip, setHasClip] = useState(false);
  const [voiceName, setVoiceName] = useState("");
  const [voiceMsg, setVoiceMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [savingVoice, setSavingVoice] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const load = () => api<Config>("/api/config").then(setCfg).catch(() => {});
  useEffect(() => { load(); }, []);
  useEffect(() => () => { if (timerRef.current) clearInterval(timerRef.current); }, []);

  const save = async (patch: Partial<Config>) => {
    try {
      await api("/api/config", { method: "POST", body: JSON.stringify(patch) });
      setSaved(true);
      setTimeout(() => setSaved(false), 1500);
      load();
    } catch {}
  };

  const startRecording = async () => {
    setVoiceMsg(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      chunksRef.current = [];
      rec.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        setHasClip(chunksRef.current.length > 0);
      };
      rec.start();
      mediaRecorderRef.current = rec;
      setRecording(true);
      setHasClip(false);
      setRecordSecs(0);
      timerRef.current = setInterval(() => setRecordSecs((s) => s + 1), 1000);
    } catch (err: any) {
      setVoiceMsg({ ok: false, text: `Microphone access failed: ${String(err)}` });
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setRecording(false);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  const saveRecordedVoice = async () => {
    if (!voiceName.trim()) {
      setVoiceMsg({ ok: false, text: "Give the voice a name first." });
      return;
    }
    if (!chunksRef.current.length) {
      setVoiceMsg({ ok: false, text: "Record a clip first." });
      return;
    }
    setSavingVoice(true);
    try {
      const blob = new Blob(chunksRef.current, { type: mediaRecorderRef.current?.mimeType || "audio/webm" });
      const fd = new FormData();
      fd.append("file", blob, "voice.webm");
      fd.append("name", voiceName.trim());
      const res = await fetch(`${API_BASE}/api/chatterbox/voices`, { method: "POST", body: fd });
      const d = await res.json();
      if (!res.ok) throw new Error(d.error || "upload failed");
      setVoiceMsg({ ok: true, text: `Saved voice "${d.name}".` });
      chunksRef.current = [];
      setHasClip(false);
      setVoiceName("");
      load();
    } catch (err: any) {
      setVoiceMsg({ ok: false, text: String(err.message || err) });
    } finally {
      setSavingVoice(false);
    }
  };

  const deleteVoice = async (name: string) => {
    if (!window.confirm(`Delete cloned voice "${name}"?`)) return;
    try {
      await api(`/api/chatterbox/voices/${encodeURIComponent(name)}`, { method: "DELETE" });
      load();
    } catch (err: any) {
      setVoiceMsg({ ok: false, text: String(err.message || err) });
    }
  };

  if (!cfg) return <div className="text-sm text-muted-foreground">Loading settings...</div>;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Settings</h1>
        <p className="text-sm text-muted-foreground">Defaults applied to new calls</p>
      </div>
      {saved && <div className="rounded-md border border-green-300 px-4 py-2 text-sm text-green-700">Saved.</div>}

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>TTS Engine</CardTitle>
            <CardDescription>Default text-to-speech engine for calls</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Label>TTS Engine</Label>
            <Select value={cfg.tts_engine} onChange={(e) => save({ tts_engine: e.target.value })}>
              <option value="piper">Piper</option>
              <option value="kokoro">Kokoro</option>
              <option value="chatterbox">Chatterbox (voice cloning)</option>
            </Select>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Kokoro Voice</CardTitle>
            <CardDescription>Which Kokoro voice to use for narration (and for spelling, if matched below)</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Label>Kokoro Voice</Label>
            <Select value={cfg.kokoro_voice} onChange={(e) => save({ kokoro_voice: e.target.value })}>
              {cfg.kokoro_voice_options.map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </Select>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Spell-out Engine</CardTitle>
            <CardDescription>Engine used for character-by-character spelling on Kokoro calls</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Label>Spell-out Engine</Label>
            <Select value={cfg.spell_engine} onChange={(e) => save({ spell_engine: e.target.value })}>
              <option value="piper">Piper (default — clearer isolated characters)</option>
              <option value="match">Match narration (Kokoro spells too — voice consistency test)</option>
            </Select>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Spell-out Pause Mode</CardTitle>
            <CardDescription>How the gap between spelled-out characters is produced (options depend on the TTS engine)</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Label>Pause Mode</Label>
            <Select value={cfg.spell_pause_mode} onChange={(e) => save({ spell_pause_mode: e.target.value })}>
              {cfg.spell_pause_mode_options.map((m) => (
                <option key={m} value={m}>{SPELL_MODE_LABELS[m] ?? m}</option>
              ))}
            </Select>
            {cfg.spell_pause_mode === "silence" && (
              <>
                <Label>Silence Duration</Label>
                <Select
                  value={String(cfg.spell_silence_s)}
                  onChange={(e) => save({ spell_silence_s: parseFloat(e.target.value) })}
                >
                  {cfg.spell_silence_options.map((s) => (
                    <option key={s} value={s}>{s}s</option>
                  ))}
                </Select>
              </>
            )}
            {cfg.spell_pause_mode === "punctuation" && (
              <>
                <Label>Punctuation Mark</Label>
                <Select value={cfg.spell_punct} onChange={(e) => save({ spell_punct: e.target.value })}>
                  {cfg.spell_punct_options.map((p) => (
                    <option key={p} value={p}>{SPELL_PUNCT_LABELS[p] ?? p}</option>
                  ))}
                </Select>
              </>
            )}
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Chatterbox Voice Cloning</CardTitle>
            <CardDescription>
              Record a short reference clip (10-20s of clear speech), name it, and pick it below as the
              active cloned voice for Chatterbox calls — narration and spelling both use it.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label>Active Cloned Voice</Label>
              <Select value={cfg.chatterbox_voice} onChange={(e) => save({ chatterbox_voice: e.target.value })}>
                <option value="">Chatterbox default (no cloning)</option>
                {cfg.chatterbox_voice_options.map((v) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </Select>
            </div>

            {cfg.chatterbox_voice_options.length > 0 && (
              <div className="space-y-1">
                <Label>Saved Voices</Label>
                <ul className="text-sm space-y-1">
                  {cfg.chatterbox_voice_options.map((v) => (
                    <li key={v} className="flex items-center justify-between rounded-md border px-3 py-1.5">
                      <span>{v}</span>
                      <Button variant="outline" onClick={() => deleteVoice(v)}>Delete</Button>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div className="space-y-3 rounded-md border p-3">
              <Label>Record a New Voice</Label>
              <div className="flex items-center gap-3">
                {!recording ? (
                  <Button variant="default" onClick={startRecording}>🎙️ Start Recording</Button>
                ) : (
                  <Button variant="destructive" onClick={stopRecording}>⏹ Stop ({recordSecs}s)</Button>
                )}
                <input
                  type="text"
                  placeholder="Voice name (e.g. agent-sarah)"
                  value={voiceName}
                  onChange={(e) => setVoiceName(e.target.value)}
                  className="flex-1 rounded-md border px-3 py-1.5 text-sm bg-background"
                />
                <Button
                  variant="secondary"
                  disabled={savingVoice || recording || !hasClip}
                  onClick={saveRecordedVoice}
                >
                  {savingVoice ? "Saving..." : "Save Clip"}
                </Button>
              </div>
              {voiceMsg && (
                <div className={`text-sm ${voiceMsg.ok ? "text-green-700" : "text-red-700"}`}>{voiceMsg.text}</div>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>VAD Engine</CardTitle>
            <CardDescription>Voice activity detection for speech segmentation</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Label>VAD Engine</Label>
            <Select value={cfg.vad_mode} onChange={(e) => save({ vad_mode: e.target.value })}>
              <option value="silero">Silero (neural)</option>
              <option value="rms">Plain RMS</option>
            </Select>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>LLM Model</CardTitle>
            <CardDescription>Model served by vLLM (switching restarts vLLM)</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Label>LLM Model</Label>
            <Select
              value={cfg.llm_model}
              onChange={(e) => {
                api("/api/switch-llm", { method: "POST", body: JSON.stringify({ model: e.target.value }) })
                  .then(load).catch(() => {});
              }}
            >
              {Object.entries(cfg.llm_options).map(([label, id]) => (
                <option key={id} value={id}>{label}</option>
              ))}
            </Select>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Stay Awake</CardTitle>
            <CardDescription>Keep the VM running while you work (disables auto-deactivation)</CardDescription>
          </CardHeader>
          <CardContent>
            <Button
              variant={cfg.stay_awake ? "default" : "outline"}
              onClick={() => save({ stay_awake: !cfg.stay_awake })}
            >
              {cfg.stay_awake ? "🌙 Stay Awake: ON" : "😴 Stay Awake: OFF"}
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
