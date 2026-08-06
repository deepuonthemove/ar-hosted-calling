"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { Button } from "@/components/ui/button";

interface Config {
  tts_engine: string;
  vad_mode: string;
  stay_awake: boolean;
  llm_model: string;
  llm_options: Record<string, string>;
}

export default function SettingsPage() {
  const [cfg, setCfg] = useState<Config | null>(null);
  const [saved, setSaved] = useState(false);

  const load = () => api<Config>("/api/config").then(setCfg).catch(() => {});
  useEffect(() => { load(); }, []);

  const save = async (patch: Partial<Config>) => {
    try {
      await api("/api/config", { method: "POST", body: JSON.stringify(patch) });
      setSaved(true);
      setTimeout(() => setSaved(false), 1500);
      load();
    } catch {}
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
            </Select>
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
