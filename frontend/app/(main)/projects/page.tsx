"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { api, type Project } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [uploading, setUploading] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = () => api<Project[]>("/api/projects").then(setProjects).catch((e) => setMsg({ ok: false, text: String(e) }));
  useEffect(() => { load(); }, []);

  const onUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setMsg(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_BASE || "https://call.ar-voice.com"}/api/upload-excel`, {
        method: "POST",
        body: fd,
      });
      const d = await res.json();
      setMsg({ ok: res.ok, text: res.ok ? `Project ${d.project_id} created (${d.count} rows)` : d.error });
      load();
    } catch (err: any) {
      setMsg({ ok: false, text: String(err) });
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Projects</h1>
        <p className="text-sm text-muted-foreground">Upload an Excel file — each upload creates a new project</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Upload Excel</CardTitle>
          <CardDescription>A unique project id is generated and returned. Use it with row numbers for calls.</CardDescription>
        </CardHeader>
        <CardContent className="flex items-center gap-3">
          <input ref={fileRef} type="file" accept=".xlsx" onChange={onUpload} disabled={uploading} />
          {uploading && <span className="text-sm text-muted-foreground">Uploading...</span>}
        </CardContent>
      </Card>

      {msg && (
        <div className={`rounded-md border px-4 py-2 text-sm ${msg.ok ? "border-green-300 text-green-700" : "border-red-300 text-red-700"}`}>
          {msg.text}
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle>All Projects</CardTitle>
          <CardDescription>Click a project to view its accounts</CardDescription>
        </CardHeader>
        <CardContent>
          {projects.length === 0 ? (
            <p className="text-sm text-muted-foreground">No projects yet.</p>
          ) : (
            <div className="space-y-2">
              {projects.map((p) => (
                <Link
                  key={p.project_id}
                  href={`/projects/${p.project_id}`}
                  className="flex items-center justify-between rounded-md border px-4 py-2.5 hover:bg-accent"
                >
                  <span className="font-mono text-sm">{p.project_id}</span>
                  <div className="flex items-center gap-3">
                    <Badge variant="secondary">{p.rows} rows</Badge>
                    <Button size="sm" variant="ghost">Open →</Button>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
