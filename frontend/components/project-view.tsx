"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { ProjectsTable } from "@/components/projects-table";
import { ArrowLeft } from "lucide-react";

export function ProjectView({ projectId }: { projectId: string }) {
  const [accounts, setAccounts] = useState<any[]>([]);
  const [err, setErr] = useState("");

  useEffect(() => {
    api<any[]>(`/api/projects/${projectId}/accounts`).then(setAccounts).catch((e) => setErr(String(e)));
  }, [projectId]);

  return (
    <div className="space-y-6">
      <Link href="/projects" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="h-4 w-4" /> Back to Projects
      </Link>
      <div>
        <h1 className="text-2xl font-bold font-mono">{projectId}</h1>
        <p className="text-sm text-muted-foreground">{accounts.length} accounts · click View to open a claim</p>
      </div>
      {err && <div className="text-sm text-red-600">{err}</div>}
      <ProjectsTable projectId={projectId} accounts={accounts} />
    </div>
  );
}
