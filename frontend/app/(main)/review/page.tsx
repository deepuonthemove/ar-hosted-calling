"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, statusVariant, type CallRecord } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Eye } from "lucide-react";

export default function ReviewPage() {
  const [calls, setCalls] = useState<CallRecord[]>([]);
  const [err, setErr] = useState("");
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [pages, setPages] = useState(1);
  const limit = 15;

  useEffect(() => {
    api<{ calls: CallRecord[]; total: number; page: number; pages: number }>(`/api/calls?page=${page}&limit=${limit}`)
      .then((d) => {
        setCalls(d.calls);
        setTotal(d.total);
        setPages(d.pages);
      })
      .catch((e) => setErr(String(e)));
  }, [page]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Review</h1>
        <p className="text-sm text-muted-foreground">All calls — click one for the full global view</p>
      </div>
      {err && <div className="text-sm text-red-600">{err}</div>}

      <Card>
        <CardHeader>
          <CardTitle>All Calls</CardTitle>
          <CardDescription>{total} calls total · {limit} per page</CardDescription>
        </CardHeader>
        <CardContent>
          {calls.length === 0 ? (
            <p className="text-sm text-muted-foreground">No calls yet.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-xs text-muted-foreground">
                    <th className="p-2">Call ID</th>
                    <th className="p-2">Status</th>
                    <th className="p-2">Payer</th>
                    <th className="p-2">Claim</th>
                    <th className="p-2">Started</th>
                    <th className="p-2">Duration</th>
                    <th className="p-2">Next Action</th>
                    <th className="p-2"></th>
                  </tr>
                </thead>
                <tbody>
                  {calls.map((c) => {
                    const id = c.call_id || c.callSid || "";
                    return (
                      <tr key={id} className="border-b last:border-0 hover:bg-accent">
                        <td className="p-2 font-mono text-xs">{id}</td>
                        <td className="p-2">
                          <Badge variant={statusVariant(c.status)}>
                            {c.status}
                          </Badge>
                        </td>
                        <td className="p-2">{c.payer || "—"}</td>
                        <td className="p-2">{c.claim_id || "—"}</td>
                        <td className="p-2 text-xs text-muted-foreground">
                          {c.started_at ? new Date(Number(c.started_at) * 1000).toLocaleString() : "—"}
                        </td>
                        <td className="p-2">{Math.round((Number(c.duration_ms) || 0) / 1000)}s</td>
                        <td className="p-2 max-w-[200px] truncate">{c.next_action || "—"}</td>
                        <td className="p-2">
                          <Link href={`/calls/${id}`} className="inline-flex items-center gap-1 text-xs text-primary hover:underline">
                            <Eye className="h-3.5 w-3.5" /> Open
                          </Link>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          <div className="mt-4 flex items-center justify-between">
            <span className="text-xs text-muted-foreground">
              Page {page} of {pages}
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
                className="rounded border px-3 py-1.5 text-xs disabled:opacity-40 hover:bg-accent"
              >
                Prev
              </button>
              <button
                type="button"
                onClick={() => setPage((p) => Math.min(pages, p + 1))}
                disabled={page >= pages}
                className="rounded border px-3 py-1.5 text-xs disabled:opacity-40 hover:bg-accent"
              >
                Next
              </button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
