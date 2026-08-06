"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Eye } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Row {
  row_num: number;
  account_uid: string;
  Patient: string;
  DOS: string;
  CPT: string;
  Billed: string;
  Payer: string;
  Account: string;
  Status: string;
}

export function ProjectsTable({ projectId, accounts }: { projectId: string; accounts: any[] }) {
  const router = useRouter();
  const [search, setSearch] = useState("");

  const rows: Row[] = useMemo(
    () =>
      accounts.map((a, i) => ({
        row_num: i + 1,
        account_uid: a.UID || "",
        Patient: a["Patient Name"] || "",
        DOS: (a["DOS"] || "").slice(0, 10),
        CPT: a.CPT || "",
        Billed: a["Billed Amount"] || "",
        Payer: a["Responsible Payer"] || "",
        Account: a["Account Number"] || "",
        Status: a["Call Status"] || "",
      })),
    [accounts]
  );

  const filtered = useMemo(() => {
    if (!search) return rows;
    const q = search.toLowerCase();
    return rows.filter((r) => Object.values(r).some((v) => String(v).toLowerCase().includes(q)));
  }, [rows, search]);

  return (
    <div className="space-y-3">
      <input
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        placeholder="Search accounts..."
        className="h-9 w-full max-w-sm rounded-md border border-input bg-background px-3 text-sm"
      />
      <div className="overflow-auto rounded-md border" style={{ maxHeight: "560px" }}>
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-muted text-left text-xs text-muted-foreground">
            <tr>
              <th className="p-2">#</th>
              <th className="p-2">Patient</th>
              <th className="p-2">DOS</th>
              <th className="p-2">CPT</th>
              <th className="p-2">Billed</th>
              <th className="p-2">Payer</th>
              <th className="p-2">Account</th>
              <th className="p-2">Status</th>
              <th className="p-2"></th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={9} className="p-4 text-center text-muted-foreground">
                  {accounts.length === 0 ? "Loading accounts..." : "No matches."}
                </td>
              </tr>
            ) : (
              filtered.map((r) => (
                <tr key={r.account_uid || r.row_num} className="border-b last:border-0 hover:bg-accent">
                  <td className="p-2">{r.row_num}</td>
                  <td className="p-2 font-medium">{r.Patient}</td>
                  <td className="p-2">{r.DOS}</td>
                  <td className="p-2">{r.CPT}</td>
                  <td className="p-2">{r.Billed}</td>
                  <td className="p-2">{r.Payer}</td>
                  <td className="p-2">{r.Account}</td>
                  <td className="p-2">{r.Status}</td>
                  <td className="p-2">
                    <Button size="sm" variant="outline" onClick={() => router.push(`/projects/${projectId}/${r.row_num}`)}>
                      <Eye className="mr-1 h-3.5 w-3.5" /> View
                    </Button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      <div className="text-xs text-muted-foreground">{filtered.length} of {rows.length} accounts</div>
    </div>
  );
}
