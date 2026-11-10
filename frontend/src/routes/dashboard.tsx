import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Activity, ClipboardList, Download, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, SeverityBadge } from "@/components/shared";
import { getScreenings, useUser, type Screening } from "@/services/store";
import { downloadReport } from "@/services/report";
import { allConditions } from "@/services/prediction";

export const Route = createFileRoute("/dashboard")({
  head: () => ({ meta: [{ title: "Dashboard — Scanio AI" }, { name: "description", content: "Your dental screening history and reports." }, { property: "og:title", content: "Dashboard — Scanio AI" }, { property: "og:description", content: "Track your dental screenings over time." }] }),
  component: Dashboard,
});

function Dashboard() {
  const { user, ready } = useUser();
  const [list, setList] = useState<Screening[]>([]);
  useEffect(() => { if (user) setList(getScreenings(user.email)); }, [user]);

  if (!ready) return null;
  if (!user) return <div className="py-20 text-center"><h1 className="text-2xl font-bold">Please log in</h1><Button className="mt-4" asChild><Link to="/login">Log in</Link></Button></div>;

  const avg = list.length ? Math.round(list.reduce((a, s) => a + s.overallScore, 0) / list.length) : "—";
  const stats = [
    { icon: ClipboardList, label: "Total screenings", value: list.length },
    { icon: Activity, label: "Average score", value: avg },
    { icon: Activity, label: "Latest score", value: list[0]?.overallScore ?? "—" },
  ];
  return (
    <div className="mx-auto max-w-6xl space-y-8 px-4 py-12">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div><h1 className="text-3xl font-bold">Hello, {user.name}</h1><p className="text-muted-foreground">Here's your screening history.</p></div>
        <Button asChild><Link to="/upload"><Upload /> New screening</Link></Button>
      </div>
      <div className="grid gap-4 sm:grid-cols-3">
        {stats.map((s) => (
          <Card key={s.label} className="flex items-center gap-4 p-5">
            <span className="grid h-12 w-12 place-items-center rounded-xl bg-accent text-primary"><s.icon /></span>
            <div><p className="text-sm text-muted-foreground">{s.label}</p><p className="font-display text-2xl font-bold">{s.value}</p></div>
          </Card>
        ))}
      </div>
      <Card className="p-0">
        <h2 className="border-b border-border p-5 text-lg font-semibold">Screening history</h2>
        {list.length === 0 ? (
          <div className="p-10 text-center text-muted-foreground">No screenings yet. <Link to="/upload" className="font-medium text-primary">Start your first one</Link>.</div>
        ) : (
          <ul className="divide-y divide-border">
            {list.map((s) => (
              <li key={s.id} className="flex flex-col gap-3 p-5 sm:flex-row sm:items-center">
                <img src={s.images.front} alt="" className="h-16 w-16 rounded-xl object-cover" />
                <div className="flex-1">
                  <p className="font-medium">{new Date(s.createdAt).toLocaleString()}</p>
                  <div className="mt-1 flex flex-wrap gap-2">{allConditions(s).map((c, i) => <span key={i} className="flex items-center gap-1 text-xs text-muted-foreground">{c.name} <SeverityBadge s={c.severity} /></span>)}</div>
                </div>
                <p className="font-display text-xl font-bold text-primary">{s.overallScore}</p>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" asChild><Link to="/results/$id" params={{ id: s.id }}>View</Link></Button>
                  <Button size="sm" variant="ghost" onClick={() => downloadReport(s, user.name)} aria-label="Download report"><Download /></Button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </div>
  );
}
