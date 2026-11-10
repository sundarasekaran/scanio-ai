import type { ReactNode } from "react";
import { AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Severity } from "@/services/prediction";

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("rounded-2xl border border-border bg-card p-6 shadow-card", className)}>{children}</div>;
}

export function Disclaimer({ mock }: { mock?: boolean }) {
  return (
    <div className="flex gap-3 rounded-2xl border border-warning/40 bg-warning/10 p-4 text-sm text-foreground">
      <AlertTriangle className="h-5 w-5 shrink-0 text-warning" />
      <p>{mock && <b>Demo results — simulated, not from a real model. </b>}{!mock && <b>AI-generated findings may be incomplete or wrong. </b>}This is an AI-assisted screening, not a medical diagnosis. Please consult a licensed dentist.</p>
    </div>
  );
}

const sev: Record<Severity, string> = { low: "bg-success/15 text-success", moderate: "bg-warning/15 text-warning", high: "bg-destructive/15 text-destructive" };
export function SeverityBadge({ s }: { s: Severity }) {
  return <span className={cn("rounded-full px-2.5 py-0.5 text-xs font-semibold capitalize", sev[s])}>{s}</span>;
}

export function AuthShell({ title, subtitle, children }: { title: string; subtitle: string; children: ReactNode }) {
  return (
    <div className="bg-gradient-soft px-4 py-16">
      <Card className="mx-auto max-w-md">
        <h1 className="text-2xl font-bold">{title}</h1>
        <p className="mb-6 mt-1 text-sm text-muted-foreground">{subtitle}</p>
        {children}
      </Card>
    </div>
  );
}
