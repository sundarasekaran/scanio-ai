import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Download, RotateCcw, Sparkles, Activity, Layers, ArrowLeftRight, CheckCircle2, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Card, Disclaimer, SeverityBadge } from "@/components/shared";
import { AnnotatedImage } from "@/components/AnnotatedImage";
import { getScreening, useUser, type Screening } from "@/services/store";
import { downloadReport } from "@/services/report";
import { VIEWS, type ModuleViewStatus } from "@/services/prediction";
import { cn } from "@/lib/utils";

const STATUS_BADGE: Record<ModuleViewStatus, { label: string; cls: string }> = {
  assessed_with_findings: {
    label: "Assessed · Findings",
    cls: "bg-amber-500/15 text-amber-800 dark:text-amber-300 border-amber-500/30",
  },
  assessed_no_findings: {
    label: "Assessed · No Findings",
    cls: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/30",
  },
  inconclusive: {
    label: "Inconclusive",
    cls: "bg-rose-500/15 text-rose-700 dark:text-rose-300 border-rose-500/30",
  },
};

function resolveModuleStatus(
  explicit: ModuleViewStatus | undefined,
  itemCount: number,
  isInconclusive: boolean
): ModuleViewStatus {
  if (explicit) return explicit;
  if (itemCount > 0) return "assessed_with_findings";
  return isInconclusive ? "inconclusive" : "assessed_no_findings";
}

export const Route = createFileRoute("/results/$id")({
  head: () => ({
    meta: [
      { title: "Screening Results — Scanio AI" },
      { name: "description", content: "Dual-module AI dental screening: Tooth Surface Conditions & Dental Alignment Analysis." },
      { property: "og:title", content: "Screening Results — Scanio AI" },
      { property: "og:description", content: "Separated tooth surface findings and orthodontic alignment analysis." },
    ],
  }),
  component: Results,
});

function Results() {
  const { id } = Route.useParams();
  const { user } = useUser();
  const [s, setS] = useState<Screening | null | undefined>(undefined);
  const [selectedTab, setSelectedTab] = useState<"all" | "conditions" | "alignment">("all");

  useEffect(() => {
    setS(getScreening(id));
  }, [id]);

  if (s === undefined) return null;
  if (!s) {
    return (
      <div className="py-20 text-center">
        <p>Screening not found.</p>
        <Button className="mt-4" asChild>
          <Link to="/dashboard">Go to dashboard</Link>
        </Button>
      </div>
    );
  }

  // Calculate totals across all views
  const allFindings = VIEWS.flatMap((v) => s.views[v.key] ?? []);
  const totalConditions = allFindings.filter((c) => c.category !== "alignment").length;
  const totalAlignment = allFindings.filter((c) => c.category === "alignment").length;
  const inconclusiveViews = s.inconclusiveViews ?? [];
  const allInconclusive = Boolean(s.inconclusive) && inconclusiveViews.length >= VIEWS.length;
  const scoreDisplay = allInconclusive ? "Inconclusive" : `${s.overallScore}/100`;

  let globalIndex = 0;

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-4 py-12">
      {/* Header and Actions */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-3xl font-bold tracking-tight">Screening results</h1>
            <span className="rounded-full bg-primary/10 px-3 py-0.5 text-xs font-semibold text-primary">
              Dual-Module Analysis
            </span>
          </div>
          <p className="text-muted-foreground mt-1">
            {new Date(s.createdAt).toLocaleString()} ·{" "}
            {s.modelProvider ? `Powered by ${s.modelProvider}` : "AI-Assisted Vision Screening"}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-xl bg-gradient-primary px-4 py-2 font-display font-bold text-primary-foreground shadow-sm">
            {allInconclusive ? "Inconclusive" : `Score ${s.overallScore}/100`}
          </span>
          <Button variant="outline" asChild>
            <Link to="/upload">
              <RotateCcw className="h-4 w-4 mr-1.5" /> New scan
            </Link>
          </Button>
          <Button onClick={() => downloadReport(s, user?.name ?? "Patient")}>
            <Download className="h-4 w-4 mr-1.5" /> Download report
          </Button>
        </div>
      </div>

      {/* Module Overview Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="rounded-2xl border border-border bg-card p-4 shadow-sm flex items-center justify-between">
          <div className="space-y-0.5">
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Overall Health</span>
            <div className="text-2xl font-bold text-primary">{scoreDisplay}</div>
            <p className="text-xs text-muted-foreground">
              {allInconclusive ? "Clearer intraoral photos required" : "Dynamic composite index"}
            </p>
          </div>
          <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary">
            <Activity className="h-5 w-5" />
          </div>
        </div>

        <div className="rounded-2xl border border-emerald-500/20 bg-emerald-50/50 dark:bg-emerald-950/20 p-4 shadow-sm flex items-center justify-between">
          <div className="space-y-0.5">
            <span className="text-xs font-semibold text-emerald-700 dark:text-emerald-400 uppercase tracking-wider">
              Module 1: Tooth Surfaces
            </span>
            <div className="text-2xl font-bold text-emerald-800 dark:text-emerald-200">
              {totalConditions} {totalConditions === 1 ? "Finding" : "Findings"}
            </div>
            <p className="text-xs text-emerald-600/80 dark:text-emerald-400/80">Enamel, cavities, tartar</p>
          </div>
          <div className="h-10 w-10 rounded-xl bg-emerald-500/20 flex items-center justify-center text-emerald-600 dark:text-emerald-400">
            <Sparkles className="h-5 w-5" />
          </div>
        </div>

        <div className="rounded-2xl border border-indigo-500/20 bg-indigo-50/50 dark:bg-indigo-950/20 p-4 shadow-sm flex items-center justify-between">
          <div className="space-y-0.5">
            <span className="text-xs font-semibold text-indigo-700 dark:text-indigo-400 uppercase tracking-wider">
              Module 2: Dental Alignment
            </span>
            <div className="text-2xl font-bold text-indigo-800 dark:text-indigo-200">
              {totalAlignment} {totalAlignment === 1 ? "Observation" : "Observations"}
            </div>
            <p className="text-xs text-indigo-600/80 dark:text-indigo-400/80">Gaps, crowding, arch form</p>
          </div>
          <div className="h-10 w-10 rounded-xl bg-indigo-500/20 flex items-center justify-center text-indigo-600 dark:text-indigo-400">
            <ArrowLeftRight className="h-5 w-5" />
          </div>
        </div>
      </div>

      {s.inconclusive && (
        <div className="flex items-start gap-3 rounded-2xl border border-amber-500/40 bg-amber-500/10 p-4 text-sm text-foreground">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-600 dark:text-amber-400" />
          <div className="space-y-1">
            <p className="font-semibold">
              {s.statusMessage || "One or more dental photos were inconclusive."}
            </p>
            {s.inconclusiveReasons && Object.keys(s.inconclusiveReasons).length > 0 && (
              <ul className="list-disc pl-4 text-xs text-muted-foreground space-y-0.5">
                {VIEWS.filter((v) => s.inconclusiveReasons?.[v.key]).map((v) => (
                  <li key={v.key}>
                    <b>{v.label}:</b> {s.inconclusiveReasons?.[v.key]}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}

      <Disclaimer mock={s.isMock} />

      {/* Global Module Filter Selector */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
        <h2 className="text-xl font-bold">Detailed Anatomical Views</h2>
        <div className="flex rounded-xl bg-muted p-1 text-xs font-medium">
          <button
            type="button"
            onClick={() => setSelectedTab("all")}
            className={cn(
              "px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5",
              selectedTab === "all" ? "bg-background text-foreground shadow-sm font-semibold" : "text-muted-foreground hover:text-foreground"
            )}
          >
            <Layers className="h-3.5 w-3.5" />
            All Modules ({allFindings.length})
          </button>
          <button
            type="button"
            onClick={() => setSelectedTab("conditions")}
            className={cn(
              "px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5",
              selectedTab === "conditions" ? "bg-background text-emerald-600 dark:text-emerald-400 shadow-sm font-semibold" : "text-muted-foreground hover:text-foreground"
            )}
          >
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            Tooth Surfaces ({totalConditions})
          </button>
          <button
            type="button"
            onClick={() => setSelectedTab("alignment")}
            className={cn(
              "px-3 py-1.5 rounded-lg transition-all flex items-center gap-1.5",
              selectedTab === "alignment" ? "bg-background text-indigo-600 dark:text-indigo-400 shadow-sm font-semibold" : "text-muted-foreground hover:text-foreground"
            )}
          >
            <span className="h-2 w-2 rounded-full bg-indigo-500" />
            Dental Alignment ({totalAlignment})
          </button>
        </div>
      </div>

      {/* Per-View Breakdown */}
      {VIEWS.map((v) => {
        const fullList = s.views[v.key] ?? [];
        const viewOffset = globalIndex;
        globalIndex += fullList.length;

        // Separate conditions and alignment for this view
        const surfaceConditions = fullList.filter((c) => c.category !== "alignment");
        const alignmentFindings = fullList.filter((c) => c.category === "alignment");
        const viewInconclusiveReason = s.inconclusiveReasons?.[v.key];
        const isViewInconclusive = inconclusiveViews.includes(v.key) || Boolean(viewInconclusiveReason);
        const condStatus = resolveModuleStatus(
          s.moduleStatuses?.toothConditions?.[v.key],
          surfaceConditions.length,
          isViewInconclusive
        );
        const alignStatus = resolveModuleStatus(
          s.moduleStatuses?.dentalAlignment?.[v.key],
          alignmentFindings.length,
          isViewInconclusive
        );
        const viewDiag = s.viewDiagnostics?.[v.key];
        const viewQuality = s.imageQuality?.[v.key];

        const showConditions = selectedTab === "all" || selectedTab === "conditions";
        const showAlignment = selectedTab === "all" || selectedTab === "alignment";

        return (
          <Card key={v.key} className="grid gap-6 lg:grid-cols-[1fr_1.3fr]">
            {/* Left: Annotated Image Viewer */}
            <div className="space-y-3">
              <div className="flex items-center justify-between gap-2">
                <h3 className="text-xl font-semibold">{v.label}</h3>
                <div className="flex items-center gap-2">
                  {viewQuality && (
                    <span className="rounded-md bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">
                      {viewQuality.width}×{viewQuality.height} · {viewQuality.qualityStatus.replaceAll("_", " ")}
                    </span>
                  )}
                  <span className="text-xs text-muted-foreground">
                    {fullList.length} total {fullList.length === 1 ? "item" : "items"}
                  </span>
                </div>
              </div>
              <AnnotatedImage
                src={s.images[v.key]}
                conditions={fullList}
                alt={`${v.label} with marked areas`}
                offset={viewOffset}
              />
              <p className="text-xs text-muted-foreground italic">{v.hint}</p>
              {viewDiag?.unassessableTeethReason && (
                <p className="rounded-lg bg-muted/70 px-2.5 py-1.5 text-[11px] text-muted-foreground">
                  <b>Visibility note:</b> {viewDiag.unassessableTeethReason}
                </p>
              )}
            </div>

            {/* Right: Segregated Module Findings */}
            <div className="space-y-5">
              {/* SECTION A: TOOTH-SURFACE CONDITION ANALYSIS */}
              {showConditions && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between border-b border-border/80 pb-2">
                    <div className="flex items-center gap-2">
                      <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
                      <h4 className="font-semibold text-sm uppercase tracking-wider text-emerald-800 dark:text-emerald-300">
                        Section A: Tooth Surface Findings
                      </h4>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={cn("rounded-full border px-2 py-0.5 text-[10px] font-semibold", STATUS_BADGE[condStatus].cls)}>
                        {STATUS_BADGE[condStatus].label}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {surfaceConditions.length} {surfaceConditions.length === 1 ? "finding" : "findings"}
                      </span>
                    </div>
                  </div>

                  {surfaceConditions.length === 0 ? (
                    condStatus === "inconclusive" ? (
                      <div className="flex items-center gap-2 rounded-xl bg-amber-500/10 p-3.5 text-xs text-amber-800 dark:text-amber-300 border border-amber-500/20">
                        <AlertTriangle className="h-4 w-4 shrink-0 text-amber-600" />
                        <span>{viewInconclusiveReason || "Inconclusive image: unable to reliably inspect tooth surfaces in this photo."}</span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 rounded-xl bg-emerald-500/10 p-3.5 text-xs text-emerald-700 dark:text-emerald-300 border border-emerald-500/20">
                        <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600" />
                        <span>No visible enamel caries, cracks, or tartar detected on visible tooth surfaces in this view.</span>
                      </div>
                    )
                  ) : (
                    surfaceConditions.map((c, i) => {
                      const itemNumber = viewOffset + fullList.indexOf(c) + 1;
                      return (
                        <div
                          key={`cond-${i}`}
                          className="space-y-2 rounded-xl border border-emerald-500/20 bg-card p-4 shadow-sm"
                        >
                          <div className="flex items-center justify-between gap-2">
                            <h5 className="font-semibold text-sm">
                              <span className="mr-2 text-emerald-600 dark:text-emerald-400 font-bold">
                                #{itemNumber}
                              </span>
                              {c.name}
                              {c.tooth && (
                                <span className="ml-2 rounded-md bg-muted px-2 py-0.5 text-xs font-semibold text-foreground">
                                  Tooth {c.tooth}
                                </span>
                              )}
                              {c.affectedRegion && (
                                <span className="ml-1 text-xs text-muted-foreground font-normal">
                                  · {c.affectedRegion}
                                </span>
                              )}
                            </h5>
                            <SeverityBadge s={c.severity} />
                          </div>

                          <p className="text-xs text-muted-foreground leading-relaxed">{c.description}</p>

                          <div>
                            <div className="mb-1 flex justify-between text-[11px]">
                              <span className="text-muted-foreground">Model Confidence</span>
                              <span className="font-semibold text-foreground">
                                {Math.round(c.confidence * 100)}%
                              </span>
                            </div>
                            <Progress value={c.confidence * 100} className="h-1.5" />
                          </div>

                          <div className="rounded-lg bg-emerald-500/10 border border-emerald-500/15 p-2.5 text-xs text-emerald-900 dark:text-emerald-100">
                            <b>Recommendation:</b> {c.recommendation}
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              )}

              {/* SECTION B: DENTAL ALIGNMENT ANALYSIS */}
              {showAlignment && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between border-b border-border/80 pb-2">
                    <div className="flex items-center gap-2">
                      <span className="h-2.5 w-2.5 rounded-full bg-indigo-500" />
                      <h4 className="font-semibold text-sm uppercase tracking-wider text-indigo-800 dark:text-indigo-300">
                        Section B: Dental Alignment Findings
                      </h4>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={cn("rounded-full border px-2 py-0.5 text-[10px] font-semibold", STATUS_BADGE[alignStatus].cls)}>
                        {STATUS_BADGE[alignStatus].label}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {alignmentFindings.length} {alignmentFindings.length === 1 ? "observation" : "observations"}
                      </span>
                    </div>
                  </div>

                  {alignmentFindings.length === 0 ? (
                    alignStatus === "inconclusive" ? (
                      <div className="flex items-center gap-2 rounded-xl bg-amber-500/10 p-3.5 text-xs text-amber-800 dark:text-amber-300 border border-amber-500/20">
                        <AlertTriangle className="h-4 w-4 shrink-0 text-amber-600" />
                        <span>{viewInconclusiveReason || "Inconclusive image: unable to reliably inspect dental alignment in this photo."}</span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 rounded-xl bg-indigo-500/10 p-3.5 text-xs text-indigo-700 dark:text-indigo-300 border border-indigo-500/20">
                        <CheckCircle2 className="h-4 w-4 shrink-0 text-indigo-600" />
                        <span>Dental arch curve and interdental spacing appear regular with no marked crowding or gaps.</span>
                      </div>
                    )
                  ) : (
                    alignmentFindings.map((c, i) => {
                      const itemNumber = viewOffset + fullList.indexOf(c) + 1;
                      return (
                        <div
                          key={`align-${i}`}
                          className="space-y-2 rounded-xl border-2 border-dashed border-indigo-500/40 bg-indigo-50/30 dark:bg-indigo-950/20 p-4 shadow-sm"
                        >
                          <div className="flex items-center justify-between gap-2">
                            <h5 className="font-semibold text-sm">
                              <span className="mr-2 text-indigo-600 dark:text-indigo-400 font-bold">
                                #{itemNumber}
                              </span>
                              {c.name}
                              {c.tooth && (
                                <span className="ml-2 rounded-md bg-indigo-500/20 text-indigo-700 dark:text-indigo-300 px-2 py-0.5 text-xs font-semibold">
                                  Teeth: {c.tooth}
                                </span>
                              )}
                            </h5>
                            <SeverityBadge s={c.severity} />
                          </div>

                          {/* Qualitative Gap / Alignment Observation */}
                          {c.gapMeasurement && (
                            <div className="flex items-center gap-2 rounded-md bg-indigo-500/20 px-2.5 py-1 text-xs font-semibold text-indigo-800 dark:text-indigo-200">
                              <span>Observation:</span>
                              <span className="underline decoration-indigo-400">{c.gapMeasurement}</span>
                            </div>
                          )}

                          <p className="text-xs text-muted-foreground leading-relaxed">{c.description}</p>

                          <div>
                            <div className="mb-1 flex justify-between text-[11px]">
                              <span className="text-muted-foreground">Alignment Confidence</span>
                              <span className="font-semibold text-foreground">
                                {Math.round(c.confidence * 100)}%
                              </span>
                            </div>
                            <Progress value={c.confidence * 100} className="h-1.5" />
                          </div>

                          <div className="rounded-lg bg-indigo-500/15 border border-indigo-500/20 p-2.5 text-xs text-indigo-950 dark:text-indigo-100">
                            <b>Orthodontic Guidance:</b> {c.recommendation}
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              )}
            </div>
          </Card>
        );
      })}
    </div>
  );
}
