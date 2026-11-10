import { useState } from "react";
import { cn } from "@/lib/utils";
import type { DetectedCondition, FindingCategory } from "@/services/prediction";
import { Activity, GitCommit, Sparkles } from "lucide-react";

const ringCondition = {
  low: "border-emerald-500 bg-emerald-500/10 hover:bg-emerald-500/20",
  moderate: "border-amber-500 bg-amber-500/15 hover:bg-amber-500/25",
  high: "border-rose-500 bg-rose-500/20 hover:bg-rose-500/30",
};

const tagCondition = {
  low: "bg-emerald-600 text-white",
  moderate: "bg-amber-600 text-white",
  high: "bg-rose-600 text-white",
};

export function AnnotatedImage({
  src,
  conditions,
  alt,
  offset = 0,
}: {
  src: string;
  conditions: DetectedCondition[];
  alt: string;
  offset?: number;
}) {
  const [activeIdx, setActiveIdx] = useState<number | null>(null);
  const [filter, setFilter] = useState<"all" | FindingCategory>("all");

  const conditionCount = conditions.filter((c) => c.category !== "alignment").length;
  const alignmentCount = conditions.filter((c) => c.category === "alignment").length;
  const hasBothCategories = conditionCount > 0 && alignmentCount > 0;

  const visibleConditions = conditions
    .map((c, originalIdx) => ({ c, originalIdx }))
    .filter(({ c }) => {
      if (filter === "all") return true;
      if (filter === "alignment") return c.category === "alignment";
      return c.category !== "alignment";
    });

  return (
    <div className="space-y-2 select-none">
      {/* Category Filter Pills (if multiple categories present) */}
      {hasBothCategories && (
        <div className="flex flex-wrap items-center gap-1.5 p-1 rounded-xl bg-muted/80 text-xs font-medium">
          <button
            type="button"
            onClick={() => setFilter("all")}
            className={cn(
              "px-2.5 py-1 rounded-lg transition-all",
              filter === "all"
                ? "bg-background text-foreground shadow-sm font-semibold"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            All Overlays ({conditions.length})
          </button>
          <button
            type="button"
            onClick={() => setFilter("condition")}
            className={cn(
              "px-2.5 py-1 rounded-lg transition-all flex items-center gap-1",
              filter === "condition"
                ? "bg-background text-emerald-600 dark:text-emerald-400 shadow-sm font-semibold"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            Tooth Surfaces ({conditionCount})
          </button>
          <button
            type="button"
            onClick={() => setFilter("alignment")}
            className={cn(
              "px-2.5 py-1 rounded-lg transition-all flex items-center gap-1",
              filter === "alignment"
                ? "bg-background text-indigo-600 dark:text-indigo-400 shadow-sm font-semibold"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <span className="h-2 w-2 rounded-full bg-indigo-500" />
            Alignment ({alignmentCount})
          </button>
        </div>
      )}

      {/* Interactive Image Frame */}
      <div className="relative overflow-hidden rounded-2xl border border-border bg-muted">
        <img src={src} alt={alt} className="block w-full h-auto object-contain" />

        {visibleConditions.map(({ c, originalIdx }) => {
          const isHovered = activeIdx === originalIdx;
          const boxNumber = offset + originalIdx + 1;
          const isNearTop = c.box.y < 0.22;
          const isAlignment = c.category === "alignment";
          const isDiastema = isAlignment && (c.findingType === "diastema" || c.name.toLowerCase().includes("gap") || c.name.toLowerCase().includes("diastema"));

          return (
            <div
              key={originalIdx}
              tabIndex={0}
              role="button"
              aria-label={`Finding #${boxNumber} [${isAlignment ? "Alignment" : "Surface"}]: ${c.name}${c.tooth ? `, Tooth ${c.tooth}` : ""}`}
              onMouseEnter={() => setActiveIdx(originalIdx)}
              onMouseLeave={() => setActiveIdx(null)}
              onFocus={() => setActiveIdx(originalIdx)}
              onBlur={() => setActiveIdx(null)}
              onClick={() => setActiveIdx(activeIdx === originalIdx ? null : originalIdx)}
              className={cn(
                "absolute rounded-lg transition-all cursor-pointer focus:outline-none",
                isAlignment
                  ? "border-2 border-dashed border-indigo-500 bg-indigo-500/15 hover:bg-indigo-500/25"
                  : cn("border-2 border-solid", ringCondition[c.severity]),
                isHovered && "ring-2 ring-primary ring-offset-1 z-30 scale-[1.02]"
              )}
              style={{
                left: `${c.box.x * 100}%`,
                top: `${c.box.y * 100}%`,
                width: `${c.box.w * 100}%`,
                height: `${c.box.h * 100}%`,
              }}
            >
              {/* Interdental Gap Marker Indicator */}
              {isDiastema && (
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-85">
                  <span className="text-[10px] md:text-xs font-black tracking-tighter text-indigo-400 bg-indigo-950/70 px-1 rounded">
                    ↔
                  </span>
                </div>
              )}

              {/* Number/Icon badge */}
              <span
                className={cn(
                  "absolute -top-3 -left-2.5 grid h-6 w-6 place-items-center rounded-full text-xs font-bold shadow-md transition-transform",
                  isAlignment ? "bg-indigo-600 text-white" : tagCondition[c.severity],
                  isHovered && "scale-110"
                )}
              >
                {boxNumber}
              </span>

              {/* Hover tooltip displaying condition label, tooth number, gap measurement, and confidence */}
              {isHovered && (
                <div
                  className={cn(
                    "absolute z-40 w-64 rounded-xl border border-border bg-popover/95 p-3 text-popover-foreground shadow-2xl backdrop-blur-md transition-opacity pointer-events-none",
                    isNearTop ? "top-[calc(100%+8px)]" : "bottom-[calc(100%+8px)]"
                  )}
                  style={{
                    left: c.box.x > 0.6 ? "auto" : "0",
                    right: c.box.x > 0.6 ? "0" : "auto",
                  }}
                >
                  <div className="flex items-center justify-between gap-1 text-[11px]">
                    <span className="font-bold text-primary">#{boxNumber}</span>
                    <span
                      className={cn(
                        "rounded px-1.5 py-0.5 font-semibold text-[10px] uppercase tracking-wider",
                        isAlignment
                          ? "bg-indigo-500/20 text-indigo-600 dark:text-indigo-400"
                          : "bg-emerald-500/20 text-emerald-600 dark:text-emerald-400"
                      )}
                    >
                      {isAlignment ? "Alignment" : "Tooth Surface"}
                    </span>
                    <span className="font-medium text-muted-foreground">
                      {Math.round(c.confidence * 100)}%
                    </span>
                  </div>

                  <p className="mt-1.5 font-semibold text-xs leading-snug">{c.name}</p>

                  {/* Anatomical Details: Teeth involved & Gap measurements */}
                  <div className="mt-1.5 flex flex-wrap gap-1 text-[11px]">
                    {c.tooth && (
                      <span className="rounded bg-muted px-1.5 py-0.5 text-foreground font-medium">
                        {isAlignment ? `Teeth: ${c.tooth}` : `Tooth ${c.tooth}`}
                      </span>
                    )}
                    {c.gapMeasurement && (
                      <span className="rounded bg-indigo-500/20 text-indigo-700 dark:text-indigo-300 px-1.5 py-0.5 font-semibold">
                        📏 {c.gapMeasurement}
                      </span>
                    )}
                    {c.affectedRegion && !c.gapMeasurement && (
                      <span className="rounded bg-muted px-1.5 py-0.5 text-muted-foreground">
                        {c.affectedRegion}
                      </span>
                    )}
                  </div>

                  <p className="mt-1.5 text-[11px] text-muted-foreground line-clamp-2">
                    {c.description}
                  </p>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
