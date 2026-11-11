import type { Screening } from "./store";
import { VIEWS, type ModuleViewStatus } from "./prediction";

const colors = { low: "#16a34a", moderate: "#d97706", high: "#dc2626" };
const esc = (str: string) =>
  str.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");

const statusLabel = (st: ModuleViewStatus) => {
  if (st === "assessed_with_findings") return "Assessed — Findings Detected";
  if (st === "inconclusive") return "Inconclusive";
  return "Assessed — No Findings Detected";
};

const statusPillStyle = (st: ModuleViewStatus) => {
  if (st === "assessed_with_findings") return "background:#fef3c7;color:#92400e;border:1px solid #fcd34d";
  if (st === "inconclusive") return "background:#fee2e2;color:#991b1b;border:1px solid #fca5a5";
  return "background:#d1fae5;color:#065f46;border:1px solid #6ee7b7";
};

const resolveStatus = (explicit: ModuleViewStatus | undefined, count: number, isInc: boolean): ModuleViewStatus => {
  if (explicit) return explicit;
  if (count > 0) return "assessed_with_findings";
  return isInc ? "inconclusive" : "assessed_no_findings";
};

export function downloadReport(s: Screening, patient: string) {
  let globalIndex = 0;

  const sections = VIEWS.map((v) => {
    const fullList = s.views[v.key] ?? [];
    const viewOffset = globalIndex;
    globalIndex += fullList.length;

    const surfaceConditions = fullList.filter((c) => c.category !== "alignment");
    const alignmentFindings = fullList.filter((c) => c.category === "alignment");
    const viewInconclusiveReason = s.inconclusiveReasons?.[v.key];
    const isViewInconclusive = Boolean(s.inconclusiveViews?.includes(v.key) || viewInconclusiveReason);
    const condStatus = resolveStatus(s.moduleStatuses?.toothConditions?.[v.key], surfaceConditions.length, isViewInconclusive);
    const alignStatus = resolveStatus(s.moduleStatuses?.dentalAlignment?.[v.key], alignmentFindings.length, isViewInconclusive);
    const viewQuality = s.imageQuality?.[v.key];
    const viewDiag = s.viewDiagnostics?.[v.key];
    const qualityBadge = viewQuality
      ? `<span style="margin-left:8px;font-size:11px;font-weight:500;color:#475569;background:#f1f5f9;padding:2px 8px;border-radius:999px">${viewQuality.width}&times;${viewQuality.height} &middot; ${esc(viewQuality.qualityStatus.replaceAll("_", " "))}</span>`
      : "";
    const visibilityNote = viewDiag?.unassessableTeethReason
      ? `<p style="font-size:12px;color:#475569;background:#f8fafc;border:1px solid #e2e8f0;padding:8px 10px;border-radius:8px;margin-top:8px"><b>Visibility note:</b> ${esc(viewDiag.unassessableTeethReason)}</p>`
      : "";

    // Overlays for image
    const boxes = fullList
      .map((c) => {
        const idx = viewOffset + fullList.indexOf(c) + 1;
        const isAlign = c.category === "alignment";
        const borderColor = isAlign ? "#6366f1" : colors[c.severity];
        const borderStyle = isAlign ? "dashed" : "solid";
        return `<div style="position:absolute;left:${c.box.x * 100}%;top:${c.box.y * 100}%;width:${c.box.w * 100}%;height:${c.box.h * 100}%;border:2px ${borderStyle} ${borderColor};border-radius:6px"><span style="position:absolute;top:-10px;left:-8px;background:${borderColor};color:#fff;border-radius:50%;width:20px;height:20px;font:bold 11px sans-serif;display:grid;place-items:center">${idx}</span></div>`;
      })
      .join("");

    // Section A table rows
    const conditionRows =
      surfaceConditions
        .map((c) => {
          const idx = viewOffset + fullList.indexOf(c) + 1;
          const toothStr = c.tooth ? `Tooth ${esc(c.tooth)}` : "—";
          return `<tr><td>#${idx}</td><td><b>${esc(c.name)}</b></td><td>${toothStr}</td><td>${Math.round(c.confidence * 100)}%</td><td>${esc(c.severity)}</td><td>${esc(c.description)}</td><td>${esc(c.recommendation)}</td></tr>`;
        })
        .join("") || (condStatus === "inconclusive"
          ? `<tr><td colspan="7" style="color:#92400e;background:#fffbeb">${esc(viewInconclusiveReason || "Inconclusive image: unable to reliably inspect tooth surfaces.")}</td></tr>`
          : `<tr><td colspan="7" style="color:#059669;background:#ecfdf5">Assessed — No tooth surface concerns detected in visible enamel.</td></tr>`);

    // Section B table rows
    const alignmentRows =
      alignmentFindings
        .map((c) => {
          const idx = viewOffset + fullList.indexOf(c) + 1;
          const teethStr = c.tooth ? `Teeth: ${esc(c.tooth)}` : "—";
          const gapStr = c.gapMeasurement ? `<br><small style="color:#4338ca"><b>Observation:</b> ${esc(c.gapMeasurement)}</small>` : "";
          return `<tr><td>#${idx}</td><td><b>${esc(c.name)}</b>${gapStr}</td><td>${teethStr}</td><td>${Math.round(c.confidence * 100)}%</td><td>${esc(c.severity)}</td><td>${esc(c.description)}</td><td>${esc(c.recommendation)}</td></tr>`;
        })
        .join("") || (alignStatus === "inconclusive"
          ? `<tr><td colspan="7" style="color:#92400e;background:#fffbeb">${esc(viewInconclusiveReason || "Inconclusive image: unable to reliably inspect dental alignment.")}</td></tr>`
          : `<tr><td colspan="7" style="color:#4338ca;background:#eef2ff">Assessed — No dental alignment or arch irregularities detected in visible teeth.</td></tr>`);

    return `
<div class="view-block">
  <h2>${esc(v.label)}${qualityBadge}</h2>
  <div style="position:relative;display:inline-block;max-width:440px;margin-bottom:16px">
    <img src="${s.images[v.key]}" style="width:100%;display:block;border-radius:12px;border:1px solid #cbd5e1" alt="${esc(v.label)}"/>
    ${boxes}
  </div>
  ${visibilityNote}

  <h3 class="sec-title sec-a">Section A: Tooth-Surface Condition Findings (Module 1) <span style="float:right;font-size:11px;padding:1px 8px;border-radius:999px;${statusPillStyle(condStatus)}">${statusLabel(condStatus)}</span></h3>
  <table>
    <thead><tr><th>#</th><th>Condition</th><th>Tooth</th><th>Confidence</th><th>Severity</th><th>Description</th><th>Recommendation</th></tr></thead>
    <tbody>${conditionRows}</tbody>
  </table>

  <h3 class="sec-title sec-b">Section B: Dental Alignment Analysis (Module 2) <span style="float:right;font-size:11px;padding:1px 8px;border-radius:999px;${statusPillStyle(alignStatus)}">${statusLabel(alignStatus)}</span></h3>
  <table>
    <thead><tr><th>#</th><th>Alignment Observation</th><th>Involved Teeth</th><th>Confidence</th><th>Severity</th><th>Description</th><th>Orthodontic Recommendation</th></tr></thead>
    <tbody>${alignmentRows}</tbody>
  </table>
</div>`;
  }).join("");

  const allFindings = VIEWS.flatMap((v) => s.views[v.key] ?? []);
  const totalCond = allFindings.filter((c) => c.category !== "alignment").length;
  const totalAlign = allFindings.filter((c) => c.category === "alignment").length;
  const allInconclusive = Boolean(s.inconclusive) && (s.inconclusiveViews?.length ?? 0) >= VIEWS.length;
  const scoreDisplay = allInconclusive ? "Inconclusive" : `${s.overallScore}/100`;

  const html = `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Scanio AI — Dual-Module Screening Report</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; max-width: 900px; margin: 36px auto; color: #1e293b; padding: 0 16px; line-height: 1.5; }
    h1 { color: #0f766e; margin-bottom: 4px; }
    .meta-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin: 20px 0 30px; display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
    .meta-item b { display: block; font-size: 11px; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em; }
    .meta-item span { font-size: 16px; font-weight: 600; color: #0f172a; }
    .view-block { margin-bottom: 40px; padding-bottom: 24px; border-bottom: 2px solid #e2e8f0; }
    .sec-title { font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; margin: 18px 0 8px; padding: 6px 10px; border-radius: 6px; }
    .sec-a { background: #ecfdf5; color: #065f46; border-left: 4px solid #10b981; }
    .sec-b { background: #eef2ff; color: #3730a3; border-left: 4px solid #6366f1; }
    table { width: 100%; border-collapse: collapse; margin: 8px 0 20px; font-size: 13px; }
    td, th { border: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; vertical-align: top; }
    th { background: #f1f5f9; font-weight: 600; color: #334155; }
    .note { background: #fffbeb; border: 1px solid #fde68a; color: #92400e; padding: 14px; border-radius: 10px; font-size: 13px; margin-top: 24px; }
  </style>
</head>
<body>
  <h1>Scanio AI — Dental Screening Report</h1>
  <p style="color: #64748b; margin-top: 0;">Comprehensive Dual-Module Analysis (Tooth Surfaces & Dental Alignment)</p>

  <div class="meta-box">
    <div class="meta-item"><b>Patient Name</b><span>${patient}</span></div>
    <div class="meta-item"><b>Report Date</b><span>${new Date(s.createdAt).toLocaleDateString()}</span></div>
    <div class="meta-item"><b>Oral Health Score</b><span style="color:#0f766e">${scoreDisplay}</span></div>
    <div class="meta-item"><b>Tooth Surface Findings</b><span>${totalCond}</span></div>
    <div class="meta-item"><b>Alignment Observations</b><span>${totalAlign}</span></div>
  </div>

  ${sections}

  <p class="note">
    <b>Disclaimer:</b> ${s.isMock ? "These results were generated by a demo simulation. " : ""}This preliminary report was generated using AI-assisted vision models. It is NOT a clinical diagnosis or treatment plan. Bounding boxes are localized to visible tooth surfaces and interdental spaces. Please schedule a clinical examination with a licensed dentist or orthodontic specialist.
  </p>
</body>
</html>`;

  const url = URL.createObjectURL(new Blob([html], { type: "text/html" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `scanio-report-${s.id.slice(0, 8)}.html`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
