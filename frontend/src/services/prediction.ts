/**
 * AI prediction service connecting to local Python FastAPI backend.
 * Endpoints:
 * - GET /health
 * - GET /api/v1/model-status
 * - POST /api/v1/predict (accepts multipart fields "upper", "lower", "front")
 */

export type Severity = "low" | "moderate" | "high";
export type ViewKey = "upper" | "lower" | "front";
export type FindingCategory = "condition" | "alignment";

export const VIEWS: { key: ViewKey; label: string; hint: string }[] = [
  { key: "upper", label: "Upper arch", hint: "Tilt head back, open wide, photograph the top teeth (a mirror helps)." },
  { key: "lower", label: "Lower arch", hint: "Tilt head down, open wide, photograph the bottom teeth." },
  { key: "front", label: "Front view", hint: "Smile with teeth together, camera straight in front." },
];

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface DetectedCondition {
  name: string;
  category: FindingCategory;
  condition?: string | undefined;
  findingType?: string | undefined;
  involvedTeeth?: string[] | undefined;
  gapMeasurement?: string | undefined;
  affectedRegion?: string | undefined;
  confidence: number; // 0..1
  severity: Severity;
  description: string;
  recommendation: string;
  box: Box;
  rawBox2d?: number[] | undefined;
  tooth?: string | undefined;
  toothNumber?: string | undefined;
}

export interface PredictionResult {
  views: Record<ViewKey, DetectedCondition[]>;
  modules?: {
    toothConditions: Record<ViewKey, DetectedCondition[]>;
    dentalAlignment: Record<ViewKey, DetectedCondition[]>;
  } | undefined;
  overallScore: number;
  isMock: boolean;
  screeningDisclaimer?: string | undefined;
  modelProvider?: string | undefined;
  inconclusive?: boolean | undefined;
  inconclusiveViews?: ViewKey[] | undefined;
  inconclusiveReasons?: Partial<Record<ViewKey, string>> | undefined;
  statusMessage?: string | undefined;
}

const DEFAULT_PREDICTION_API_URL = "http://127.0.0.1:8000/api/v1/predict";
const REQUEST_TIMEOUT_MS = 90_000;

/**
 * Returns the configured prediction API endpoint.
 * Reads VITE_PREDICTION_API_URL from environment variables (.env.local),
 * falling back to the standard local backend endpoint.
 */
export function getPredictionApiUrl(): string {
  const envUrl = (import.meta.env["VITE_PREDICTION_API_URL"] as string | undefined)?.trim();
  return envUrl && envUrl.length > 0 ? envUrl : DEFAULT_PREDICTION_API_URL;
}

interface RawFindingInput {
  name?: unknown;
  category?: unknown;
  condition?: unknown;
  findingType?: unknown;
  involvedTeeth?: unknown;
  gapMeasurement?: unknown;
  affectedRegion?: unknown;
  confidence?: unknown;
  severity?: unknown;
  description?: unknown;
  recommendation?: unknown;
  tooth?: unknown;
  toothNumber?: unknown;
  box?: unknown;
  boundingBox?: unknown;
  rawBox2d?: unknown;
}

function parseStrictBox(raw: unknown): Box | null {
  if (!raw || typeof raw !== "object") return null;
  const obj = raw as Record<string, unknown>;
  const x = obj["x"];
  const y = obj["y"];
  const w = obj["w"] ?? obj["width"];
  const h = obj["h"] ?? obj["height"];

  if (
    typeof x !== "number" ||
    typeof y !== "number" ||
    typeof w !== "number" ||
    typeof h !== "number"
  ) {
    return null;
  }
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(w) || !Number.isFinite(h)) {
    return null;
  }
  if (x < 0 || y < 0 || x > 1 || y > 1 || w <= 0 || h <= 0) {
    return null;
  }
  if (x + w > 1.001 || y + h > 1.001) {
    return null;
  }

  return {
    x,
    y,
    w: x + w > 1 ? Math.max(0, 1 - x) : w,
    h: y + h > 1 ? Math.max(0, 1 - y) : h,
  };
}

function normalizeCondition(raw: unknown): DetectedCondition | null {
  if (!raw || typeof raw !== "object") return null;
  const c = raw as RawFindingInput;

  const box = parseStrictBox(c.boundingBox ?? c.box);
  if (!box) return null;

  const rawConf = c.confidence ?? 0.85;
  if (typeof rawConf !== "number" || !Number.isFinite(rawConf) || rawConf < 0 || rawConf > 1) {
    return null;
  }

  let rawSev = typeof c.severity === "string" ? c.severity.toLowerCase().trim() : "low";
  if (rawSev === "mild") rawSev = "low";
  if (rawSev === "med" || rawSev === "medium") rawSev = "moderate";
  if (rawSev === "severe") rawSev = "high";
  const severityVal: Severity = ["low", "moderate", "high"].includes(rawSev)
    ? (rawSev as Severity)
    : "low";

  const categoryVal: FindingCategory = c.category === "alignment" ? "alignment" : "condition";
  const nameStr = typeof c.name === "string" ? c.name.trim() : "";
  const condStr = typeof c.condition === "string" ? c.condition.trim() : "";
  const resolvedName =
    nameStr || condStr || (categoryVal === "alignment" ? "Alignment Observation" : "Tooth Finding");

  const involvedTeeth = Array.isArray(c.involvedTeeth)
    ? c.involvedTeeth.filter((t): t is string => typeof t === "string" && t.trim().length > 0)
    : undefined;

  const cond: DetectedCondition = {
    name: resolvedName,
    category: categoryVal,
    condition: condStr || nameStr || undefined,
    findingType: typeof c.findingType === "string" && c.findingType ? c.findingType : undefined,
    involvedTeeth,
    gapMeasurement: typeof c.gapMeasurement === "string" && c.gapMeasurement ? c.gapMeasurement : undefined,
    affectedRegion: typeof c.affectedRegion === "string" && c.affectedRegion ? c.affectedRegion : undefined,
    confidence: rawConf,
    severity: severityVal,
    description: typeof c.description === "string" ? c.description : "",
    recommendation: typeof c.recommendation === "string" ? c.recommendation : "",
    box,
    rawBox2d:
      Array.isArray(c.rawBox2d) && c.rawBox2d.every((n) => typeof n === "number")
        ? c.rawBox2d
        : undefined,
  };

  const rawTooth = c.toothNumber ?? c.tooth;
  const toothVal =
    typeof rawTooth === "string" && rawTooth.trim()
      ? rawTooth.trim()
      : involvedTeeth && involvedTeeth.length > 0
        ? involvedTeeth.join(", ")
        : undefined;
  if (toothVal) {
    cond.tooth = toothVal;
    cond.toothNumber = toothVal;
  }

  return cond;
}

function normalizeConditionList(list: unknown): DetectedCondition[] {
  if (!Array.isArray(list)) return [];
  return list.map(normalizeCondition).filter((item): item is DetectedCondition => item !== null);
}

function fallbackStatusError(status: number): string {
  switch (status) {
    case 401:
      return "Invalid OpenRouter API key. Check OPENROUTER_API_KEY in the backend .env file.";
    case 402:
      return "OpenRouter account has insufficient credits or payment is required. Check your OpenRouter balance and key limits.";
    case 403:
      return "OpenRouter model or account permission restriction. Verify model access for your API key.";
    case 404:
      return "Configured OpenRouter vision model is unavailable or not found. Check OPENROUTER_MODEL in backend .env.";
    case 413:
      return "Uploaded image exceeds the maximum allowed size limit.";
    case 415:
      return "Unsupported image format. Please upload clear JPG, PNG, or WEBP photos.";
    case 429:
      return "OpenRouter rate limit exceeded. Please wait a moment and try again.";
    case 502:
      return "Vision provider returned a malformed or upstream error response. Please retry with clearer photos.";
    case 503:
      return "Screening service is not configured yet. Verify OPENROUTER_API_KEY in backend .env.";
    case 504:
      return "Dental screening request timed out while waiting for the OpenRouter vision model. Please try again.";
    case 500:
      return "Backend internal error during dental screening. Please check the backend terminal logs.";
    default:
      return `Screening service returned status ${status}.`;
  }
}

async function extractErrorMessage(res: Response): Promise<string> {
  const fallback = fallbackStatusError(res.status);
  const text = await res.text().catch(() => "");
  if (!text) return fallback;

  try {
    const errBody = JSON.parse(text) as { detail?: unknown; message?: unknown };
    const detail = errBody?.detail ?? errBody?.message;
    if (typeof detail === "string" && detail.trim()) {
      return detail.trim();
    }
    if (Array.isArray(detail) && detail.length > 0) {
      return detail
        .map((item) =>
          item && typeof item === "object" && "msg" in item
            ? String((item as { msg: unknown }).msg)
            : JSON.stringify(item)
        )
        .join("; ");
    }
    if (detail && typeof detail === "object" && "message" in detail) {
      const msg = (detail as { message?: unknown }).message;
      if (typeof msg === "string" && msg.trim()) return msg.trim();
    }
  } catch {
    if (!text.trim().startsWith("<")) {
      return text.trim();
    }
  }
  return fallback;
}

/**
 * Runs the screening through the local Python FastAPI backend.
 * Sends multipart form-data with fields: upper, lower, front.
 */
export async function analyzeDentalImages(
  files: Record<ViewKey, File>,
  _images?: Record<ViewKey, string>
): Promise<PredictionResult> {
  const apiUrl = getPredictionApiUrl();

  if (!files?.upper || !files?.lower || !files?.front) {
    throw new Error("All three dental images (upper, lower, and front) are required for screening.");
  }

  const formData = new FormData();
  formData.append("upper", files.upper);
  formData.append("lower", files.lower);
  formData.append("front", files.front);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let res: Response;
  try {
    // Do not set Content-Type manually so fetch adds the multipart boundary.
    res = await fetch(apiUrl, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
  } catch (err: unknown) {
    if (err instanceof Error && err.name === "AbortError") {
      throw new Error("Screening request timed out after 90 seconds. Please try again.");
    }
    const errorMsg = err instanceof Error ? err.message : String(err);
    if (errorMsg.includes("Failed to fetch") || errorMsg.includes("NetworkError") || errorMsg.includes("fetch")) {
      throw new Error(
        `Unable to reach the Python backend at ${apiUrl}. Please ensure the FastAPI backend server is running on http://127.0.0.1:8000.`
      );
    }
    throw new Error(`Connection error: ${errorMsg}`);
  } finally {
    clearTimeout(timeoutId);
  }

  if (!res.ok) {
    const message = await extractErrorMessage(res);
    throw new Error(message);
  }

  let rawData: unknown;
  try {
    rawData = await res.json();
  } catch {
    throw new Error("Screening service returned malformed JSON.");
  }

  if (!rawData || typeof rawData !== "object" || !("views" in rawData)) {
    throw new Error("Screening service returned an unexpected response structure.");
  }

  const data = rawData as Record<string, unknown>;
  const rawViews = data["views"] as Record<string, unknown> | undefined;
  if (!rawViews || typeof rawViews !== "object") {
    throw new Error("Screening response is missing view findings.");
  }

  const views: Record<ViewKey, DetectedCondition[]> = {
    upper: normalizeConditionList(rawViews["upper"]),
    lower: normalizeConditionList(rawViews["lower"]),
    front: normalizeConditionList(rawViews["front"]),
  };

  const rawModules = data["modules"] as Record<string, Record<string, unknown>> | undefined;
  const modules = rawModules && typeof rawModules === "object"
    ? {
        toothConditions: {
          upper: normalizeConditionList(rawModules["toothConditions"]?.["upper"]),
          lower: normalizeConditionList(rawModules["toothConditions"]?.["lower"]),
          front: normalizeConditionList(rawModules["toothConditions"]?.["front"]),
        },
        dentalAlignment: {
          upper: normalizeConditionList(rawModules["dentalAlignment"]?.["upper"]),
          lower: normalizeConditionList(rawModules["dentalAlignment"]?.["lower"]),
          front: normalizeConditionList(rawModules["dentalAlignment"]?.["front"]),
        },
      }
    : undefined;

  const rawScore = data["overallScore"];
  const overallScore =
    typeof rawScore === "number" && Number.isFinite(rawScore)
      ? Math.max(0, Math.min(100, Math.round(rawScore)))
      : 0;

  const inconclusiveViews = Array.isArray(data["inconclusiveViews"])
    ? data["inconclusiveViews"].filter(
        (v): v is ViewKey => v === "upper" || v === "lower" || v === "front"
      )
    : undefined;

  const rawReasons = data["inconclusiveReasons"];
  const inconclusiveReasons =
    rawReasons && typeof rawReasons === "object"
      ? (rawReasons as Partial<Record<ViewKey, string>>)
      : undefined;

  return {
    views,
    modules,
    overallScore,
    isMock: false,
    screeningDisclaimer:
      typeof data["screeningDisclaimer"] === "string" ? data["screeningDisclaimer"] : undefined,
    modelProvider: typeof data["modelProvider"] === "string" ? data["modelProvider"] : undefined,
    inconclusive: Boolean(data["inconclusive"]),
    inconclusiveViews,
    inconclusiveReasons,
    statusMessage: typeof data["statusMessage"] === "string" ? data["statusMessage"] : undefined,
  };
}

export const allConditions = (r: PredictionResult) =>
  VIEWS.flatMap((v) => (r.views[v.key] ?? []).map((c) => ({ ...c, view: v.label })));
