import { afterEach, describe, expect, it, vi } from "vitest";
import { analyzeDentalImages, wakeBackend } from "./prediction";

afterEach(() => {
  vi.unstubAllGlobals();
});

const makeFiles = () => ({
  upper: new File(["u"], "upper.jpg", { type: "image/jpeg" }),
  lower: new File(["l"], "lower.jpg", { type: "image/jpeg" }),
  front: new File(["f"], "front.jpg", { type: "image/jpeg" }),
});

describe("prediction service", () => {
  it("sends multipart upper, lower, and front files and drops findings with invalid boxes", async () => {
    const payload = {
      overallScore: 86,
      inconclusive: false,
      inconclusiveViews: [],
      inconclusiveReasons: {},
      views: {
        upper: [
          {
            name: "Enamel Discoloration",
            category: "condition",
            confidence: 0.88,
            severity: "low",
            description: "Surface staining on molar enamel.",
            recommendation: "Schedule routine polish.",
            tooth: "16",
            box: { x: 0.72, y: 0.42, w: 0.1, h: 0.12 },
          },
          {
            name: "Invalid Negative Box",
            category: "condition",
            confidence: 0.9,
            severity: "low",
            description: "Should be rejected instead of clamped.",
            recommendation: "N/A",
            box: { x: -0.1, y: 0.2, w: 0.1, h: 0.1 },
          },
        ],
        lower: [],
        front: [
          {
            name: "Midline Diastema (Gap)",
            category: "alignment",
            findingType: "diastema",
            involvedTeeth: ["11", "21"],
            confidence: 0.91,
            severity: "low",
            description: "Visible spacing between upper central incisors.",
            recommendation: "Orthodontic review if desired.",
            box: { x: 0.46, y: 0.2, w: 0.08, h: 0.1 },
          },
          {
            name: "Missing Box Item",
            category: "alignment",
            confidence: 0.7,
            severity: "low",
            description: "Missing box must be rejected.",
            recommendation: "N/A",
          },
        ],
      },
    };

    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(Response.json(payload));
    vi.stubGlobal("fetch", fetchMock);

    const res = await analyzeDentalImages(makeFiles());
    expect(fetchMock).toHaveBeenCalledTimes(1);

    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toContain("/api/v1/predict");
    expect(init?.method).toBe("POST");
    expect(init?.body).toBeInstanceOf(FormData);
    const fd = init?.body as FormData;
    expect([...fd.keys()]).toEqual(["upper", "lower", "front"]);

    expect(res.overallScore).toBe(86);
    expect(res.views.upper).toHaveLength(1);
    expect(res.views.upper[0]?.name).toBe("Enamel Discoloration");
    expect(res.views.front).toHaveLength(1);
    expect(res.views.front[0]?.category).toBe("alignment");
  });

  it("surfaces clear backend error messages for OpenRouter API failures", async () => {
    const cases: Array<[number, string, RegExp]> = [
      [401, "Invalid OpenRouter API key.", /Invalid OpenRouter API key/],
      [402, "OpenRouter account has insufficient credits or payment is required.", /insufficient credits/],
      [403, "OpenRouter model or account permission restriction.", /permission restriction/],
      [404, "Configured OpenRouter model 'google/gemini-2.5-flash' is unavailable.", /unavailable/],
      [429, "OpenRouter rate limit exceeded.", /rate limit exceeded/],
      [504, "OpenRouter screening request timed out.", /timed out/],
      [415, "Unsupported image format.", /Unsupported image format/],
      [502, "OpenRouter returned malformed JSON.", /malformed JSON/],
    ];

    for (const [status, detail, expected] of cases) {
      const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
        new Response(JSON.stringify({ detail }), {
          status,
          headers: { "Content-Type": "application/json" },
        })
      );
      vi.stubGlobal("fetch", fetchMock);
      await expect(analyzeDentalImages(makeFiles())).rejects.toThrow(expected);
    }
  });

  it("provides status-specific fallback messages when backend error body has no detail", async () => {
    const cases: Array<[number, RegExp]> = [
      [401, /Invalid vision API key/],
      [402, /insufficient credits or quota/],
      [403, /permission restriction/],
      [404, /NVIDIA_MODEL/],
      [429, /rate limit exceeded/],
      [502, /malformed or upstream error/],
      [503, /NVIDIA_API_KEY/],
    ];
    for (const [status, expected] of cases) {
      vi.stubGlobal("fetch", vi.fn<typeof fetch>().mockResolvedValue(new Response("", { status })));
      await expect(analyzeDentalImages(makeFiles())).rejects.toThrow(expected);
    }
  });

  it("preserves inconclusive status fields", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      Response.json({
        overallScore: 0,
        inconclusive: true,
        inconclusiveViews: ["upper", "lower", "front"],
        inconclusiveReasons: { front: "Blurry photo" },
        statusMessage: "Screening inconclusive",
        views: { upper: [], lower: [], front: [] },
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    const res = await analyzeDentalImages(makeFiles());
    expect(res.inconclusive).toBe(true);
    expect(res.inconclusiveViews).toEqual(["upper", "lower", "front"]);
    expect(res.statusMessage).toBe("Screening inconclusive");
  });

  it("preserves explicit viewStatuses, moduleStatuses, imageQuality, and viewDiagnostics", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      Response.json({
        overallScore: 84,
        inconclusive: false,
        viewStatuses: {
          upper: "assessed_with_findings",
          lower: "assessed_with_findings",
          front: "assessed_no_findings",
        },
        moduleStatuses: {
          toothConditions: {
            upper: "assessed_with_findings",
            lower: "assessed_no_findings",
            front: "assessed_no_findings",
          },
          dentalAlignment: {
            upper: "assessed_no_findings",
            lower: "assessed_with_findings",
            front: "assessed_no_findings",
          },
        },
        imageQuality: {
          lower: {
            width: 1600,
            height: 1395,
            aspectRatio: 1.147,
            minDimension: 1395,
            laplacianVariance: 23.3,
            edgeContrast: 12.1,
            meanBrightness: 148.9,
            qualityStatus: "soft_focus",
            qualityNotes: "Moderate softness",
          },
        },
        viewDiagnostics: {
          lower: {
            viewStatus: "assessed_with_findings",
            conditionStatus: "assessed_no_findings",
            alignmentStatus: "assessed_with_findings",
            unassessableTeethReason: "Third molars out of frame",
            rawConditionsCount: 0,
            rawAlignmentCount: 1,
            rejectedCount: 0,
            cropPassUsed: true,
            cropsAnalyzed: ["anterior", "left_posterior", "right_posterior"],
          },
        },
        views: { upper: [], lower: [], front: [] },
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    const res = await analyzeDentalImages(makeFiles());
    expect(res.viewStatuses?.lower).toBe("assessed_with_findings");
    expect(res.moduleStatuses?.dentalAlignment?.lower).toBe("assessed_with_findings");
    expect(res.imageQuality?.lower?.qualityStatus).toBe("soft_focus");
    expect(res.viewDiagnostics?.lower?.cropPassUsed).toBe(true);
  });

  it("pings backend /health endpoint quietly on wakeBackend()", () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(new Response('{"status":"ok"}'));
    vi.stubGlobal("fetch", fetchMock);
    wakeBackend();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]?.[0]).toMatch(/\/health$/);
  });
});
