import { createServerFn } from "@tanstack/react-start";

type Images = { upper: string; lower: string; front: string };

const condition = {
  type: "object",
  additionalProperties: false,
  required: ["name", "confidence", "severity", "description", "recommendation", "tooth", "box"],
  properties: {
    name: { type: "string" },
    confidence: { type: "number" },
    severity: { type: "string", enum: ["low", "moderate", "high"] },
    description: { type: "string" },
    recommendation: { type: "string" },
    tooth: { type: ["string", "null"] },
    box: {
      type: "object",
      additionalProperties: false,
      required: ["x", "y", "w", "h"],
      properties: { x: { type: "number" }, y: { type: "number" }, w: { type: "number" }, h: { type: "number" } },
    },
  },
};
const schema = {
  type: "object",
  additionalProperties: false,
  required: ["overallScore", "views"],
  properties: {
    overallScore: { type: "number" },
    views: {
      type: "object",
      additionalProperties: false,
      required: ["upper", "lower", "front"],
      properties: {
        upper: { type: "array", items: condition },
        lower: { type: "array", items: condition },
        front: { type: "array", items: condition },
      },
    },
  },
};

const PROMPT = `You are a dental screening assistant. You receive three photos of the same person: UPPER ARCH, LOWER ARCH and FRONT VIEW.
For EACH photo, carefully inspect every visible tooth and list ALL visible concerns, for example:
gaps between teeth (diastema / spacing), crowding, misalignment or rotated teeth, overbite/underbite signs, possible cavities or dark spots,
staining/discoloration, plaque, tartar/calculus, chipped or cracked teeth, worn enamel, missing teeth, gum redness/swelling or recession.
Report each separate issue as its own item. Do not invent issues that are not visible; return an empty list for a view if nothing is visible.
For every item give a bounding box in normalized coordinates (0..1, x,y = top-left corner, relative to that photo's width/height)
that tightly surrounds ONLY the affected tooth/teeth (or, for a gap, the space between the two teeth). Never place a box on lips, skin or background.
Give tooth as FDI number(s) if you can tell, else null. confidence is 0..1. Use plain, non-alarming language; this is a screening, not a diagnosis.
overallScore is an oral health score 0-100.`;

export const analyzeTeeth = createServerFn({ method: "POST" })
  .inputValidator((d: Images) => {
    for (const k of ["upper", "lower", "front"] as const) {
      if (typeof d?.[k] !== "string" || !/^data:image\/(jpeg|jpg|png);base64,/.test(d[k])) throw new Error(`Invalid ${k} image`);
      if (d[k].length > 4_000_000) throw new Error(`${k} image too large`);
    }
    return d;
  })
  .handler(async ({ data }) => {
    const key = process.env["LOVABLE_API_KEY"];
    if (!key) throw new Error("AI service is not configured");
    const res = await fetch("https://ai.gateway.lovable.dev/v1/responses", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Lovable-API-Key": key, "X-Lovable-AIG-SDK": "fetch" },
      body: JSON.stringify({
        model: "openai/gpt-6-astra",
        stream: true,
        store: false,
        reasoning: { effort: "medium", summary: "auto" },
        include: ["reasoning.encrypted_content"],
        text: { format: { type: "json_schema", name: "dental_screening", strict: true, schema } },
        input: [{
          role: "user",
          content: [
            { type: "input_text", text: PROMPT },
            { type: "input_text", text: "UPPER ARCH:" }, { type: "input_image", image_url: data.upper, detail: "high" },
            { type: "input_text", text: "LOWER ARCH:" }, { type: "input_image", image_url: data.lower, detail: "high" },
            { type: "input_text", text: "FRONT VIEW:" }, { type: "input_image", image_url: data.front, detail: "high" },
          ],
        }],
      }),
    });
    if (!res.ok || !res.body) {
      const body = await res.text().catch(() => "");
      console.error("AI gateway error", res.status, body);
      if (res.status === 429) throw new Error("Too many screenings right now — please try again in a minute.");
      if (res.status === 402) throw new Error("AI credits are used up. Please add credits to continue.");
      throw new Error(`Screening failed (${res.status})`);
    }
    // Consume the SSE stream and collect the output text.
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "", text = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload || payload === "[DONE]") continue;
        try {
          const ev = JSON.parse(payload);
          if (ev.type === "response.output_text.delta") text += ev.delta;
          if (ev.type === "response.failed" || ev.type === "error") throw new Error("Screening failed");
        } catch (e) { if ((e as Error).message === "Screening failed") throw e; }
      }
    }
    if (!text) throw new Error("The AI could not analyze these photos. Please try clearer photos.");
    type C = { name: string; confidence: number; severity: "low" | "moderate" | "high"; description: string; recommendation: string; tooth: string | null; box: { x: number; y: number; w: number; h: number } };
    return JSON.parse(text) as { overallScore: number; views: { upper: C[]; lower: C[]; front: C[] } };
  });
