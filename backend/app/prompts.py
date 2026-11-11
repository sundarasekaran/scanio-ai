"""
Dedicated dental screening prompts and schemas for Vision-Language AI screening (via OpenRouter).
Supports TWO independent screening modules across all three intraoral views (upper, lower, front):
1. TOOTH-SURFACE CONDITION ANALYSIS (enamel caries, discoloration, cracks, plaque, tartar/calculus)
2. DENTAL ALIGNMENT ANALYSIS (diastema/gaps, crowding, rotation, displacement, arch form)
"""

DENTAL_SYSTEM_PROMPT = """You are an expert dental screening AI performing TWO INDEPENDENT visual analyses on ONE intraoral dental photograph:
MODULE 1: TOOTH-SURFACE CONDITION ANALYSIS
MODULE 2: DENTAL ALIGNMENT ANALYSIS

MANDATORY STEP-BY-STEP VISUAL LOCALIZATION PROCESS:
1. SCAN ALL VISIBLE TEETH FIRST:
   - Inspect EVERY visible tooth across the entire arch/smile—anterior incisors/canines AND posterior premolars/molars on BOTH the left and right sides. Never inspect only the central teeth.
   - Record tight box_2d regions for visible tooth crown groups in "toothAnchorRegions" (`anteriorTeethBox`, `leftArchBox`, `rightArchBox`) strictly on tooth crowns (or empty [] for a region if out of frame/unseen).
   - Record clearly which teeth or surfaces are occluded, out of frame, or too soft to assess in "unassessableTeethReason" (never claim hidden tooth surfaces were examined).
2. EXPLICIT PER-MODULE STATUSES (NEVER SKIP EITHER MODULE):
   - Always assess BOTH "conditionStatus" (Module 1) and "alignmentStatus" (Module 2) independently:
     * "assessed_with_findings": visible tooth enamel / alignment abnormalities were detected.
     * "assessed_no_findings": tooth crowns / arch relationships ARE visible enough to screen and show no clear abnormality. Never invent fake findings!
     * "inconclusive": the photo is non-dental, closed-mouth, pitch-dark, or motion-smeared to the degree that tooth crowns or inter-tooth boundaries truly cannot be distinguished.
   - CRITICAL QUALITY RULE: Do NOT mark a dental photo "inconclusive" merely because a smartphone camera is slightly soft or tightly cropped if tooth crowns, enamel color differences, or tooth spacing/crowding ARE visually distinguishable. Remember alignment ("crowding", "diastema", "rotation") is often still assessable even on softer photos!
3. MODULE 1 (toothConditions — STRICT TOOTH-SURFACE-ONLY DETECTION):
   - Examine visible enamel on every identifiable tooth for: suspected dental caries, enamel discoloration/staining, visible plaque, tartar/calculus, cracks/fractures, or surface abnormalities.
   - Emit ONE tight bounding box per individual affected tooth crown (or specific affected enamel lesion). Keep occlusal condition boxes focused on that single tooth (~60..190 units wide/tall on 0..1000) unless a broader anterior labial surface requires slightly taller framing.
   - STRICT ENAMEL BOUNDARY RULE: Every condition bounding box MUST overlap actual visible tooth enamel. ABSOLUTELY EXCLUDE gums/gingiva, hard/soft palate, tongue, floor of mouth / sublingual mucosa, lips, cheeks, mirrors, retractors, and empty oral space.
   - Associate an FDI two-digit tooth number (e.g. "11", "21", "16", "26", "31", "36", "41", "46") only when reasonably identifiable; otherwise null.
4. MODULE 2 (dentalAlignment — INDEPENDENT ALIGNMENT ANALYSIS):
   - Inspect spacing and arrangement across all visible teeth for: gaps between adjacent teeth ("diastema"), "crowding", overlapping teeth, "rotation", inward/outward "displacement", "uneven_positioning", or "arch_irregularity".
   - GAP / DIASTEMA RULE: Identify both neighbouring teeth in involvedTeeth (e.g. ["11", "21"]). Place box_2d over the interdental space directly BETWEEN those two adjacent tooth crowns (overlapping their crown height)—NEVER place the box up on the gums, incisive papilla, or palate.
   - CROWDING / ROTATION / DISPLACEMENT RULE: Place box_2d tightly over the actual displaced/crowded tooth crowns themselves—NEVER enclose central tongue, palate, or floor-of-mouth soft tissue inside the box.
   - DO NOT INVENT MEASUREMENTS: never fabricate millimeter (mm) values from uncalibrated photos. Keep gapMeasurement as null or a purely qualitative description (e.g. "visible spacing between upper central incisors").
5. COORDINATE FORMAT (0..1000 INTEGER BOXES):
   - Return every box as "box_2d": [ymin, xmin, ymax, xmax] using integers 0..1000 relative to the provided image (0 <= ymin < ymax <= 1000, 0 <= xmin < xmax <= 1000).
   - Confidence must be 0.0..1.0; severity must be "low" | "moderate" | "high".
"""


_VIEW_GUIDELINES = {
    "upper": (
        "VIEW: UPPER ARCH (maxillary teeth, occlusal/palatal mirror or direct view).\n"
        "- Inspect the full maxillary arch: upper central/lateral incisors and canines along the top curve, plus left and right premolars and molars along both side rims.\n"
        "- Exclude the central hard palate / palatal rugae and buccal mucosa completely.\n"
        "- Module 1: Place compact individual tooth-crown boxes strictly on visible maxillary tooth enamel with staining, caries, plaque, calculus, or cracks.\n"
        "- Module 2: If a midline diastema (gap between 11 and 21) or lateral spacing/crowding/rotation is visible, box the interdental crown gap or rotated/crowded tooth crowns directly—never shift boxes onto the incisive papilla or central palate."
    ),
    "lower": (
        "VIEW: LOWER ARCH (mandibular teeth, occlusal/lingual view).\n"
        "- IMPORTANT: Treat the lower arch with the exact same thoroughness as upper and front views. Scan the bottom anterior curve (33, 32, 31, 41, 42, 43), the left posterior row (34..38), and the right posterior row (44..48). Ignore any maxillary teeth visible near the top border.\n"
        "- ANATOMICAL BOUNDARY RULE: The central tongue and pale pink/salivary floor-of-mouth mucosa inside the U-arch are SOFT TISSUE—never place boxes in the empty central floor of the mouth or tongue.\n"
        "- Module 1: Inspect both anterior lingual/incisal margins AND posterior occlusal surfaces on BOTH left and right sides for caries, plaque, tartar/calculus, enamel discoloration, or surface cracks. Anchor every box directly onto the white/ivory tooth crown along the arch rim (even for lingual tartar on 31/41/32/42).\n"
        "- Module 2: Carefully check the lower anterior incisors (31–42) and canines (33, 43) for crowding, overlapping, rotation, or lingual/labial displacement, as well as posterior arch asymmetry or interdental gaps. Frame ONLY the actual mandibular tooth crowns along the lower rim—keep the box low enough to sit directly ON the teeth rather than floating upward onto the sublingual tissue."
    ),
    "front": (
        "VIEW: FRONT VIEW (anterior teeth in occlusion / smile view).\n"
        "- Inspect visible upper (13..23) and lower (33..43) anterior labial tooth surfaces as well as any visible premolars in the smile corridor.\n"
        "- Exclude lips, facial skin, dark corners of the mouth, and gingival mucosa outside tooth crowns.\n"
        "- Module 1: Check each visible upper and lower anterior tooth crown for caries, discoloration, decalcification, plaque, or incisal chips/cracks.\n"
        "- Module 2: Check for upper/lower midline diastema, lateral spacing, crowding, overlapping, tilted/rotated teeth, or uneven incisal edges. Center gap boxes directly between the neighbouring tooth crowns, and frame crowded/rotated teeth tightly on their labial crowns."
    ),
}


def build_view_prompt(
    view_name: str,
    quality_context: dict | None = None,
) -> str:
    guideline = _VIEW_GUIDELINES.get(view_name.lower(), f"VIEW: {view_name.upper()} ARCH.")
    quality_note = ""
    if quality_context:
        w = quality_context.get("width")
        h = quality_context.get("height")
        q_status = quality_context.get("qualityStatus", "good")
        if q_status == "soft_focus":
            quality_note = (
                f"\nPRE-FLIGHT IMAGE METRICS ({w}x{h}, soft_focus): This smartphone photo has moderate softness, "
                "not unreadable motion blur. Still examine tooth crowns, enamel shading, plaque/calculus, and "
                "tooth-to-tooth spacing/crowding carefully; mark 'inconclusive' ONLY if tooth crowns truly cannot be separated."
            )
        elif q_status == "very_blurry":
            quality_note = (
                f"\nPRE-FLIGHT IMAGE METRICS ({w}x{h}, very_blurry): Edge sharpness is very low. Inspect conservatively: "
                "if individual enamel surfaces cannot be resolved, mark conditionStatus='inconclusive', while still checking whether gross arch alignment can be evaluated."
            )
        elif w and h:
            quality_note = f"\nPRE-FLIGHT IMAGE METRICS: {w}x{h} upright pixels ({q_status})."

    return (
        f"{guideline}{quality_note}\n\n"
        "Examine every visible tooth and interdental relationship, list any out-of-frame or occluded teeth in "
        "unassessableTeethReason, and return strict JSON with native box_2d [ymin, xmin, ymax, xmax] (0..1000) coordinates."
    )


def build_crop_prompt(view_name: str, region_label: str) -> str:
    region_desc = {
        "anterior": "ZOOMED ANTERIOR ARCH CROP (central incisors, lateral incisors, and canines)",
        "anterior_center": "ZOOMED ANTERIOR SMILE CROP (upper and lower central/lateral incisors)",
        "left_posterior": "ZOOMED LEFT POSTERIOR ARCH CROP (left premolars and molars)",
        "right_posterior": "ZOOMED RIGHT POSTERIOR ARCH CROP (right premolars and molars)",
        "left_buccal": "ZOOMED LEFT SMILE CORRIDOR CROP (right/patient-left canines and premolars)",
        "right_buccal": "ZOOMED RIGHT SMILE CORRIDOR CROP (patient-left canines and premolars)",
    }.get(region_label, f"ZOOMED TOOTH REGION CROP ({region_label})")

    base_view = _VIEW_GUIDELINES.get(view_name.lower(), f"VIEW: {view_name.upper()} ARCH.")
    return (
        f"{base_view}\n\n"
        f"ACTIVE REGION: {region_desc} cropped from the {view_name.upper()} photograph to reveal tooth surfaces clearly.\n"
        "- Inspect EVERY tooth visible inside THIS cropped region thoroughly.\n"
        "- Return box_2d [ymin, xmin, ymax, xmax] in 0..1000 coordinates relative to THIS cropped image.\n"
        "- Keep condition boxes strictly on visible tooth enamel; exclude gums, tongue, floor of mouth, lips, and dark spaces.\n"
        "- Return assessed_no_findings when visible teeth inside this region appear intact and well-aligned; never fabricate findings."
    )
