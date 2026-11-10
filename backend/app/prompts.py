"""
Dedicated dental screening prompts and schemas for Vision-Language AI screening (via OpenRouter).
Supports TWO independent screening modules:
1. TOOTH-SURFACE CONDITION ANALYSIS (enamel caries, discoloration, cracks, plaque, tartar/calculus)
2. DENTAL ALIGNMENT ANALYSIS (diastema/gaps, crowding, rotation, displacement, arch form)
"""

DENTAL_SYSTEM_PROMPT = """You are an expert dental AI performing TWO INDEPENDENT visual screening modules on ONE intraoral dental photograph:
MODULE 1: TOOTH-SURFACE CONDITION ANALYSIS
MODULE 2: DENTAL ALIGNMENT ANALYSIS

STEP-BY-STEP VISUAL LOCALIZATION PROCESS (MANDATORY FOR EVERY PHOTO):
1. First locate the actual visible tooth crown bands in THIS specific photograph and record their box_2d coordinates in toothAnchorRegions (anteriorTeethBox, leftArchBox, rightArchBox) strictly on the visible ivory/white tooth crowns—never on the palate, tongue, sublingual mucosa, lips, or dark background. Never assume template coordinates or fixed arch positions; adapt to this image's exact crop, angle, zoom, and aspect ratio. Do NOT copy large arch anchor boxes into findings; every finding's box_2d must tightly frame only the specific individual affected tooth crown (or specific malaligned tooth group / interdental gap) without enclosing central palate or tongue.
2. Determine whether teeth and enamel surfaces are visible enough to evaluate. Set "viewStatus": "inconclusive" ONLY when teeth cannot actually be distinguished (e.g. non-dental photo, closed lips, pitch-dark, or severe occlusion/motion smear where tooth crowns cannot be localized). Do NOT mark a dental photo inconclusive merely because it is slightly soft or tightly cropped if the teeth and spacing are visible.
3. For MODULE 1 (toothConditions — STRICT TOOTH-SURFACE-ONLY DETECTION):
   - Analyse ONLY visible tooth enamel surfaces (occlusal fissures/pits, labial/buccal enamel, lingual surfaces, incisal edges, cervical tooth margins).
   - Detect ONLY clearly visible suspected surface conditions (caries/cavities, enamel discoloration/staining, plaque, tartar/calculus, or enamel cracks/chips).
   - Emit ONE tight bounding box per individual affected tooth (or specific affected enamel area on that tooth). Never bundle multiple teeth plus surrounding palate/gums into one large box when individual teeth are distinguishable.
   - EVERY condition bounding box MUST strictly overlap actual visible tooth enamel.
   - ABSOLUTELY EXCLUDE: gums/gingiva, hard or soft palate (roof of mouth), tongue, floor of mouth, lips, cheeks, facial skin, cheek retractors, mirrors, empty oral cavity space, and dark shadows outside the teeth.
   - Associate an FDI two-digit tooth number (e.g. "11", "21", "16", "26", "36", "46") only when reasonably identifiable; otherwise null.
   - If visible enamel shows no clear conditions, return "toothConditions": [].
4. For MODULE 2 (dentalAlignment — INDEPENDENT ALIGNMENT ANALYSIS):
   - Identify visible gaps between neighbouring teeth ("diastema"), apparent "crowding", "rotation", "displacement", "uneven_positioning", or "arch_irregularity".
   - GAP / DIASTEMA LOCALIZATION RULE: Identify both neighbouring teeth in involvedTeeth (e.g. ["11", "21"]). Place box_2d tightly over the actual interdental space BETWEEN the two neighbouring tooth crowns (sharing the vertical crown span of those two teeth)—NEVER place the box over the gums, incisive papilla, or palate behind/above the crowns.
   - CROWDING / ROTATION / DISPLACEMENT RULE: Frame ONLY the involved tooth crowns tightly; do NOT cover the tongue, palate, or empty mouth interior.
   - DO NOT INVENT MEASUREMENTS: never fabricate millimeter (mm) measurements from uncalibrated photographs. Leave gapMeasurement as null or a purely qualitative observation (e.g. "visible spacing between upper central incisors").
5. COORDINATE FORMAT (GEMINI NATIVE box_2d):
   - Return EVERY bounding box as "box_2d": [ymin, xmin, ymax, xmax] with integers from 0 to 1000 relative to the full image (top-left is [0, 0], bottom-right is [1000, 1000]).
   - Must satisfy 0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000.
   - Confidence must be 0.0..1.0 and severity must be "low" | "moderate" | "high".
"""


def build_view_prompt(view_name: str) -> str:
    view_guidelines = {
        "upper": (
            "VIEW: UPPER ARCH (maxillary teeth, occlusal/palatal view).\n"
            "- Locate the actual visible upper tooth crowns along this photo's peripheral arch first.\n"
            "- The central area inside the arch is the hard palate / palatal rugae (soft tissue): NEVER place condition or alignment boxes over the palate or palatal mucosa.\n"
            "- Module 1: Frame ONLY the individual visible tooth crowns showing enamel conditions (keep boxes tight to each individual tooth crown, roughly 70..180 units wide/tall on 0..1000, so they never span an entire molar row or spill onto the palate/cheek; emit separate boxes if multiple teeth are affected).\n"
            "- Module 2: If there is a midline diastema (gap between upper central incisors 11 and 21), set box_2d directly across the interdental crown gap between the two white central incisors themselves (overlapping the adjacent incisor crown height near the top of the arch, NEVER shifted downward onto the pink incisive papilla or palate)."
        ),
        "lower": (
            "VIEW: LOWER ARCH (mandibular teeth, occlusal/lingual view).\n"
            "- Locate the actual visible lower mandibular tooth crowns along this photo's outer U-shaped arch first (near the bottom border above the lower lip for incisors, and along the left/right edges for premolars/molars; ignore any upper teeth reflected or visible at the top edge).\n"
            "- CRITICAL FOR LOWER ARCH: The central tongue and the pale pink/whitish sublingual mucosa/salivary highlights inside the U-curve behind the lower front teeth are SOFT TISSUE, NOT teeth or tartar. NEVER place any box inside that inner pink floor-of-mouth area.\n"
            "- Module 1: Only report tartar/calculus, discoloration, or caries when clearly visible directly ON an ivory/white tooth crown along the outer arch (compact box ~70..180 units on 0..1000 per individual tooth; never span a multi-tooth row).\n"
            "- Module 2: If lower anterior crowding/rotation is visible, frame ONLY the actual lower incisor tooth crowns along the bottom arch rim (overlapping the white/ivory crowns themselves, NOT shifted inward/upward onto the sublingual mucosa or tongue)."
        ),
        "front": (
            "VIEW: FRONT VIEW (anterior teeth in occlusion / smile view).\n"
            "- Locate the visible labial tooth crowns across the smile opening (note that tightly cropped front photos are horizontally wide so individual tooth crowns span a larger fraction of image height).\n"
            "- Exclude the upper/lower lips, facial skin, dark buccal corridors, and corner cheek mucosa at the left/right edges.\n"
            "- Module 1: Frame ONLY specific tooth crowns with visible enamel discoloration, decay, chips, or plaque.\n"
            "- Module 2: For a midline diastema or spacing, place box_2d tightly over the gap between the two adjacent tooth crowns (covering the vertical crown height, NOT shifted onto the upper gums or lip)."
        ),
    }

    guideline = view_guidelines.get(view_name.lower(), f"VIEW: {view_name.upper()} ARCH.")

    return f"""{guideline}

Perform step-by-step tooth-first localization on this exact image and return valid JSON with native box_2d [ymin, xmin, ymax, xmax] (0..1000) coordinates."""
