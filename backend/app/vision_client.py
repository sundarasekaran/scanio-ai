"""
OpenRouter Vision AI Client for Scanio AI Dental Screening.

Supports TWO independent screening modules:
1. TOOTH-SURFACE CONDITION ANALYSIS (enamel cavities, discoloration, cracks, plaque, tartar/calculus)
2. DENTAL ALIGNMENT ANALYSIS (diastema/gaps, crowding, rotation, displacement, arch form)

Enforces:
- OpenRouter Chat Completions API (POST {OPENROUTER_BASE_URL}/chat/completions) with Bearer auth.
- OpenRouter key & model catalog verification (existence, vision/image input, structured JSON support).
- Multimodal image_url payload with base64 data URL and exact detected MIME type.
- Native spatial box_2d ([ymin, xmin, ymax, xmax] 0..1000) coordinate validation without clamping/shifting.
- EXIF orientation normalization & pixel-level tooth-enamel overlap validation.
- Strict API key sanitization (keys are never logged or returned to clients).
- Zero mock or heuristic fallback.
"""
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
import base64
import json
import logging
import math
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dotenv import load_dotenv

from app.prompts import DENTAL_SYSTEM_PROMPT, build_view_prompt

logger = logging.getLogger("scanio.openrouter")

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE_PATH = BASE_DIR / ".env"

DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MAX_TOOTH_BOX_WIDTH = 0.35
MAX_TOOTH_BOX_HEIGHT = 0.35
MAX_OCCLUSAL_CONDITION_SPAN = 0.28
MAX_CONDITION_SPAN = 0.45
MAX_FRONT_CONDITION_HEIGHT = 0.60
MAX_ALIGNMENT_WIDTH = 0.65
MAX_ALIGNMENT_HEIGHT = 0.45
MIN_TOOTH_BOX_SIZE = 0.02
MIN_FINDING_CONFIDENCE = 0.30
OPENROUTER_REQUEST_TIMEOUT_SEC = 60
OPENROUTER_MAX_TOKENS = 6144
MAX_TRANSIENT_RETRIES = 2

_SOFT_TISSUE_ONLY_RE = re.compile(
    r"\b(hard palate|soft palate|palatal rugae|roof of (?:the )?mouth|tongue|uvula|cheek mucosa|lip|facial skin|empty mouth|empty oral cavity)\b",
    re.IGNORECASE,
)

OPENROUTER_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "viewStatus": {
            "type": "string",
            "enum": ["assessed", "inconclusive"],
        },
        "inconclusiveReason": {
            "type": ["string", "null"],
        },
        "visibleToothSummary": {
            "type": "string",
            "description": "Brief summary of visible tooth crowns localized in this photograph before assigning boxes.",
        },
        "toothAnchorRegions": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "anteriorTeethBox": {"type": "array", "items": {"type": "integer"}},
                "leftArchBox": {"type": "array", "items": {"type": "integer"}},
                "rightArchBox": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["anteriorTeethBox", "leftArchBox", "rightArchBox"],
        },
        "toothConditions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "toothNumber": {"type": ["string", "null"]},
                    "condition": {"type": "string"},
                    "affectedRegion": {"type": ["string", "null"]},
                    "confidence": {"type": "number"},
                    "severity": {"type": "string", "enum": ["low", "moderate", "high"]},
                    "description": {"type": "string"},
                    "recommendation": {"type": "string"},
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": [
                    "toothNumber",
                    "condition",
                    "affectedRegion",
                    "confidence",
                    "severity",
                    "description",
                    "recommendation",
                    "box_2d",
                ],
            },
        },
        "dentalAlignment": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "findingType": {
                        "type": "string",
                        "enum": [
                            "diastema",
                            "crowding",
                            "rotation",
                            "displacement",
                            "uneven_positioning",
                            "arch_irregularity",
                        ],
                    },
                    "name": {"type": "string"},
                    "involvedTeeth": {"type": "array", "items": {"type": "string"}},
                    "gapMeasurement": {"type": ["string", "null"]},
                    "confidence": {"type": "number"},
                    "severity": {"type": "string", "enum": ["low", "moderate", "high"]},
                    "description": {"type": "string"},
                    "recommendation": {"type": "string"},
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": [
                    "findingType",
                    "name",
                    "involvedTeeth",
                    "gapMeasurement",
                    "confidence",
                    "severity",
                    "description",
                    "recommendation",
                    "box_2d",
                ],
            },
        },
    },
    "required": [
        "viewStatus",
        "inconclusiveReason",
        "visibleToothSummary",
        "toothAnchorRegions",
        "toothConditions",
        "dentalAlignment",
    ],
}


class VisionClientError(RuntimeError):
    """Base exception for vision provider failures."""


class OpenRouterConfigError(VisionClientError):
    pass


class OpenRouterAuthError(VisionClientError):
    pass


class OpenRouterPaymentError(VisionClientError):
    pass


class OpenRouterPermissionError(VisionClientError):
    pass


class OpenRouterModelUnavailableError(VisionClientError):
    pass


class OpenRouterRateLimitError(VisionClientError):
    pass


class OpenRouterProviderError(VisionClientError):
    pass


class OpenRouterTimeoutError(VisionClientError):
    pass


class OpenRouterUnsupportedImageError(VisionClientError):
    pass


class OpenRouterMalformedResponseError(VisionClientError):
    pass


def sanitize_secret(text: str, secret: str = "") -> str:
    """Removes API keys, bearer tokens, and URL query secrets from messages and logs."""
    if not text:
        return ""
    clean = str(text)
    if secret and secret in clean:
        clean = clean.replace(secret, "[REDACTED_API_KEY]")
    clean = re.sub(r"sk-or-v1-[0-9A-Za-z\-_]+", "[REDACTED_OPENROUTER_KEY]", clean)
    clean = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", clean, flags=re.IGNORECASE)
    clean = re.sub(r"key=AIza[0-9A-Za-z\-_]{35}", "key=[REDACTED]", clean)
    clean = re.sub(r"AIza[0-9A-Za-z\-_]{35}", "[REDACTED_KEY]", clean)
    clean = re.sub(r"AQ\.[0-9A-Za-z\-_]{20,}", "[REDACTED_KEY]", clean)
    return clean


def load_backend_env() -> dict:
    """Loads OpenRouter backend environment from backend .env independently of CWD."""
    provider = "openrouter"
    model = DEFAULT_OPENROUTER_MODEL
    base_url = DEFAULT_OPENROUTER_BASE_URL
    file_key = ""

    if ENV_FILE_PATH.is_file():
        try:
            load_dotenv(ENV_FILE_PATH, override=True)
            content = ENV_FILE_PATH.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k == "VISION_PROVIDER" and v:
                    provider = v.lower()
                elif k == "OPENROUTER_MODEL" and v:
                    model = v
                elif k == "OPENROUTER_BASE_URL" and v:
                    base_url = v.rstrip("/")
                elif k == "OPENROUTER_API_KEY" and v:
                    file_key = v
        except Exception as exc:
            logger.warning("Error reading .env file at %s: %s", ENV_FILE_PATH, exc)

    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip().strip('"').strip("'")
    effective_key = file_key or env_key

    os.environ["VISION_PROVIDER"] = provider
    os.environ["OPENROUTER_MODEL"] = model
    os.environ["OPENROUTER_BASE_URL"] = base_url
    if effective_key:
        os.environ["OPENROUTER_API_KEY"] = effective_key

    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": effective_key,
        "has_key": bool(effective_key),
        "env_path": str(ENV_FILE_PATH),
        "env_exists": ENV_FILE_PATH.is_file(),
    }


def get_openrouter_config() -> dict:
    return load_backend_env()


def reload_environment() -> dict:
    return load_backend_env()


def _parse_openrouter_error_body(exc: urllib.error.HTTPError) -> str:
    raw_msg = f"HTTP {exc.code} {exc.reason}"
    try:
        body = exc.read().decode("utf-8", errors="replace")
        err_json = json.loads(body)
        err_obj = err_json.get("error", {}) if isinstance(err_json, dict) else {}
        if isinstance(err_obj, dict):
            msg = err_obj.get("message")
            if isinstance(msg, str) and msg.strip():
                raw_msg = msg.strip()
    except Exception:
        pass
    return raw_msg


def _classify_openrouter_http_error(
    code: int,
    raw_msg: str,
    api_key: str,
    model: str,
    view_name: str | None = None,
) -> VisionClientError:
    safe_msg = sanitize_secret(raw_msg, api_key)
    lower = safe_msg.lower()
    view_ctx = f" ({view_name})" if view_name else ""

    if code == 401:
        return OpenRouterAuthError(
            "Invalid OpenRouter API key. Check OPENROUTER_API_KEY in the backend .env file."
        )
    if code == 402:
        return OpenRouterPaymentError(
            "OpenRouter account has insufficient credits or payment is required. "
            "Add credits or increase the key budget in OpenRouter."
        )
    if code == 403:
        return OpenRouterPermissionError(
            f"OpenRouter model or account permission restriction for '{model}'. "
            "Check your OpenRouter key permissions and model access."
        )
    if code == 404 or "no endpoints found" in lower or "not a valid model" in lower:
        return OpenRouterModelUnavailableError(
            f"Configured OpenRouter model '{model}' is unavailable or was not found in the OpenRouter catalog."
        )
    if code == 429:
        return OpenRouterRateLimitError(
            "OpenRouter rate limit exceeded. Please wait a moment and try again."
        )
    if code in (500, 502, 503, 504):
        return OpenRouterProviderError(
            f"OpenRouter provider or server error (HTTP {code}) while running '{model}'{view_ctx}. Please try again shortly."
        )
    if code == 400 and any(tok in lower for tok in ("image", "mime", "invalid_image", "base64", "media")):
        return OpenRouterUnsupportedImageError(
            f"OpenRouter model '{model}' could not process the uploaded image{view_ctx}. "
            "Please upload a clear JPG, PNG, or WEBP photo."
        )

    return VisionClientError(f"OpenRouter API error{view_ctx} (HTTP {code}): {safe_msg}")


def _is_timeout_exception(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return True
        if reason and "timed out" in str(reason).lower():
            return True
    return "timed out" in str(exc).lower()


def _check_openrouter_key(api_key: str, base_url: str, model: str) -> tuple[bool, str | None, str | None]:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/key",
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "ScanioAI/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            json.loads(resp.read().decode("utf-8"))
        return True, None, None
    except urllib.error.HTTPError as exc:
        raw_msg = _parse_openrouter_error_body(exc)
        classified = _classify_openrouter_http_error(exc.code, raw_msg, api_key, model)
        status_map = {
            401: "invalid_api_key",
            402: "insufficient_credits",
            403: "permission_denied",
            429: "rate_limited",
        }
        return False, status_map.get(exc.code, f"http_{exc.code}"), sanitize_secret(str(classified), api_key)
    except Exception as exc:
        status_label = "timeout" if _is_timeout_exception(exc) else "network_error"
        return False, status_label, sanitize_secret(str(exc), api_key)


def _fetch_openrouter_model_entry(api_key: str, model: str, base_url: str) -> dict | None:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "ScanioAI/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        catalog = json.loads(resp.read().decode("utf-8"))
    models = catalog.get("data", []) if isinstance(catalog, dict) else []
    for entry in models:
        if isinstance(entry, dict) and entry.get("id") == model:
            return entry
    return None


def verify_openrouter_model_availability(
    api_key: str,
    model: str,
    base_url: str = DEFAULT_OPENROUTER_BASE_URL,
) -> dict:
    """
    Verifies:
    1. OpenRouter API key authentication (`GET /key`).
    2. Configured model exists in OpenRouter catalog (`GET /models`).
    3. Model supports image/vision input.
    4. Structured JSON / response_format support status.
    """
    if not api_key:
        return {
            "valid": False,
            "status": "missing_key",
            "error": f"OPENROUTER_API_KEY is not set in backend .env at {ENV_FILE_PATH}.",
        }

    key_ok, key_status, key_err = _check_openrouter_key(api_key, base_url, model)
    if not key_ok:
        return {
            "valid": False,
            "status": key_status or "auth_error",
            "error": key_err or "OpenRouter API key verification failed.",
        }

    try:
        entry = _fetch_openrouter_model_entry(api_key, model, base_url)
    except urllib.error.HTTPError as exc:
        raw_msg = _parse_openrouter_error_body(exc)
        classified = _classify_openrouter_http_error(exc.code, raw_msg, api_key, model)
        return {
            "valid": False,
            "status": f"http_{exc.code}",
            "error": sanitize_secret(str(classified), api_key),
        }
    except Exception as exc:
        status_label = "timeout" if _is_timeout_exception(exc) else "network_error"
        return {
            "valid": False,
            "status": status_label,
            "error": sanitize_secret(str(exc), api_key),
        }

    if not entry:
        return {
            "valid": False,
            "status": "model_not_found",
            "error": f"OpenRouter model '{model}' was not found in the OpenRouter catalog.",
        }

    arch = entry.get("architecture") or {}
    input_modalities = arch.get("input_modalities") or []
    modality_str = str(arch.get("modality") or "").lower()
    vision_supported = (
        ("image" in [str(m).lower() for m in input_modalities])
        or ("image" in modality_str)
    )
    if not vision_supported:
        return {
            "valid": False,
            "status": "unsupported_vision",
            "vision_supported": False,
            "error": f"OpenRouter model '{model}' does not support image/vision input.",
        }

    supported_params = [str(p) for p in (entry.get("supported_parameters") or [])]
    supports_structured = "structured_outputs" in supported_params
    supports_response_format = "response_format" in supported_params
    supports_reasoning = "reasoning" in supported_params

    return {
        "valid": True,
        "status": "available",
        "model": entry.get("id", model),
        "displayName": entry.get("name", model),
        "visionSupported": True,
        "structuredOutputSupported": supports_structured,
        "responseFormatSupported": supports_response_format,
        "supportsReasoning": supports_reasoning,
        "supportedParameters": supported_params,
    }


def detect_image_mime_type(image_bytes: bytes) -> str:
    """Validates image integrity and returns the exact supported MIME type."""
    if not image_bytes:
        raise OpenRouterUnsupportedImageError("Uploaded image is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img.verify()
        with Image.open(BytesIO(image_bytes)) as img:
            fmt = (img.format or "").upper()
            width, height = img.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise OpenRouterUnsupportedImageError("Uploaded file is not a valid image.") from exc

    if width <= 0 or height <= 0:
        raise OpenRouterUnsupportedImageError("Uploaded image has invalid dimensions.")

    mime_map = {
        "JPEG": "image/jpeg",
        "MPO": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }
    mime = mime_map.get(fmt)
    if not mime:
        raise OpenRouterUnsupportedImageError(
            f"Unsupported image format '{fmt or 'unknown'}'. Please upload JPG, PNG, or WEBP."
        )
    return mime


def normalize_image_orientation(image_bytes: bytes, mime_type: str) -> bytes:
    """
    Applies EXIF Orientation rotation if needed so the model analyzes the exact upright
    orientation rendered by browser <img> and <canvas> elements.
    """
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            exif = img.getexif()
            orientation = exif.get(274, 1) if exif else 1
            if orientation == 1:
                return image_bytes
            transposed = ImageOps.exif_transpose(img)
            out = BytesIO()
            fmt = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}.get(mime_type, "JPEG")
            if fmt == "JPEG" and transposed.mode not in ("RGB", "L"):
                transposed = transposed.convert("RGB")
            transposed.save(out, format=fmt, quality=92)
            return out.getvalue()
    except Exception:
        return image_bytes


def compute_tooth_enamel_mask(image_bytes: bytes) -> dict | None:
    """Computes a downscaled (max 256px) colorimetric tooth-enamel vs oral-mucosa mask."""
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((256, 256))
            w, h = img.size
            if w <= 0 or h <= 0:
                return None
            pixels = list(img.getdata())
    except Exception:
        return None

    enamel = []
    mucosa = []
    for r, g, b in pixels:
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        g_over_r = g / max(r, 1)
        rg_diff = r - g
        pink_excess = r - 2 * g + b
        is_enamel = (lum >= 80.0) and (g_over_r >= 0.74) and (rg_diff <= 45) and (pink_excess <= 28)
        is_mucosa = (rg_diff >= 48) or (pink_excess >= 32 and r > 80) or (g_over_r < 0.71 and r > 70)
        enamel.append(is_enamel)
        mucosa.append(is_mucosa)

    overall_enamel = sum(enamel) / max(len(enamel), 1)
    return {
        "width": w,
        "height": h,
        "enamel": enamel,
        "mucosa": mucosa,
        "overall_enamel": overall_enamel,
    }


def _box_region_ratios(mask: dict, x: float, y: float, bw: float, bh: float) -> tuple[float, float, int]:
    w = mask["width"]
    h = mask["height"]
    enamel = mask["enamel"]
    mucosa = mask["mucosa"]

    x0 = max(0, min(w - 1, int(math.floor(x * w))))
    y0 = max(0, min(h - 1, int(math.floor(y * h))))
    x1 = max(x0 + 1, min(w, int(math.ceil((x + bw) * w))))
    y1 = max(y0 + 1, min(h, int(math.ceil((y + bh) * h))))

    tot = 0
    hit_e = 0
    hit_m = 0
    for yy in range(y0, y1):
        row = yy * w
        for xx in range(x0, x1):
            tot += 1
            if enamel[row + xx]:
                hit_e += 1
            if mucosa[row + xx]:
                hit_m += 1
    if tot == 0:
        return 0.0, 0.0, 0
    return hit_e / tot, hit_m / tot, tot


def evaluate_box_enamel_overlap(
    mask: dict | None,
    box: dict,
    category: str,
    finding_type: str | None = None,
) -> tuple[bool, float, float]:
    """Verifies whether a candidate box overlaps visible tooth enamel vs oral mucosa."""
    if not mask or mask.get("overall_enamel", 0.0) < 0.03:
        return True, 1.0, 0.0

    x, y, bw, bh = box["x"], box["y"], box["w"], box["h"]
    enamel_ratio, mucosa_ratio, _ = _box_region_ratios(mask, x, y, bw, bh)

    core_x = x + bw * 0.20
    core_y = y + bh * 0.20
    core_w = bw * 0.60
    core_h = bh * 0.60
    core_enamel, _, _ = _box_region_ratios(mask, core_x, core_y, core_w, core_h)

    if category == "condition":
        if enamel_ratio < 0.25 or mucosa_ratio > 0.72:
            return False, enamel_ratio, mucosa_ratio
        if core_enamel < 0.20 and enamel_ratio < 0.35:
            return False, enamel_ratio, mucosa_ratio
        return True, enamel_ratio, mucosa_ratio

    if finding_type == "diastema":
        if enamel_ratio < 0.20 or mucosa_ratio > 0.78:
            return False, enamel_ratio, mucosa_ratio
        return True, enamel_ratio, mucosa_ratio

    if enamel_ratio < 0.24 or mucosa_ratio > 0.74:
        return False, enamel_ratio, mucosa_ratio
    if bw >= 0.25 and bh >= 0.25 and (enamel_ratio < 0.45 or core_enamel < 0.35):
        return False, enamel_ratio, mucosa_ratio
    return True, enamel_ratio, mucosa_ratio


def is_in_palatal_exclusion_zone(box: dict, view_name: str) -> bool:
    """Checks if a bounding box falls in the central hard palate vault in an upper occlusal image."""
    if view_name.lower() != "upper":
        return False
    center_x = box["x"] + (box["w"] / 2.0)
    center_y = box["y"] + (box["h"] / 2.0)
    return 0.30 <= center_x <= 0.70 and 0.32 <= center_y <= 0.76


def is_in_lower_tongue_exclusion_zone(box: dict, view_name: str) -> bool:
    """Checks if a condition box falls inside the central tongue / sublingual mucosa in a lower image."""
    if view_name.lower() != "lower":
        return False
    center_x = box["x"] + (box["w"] / 2.0)
    center_y = box["y"] + (box["h"] / 2.0)
    return 0.30 <= center_x <= 0.70 and 0.20 <= center_y <= 0.68


def is_valid_tooth_box(box: dict, category: str = "condition", view_name: str = "upper") -> bool:
    """Validates strict normalized coordinates in [0, 1] and localized dimensions."""
    try:
        raw_x = box.get("x")
        raw_y = box.get("y")
        raw_w = box.get("w", box.get("width"))
        raw_h = box.get("h", box.get("height"))
        if raw_x is None or raw_y is None or raw_w is None or raw_h is None:
            return False

        x = float(raw_x)
        y = float(raw_y)
        w = float(raw_w)
        h = float(raw_h)
    except (ValueError, TypeError, AttributeError):
        return False

    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return False
    if x < 0.0 or y < 0.0 or x > 1.0 or y > 1.0 or w <= 0.0 or h <= 0.0:
        return False
    if x + w > 1.001 or y + h > 1.001:
        return False
    if w < MIN_TOOTH_BOX_SIZE or h < MIN_TOOTH_BOX_SIZE:
        return False

    if category == "alignment":
        if w > MAX_ALIGNMENT_WIDTH or h > MAX_ALIGNMENT_HEIGHT:
            logger.warning("Rejected oversized alignment bounding box (w=%.2f, h=%.2f).", w, h)
            return False
        if w > 0.45 and h > 0.40:
            return False
        return True

    if w > MAX_TOOTH_BOX_WIDTH and h > MAX_TOOTH_BOX_HEIGHT:
        logger.warning("Rejected non-tooth bounding box (spans too large an area: w=%.2f, h=%.2f).", w, h)
        return False

    is_front = view_name.lower() == "front"
    max_w = MAX_CONDITION_SPAN if is_front else MAX_OCCLUSAL_CONDITION_SPAN
    max_h = (
        MAX_FRONT_CONDITION_HEIGHT
        if is_front and w <= 0.28
        else (MAX_CONDITION_SPAN if is_front else MAX_OCCLUSAL_CONDITION_SPAN)
    )
    if w > max_w or h > max_h:
        logger.warning(
            "Rejected oversized condition bounding box (%s: w=%.2f, h=%.2f, limits=%.2fx%.2f).",
            view_name,
            w,
            h,
            max_w,
            max_h,
        )
        return False

    return True


def _parse_strict_box(
    item: dict,
    category: str,
    view_name: str,
) -> tuple[dict | None, list[int] | None]:
    """Parses box_2d ([ymin, xmin, ymax, xmax] 0..1000) or {x, y, w/width, h/height} (0..1) strictly."""
    raw_box_2d = item.get("box_2d")
    if raw_box_2d is not None:
        if not isinstance(raw_box_2d, (list, tuple)) or len(raw_box_2d) != 4:
            return None, None
        try:
            ymin, xmin, ymax, xmax = [float(v) for v in raw_box_2d]
        except (ValueError, TypeError):
            return None, None
        if not all(math.isfinite(v) for v in (ymin, xmin, ymax, xmax)):
            return None, None
        if ymin < 0.0 or xmin < 0.0 or ymax > 1000.0 or xmax > 1000.0:
            return None, None
        if ymin >= ymax or xmin >= xmax:
            return None, None

        x = round(xmin / 1000.0, 4)
        y = round(ymin / 1000.0, 4)
        w = round((xmax - xmin) / 1000.0, 4)
        h = round((ymax - ymin) / 1000.0, 4)
        box = {"x": x, "y": y, "w": w, "h": h}
        if not is_valid_tooth_box(box, category=category, view_name=view_name):
            return None, None
        return box, [int(round(ymin)), int(round(xmin)), int(round(ymax)), int(round(xmax))]

    raw_box = item.get("boundingBox") if item.get("boundingBox") is not None else item.get("box")
    if not isinstance(raw_box, dict):
        return None, None

    raw_x = raw_box.get("x")
    raw_y = raw_box.get("y")
    raw_w = raw_box.get("w", raw_box.get("width"))
    raw_h = raw_box.get("h", raw_box.get("height"))
    if raw_x is None or raw_y is None or raw_w is None or raw_h is None:
        return None, None

    try:
        x = float(raw_x)
        y = float(raw_y)
        w = float(raw_w)
        h = float(raw_h)
    except (ValueError, TypeError):
        return None, None

    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None, None
    if x < 0.0 or y < 0.0 or x > 1.0 or y > 1.0 or w <= 0.0 or h <= 0.0:
        return None, None
    if x + w > 1.001 or y + h > 1.001:
        return None, None

    if x + w > 1.0:
        w = max(0.0, 1.0 - x)
    if y + h > 1.0:
        h = max(0.0, 1.0 - y)

    box = {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)}
    if not is_valid_tooth_box(box, category=category, view_name=view_name):
        return None, None
    return box, None


def _is_soft_tissue_only_finding(item: dict, name: str) -> bool:
    affected = str(item.get("affectedRegion") or "")
    combined = f"{name} {affected}".strip()
    if _SOFT_TISSUE_ONLY_RE.search(combined) and "enamel" not in combined.lower() and "tooth" not in combined.lower():
        return True
    return False


def sanitize_and_normalize_finding(
    item: dict,
    view_name: str,
    category: str = "condition",
    enamel_mask: dict | None = None,
) -> dict | None:
    if not isinstance(item, dict):
        return None

    box, raw_box_2d = _parse_strict_box(item, category=category, view_name=view_name)
    if box is None:
        return None

    finding_type = item.get("findingType")
    if finding_type is not None:
        finding_type = str(finding_type).strip() or None

    if category == "condition":
        if is_in_palatal_exclusion_zone(box, view_name):
            logger.warning(
                "Filtered condition box inside palatal vault (%s): center=(%.2f, %.2f)",
                view_name,
                box["x"] + box["w"] / 2.0,
                box["y"] + box["h"] / 2.0,
            )
            return None
        if is_in_lower_tongue_exclusion_zone(box, view_name):
            logger.warning(
                "Filtered condition box inside lower tongue/floor (%s): center=(%.2f, %.2f)",
                view_name,
                box["x"] + box["w"] / 2.0,
                box["y"] + box["h"] / 2.0,
            )
            return None

    if category == "alignment" and view_name.lower() == "upper":
        cx = box["x"] + box["w"] / 2.0
        cy = box["y"] + box["h"] / 2.0
        if 0.33 <= cx <= 0.67 and 0.36 <= cy <= 0.74:
            logger.warning("Filtered alignment box placed inside upper palatal vault: center=(%.2f, %.2f)", cx, cy)
            return None

    overlaps_tooth, enamel_ratio, mucosa_ratio = evaluate_box_enamel_overlap(
        enamel_mask,
        box,
        category=category,
        finding_type=finding_type,
    )
    if not overlaps_tooth:
        logger.warning(
            "Rejected %s box outside tooth enamel (%s): box=%s enamel=%.2f mucosa=%.2f",
            category,
            view_name,
            box,
            enamel_ratio,
            mucosa_ratio,
        )
        return None

    raw_name = item.get("name") or item.get("condition")
    name = str(raw_name).strip() if raw_name is not None else ""
    if not name:
        name = (
            (finding_type or "Dental Alignment Observation").replace("_", " ").title()
            if category == "alignment"
            else "Tooth Surface Finding"
        )

    if _is_soft_tissue_only_finding(item, name):
        logger.warning("Filtered non-tooth soft-tissue finding: %s", name)
        return None

    raw_conf = item.get("confidence", 0.85)
    try:
        confidence = float(raw_conf)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
        return None
    if confidence < MIN_FINDING_CONFIDENCE:
        logger.info("Filtered low-confidence finding (%s: %.2f).", name, confidence)
        return None

    raw_severity = str(item.get("severity") or "low").strip().lower()
    if raw_severity in ("mild", "low"):
        severity = "low"
    elif raw_severity in ("moderate", "med", "medium"):
        severity = "moderate"
    elif raw_severity in ("severe", "high"):
        severity = "high"
    else:
        severity = "low"

    raw_involved = item.get("involvedTeeth")
    if isinstance(raw_involved, list):
        involved_teeth = [str(t).strip() for t in raw_involved if t is not None and str(t).strip()]
    elif isinstance(raw_involved, str):
        involved_teeth = [t.strip() for t in raw_involved.split(",") if t.strip()]
    else:
        involved_teeth = []

    tooth = item.get("toothNumber") or item.get("tooth") or None
    if tooth is not None:
        tooth = str(tooth).strip()
        if (
            not tooth
            or tooth.lower() in ("null", "none", "n/a", "unknown")
            or len(tooth) > 20
            or any(bad in tooth.lower() for bad in ("mouth", "jaw", "arch", "palate", "tongue"))
        ):
            tooth = None
    if not tooth and involved_teeth:
        tooth = ", ".join(involved_teeth)

    gap_measurement = item.get("gapMeasurement")
    if gap_measurement is not None:
        gap_measurement = str(gap_measurement).strip() or None

    affected_region = item.get("affectedRegion")
    if affected_region is not None:
        affected_region = str(affected_region).strip() or None

    description = str(item.get("description") or "").strip() or f"Localized {category} observation in {view_name} view."
    default_rec = (
        "Consult with an orthodontic specialist for alignment evaluation."
        if category == "alignment"
        else "Maintain routine oral hygiene and schedule a clinical dental examination."
    )
    recommendation = str(item.get("recommendation") or "").strip() or default_rec

    return {
        "name": name,
        "category": category,
        "condition": name,
        "findingType": finding_type,
        "involvedTeeth": involved_teeth,
        "gapMeasurement": gap_measurement,
        "affectedRegion": affected_region,
        "confidence": round(confidence, 2),
        "severity": severity,
        "description": description,
        "recommendation": recommendation,
        "box": box,
        "rawBox2d": raw_box_2d,
        "tooth": tooth,
        "toothNumber": tooth,
    }


def _extract_openrouter_text(result: dict, view_name: str) -> str:
    if not isinstance(result, dict):
        raise OpenRouterMalformedResponseError(f"OpenRouter returned an unexpected payload ({view_name}).")

    if "error" in result and isinstance(result["error"], dict):
        err_msg = result["error"].get("message") or "OpenRouter returned an error payload."
        raise OpenRouterProviderError(sanitize_secret(str(err_msg)))

    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenRouterMalformedResponseError(
            f"OpenRouter returned no choices for the {view_name} image."
        )

    first = choices[0] if isinstance(choices[0], dict) else {}
    finish_reason = str(first.get("finish_reason") or "").lower()
    if finish_reason in ("content_filter", "error"):
        raise OpenRouterMalformedResponseError(
            f"OpenRouter halted screening for the {view_name} image ({finish_reason})."
        )

    message = first.get("message")
    if not isinstance(message, dict):
        raise OpenRouterMalformedResponseError(
            f"OpenRouter response is missing assistant message for {view_name}."
        )

    content = message.get("content")
    if isinstance(content, str):
        text_content = content.strip()
    elif isinstance(content, list):
        chunks = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") in ("text", "output_text", None) and isinstance(block.get("text"), str)
        ]
        text_content = "\n".join(c for c in chunks if c.strip()).strip()
    else:
        text_content = ""

    if not text_content:
        raise OpenRouterMalformedResponseError(
            f"OpenRouter response for the {view_name} image did not contain text/JSON output."
        )
    if finish_reason == "length":
        logger.warning("OpenRouter output for %s hit max_tokens length limit; attempting JSON parse.", view_name)
    return text_content


def _parse_json_response(text_content: str, view_name: str) -> dict | list:
    stripped = text_content.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, (dict, list)):
            return parsed
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, re.IGNORECASE)
    if fence_match:
        inner = fence_match.group(1).strip()
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(stripped):
        if ch in ("{", "["):
            try:
                parsed, _ = decoder.raw_decode(stripped[idx:])
                if isinstance(parsed, (dict, list)):
                    return parsed
            except json.JSONDecodeError:
                continue

    raise OpenRouterMalformedResponseError(
        f"OpenRouter returned malformed JSON for the {view_name} image. Please retry."
    )


def _send_openrouter_chat_request(
    url: str,
    payload: dict,
    api_key: str,
    model: str,
    view_name: str,
) -> dict:
    current_payload = dict(payload)
    last_error: Exception | None = None

    for attempt in range(MAX_TRANSIENT_RETRIES + 1):
        body_bytes = json.dumps(current_payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8080",
                "X-Title": "Scanio AI",
                "User-Agent": "ScanioAI/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=OPENROUTER_REQUEST_TIMEOUT_SEC) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            raw_msg = _parse_openrouter_error_body(exc)
            lower = raw_msg.lower()

            if exc.code in (500, 502, 503, 504) and attempt < MAX_TRANSIENT_RETRIES:
                delay = 1.5 * (attempt + 1)
                logger.warning(
                    "Transient OpenRouter HTTP %s for %s (attempt %s/%s); retrying in %.1fs.",
                    exc.code,
                    view_name,
                    attempt + 1,
                    MAX_TRANSIENT_RETRIES + 1,
                    delay,
                )
                time.sleep(delay)
                last_error = _classify_openrouter_http_error(exc.code, raw_msg, api_key, model, view_name)
                continue

            # Compatibility fallback if upstream rejects reasoning or strict json_schema
            if exc.code == 400 and "reasoning" in lower and "reasoning" in current_payload:
                current_payload = {k: v for k, v in current_payload.items() if k != "reasoning"}
                continue

            if exc.code == 400 and ("response_format" in lower or "json_schema" in lower) and "response_format" in current_payload:
                rf = current_payload.get("response_format", {})
                if isinstance(rf, dict) and rf.get("type") == "json_schema":
                    current_payload = dict(current_payload)
                    current_payload["response_format"] = {"type": "json_object"}
                    continue
                current_payload = {k: v for k, v in current_payload.items() if k != "response_format"}
                continue

            raise _classify_openrouter_http_error(exc.code, raw_msg, api_key, model, view_name) from exc
        except json.JSONDecodeError as exc:
            raise OpenRouterMalformedResponseError(
                f"OpenRouter API returned an invalid JSON envelope for the {view_name} image."
            ) from exc
        except Exception as exc:
            if _is_timeout_exception(exc):
                raise OpenRouterTimeoutError(
                    f"OpenRouter screening request timed out while analyzing the {view_name} image. Please try again."
                ) from exc
            sanitized = sanitize_secret(str(exc), api_key)
            raise VisionClientError(f"OpenRouter connection error ({view_name}): {sanitized}") from exc

    if isinstance(last_error, VisionClientError):
        raise last_error
    raise OpenRouterProviderError(f"OpenRouter provider error for '{model}'.")


def analyze_with_openrouter(
    image_bytes: bytes,
    view_name: str,
    api_key: str,
    model: str,
    base_url: str = DEFAULT_OPENROUTER_BASE_URL,
    mime_type: str | None = None,
    capabilities: dict | None = None,
) -> dict:
    """
    Submits dental image bytes to OpenRouter Chat Completions API using multimodal image_url
    content with base64 data URL and exact MIME type.
    """
    detected_mime = mime_type or detect_image_mime_type(image_bytes)
    upright_bytes = normalize_image_orientation(image_bytes, detected_mime)
    enamel_mask = compute_tooth_enamel_mask(upright_bytes)

    b64_image = base64.b64encode(upright_bytes).decode("utf-8")
    data_url = f"data:{detected_mime};base64,{b64_image}"
    url = f"{base_url.rstrip('/')}/chat/completions"

    payload: dict = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": OPENROUTER_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": DENTAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_view_prompt(view_name)},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }

    caps = capabilities or {}
    if caps.get("structuredOutputSupported", True):
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "dental_screening",
                "strict": True,
                "schema": OPENROUTER_JSON_SCHEMA,
            },
        }
    elif caps.get("responseFormatSupported"):
        payload["response_format"] = {"type": "json_object"}

    if caps.get("supportsReasoning", "gemini-2.5" in model.lower() or "gemini-3" in model.lower()):
        payload["reasoning"] = {"max_tokens": 2048}

    result = _send_openrouter_chat_request(url, payload, api_key, model, view_name)
    text_content = _extract_openrouter_text(result, view_name)
    parsed = _parse_json_response(text_content, view_name)

    view_status = "assessed"
    inconclusive_reason: str | None = None
    raw_conditions: list = []
    raw_alignment: list = []

    if isinstance(parsed, dict):
        raw_status = str(parsed.get("viewStatus") or "assessed").strip().lower()
        if raw_status == "inconclusive":
            view_status = "inconclusive"
            reason = parsed.get("inconclusiveReason")
            inconclusive_reason = (
                str(reason).strip()
                if reason
                else f"The {view_name} photo did not provide enough clarity for reliable inspection."
            )

        cond_val = parsed.get("toothConditions")
        align_val = parsed.get("dentalAlignment")
        if cond_val is not None and not isinstance(cond_val, list):
            raise OpenRouterMalformedResponseError(
                f"OpenRouter returned an invalid 'toothConditions' structure for {view_name}."
            )
        if align_val is not None and not isinstance(align_val, list):
            raise OpenRouterMalformedResponseError(
                f"OpenRouter returned an invalid 'dentalAlignment' structure for {view_name}."
            )

        raw_conditions = cond_val or []
        raw_alignment = align_val or []

        if not raw_conditions and not raw_alignment and isinstance(parsed.get("findings"), list):
            raw_conditions = parsed["findings"]
    elif isinstance(parsed, list):
        raw_conditions = parsed
    else:
        raise OpenRouterMalformedResponseError(
            f"OpenRouter returned an unexpected JSON type for {view_name}."
        )

    valid_conditions: list[dict] = []
    rejected_count = 0
    for item in raw_conditions:
        norm = sanitize_and_normalize_finding(
            item,
            view_name,
            category="condition",
            enamel_mask=enamel_mask,
        )
        if norm:
            valid_conditions.append(norm)
        else:
            rejected_count += 1

    valid_alignment: list[dict] = []
    for item in raw_alignment:
        norm = sanitize_and_normalize_finding(
            item,
            view_name,
            category="alignment",
            enamel_mask=enamel_mask,
        )
        if norm:
            valid_alignment.append(norm)
        else:
            rejected_count += 1

    all_findings = valid_conditions + valid_alignment

    if (len(raw_conditions) + len(raw_alignment)) > 0 and not all_findings and rejected_count > 0:
        view_status = "inconclusive"
        if not inconclusive_reason:
            inconclusive_reason = (
                f"Model detections for the {view_name} view did not overlap visible tooth surfaces and were rejected."
            )

    return {
        "viewStatus": view_status,
        "inconclusiveReason": inconclusive_reason,
        "toothConditions": valid_conditions,
        "dentalAlignment": valid_alignment,
        "all": all_findings,
        "rejectedCount": rejected_count,
        "usage": result.get("usage", {}),
    }
