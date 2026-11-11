"""
Multimodal Vision AI Client for Scanio AI Dental Screening (NVIDIA NIM & OpenRouter).

Supports TWO independent screening modules:
1. TOOTH-SURFACE CONDITION ANALYSIS (enamel caries, discoloration, cracks, plaque, tartar/calculus)
2. DENTAL ALIGNMENT ANALYSIS (diastema/gaps, crowding, rotation, displacement, arch form)

Enforces:
- NVIDIA NIM API Catalog (`POST {NVIDIA_BASE_URL}/chat/completions`) and OpenRouter Chat Completions API.
- Key & vision-model capability verification (catalog presence, multimodal vision support).
- Multimodal image_url payload with base64 data URL and preserved MIME type (JPEG, PNG, WEBP).
- Native spatial box_2d ([ymin, xmin, ymax, xmax] 0..1000) coordinate validation without clamping/shifting.
- EXIF orientation normalization, quality assessment, regional crop recovery, & enamel validation.
- Strict API key sanitization (keys are never logged or returned to clients).
- Zero mock or heuristic fallback.
"""
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError
import base64
import hashlib
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

from app.prompts import DENTAL_SYSTEM_PROMPT, build_crop_prompt, build_view_prompt

logger = logging.getLogger("scanio.vision")

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE_PATH = BASE_DIR / ".env"

DEFAULT_VISION_PROVIDER = "nvidia"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_OPENROUTER_MODEL = "google/gemini-2.5-flash"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
NVIDIA_INLINE_RAW_LIMIT_BYTES = 145_000
NVIDIA_AUTH_PROBE_TTL_SEC = 300
_NVIDIA_AUTH_CACHE: dict[str, tuple[float, dict]] = {}

NVIDIA_VISION_MODELS: dict[str, dict] = {
    "nvidia/nemotron-3.5-super-vl-120b-a12b": {
        "displayName": "NVIDIA Nemotron 3.5 Super VL 120B",
        "supportsStructured": True,
        "supportsResponseFormat": True,
    },
    "google/gemma-4-31b-it": {
        "displayName": "Google Gemma 4 31B IT",
        "supportsStructured": False,
        "supportsResponseFormat": True,
    },
    "google/gemma-3-12b-it": {
        "displayName": "Google Gemma 3 12B IT",
        "supportsStructured": False,
        "supportsResponseFormat": True,
    },
    "google/gemma-3-4b-it": {
        "displayName": "Google Gemma 3 4B IT",
        "supportsStructured": False,
        "supportsResponseFormat": True,
    },
    "meta/llama-3.2-90b-vision-instruct": {
        "displayName": "Meta Llama 3.2 90B Vision Instruct",
        "supportsStructured": False,
        "supportsResponseFormat": True,
    },
    "meta/llama-3.2-11b-vision-instruct": {
        "displayName": "Meta Llama 3.2 11B Vision Instruct",
        "supportsStructured": False,
        "supportsResponseFormat": True,
    },
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": {
        "displayName": "NVIDIA Nemotron 3 Nano Omni 30B",
        "supportsStructured": False,
        "supportsResponseFormat": True,
    },
    "nvidia/cosmos-reason2-8b": {
        "displayName": "NVIDIA Cosmos Reason 2 8B",
        "supportsStructured": False,
        "supportsResponseFormat": True,
    },
    "microsoft/phi-3-vision-128k-instruct": {
        "displayName": "Microsoft Phi-3 Vision 128K Instruct",
        "supportsStructured": False,
        "supportsResponseFormat": False,
    },
    "microsoft/phi-4-multimodal-instruct": {
        "displayName": "Microsoft Phi-4 Multimodal Instruct",
        "supportsStructured": False,
        "supportsResponseFormat": False,
    },
    "nvidia/neva-22b": {
        "displayName": "NVIDIA NeVA 22B",
        "supportsStructured": False,
        "supportsResponseFormat": False,
    },
    "nvidia/vila": {
        "displayName": "NVIDIA VILA",
        "supportsStructured": False,
        "supportsResponseFormat": False,
    },
    "adept/fuyu-8b": {
        "displayName": "Adept Fuyu 8B",
        "supportsStructured": False,
        "supportsResponseFormat": False,
    },
}

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
            "enum": ["assessed", "assessed_with_findings", "assessed_no_findings", "inconclusive"],
        },
        "conditionStatus": {
            "type": "string",
            "enum": ["assessed_with_findings", "assessed_no_findings", "inconclusive"],
        },
        "alignmentStatus": {
            "type": "string",
            "enum": ["assessed_with_findings", "assessed_no_findings", "inconclusive"],
        },
        "inconclusiveReason": {
            "type": ["string", "null"],
        },
        "visibleToothSummary": {
            "type": "string",
            "description": "Brief summary of visible tooth crowns localized in this photograph before assigning boxes.",
        },
        "unassessableTeethReason": {
            "type": ["string", "null"],
            "description": "Teeth or surfaces occluded, out of frame, or too blurry to assess reliably; null if all arch teeth are visible.",
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
        "conditionStatus",
        "alignmentStatus",
        "inconclusiveReason",
        "visibleToothSummary",
        "unassessableTeethReason",
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
    clean = re.sub(r"nvapi-[0-9A-Za-z\-_]+", "[REDACTED_NVIDIA_KEY]", clean)
    clean = re.sub(r"sk-or-v1-[0-9A-Za-z\-_]+", "[REDACTED_OPENROUTER_KEY]", clean)
    clean = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", clean, flags=re.IGNORECASE)
    clean = re.sub(r"key=AIza[0-9A-Za-z\-_]{35}", "key=[REDACTED]", clean)
    clean = re.sub(r"AIza[0-9A-Za-z\-_]{35}", "[REDACTED_KEY]", clean)
    clean = re.sub(r"AQ\.[0-9A-Za-z\-_]{20,}", "[REDACTED_KEY]", clean)
    return clean


def _read_env_file_vars() -> dict[str, str]:
    values: dict[str, str] = {}
    candidates = [ENV_FILE_PATH, BASE_DIR.parent / ".env", Path("/etc/secrets/.env")]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            # Local backend/.env overrides shell env during dev; other candidates only fill missing keys.
            load_dotenv(path, override=(path == ENV_FILE_PATH))
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                norm_k = k.strip()
                if norm_k and (path == ENV_FILE_PATH or norm_k not in values):
                    values[norm_k] = v.strip().strip('"').strip("'")
        except Exception as exc:
            logger.warning("Error reading .env file at %s: %s", path, exc)
    return values


def _lookup_os_env(key: str) -> str:
    if key in os.environ:
        return os.environ[key]
    target = key.strip().upper()
    for env_k, env_v in os.environ.items():
        if env_k.strip().upper() == target:
            return env_v
    return ""


def _clean_api_key_value(raw_val: str) -> str:
    cleaned = raw_val.strip().strip('"').strip("'")
    if cleaned.lower().startswith("bearer "):
        cleaned = cleaned[7:].strip()
    return cleaned


def load_backend_env() -> dict:
    """Loads NVIDIA / OpenRouter vision backend configuration independently of CWD."""
    file_vars = _read_env_file_vars()

    def _pick(key: str, default: str = "") -> str:
        val = file_vars.get(key) or _lookup_os_env(key)
        return val.strip().strip('"').strip("'") or default

    raw_provider = _pick("VISION_PROVIDER", "").lower()
    nv_key = _clean_api_key_value(_pick("NVIDIA_API_KEY", ""))
    nv_model = _pick("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL)
    nv_base_url = _pick("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL).rstrip("/")

    or_key = _clean_api_key_value(_pick("OPENROUTER_API_KEY", ""))
    or_model = _pick("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    or_base_url = _pick("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL).rstrip("/")

    # Default to NVIDIA; also migrate stale VISION_PROVIDER=openrouter cleanly when NVIDIA_API_KEY
    # is present without a valid OpenRouter key.
    if raw_provider == "openrouter":
        if nv_key and (not or_key or or_key.startswith("nvapi-")):
            provider = "nvidia"
        else:
            provider = "openrouter"
    elif raw_provider:
        provider = raw_provider
    elif nv_key or not or_key:
        provider = DEFAULT_VISION_PROVIDER
    else:
        provider = "openrouter"

    if provider == "nvidia":
        effective_key = nv_key
        model = nv_model
        base_url = nv_base_url
    else:
        effective_key = or_key
        model = or_model
        base_url = or_base_url

    os.environ["VISION_PROVIDER"] = provider
    if nv_key:
        os.environ["NVIDIA_API_KEY"] = nv_key
    os.environ["NVIDIA_MODEL"] = nv_model
    os.environ["NVIDIA_BASE_URL"] = nv_base_url
    os.environ["OPENROUTER_MODEL"] = or_model
    os.environ["OPENROUTER_BASE_URL"] = or_base_url
    if or_key:
        os.environ["OPENROUTER_API_KEY"] = or_key

    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": effective_key,
        "has_key": bool(effective_key),
        "nvidia_configured": bool(nv_key),
        "openrouter_configured": bool(or_key),
        "openrouter_has_nvapi_key": bool(or_key.startswith("nvapi-")),
        "env_path": str(ENV_FILE_PATH),
        "env_exists": ENV_FILE_PATH.is_file(),
    }


def get_vision_config() -> dict:
    return load_backend_env()


def get_openrouter_config() -> dict:
    return load_backend_env()


def reload_environment() -> dict:
    return load_backend_env()


def _parse_openrouter_error_body(exc: urllib.error.HTTPError) -> str:
    raw_msg = f"HTTP {exc.code} {exc.reason}"
    try:
        body = exc.read().decode("utf-8", errors="replace")
        err_json = json.loads(body)
        if isinstance(err_json, dict):
            err_obj = err_json.get("error")
            if isinstance(err_obj, dict):
                msg = err_obj.get("message")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
            for k in ("detail", "message", "title"):
                val = err_json.get(k)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        if body.strip():
            raw_msg = body.strip()[:300]
    except Exception:
        pass
    return raw_msg


def _classify_provider_http_error(
    code: int,
    raw_msg: str,
    api_key: str,
    model: str,
    view_name: str | None = None,
    provider: str = "nvidia",
) -> VisionClientError:
    safe_msg = sanitize_secret(raw_msg, api_key)
    lower = safe_msg.lower()
    view_ctx = f" ({view_name})" if view_name else ""
    label = "NVIDIA API" if provider == "nvidia" else "OpenRouter"
    key_var = "NVIDIA_API_KEY" if provider == "nvidia" else "OPENROUTER_API_KEY"

    if code == 401:
        return OpenRouterAuthError(
            f"Invalid {label} key. Check {key_var} in the backend environment configuration."
        )
    if code == 402:
        return OpenRouterPaymentError(
            f"{label} account has insufficient credits or quota exhausted. "
            f"Check your {label} billing/credits."
        )
    if code == 403:
        return OpenRouterPermissionError(
            f"{label} model or account permission restriction for '{model}'. "
            "Check your API key permissions and model access."
        )
    if code == 404 or "no endpoints found" in lower or "not a valid model" in lower:
        return OpenRouterModelUnavailableError(
            f"Configured {label} model '{model}' is unavailable or was not found in the {label} catalog."
        )
    if code == 429:
        return OpenRouterRateLimitError(
            f"{label} rate limit exceeded. Please wait a moment and try again."
        )
    if code in (500, 502, 503, 504):
        return OpenRouterProviderError(
            f"{label} provider or server error (HTTP {code}) while running '{model}'{view_ctx}. Please try again shortly."
        )
    if code == 400 and any(tok in lower for tok in ("image", "mime", "invalid_image", "base64", "media", "asset")):
        return OpenRouterUnsupportedImageError(
            f"{label} model '{model}' could not process the uploaded image{view_ctx}. "
            "Please upload a clear JPG, PNG, or WEBP photo."
        )

    return VisionClientError(f"{label} error{view_ctx} (HTTP {code}): {safe_msg}")


def _classify_openrouter_http_error(
    code: int,
    raw_msg: str,
    api_key: str,
    model: str,
    view_name: str | None = None,
) -> VisionClientError:
    return _classify_provider_http_error(code, raw_msg, api_key, model, view_name=view_name, provider="openrouter")


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


def is_nvidia_vision_model(model_id: str) -> bool:
    """Checks whether an NVIDIA NIM catalog model ID supports multimodal vision-language chat."""
    norm = (model_id or "").strip().lower()
    if not norm:
        return False
    if norm in NVIDIA_VISION_MODELS:
        return True
    if any(bad in norm for bad in ("embed", "parse", "nvclip", "retriever", "safety", "guard", "reward", "translate", "coder", "video-detector", "chatqa")):
        return False
    return any(tok in norm for tok in ("-vl", "vision", "-omni", "gemma-3", "gemma-4", "neva", "vila", "cosmos-reason", "fuyu"))


def _fetch_nvidia_catalog(api_key: str, base_url: str) -> list[dict]:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "ScanioAI/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        catalog = json.loads(resp.read().decode("utf-8"))
    return catalog.get("data", []) if isinstance(catalog, dict) else []


def _probe_nvidia_auth(api_key: str, model: str, base_url: str) -> tuple[bool, str | None, str | None]:
    cache_key = hashlib.sha256(f"{base_url}|{model}|{api_key}".encode("utf-8")).hexdigest()
    now = time.time()
    cached = _NVIDIA_AUTH_CACHE.get(cache_key)
    if cached and (now - cached[0]) < NVIDIA_AUTH_PROBE_TTL_SEC:
        res = cached[1]
        return res["ok"], res.get("status"), res.get("error")

    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ScanioAI/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
        _NVIDIA_AUTH_CACHE[cache_key] = (now, {"ok": True})
        return True, None, None
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            _NVIDIA_AUTH_CACHE[cache_key] = (now, {"ok": True})
            return True, None, None
        raw_msg = _parse_openrouter_error_body(exc)
        classified = _classify_provider_http_error(exc.code, raw_msg, api_key, model, provider="nvidia")
        status_map = {
            401: "invalid_api_key",
            402: "insufficient_credits",
            403: "permission_denied",
            404: "model_not_found",
            429: "rate_limited",
        }
        st = status_map.get(exc.code, f"http_{exc.code}")
        err = sanitize_secret(str(classified), api_key)
        _NVIDIA_AUTH_CACHE[cache_key] = (now, {"ok": False, "status": st, "error": err})
        return False, st, err
    except Exception as exc:
        st = "timeout" if _is_timeout_exception(exc) else "network_error"
        return False, st, sanitize_secret(str(exc), api_key)


def verify_nvidia_model_availability(
    api_key: str,
    model: str,
    base_url: str = DEFAULT_NVIDIA_BASE_URL,
    run_auth_probe: bool = False,
) -> dict:
    """
    Verifies NVIDIA NIM model availability and multimodal vision support.
    Optionally performs a cached lightweight auth probe for /api/v1/model-status.
    """
    if not api_key:
        return {
            "valid": False,
            "status": "missing_key",
            "error": "NVIDIA_API_KEY is not configured in the backend environment.",
        }
    if api_key.startswith("sk-or-v1-"):
        return {
            "valid": False,
            "status": "misconfigured_provider_key",
            "error": (
                "Detected an OpenRouter key (sk-or-v1-...) inside NVIDIA_API_KEY while VISION_PROVIDER=nvidia. "
                "Configure your NVIDIA key (nvapi-...) in NVIDIA_API_KEY."
            ),
        }

    try:
        catalog = _fetch_nvidia_catalog(api_key, base_url)
    except urllib.error.HTTPError as exc:
        raw_msg = _parse_openrouter_error_body(exc)
        classified = _classify_provider_http_error(exc.code, raw_msg, api_key, model, provider="nvidia")
        return {
            "valid": False,
            "status": f"http_{exc.code}",
            "error": sanitize_secret(str(classified), api_key),
        }
    except Exception as exc:
        st = "timeout" if _is_timeout_exception(exc) else "network_error"
        return {
            "valid": False,
            "status": st,
            "error": sanitize_secret(str(exc), api_key),
        }

    entry = next((m for m in catalog if isinstance(m, dict) and m.get("id") == model), None)
    if not entry:
        return {
            "valid": False,
            "status": "model_not_found",
            "error": f"NVIDIA model '{model}' was not found in the NVIDIA NIM catalog at {base_url}.",
        }

    if not is_nvidia_vision_model(model):
        return {
            "valid": False,
            "status": "unsupported_vision",
            "vision_supported": False,
            "error": (
                f"Configured NVIDIA model '{model}' does not support image/vision input. "
                f"Use a verified NVIDIA vision model such as '{DEFAULT_NVIDIA_MODEL}', "
                "'google/gemma-4-31b-it', or 'meta/llama-3.2-90b-vision-instruct'."
            ),
        }

    if run_auth_probe:
        auth_ok, auth_status, auth_err = _probe_nvidia_auth(api_key, model, base_url)
        if not auth_ok:
            return {
                "valid": False,
                "status": auth_status or "auth_error",
                "error": auth_err or "NVIDIA API key authentication failed.",
            }

    meta = NVIDIA_VISION_MODELS.get(model, {})
    return {
        "valid": True,
        "status": "available",
        "provider": "nvidia",
        "model": entry.get("id", model),
        "displayName": meta.get("displayName", model),
        "visionSupported": True,
        "structuredOutputSupported": meta.get("supportsStructured", False),
        "responseFormatSupported": meta.get("supportsResponseFormat", True),
        "supportsReasoning": False,
    }


def verify_vision_model_availability(
    provider: str,
    api_key: str,
    model: str,
    base_url: str,
    run_auth_probe: bool = False,
) -> dict:
    if provider == "nvidia":
        return verify_nvidia_model_availability(api_key, model, base_url, run_auth_probe=run_auth_probe)
    return verify_openrouter_model_availability(api_key, model, base_url)


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
    if api_key.startswith("nvapi-"):
        return {
            "valid": False,
            "status": "misconfigured_provider_key",
            "error": (
                "Detected an NVIDIA API key (nvapi-...) inside OPENROUTER_API_KEY while VISION_PROVIDER=openrouter. "
                "Move the key to NVIDIA_API_KEY and set VISION_PROVIDER=nvidia."
            ),
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
    orientation rendered by browser <img> and <canvas> elements without unnecessary recompression.
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
            save_kwargs = {"format": fmt}
            if fmt in ("JPEG", "WEBP"):
                save_kwargs["quality"] = 95
                if fmt == "JPEG":
                    save_kwargs["subsampling"] = 0
            transposed.save(out, **save_kwargs)
            return out.getvalue()
    except Exception:
        return image_bytes


_LAPLACIAN_KERNEL = ImageFilter.Kernel(
    (3, 3),
    [0, 1, 0, 1, -4, 1, 0, 1, 0],
    scale=1,
    offset=128,
)


def assess_image_quality(image_bytes: bytes) -> dict:
    """
    Measures upright resolution, aspect ratio, brightness, edge contrast, and
    Laplacian variance sharpness before calling the vision AI model.
    """
    try:
        with Image.open(BytesIO(image_bytes)) as raw:
            upright = ImageOps.exif_transpose(raw)
            w, h = upright.size
            gray = upright.convert("L")
            norm = gray.copy()
            norm.thumbnail((640, 640))
            edges = norm.filter(ImageFilter.FIND_EDGES)
            lap = norm.filter(_LAPLACIAN_KERNEL)
            if norm.width > 8 and norm.height > 8:
                inner = (2, 2, norm.width - 2, norm.height - 2)
                edges = edges.crop(inner)
                lap = lap.crop(inner)
            edge_contrast = round(float(ImageStat.Stat(edges).stddev[0]), 2)
            lap_var = round(float(ImageStat.Stat(lap).var[0]), 2)
            mean_luma = round(float(ImageStat.Stat(norm).mean[0]), 1)
    except Exception:
        return {
            "width": 0,
            "height": 0,
            "aspectRatio": 0.0,
            "minDimension": 0,
            "laplacianVariance": 0.0,
            "edgeContrast": 0.0,
            "meanBrightness": 0.0,
            "qualityStatus": "unknown",
            "qualityNotes": "Unable to measure image quality metrics.",
        }

    min_dim = min(w, h)
    aspect_ratio = round(w / max(h, 1), 3)
    notes: list[str] = []

    if min_dim < 64:
        q_status = "too_small"
        notes.append("Image resolution is too small for reliable tooth-surface inspection.")
    elif mean_luma < 32.0:
        q_status = "dark"
        notes.append("Photo appears very dark; tooth details may be underexposed.")
    elif mean_luma > 248.0 and edge_contrast < 4.0:
        q_status = "overexposed"
        notes.append("Photo appears flat or overexposed.")
    elif lap_var < 6.0 and edge_contrast < 5.0:
        q_status = "very_blurry"
        notes.append("Severe blur or motion smear detected; enamel details may be unclear.")
    elif lap_var < 22.0:
        q_status = "soft_focus"
        notes.append("Moderate smartphone softness detected; crowns and spacing remain evaluable.")
    else:
        q_status = "good"
        notes.append("Adequate resolution and sharpness for intraoral screening.")

    return {
        "width": w,
        "height": h,
        "aspectRatio": aspect_ratio,
        "minDimension": min_dim,
        "laplacianVariance": lap_var,
        "edgeContrast": edge_contrast,
        "meanBrightness": mean_luma,
        "qualityStatus": q_status,
        "qualityNotes": " ".join(notes),
    }


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
    return 0.32 <= center_x <= 0.68 and 0.22 <= center_y <= 0.63


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


def _merge_system_into_user_message(payload: dict) -> dict | None:
    """Folds system prompt into the user text part for NIM models that forbid system role with images."""
    msgs = payload.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None
    sys_msg = msgs[0] if isinstance(msgs[0], dict) and msgs[0].get("role") == "system" else None
    usr_msg = msgs[1] if isinstance(msgs[1], dict) and msgs[1].get("role") == "user" else None
    if not sys_msg or not usr_msg:
        return None
    sys_text = str(sys_msg.get("content") or "").strip()
    usr_content = usr_msg.get("content")
    if isinstance(usr_content, list):
        new_parts = []
        merged = False
        for part in usr_content:
            if not merged and isinstance(part, dict) and part.get("type") == "text":
                new_parts.append({"type": "text", "text": f"{sys_text}\n\n{part.get('text', '')}".strip()})
                merged = True
            else:
                new_parts.append(part)
        if not merged:
            new_parts.insert(0, {"type": "text", "text": sys_text})
        new_usr = {"role": "user", "content": new_parts}
    else:
        new_usr = {"role": "user", "content": f"{sys_text}\n\n{usr_content}".strip()}
    out = dict(payload)
    out["messages"] = [new_usr, *msgs[2:]]
    return out


def _send_chat_request(
    url: str,
    payload: dict,
    api_key: str,
    model: str,
    view_name: str,
    provider: str = "openrouter",
) -> dict:
    current_payload = dict(payload)
    last_error: Exception | None = None
    label = "NVIDIA API" if provider == "nvidia" else "OpenRouter"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "ScanioAI/1.0",
    }
    if provider == "openrouter":
        headers["HTTP-Referer"] = "http://localhost:8080"
        headers["X-Title"] = "Scanio AI"

    for attempt in range(MAX_TRANSIENT_RETRIES + 3):
        body_bytes = json.dumps(current_payload).encode("utf-8")
        req = urllib.request.Request(url, data=body_bytes, headers=headers)
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
                    "Transient %s HTTP %s for %s (attempt %s/%s); retrying in %.1fs.",
                    label,
                    exc.code,
                    view_name,
                    attempt + 1,
                    MAX_TRANSIENT_RETRIES + 1,
                    delay,
                )
                time.sleep(delay)
                last_error = _classify_provider_http_error(
                    exc.code, raw_msg, api_key, model, view_name=view_name, provider=provider
                )
                continue

            if exc.code == 400 and "reasoning" in lower and "reasoning" in current_payload:
                current_payload = {k: v for k, v in current_payload.items() if k != "reasoning"}
                continue

            if exc.code == 400 and any(tok in lower for tok in ("response_format", "json_schema", "strict", "grammar", "guided")) and "response_format" in current_payload:
                rf = current_payload.get("response_format", {})
                if isinstance(rf, dict) and rf.get("type") == "json_schema":
                    js = rf.get("json_schema") or {}
                    if isinstance(js, dict) and js.get("strict") and provider == "nvidia":
                        current_payload = dict(current_payload)
                        current_payload["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {k: v for k, v in js.items() if k != "strict"},
                        }
                        continue
                    current_payload = dict(current_payload)
                    current_payload["response_format"] = {"type": "json_object"}
                    continue
                current_payload = {k: v for k, v in current_payload.items() if k != "response_format"}
                continue

            if exc.code == 400 and "system" in lower:
                merged = _merge_system_into_user_message(current_payload)
                if merged is not None:
                    current_payload = merged
                    continue

            raise _classify_provider_http_error(
                exc.code, raw_msg, api_key, model, view_name=view_name, provider=provider
            ) from exc
        except json.JSONDecodeError as exc:
            raise OpenRouterMalformedResponseError(
                f"{label} returned an invalid JSON envelope for the {view_name} image."
            ) from exc
        except Exception as exc:
            if _is_timeout_exception(exc):
                raise OpenRouterTimeoutError(
                    f"{label} screening request timed out while analyzing the {view_name} image. Please try again."
                ) from exc
            sanitized = sanitize_secret(str(exc), api_key)
            raise VisionClientError(f"{label} connection error ({view_name}): {sanitized}") from exc

    if isinstance(last_error, VisionClientError):
        raise last_error
    raise OpenRouterProviderError(f"{label} provider error for '{model}'.")


def _send_openrouter_chat_request(
    url: str,
    payload: dict,
    api_key: str,
    model: str,
    view_name: str,
) -> dict:
    return _send_chat_request(url, payload, api_key, model, view_name, provider="openrouter")


def compute_box_iou(box_a: dict, box_b: dict) -> float:
    """Computes normalized Intersection over Union (IoU) for two {x, y, w, h} boxes."""
    ax1, ay1 = float(box_a["x"]), float(box_a["y"])
    ax2, ay2 = ax1 + float(box_a["w"]), ay1 + float(box_a["h"])
    bx1, by1 = float(box_b["x"]), float(box_b["y"])
    bx2, by2 = bx1 + float(box_b["w"]), by1 + float(box_b["h"])

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def map_crop_box_to_original(
    box: dict,
    crop_rect: tuple[float, float, float, float],
) -> tuple[dict, list[int]]:
    """Maps normalized [0, 1] crop coordinates back into full-image normalized [0, 1] and 0..1000 box_2d."""
    cx0, cy0, cw, ch = crop_rect
    ox = max(0.0, min(1.0, cx0 + float(box["x"]) * cw))
    oy = max(0.0, min(1.0, cy0 + float(box["y"]) * ch))
    ow = max(0.0, min(1.0 - ox, float(box["w"]) * cw))
    oh = max(0.0, min(1.0 - oy, float(box["h"]) * ch))
    mapped = {
        "x": round(ox, 4),
        "y": round(oy, 4),
        "w": round(ow, 4),
        "h": round(oh, 4),
    }
    raw_b2d = [
        int(round(oy * 1000.0)),
        int(round(ox * 1000.0)),
        int(round((oy + oh) * 1000.0)),
        int(round((ox + ow) * 1000.0)),
    ]
    return mapped, raw_b2d


def _deduplicate_findings(findings: list[dict], iou_threshold: float = 0.42) -> list[dict]:
    """Deduplicates overlapping findings of the same category across full-frame and regional crop passes."""
    ranked = sorted(findings, key=lambda f: float(f.get("confidence", 0.0)), reverse=True)
    kept: list[dict] = []
    for cand in ranked:
        c_cat = cand.get("category")
        c_tooth = str(cand.get("tooth") or "").strip().lower()
        c_name = str(cand.get("findingType") or cand.get("name") or "").strip().lower()
        c_box = cand["box"]
        is_dup = False
        for existing in kept:
            if existing.get("category") != c_cat:
                continue
            iou = compute_box_iou(existing["box"], c_box)
            if iou >= iou_threshold:
                is_dup = True
                break
            e_tooth = str(existing.get("tooth") or "").strip().lower()
            e_name = str(existing.get("findingType") or existing.get("name") or "").strip().lower()
            if c_tooth and e_tooth == c_tooth and c_name == e_name and iou >= 0.15:
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)
    return kept


def _nominal_crop_specs(view_name: str) -> list[tuple[str, tuple[float, float, float, float]]]:
    v = view_name.lower()
    if v == "lower":
        return [
            ("anterior", (0.14, 0.48, 0.72, 0.50)),
            ("left_posterior", (0.0, 0.16, 0.46, 0.74)),
            ("right_posterior", (0.54, 0.16, 0.46, 0.74)),
        ]
    if v == "upper":
        return [
            ("anterior", (0.14, 0.02, 0.72, 0.46)),
            ("left_posterior", (0.0, 0.12, 0.46, 0.84)),
            ("right_posterior", (0.54, 0.12, 0.46, 0.84)),
        ]
    return [
        ("anterior_center", (0.18, 0.04, 0.64, 0.92)),
        ("left_buccal", (0.0, 0.04, 0.50, 0.92)),
        ("right_buccal", (0.50, 0.04, 0.50, 0.92)),
    ]


def generate_arch_crops(
    image_bytes: bytes,
    mime_type: str,
    view_name: str,
) -> list[dict]:
    """Creates overlapping regional crops covering posterior and anterior tooth bands without clipping crowns."""
    crops: list[dict] = []
    fmt = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}.get(mime_type, "JPEG")
    with Image.open(BytesIO(image_bytes)) as img:
        w, h = img.size
        if w < 100 or h < 100:
            return crops
        for region_label, (nx, ny, nw, nh) in _nominal_crop_specs(view_name):
            px0 = max(0, int(math.floor(nx * w)))
            py0 = max(0, int(math.floor(ny * h)))
            px1 = min(w, int(math.ceil((nx + nw) * w)))
            py1 = min(h, int(math.ceil((ny + nh) * h)))
            if px1 - px0 < 40 or py1 - py0 < 40:
                continue
            cropped = img.crop((px0, py0, px1, py1))
            if fmt == "JPEG" and cropped.mode not in ("RGB", "L"):
                cropped = cropped.convert("RGB")
            buf = BytesIO()
            save_kwargs = {"format": fmt}
            if fmt in ("JPEG", "WEBP"):
                save_kwargs["quality"] = 95
                if fmt == "JPEG":
                    save_kwargs["subsampling"] = 0
            cropped.save(buf, **save_kwargs)
            actual_rect = (
                px0 / float(w),
                py0 / float(h),
                (px1 - px0) / float(w),
                (py1 - py0) / float(h),
            )
            crops.append(
                {
                    "region": region_label,
                    "rect": actual_rect,
                    "bytes": buf.getvalue(),
                }
            )
    return crops


def prepare_image_for_provider(
    image_bytes: bytes,
    mime_type: str,
    provider: str = "nvidia",
) -> tuple[bytes, str]:
    """
    Preserves original bytes and MIME type (JPEG/PNG/WEBP) unless NVIDIA NIM's inline
    base64 payload limit requires aspect-preserving size reduction (never cropping teeth).
    """
    if provider != "nvidia" or len(image_bytes) <= NVIDIA_INLINE_RAW_LIMIT_BYTES:
        return image_bytes, mime_type

    fmt_map = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
    fmt = fmt_map.get(mime_type, "JPEG")
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            base = ImageOps.exif_transpose(img)
            w, h = base.size
            smallest: tuple[bytes, str] | None = None
            for max_side in (1600, 1280, 1024, 860):
                scaled = base.copy()
                if max(w, h) > max_side:
                    scaled.thumbnail((max_side, max_side))
                if fmt in ("JPEG", "WEBP"):
                    out_im = scaled.convert("RGB") if fmt == "JPEG" and scaled.mode not in ("RGB", "L") else scaled
                    for q in (90, 82, 74, 65):
                        buf = BytesIO()
                        kwargs = {"format": fmt, "quality": q}
                        if fmt == "JPEG":
                            kwargs["optimize"] = True
                        out_im.save(buf, **kwargs)
                        data = buf.getvalue()
                        if smallest is None or len(data) < len(smallest[0]):
                            smallest = (data, mime_type)
                        if len(data) <= NVIDIA_INLINE_RAW_LIMIT_BYTES:
                            return data, mime_type
                else:
                    buf_png = BytesIO()
                    scaled.save(buf_png, format="PNG", optimize=True)
                    png_data = buf_png.getvalue()
                    if smallest is None or len(png_data) < len(smallest[0]):
                        smallest = (png_data, "image/png")
                    if len(png_data) <= NVIDIA_INLINE_RAW_LIMIT_BYTES:
                        return png_data, "image/png"
            # Final high-efficiency JPEG fallback preserving full field of view under limit
            for max_side in (1024, 860, 720, 640):
                scaled = base.copy()
                if max(w, h) > max_side:
                    scaled.thumbnail((max_side, max_side))
                out_im = scaled.convert("RGB") if scaled.mode not in ("RGB", "L") else scaled
                for q in (82, 72, 62):
                    buf = BytesIO()
                    out_im.save(buf, format="JPEG", quality=q, optimize=True)
                    data = buf.getvalue()
                    if smallest is None or len(data) < len(smallest[0]):
                        smallest = (data, "image/jpeg")
                    if len(data) <= NVIDIA_INLINE_RAW_LIMIT_BYTES:
                        return data, "image/jpeg"
            if smallest is not None:
                return smallest
    except Exception:
        pass
    return image_bytes, mime_type


def _build_chat_payload(
    prompt_text: str,
    image_bytes: bytes,
    mime_type: str,
    model: str,
    caps: dict,
    provider: str = "openrouter",
) -> dict:
    prepared_bytes, effective_mime = prepare_image_for_provider(image_bytes, mime_type, provider=provider)
    b64_image = base64.b64encode(prepared_bytes).decode("utf-8")
    data_url = f"data:{effective_mime};base64,{b64_image}"
    payload: dict = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": OPENROUTER_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": DENTAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }
    if caps.get("structuredOutputSupported", provider != "nvidia"):
        json_schema_cfg: dict = {
            "name": "dental_screening",
            "schema": OPENROUTER_JSON_SCHEMA,
        }
        if provider == "openrouter":
            json_schema_cfg["strict"] = True
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": json_schema_cfg,
        }
    elif caps.get("responseFormatSupported"):
        payload["response_format"] = {"type": "json_object"}

    if provider == "openrouter" and caps.get(
        "supportsReasoning", "gemini-2.5" in model.lower() or "gemini-3" in model.lower()
    ):
        payload["reasoning"] = {"max_tokens": 2048}
    return payload


def _parse_parsed_payload(parsed: dict | list, view_name: str) -> dict:
    view_inconclusive = False
    inconclusive_reason: str | None = None
    condition_status: str | None = None
    alignment_status: str | None = None
    visible_summary: str | None = None
    unassessable_reason: str | None = None
    raw_conditions: list = []
    raw_alignment: list = []

    if isinstance(parsed, dict):
        raw_status = str(parsed.get("viewStatus") or "assessed").strip().lower()
        if raw_status == "inconclusive":
            view_inconclusive = True
            reason = parsed.get("inconclusiveReason")
            inconclusive_reason = (
                str(reason).strip()
                if reason
                else f"The {view_name} photo did not provide enough clarity for reliable inspection."
            )

        cond_stat_val = parsed.get("conditionStatus")
        if isinstance(cond_stat_val, str) and cond_stat_val.strip():
            condition_status = cond_stat_val.strip().lower()
        align_stat_val = parsed.get("alignmentStatus")
        if isinstance(align_stat_val, str) and align_stat_val.strip():
            alignment_status = align_stat_val.strip().lower()

        if isinstance(parsed.get("visibleToothSummary"), str):
            visible_summary = parsed["visibleToothSummary"].strip() or None
        if isinstance(parsed.get("unassessableTeethReason"), str):
            unassessable_reason = parsed["unassessableTeethReason"].strip() or None

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

    return {
        "viewInconclusive": view_inconclusive,
        "inconclusiveReason": inconclusive_reason,
        "conditionStatus": condition_status,
        "alignmentStatus": alignment_status,
        "visibleToothSummary": visible_summary,
        "unassessableTeethReason": unassessable_reason,
        "rawConditions": raw_conditions,
        "rawAlignment": raw_alignment,
    }


def _derive_module_status(
    findings: list[dict],
    reported_status: str | None,
    view_inconclusive: bool,
) -> str:
    if findings:
        return "assessed_with_findings"
    if reported_status == "inconclusive" or (reported_status is None and view_inconclusive):
        return "inconclusive"
    return "assessed_no_findings"


def analyze_with_vision_provider(
    image_bytes: bytes,
    view_name: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: str = "nvidia",
    mime_type: str | None = None,
    capabilities: dict | None = None,
    enable_crop_fallback: bool = True,
) -> dict:
    """
    Submits dental image bytes to NVIDIA NIM or OpenRouter Chat Completions API using
    multimodal image_url content with preserved MIME type, quality metrics, and optional
    overlapping regional crop refinement.
    """
    detected_mime = mime_type or detect_image_mime_type(image_bytes)
    upright_bytes = normalize_image_orientation(image_bytes, detected_mime)
    quality = assess_image_quality(upright_bytes)
    enamel_mask = compute_tooth_enamel_mask(upright_bytes)

    url = f"{base_url.rstrip('/')}/chat/completions"
    caps = capabilities or {}
    payload = _build_chat_payload(
        build_view_prompt(view_name, quality_context=quality),
        upright_bytes,
        detected_mime,
        model,
        caps,
        provider=provider,
    )

    if provider == "openrouter":
        result = _send_openrouter_chat_request(url, payload, api_key, model, view_name)
    else:
        result = _send_chat_request(url, payload, api_key, model, view_name, provider=provider)
    text_content = _extract_openrouter_text(result, view_name)
    parsed = _parse_json_response(text_content, view_name)
    extracted = _parse_parsed_payload(parsed, view_name)

    valid_conditions: list[dict] = []
    rejected_count = 0
    for item in extracted["rawConditions"]:
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
    for item in extracted["rawAlignment"]:
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

    raw_conditions_count = len(extracted["rawConditions"])
    raw_alignment_count = len(extracted["rawAlignment"])
    ai_inconclusive = bool(extracted["viewInconclusive"])
    reported_cond_status = extracted["conditionStatus"]
    reported_align_status = extracted["alignmentStatus"]
    inconclusive_reason = extracted["inconclusiveReason"]
    crop_pass_used = False
    crops_analyzed: list[str] = []

    needs_crop_fallback = (
        enable_crop_fallback
        and quality.get("width", 0) >= 200
        and quality.get("height", 0) >= 200
        and quality.get("qualityStatus") != "too_small"
        and (
            (ai_inconclusive and quality.get("qualityStatus") in ("good", "soft_focus", "dark", "overexposed"))
            or (rejected_count > 0 and not (valid_conditions or valid_alignment))
            or (view_name.lower() == "lower" and (not valid_conditions or not valid_alignment))
        )
    )

    if needs_crop_fallback:
        crop_specs = generate_arch_crops(upright_bytes, detected_mime, view_name)
        if crop_specs:
            crop_pass_used = True
            crop_non_inconclusive = 0
            crop_cond_assessed = 0
            crop_align_assessed = 0
            for spec in crop_specs:
                region = spec["region"]
                crops_analyzed.append(region)
                crop_payload = _build_chat_payload(
                    build_crop_prompt(view_name, region),
                    spec["bytes"],
                    detected_mime,
                    model,
                    caps,
                    provider=provider,
                )
                try:
                    if provider == "openrouter":
                        crop_res = _send_openrouter_chat_request(
                            url, crop_payload, api_key, model, f"{view_name}:{region}"
                        )
                    else:
                        crop_res = _send_chat_request(
                            url, crop_payload, api_key, model, f"{view_name}:{region}", provider=provider
                        )
                    crop_text = _extract_openrouter_text(crop_res, f"{view_name}:{region}")
                    crop_parsed = _parse_json_response(crop_text, f"{view_name}:{region}")
                    crop_ext = _parse_parsed_payload(crop_parsed, f"{view_name}:{region}")
                except VisionClientError as exc:
                    logger.warning("Crop pass %s:%s failed (%s); keeping full-frame findings.", view_name, region, exc)
                    continue

                if not crop_ext["viewInconclusive"]:
                    crop_non_inconclusive += 1
                if crop_ext["conditionStatus"] and crop_ext["conditionStatus"] != "inconclusive":
                    crop_cond_assessed += 1
                elif not crop_ext["viewInconclusive"]:
                    crop_cond_assessed += 1
                if crop_ext["alignmentStatus"] and crop_ext["alignmentStatus"] != "inconclusive":
                    crop_align_assessed += 1
                elif not crop_ext["viewInconclusive"]:
                    crop_align_assessed += 1

                for cat, raw_list, target_list in (
                    ("condition", crop_ext["rawConditions"], valid_conditions),
                    ("alignment", crop_ext["rawAlignment"], valid_alignment),
                ):
                    raw_conditions_count += len(raw_list) if cat == "condition" else 0
                    raw_alignment_count += len(raw_list) if cat == "alignment" else 0
                    for item in raw_list:
                        if not isinstance(item, dict):
                            rejected_count += 1
                            continue
                        local_box, _ = _parse_strict_box(item, category=cat, view_name=view_name)
                        if local_box is None:
                            rejected_count += 1
                            continue
                        orig_box, orig_b2d = map_crop_box_to_original(local_box, spec["rect"])
                        remapped_item = dict(item)
                        remapped_item.pop("boundingBox", None)
                        remapped_item["box"] = orig_box
                        remapped_item["box_2d"] = orig_b2d
                        norm = sanitize_and_normalize_finding(
                            remapped_item,
                            view_name,
                            category=cat,
                            enamel_mask=enamel_mask,
                        )
                        if norm:
                            target_list.append(norm)
                        else:
                            rejected_count += 1

            valid_conditions = _deduplicate_findings(valid_conditions)
            valid_alignment = _deduplicate_findings(valid_alignment)
            if valid_conditions or valid_alignment or crop_non_inconclusive >= 2:
                ai_inconclusive = False
                inconclusive_reason = None
            if crop_cond_assessed > 0 and reported_cond_status == "inconclusive":
                reported_cond_status = "assessed_with_findings" if valid_conditions else "assessed_no_findings"
            if crop_align_assessed > 0 and reported_align_status == "inconclusive":
                reported_align_status = "assessed_with_findings" if valid_alignment else "assessed_no_findings"

    condition_status = _derive_module_status(
        valid_conditions,
        reported_cond_status,
        ai_inconclusive,
    )
    alignment_status = _derive_module_status(
        valid_alignment,
        reported_align_status,
        ai_inconclusive,
    )

    all_findings = valid_conditions + valid_alignment

    if all_findings:
        explicit_view_status = "assessed_with_findings"
        view_status = "assessed"
        inconclusive_reason = None
    elif ai_inconclusive and not crop_pass_used and (condition_status == "inconclusive" and alignment_status == "inconclusive"):
        explicit_view_status = "inconclusive"
        view_status = "inconclusive"
    elif condition_status == "inconclusive" and alignment_status == "inconclusive":
        explicit_view_status = "inconclusive"
        view_status = "inconclusive"
    else:
        explicit_view_status = "assessed_no_findings"
        view_status = "assessed"
        inconclusive_reason = None

    return {
        "viewStatus": view_status,
        "explicitViewStatus": explicit_view_status,
        "conditionStatus": condition_status,
        "alignmentStatus": alignment_status,
        "inconclusiveReason": inconclusive_reason,
        "toothConditions": valid_conditions,
        "dentalAlignment": valid_alignment,
        "all": all_findings,
        "rejectedCount": rejected_count,
        "rawConditionsCount": raw_conditions_count,
        "rawAlignmentCount": raw_alignment_count,
        "imageQuality": quality,
        "visibleToothSummary": extracted["visibleToothSummary"],
        "unassessableTeethReason": extracted["unassessableTeethReason"],
        "cropPassUsed": crop_pass_used,
        "cropsAnalyzed": crops_analyzed,
        "usage": result.get("usage", {}),
    }


def analyze_with_nvidia(
    image_bytes: bytes,
    view_name: str,
    api_key: str,
    model: str = DEFAULT_NVIDIA_MODEL,
    base_url: str = DEFAULT_NVIDIA_BASE_URL,
    mime_type: str | None = None,
    capabilities: dict | None = None,
    enable_crop_fallback: bool = True,
) -> dict:
    return analyze_with_vision_provider(
        image_bytes,
        view_name,
        api_key,
        model,
        base_url,
        provider="nvidia",
        mime_type=mime_type,
        capabilities=capabilities,
        enable_crop_fallback=enable_crop_fallback,
    )


def analyze_with_openrouter(
    image_bytes: bytes,
    view_name: str,
    api_key: str,
    model: str,
    base_url: str = DEFAULT_OPENROUTER_BASE_URL,
    mime_type: str | None = None,
    capabilities: dict | None = None,
    enable_crop_fallback: bool = True,
) -> dict:
    return analyze_with_vision_provider(
        image_bytes,
        view_name,
        api_key,
        model,
        base_url,
        provider="openrouter",
        mime_type=mime_type,
        capabilities=capabilities,
        enable_crop_fallback=enable_crop_fallback,
    )
