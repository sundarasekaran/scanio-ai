"""
Inference coordinator for Scanio AI dental screening (NVIDIA NIM & OpenRouter).

Supports TWO independent screening modules:
1. TOOTH-SURFACE CONDITION ANALYSIS (enamel caries, discoloration, cracks, plaque, tartar/calculus)
2. DENTAL ALIGNMENT ANALYSIS (diastema/gaps, crowding, rotation, displacement, arch form)
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging

from app.vision_client import (
    OpenRouterAuthError,
    OpenRouterConfigError,
    OpenRouterModelUnavailableError,
    OpenRouterPaymentError,
    OpenRouterPermissionError,
    OpenRouterProviderError,
    OpenRouterRateLimitError,
    OpenRouterTimeoutError,
    VisionClientError,
    analyze_with_nvidia,
    analyze_with_openrouter,
    assess_image_quality,
    get_vision_config,
    sanitize_secret,
    verify_nvidia_model_availability,
    verify_openrouter_model_availability,
)

logger = logging.getLogger("scanio.inference")

BASE_DIR = Path(__file__).resolve().parents[1]
SUPPORTED_PROVIDERS = ("nvidia", "openrouter")


class ModelNotReady(RuntimeError):
    pass


class OpenRouterAPIError(VisionClientError):
    pass


def _raise_verification_error(verification: dict, api_key: str) -> None:
    status = str(verification.get("status") or "").lower()
    err = sanitize_secret(verification.get("error", "Vision provider verification failed"), api_key)

    if status == "invalid_api_key":
        raise OpenRouterAuthError(err)
    if status == "insufficient_credits":
        raise OpenRouterPaymentError(err)
    if status == "permission_denied":
        raise OpenRouterPermissionError(err)
    if status in ("model_not_found", "unsupported_vision"):
        raise OpenRouterModelUnavailableError(err)
    if status == "rate_limited":
        raise OpenRouterRateLimitError(err)
    if status == "timeout":
        raise OpenRouterTimeoutError(err)
    raise OpenRouterAPIError(err)


def model_status() -> dict:
    """Returns active vision provider and model status for GET /api/v1/model-status (without secrets)."""
    config = get_vision_config()
    provider = config["provider"]
    model = config["model"]
    base_url = config["base_url"]
    has_key = config["has_key"]
    api_key = config["api_key"]
    env_path = config["env_path"]
    env_exists = config["env_exists"]

    if provider not in SUPPORTED_PROVIDERS:
        return {
            "vision_provider": provider,
            "selected_model": model,
            "base_url": base_url,
            "api_key_configured": has_key,
            "screening_ready": False,
            "model_availability": "unsupported_provider",
            "env_file_loaded": env_exists,
            "env_file_path": env_path,
            "message": f"Unsupported VISION_PROVIDER '{provider}'. Set VISION_PROVIDER=nvidia or openrouter.",
        }

    key_var = "NVIDIA_API_KEY" if provider == "nvidia" else "OPENROUTER_API_KEY"
    provider_label = "NVIDIA API" if provider == "nvidia" else "OpenRouter"

    status_response = {
        "vision_provider": provider,
        "selected_model": model,
        "base_url": base_url,
        "api_key_configured": has_key,
        "screening_ready": False,
        "model_availability": "not_configured",
        "env_file_loaded": env_exists,
        "env_file_path": env_path,
        "message": "",
    }

    if not has_key:
        status_response["model_availability"] = "missing_key"
        if provider == "nvidia" and config.get("openrouter_has_nvapi_key"):
            status_response["message"] = (
                "NVIDIA_API_KEY is missing or empty, while OPENROUTER_API_KEY contains an nvapi-... key. "
                "Add NVIDIA_API_KEY in Render Environment Variables and redeploy the service."
            )
        else:
            status_response["message"] = (
                f"{key_var} is missing or empty. Configure {key_var} in the backend environment."
            )
        return status_response

    if provider == "nvidia":
        verification = verify_nvidia_model_availability(api_key, model, base_url, run_auth_probe=True)
    else:
        verification = verify_openrouter_model_availability(api_key, model, base_url)

    if verification.get("valid"):
        status_response["screening_ready"] = True
        status_response["model_availability"] = "available"
        status_response["display_name"] = verification.get("displayName", model)
        status_response["vision_supported"] = verification.get("visionSupported", True)
        status_response["structured_output_supported"] = verification.get("structuredOutputSupported", False)
        status_response["message"] = (
            f"{provider_label} vision model '{model}' is verified and ready for dual-module dental screening."
        )
    else:
        status_response["screening_ready"] = False
        status_response["model_availability"] = verification.get("status", "error")
        sanitized_err = sanitize_secret(verification.get("error", "Unknown error"), api_key)
        status_response["message"] = sanitized_err

    return status_response


def run_screening(images: dict[str, bytes]) -> dict:
    """
    Submits uploaded upper, lower, and front dental photos to the configured Vision API
    (NVIDIA NIM or OpenRouter). Validates tooth-level boxes, aggregates dual-module
    findings, flags inconclusive views, and calculates oral health scores.
    """
    config = get_vision_config()
    provider = config["provider"]
    if provider not in SUPPORTED_PROVIDERS:
        raise OpenRouterConfigError(
            f"Unsupported VISION_PROVIDER '{provider}'. Set VISION_PROVIDER=nvidia or openrouter."
        )

    key_var = "NVIDIA_API_KEY" if provider == "nvidia" else "OPENROUTER_API_KEY"
    if not config["has_key"]:
        if provider == "nvidia" and config.get("openrouter_has_nvapi_key"):
            raise ModelNotReady(
                "NVIDIA_API_KEY is not configured or empty (OPENROUTER_API_KEY still holds an nvapi-... key). "
                "Add NVIDIA_API_KEY in Render Environment Variables and redeploy the service."
            )
        raise ModelNotReady(
            f"{key_var} is not configured or empty. Configure {key_var} in Render or backend .env."
        )

    api_key = config["api_key"]
    model = config["model"]
    base_url = config["base_url"]
    provider_label = "NVIDIA NIM" if provider == "nvidia" else "OpenRouter"

    if provider == "nvidia":
        verification = verify_nvidia_model_availability(api_key, model, base_url, run_auth_probe=False)
        analyzer = analyze_with_nvidia
    else:
        verification = verify_openrouter_model_availability(api_key, model, base_url)
        analyzer = analyze_with_openrouter

    if not verification.get("valid"):
        _raise_verification_error(verification, api_key)

    views_order = ("upper", "lower", "front")
    logger.info("Submitting upper, lower, and front images to %s (%s)...", provider_label, model)

    view_results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            view: executor.submit(
                analyzer,
                images[view],
                view,
                api_key,
                model,
                base_url,
                None,
                verification,
            )
            for view in views_order
        }
        for view, future in futures.items():
            view_results[view] = future.result()

    upper_res = view_results["upper"]
    lower_res = view_results["lower"]
    front_res = view_results["front"]

    upper_findings = upper_res.get("all", [])
    lower_findings = lower_res.get("all", [])
    front_findings = front_res.get("all", [])

    all_findings = upper_findings + lower_findings + front_findings

    inconclusive_views: list[str] = []
    inconclusive_reasons: dict[str, str] = {}
    view_statuses: dict[str, str] = {}
    condition_statuses: dict[str, str] = {}
    alignment_statuses: dict[str, str] = {}
    image_quality_map: dict[str, dict] = {}
    view_diagnostics: dict[str, dict] = {}
    rejected_total = 0

    for view in views_order:
        res = view_results[view]
        cond_list = res.get("toothConditions", [])
        align_list = res.get("dentalAlignment", [])
        rejected_count = int(res.get("rejectedCount", 0) or 0)
        rejected_total += rejected_count

        is_inc = res.get("viewStatus") == "inconclusive" or res.get("explicitViewStatus") == "inconclusive"
        cond_status = res.get("conditionStatus") or (
            "assessed_with_findings"
            if cond_list
            else ("inconclusive" if is_inc else "assessed_no_findings")
        )
        align_status = res.get("alignmentStatus") or (
            "assessed_with_findings"
            if align_list
            else ("inconclusive" if is_inc else "assessed_no_findings")
        )
        explicit_status = res.get("explicitViewStatus") or (
            "assessed_with_findings"
            if (cond_list or align_list)
            else ("inconclusive" if (is_inc or (cond_status == "inconclusive" and align_status == "inconclusive")) else "assessed_no_findings")
        )

        view_statuses[view] = explicit_status
        condition_statuses[view] = cond_status
        alignment_statuses[view] = align_status

        q_info = res.get("imageQuality") or assess_image_quality(images[view])
        image_quality_map[view] = q_info

        view_diagnostics[view] = {
            "viewStatus": explicit_status,
            "conditionStatus": cond_status,
            "alignmentStatus": align_status,
            "visibleToothSummary": res.get("visibleToothSummary"),
            "unassessableTeethReason": res.get("unassessableTeethReason"),
            "rawConditionsCount": int(res.get("rawConditionsCount", len(cond_list)) or 0),
            "rawAlignmentCount": int(res.get("rawAlignmentCount", len(align_list)) or 0),
            "rejectedCount": rejected_count,
            "cropPassUsed": bool(res.get("cropPassUsed", False)),
            "cropsAnalyzed": list(res.get("cropsAnalyzed") or []),
        }

        if explicit_status == "inconclusive" or cond_status == "inconclusive" or align_status == "inconclusive":
            inconclusive_views.append(view)
            inconclusive_reasons[view] = (
                res.get("inconclusiveReason")
                or f"The {view} image could not be reliably evaluated for all modules."
            )

    all_inconclusive = (
        sum(1 for v in views_order if view_statuses[v] == "inconclusive") == len(views_order)
    )
    is_inconclusive = len(inconclusive_views) > 0

    if all_inconclusive:
        overall_score = 0.0
        status_message = (
            "Screening inconclusive: unable to reliably inspect teeth in the uploaded photos. "
            "Please upload clearer, well-lit close-up dental photos."
        )
    else:
        cond_deductions = {"high": 15.0, "moderate": 7.0, "low": 3.0}
        align_deductions = {"high": 8.0, "moderate": 4.0, "low": 2.0}

        penalty = 0.0
        for finding in all_findings:
            sev = finding.get("severity", "low")
            if finding.get("category") == "alignment":
                penalty += align_deductions.get(sev, 2.0)
            else:
                penalty += cond_deductions.get(sev, 3.0)

        overall_score = round(max(40.0, min(100.0, 100.0 - penalty)), 1)
        status_message = (
            f"Partial screening: {', '.join(inconclusive_views)} view(s) inconclusive due to image clarity or visibility."
            if is_inconclusive
            else None
        )

    return {
        "overallScore": float(overall_score),
        "views": {
            "upper": upper_findings,
            "lower": lower_findings,
            "front": front_findings,
        },
        "modules": {
            "toothConditions": {
                "upper": upper_res.get("toothConditions", []),
                "lower": lower_res.get("toothConditions", []),
                "front": front_res.get("toothConditions", []),
            },
            "dentalAlignment": {
                "upper": upper_res.get("dentalAlignment", []),
                "lower": lower_res.get("dentalAlignment", []),
                "front": front_res.get("dentalAlignment", []),
            },
        },
        "viewStatuses": view_statuses,
        "moduleStatuses": {
            "toothConditions": condition_statuses,
            "dentalAlignment": alignment_statuses,
        },
        "imageQuality": image_quality_map,
        "viewDiagnostics": view_diagnostics,
        "screeningDisclaimer": (
            f"AI-assisted preliminary screening via {provider_label} ({model}). "
            "Separates Tooth-Surface Conditions (Module 1) and Dental Alignment (Module 2). "
            "Bounding boxes localize individual visible teeth, surfaces, and interdental alignment features only. "
            "This result is not a diagnosis and must be reviewed by a qualified dental professional."
        ),
        "isMock": False,
        "modelProvider": f"{provider} ({model})",
        "inconclusive": is_inconclusive,
        "inconclusiveViews": inconclusive_views,
        "inconclusiveReasons": inconclusive_reasons,
        "statusMessage": status_message,
        "rejectedFindingsCount": rejected_total,
    }
