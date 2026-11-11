import logging
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import UnidentifiedImageError

from app.vision_client import (
    OpenRouterAuthError,
    OpenRouterConfigError,
    OpenRouterMalformedResponseError,
    OpenRouterModelUnavailableError,
    OpenRouterPaymentError,
    OpenRouterPermissionError,
    OpenRouterProviderError,
    OpenRouterRateLimitError,
    OpenRouterTimeoutError,
    OpenRouterUnsupportedImageError,
    VisionClientError,
    detect_image_mime_type,
    reload_environment,
    sanitize_secret,
)

reload_environment()

from app.inference import (  # noqa: E402
    ModelNotReady,
    OpenRouterAPIError,
    model_status as get_model_status,
    run_screening,
)
from app.schemas import ScreeningResponse  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("scanio.main")

APP_NAME = "Scanio AI Screening API"
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", "8000000"))
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

app = FastAPI(title=APP_NAME, version="0.1.0")

origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080",
).split(",")
cors_origin_regex = os.getenv("CORS_ORIGIN_REGEX", r"^https://.*\.vercel\.app$").strip() or None
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins if origin.strip()],
    allow_origin_regex=cors_origin_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


async def validate_image(file: UploadFile, view_name: str) -> bytes:
    content_type = (file.content_type or "").lower().strip()
    if content_type and content_type != "application/octet-stream" and content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"{view_name}: unsupported image type. Use JPG, PNG, or WEBP.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"{view_name}: uploaded image is empty.")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{view_name}: image exceeds the {MAX_IMAGE_BYTES // 1_000_000} MB size limit.",
        )
    try:
        detect_image_mime_type(content)
    except OpenRouterUnsupportedImageError as exc:
        raise HTTPException(status_code=415, detail=f"{view_name}: {exc}") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"{view_name}: file is not a valid image.") from exc
    return content


@app.get("/health")
def health():
    return {"status": "ok", "service": APP_NAME}


@app.get("/api/v1/model-status")
def model_status():
    return get_model_status()


@app.post("/api/v1/predict", response_model=ScreeningResponse)
async def predict(
    upper: UploadFile = File(...),
    lower: UploadFile = File(...),
    front: UploadFile = File(...),
):
    images = {
        "upper": await validate_image(upper, "Upper arch"),
        "lower": await validate_image(lower, "Lower arch"),
        "front": await validate_image(front, "Front view"),
    }
    try:
        result = run_screening(images)
        return ScreeningResponse.model_validate(result)
    except (ModelNotReady, OpenRouterConfigError) as exc:
        logger.warning("Screening blocked - OpenRouter not configured: %s", sanitize_secret(str(exc)))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OpenRouterAuthError as exc:
        logger.error("OpenRouter authentication failure: %s", sanitize_secret(str(exc)))
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except OpenRouterPaymentError as exc:
        logger.warning("OpenRouter payment/credits required: %s", sanitize_secret(str(exc)))
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    except OpenRouterPermissionError as exc:
        logger.warning("OpenRouter permission error: %s", sanitize_secret(str(exc)))
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except OpenRouterModelUnavailableError as exc:
        logger.warning("OpenRouter model unavailable: %s", sanitize_secret(str(exc)))
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OpenRouterRateLimitError as exc:
        logger.warning("OpenRouter rate limit exceeded: %s", sanitize_secret(str(exc)))
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except OpenRouterTimeoutError as exc:
        logger.warning("OpenRouter request timed out: %s", sanitize_secret(str(exc)))
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except OpenRouterUnsupportedImageError as exc:
        logger.warning("OpenRouter rejected image: %s", sanitize_secret(str(exc)))
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except OpenRouterMalformedResponseError as exc:
        logger.error("Malformed OpenRouter response: %s", sanitize_secret(str(exc)))
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (OpenRouterProviderError, OpenRouterAPIError, VisionClientError, RuntimeError) as exc:
        safe_msg = sanitize_secret(str(exc))
        logger.error("OpenRouter upstream error: %s", safe_msg, exc_info=True)
        raise HTTPException(status_code=502, detail=safe_msg) from exc
    except ValueError as exc:
        safe_msg = sanitize_secret(str(exc))
        logger.warning("Screening validation error: %s", safe_msg)
        raise HTTPException(status_code=422, detail=safe_msg) from exc
    except Exception as exc:
        safe_msg = sanitize_secret(str(exc))
        logger.error("Internal prediction failure: %s", safe_msg, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal backend error while processing the screening. Check the backend terminal logs.",
        ) from exc
