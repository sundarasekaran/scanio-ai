# Scanio AI — FastAPI Screening Backend

FastAPI backend for **Scanio AI**, supporting two independent dental screening modules across all three intraoral views (`upper`, `lower`, `front`):

1. **Module 1 — Tooth-Surface Condition Analysis**: Caries-like lesions, enamel discoloration, visible plaque, tartar/calculus, visible cracks/fractures, and surface abnormalities localized strictly to visible tooth enamel.
2. **Module 2 — Dental Alignment Analysis**: Diastema/gaps, crowding, overlapping, rotated teeth, displaced teeth, uneven positioning, and irregular arch alignment with FDI labels where visually supported.

## Active Vision Provider (NVIDIA NIM API)

The backend defaults to **NVIDIA NIM** via the verified multimodal endpoint `https://integrate.api.nvidia.com/v1/chat/completions` and default vision model `nvidia/nemotron-3.5-super-vl-120b-a12b`.

Required backend environment variables (copy `.env.example` to `.env` for local runs, or set them directly in Render Dashboard):

```env
VISION_PROVIDER=nvidia
NVIDIA_API_KEY=YOUR_NVIDIA_API_KEY
NVIDIA_MODEL=nvidia/nemotron-3.5-super-vl-120b-a12b
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
```

Optional fallback (`VISION_PROVIDER=openrouter`) remains supported for compatibility when `OPENROUTER_API_KEY` is configured. Never commit `.env` or `.env.local` and never expose API keys in frontend `VITE_*` variables or API responses.

## Run Locally (Windows PowerShell)

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env locally and insert your real NVIDIA_API_KEY
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET /health` — Render health check (`{"status": "ok", "service": "Scanio AI Screening API"}`)
- `GET /api/v1/model-status` — Active vision provider (`nvidia`), selected model, sanitized availability check (never leaks keys)
- `POST /api/v1/predict` — Multipart upload (`upper`, `lower`, `front`) returning structured findings, per-view statuses (`assessed_with_findings`, `assessed_no_findings`, `inconclusive`), image quality metrics, and overall score/summary
- `GET /docs` — OpenAPI documentation

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```
