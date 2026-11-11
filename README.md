# Scanio AI

AI-assisted dental screening web application combining a **React + TypeScript + TanStack Start** frontend with a **Python + FastAPI** backend powered by **NVIDIA NIM Vision API** (`nvidia/nemotron-3.5-super-vl-120b-a12b`).

## Architecture

- **`frontend/`**: React 19, TypeScript, TanStack Start, Tailwind CSS, Vitest (deployed on Vercel).
- **`backend/`**: FastAPI, Pydantic, Pillow, NVIDIA NIM Vision API (`https://integrate.api.nvidia.com/v1`) and OpenRouter compatibility (`POST /api/v1/predict`, `GET /api/v1/model-status`, `GET /health`).
- **Dual Independent Screening Modules**:
  1. **Tooth-Surface Condition Analysis**: Detects enamel surface conditions within visible tooth boundaries (excluding gums, palate, tongue, lips, and empty oral space).
  2. **Dental Alignment Analysis**: Identifies visible interdental spacing/diastema gaps, crowding, rotation, and displacement without fabricated measurements.

## Local Setup

### 1. Backend (`http://127.0.0.1:8000`)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit backend/.env and set your private NVIDIA_API_KEY (never commit .env!)
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 2. Frontend (`http://localhost:8080`)

```powershell
cd frontend
npm install
Copy-Item .env.example .env.local
npm run dev
```

## Verification

```powershell
# Backend unit tests
cd backend
python -m unittest discover -s tests -p "test_*.py" -v

# Frontend typecheck, unit tests, and build
cd ../frontend
npx tsc --noEmit
npm test
npm run build
```

## Deploy as a Live Website

### 1. Backend on Render (Free Tier via `render.yaml`)

1. Sign in at [Render](https://render.com) with GitHub.
2. Create or update the `scanio-ai-backend` web service from `sundarasekaran/scanio-ai` (`rootDir: backend`).
3. Configure these Render **Environment Variables** (keep `NVIDIA_API_KEY` secret in Render—never commit it to GitHub):
   - `VISION_PROVIDER` = `nvidia`
   - `NVIDIA_API_KEY` = `nvapi-...` *(your real NVIDIA API key)*
   - `NVIDIA_MODEL` = `nvidia/nemotron-3.5-super-vl-120b-a12b` *(or `google/gemma-4-31b-it`)*
   - `NVIDIA_BASE_URL` = `https://integrate.api.nvidia.com/v1`
   - `CORS_ORIGIN_REGEX` = `^https://.*\.vercel\.app$`
   - `MAX_IMAGE_BYTES` = `8000000`
4. Verify `https://<your-backend>.onrender.com/health` returns `{"status":"ok"}` and `/api/v1/model-status` reports `"vision_provider":"nvidia"` and `"screening_ready":true`.

### 2. Frontend on Vercel (Free Hobby Tier)

1. Sign in at [Vercel](https://vercel.com) with GitHub and import `sundarasekaran/scanio-ai`.
2. Set **Root Directory** to **`frontend`** (`frontend/vercel.json` configures the `tanstack-start` preset automatically).
3. Add Production Environment Variable:
   - `VITE_PREDICTION_API_URL` = `https://<your-render-service>.onrender.com/api/v1/predict`
4. Redeploy Vercel after updating environment variables so the compiled bundle picks up the Render backend URL.

> **Security Note**: Never commit `.env` or `.env.local` files or expose `NVIDIA_API_KEY` in frontend `VITE_*` variables, browser code, API responses, or logs.
