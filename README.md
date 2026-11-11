# Scanio AI

AI-assisted dental screening web application combining a **React + TypeScript + TanStack Start** frontend with a **Python + FastAPI** backend powered by **OpenRouter Vision** (`google/gemini-2.5-flash`).

## Architecture

- **`frontend/`**: React 19, TypeScript, TanStack Start, Tailwind CSS, Vitest.
- **`backend/`**: FastAPI, Pydantic, Pillow, OpenRouter Chat Completions API (`POST /api/v1/predict`, `GET /api/v1/model-status`, `GET /health`).
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
# Edit backend/.env and set OPENROUTER_API_KEY
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
python -m unittest tests/test_backend.py -v

# Frontend typecheck, unit tests, and build
cd ../frontend
npx tsc --noEmit
npm test
npm run build
```

## Deploy as a Live Website

### 1. Backend on Render (Free Tier via `render.yaml`)

1. Sign in at [Render](https://render.com) with GitHub.
2. Click **New → Blueprint** and select `sundarasekaran/scanio-ai`.
3. Render reads [`render.yaml`](./render.yaml) (`plan: free`, `rootDir: backend`).
4. Paste your **`OPENROUTER_API_KEY`** when prompted and click **Apply**.
5. Once live, copy your backend URL (e.g., `https://scanio-ai-backend.onrender.com`) and confirm `https://<your-backend>.onrender.com/health` returns `{"status":"ok"}`.

### 2. Frontend on Vercel (Free Hobby Tier)

1. Sign in at [Vercel](https://vercel.com) with GitHub.
2. Click **Add New… → Project** and import `sundarasekaran/scanio-ai`.
3. Set **Root Directory** to **`frontend`** (`frontend/vercel.json` configures the `tanstack-start` preset automatically).
4. Add Environment Variable:
   - `VITE_PREDICTION_API_URL` = `https://<your-render-service>.onrender.com/api/v1/predict`
5. Click **Deploy**. Any `*.vercel.app` domain is allowed by `CORS_ORIGIN_REGEX` on the backend by default (set `CORS_ORIGINS` on Render if you add a custom domain).

> **Security Note**: Never commit `.env` or `.env.local` files or expose `OPENROUTER_API_KEY` in frontend `VITE_*` variables.
