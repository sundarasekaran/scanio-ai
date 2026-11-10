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

> **Security Note**: Never commit `.env` or `.env.local` files or expose `OPENROUTER_API_KEY` in frontend `VITE_*` variables.
