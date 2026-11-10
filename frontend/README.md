# Scanio AI — Frontend & Local Python Backend Integration

Scanio AI is an AI-assisted dental screening web application built with React, TypeScript, TanStack Start, and Tailwind CSS, connected to a local Python FastAPI backend.

---

## Architecture Overview

- **Frontend:** React 19 + TanStack Start + Tailwind CSS (runs by default on `http://localhost:8080`).
- **Backend:** Python FastAPI server (runs by default on `http://127.0.0.1:8000`).
- **Screening Endpoint:** `POST http://127.0.0.1:8000/api/v1/predict`
  - Accepts multipart form-data with three image fields: `upper`, `lower`, and `front`.
  - Validates image types (`image/jpeg`, `image/png`, `image/webp`), size limits, and image validity.
  - Returns honest HTTP 503 when trained model weights are not installed yet, preventing fabricated findings.

---

## Quick Start (Running Both Services Locally)

### 1. Start the Python FastAPI Backend

Open a terminal in the backend directory:

```powershell
cd "c:\Users\sunda\Downloads\scanio-ai-python-backend\scanio-ai-python-backend"

# Create virtual environment (if not already created)
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
# (or if PowerShell scripts are disabled: .venv\Scripts\activate.bat in cmd)

# Install requirements
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Verify backend health in your browser or terminal:
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Model status: [http://127.0.0.1:8000/api/v1/model-status](http://127.0.0.1:8000/api/v1/model-status)
- Interactive API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 2. Configure & Start the Frontend

In the root of `scanio-ai-export`, ensure `.env.local` contains:

```env
VITE_PREDICTION_API_URL=http://127.0.0.1:8000/api/v1/predict
```

> **Note:** If you add or change `.env.local`, the frontend dev server **must be restarted** for Vite to pick up the updated environment variables.

Install dependencies and start the dev server:

```powershell
cd "c:\Users\sunda\Downloads\scanio-ai-export"
npm install
npm run dev
```

Open the application:
- Local Application: [http://localhost:8080/](http://localhost:8080/)

---

## Model Status & Adding Trained Weights

The backend is currently in **scaffold / starter mode**. It explicitly does not fabricate dummy bounding boxes or artificial confidence scores.

When screening photos are uploaded without trained model weights:
- Backend returns `HTTP 503 (Service Unavailable)`
- Frontend clearly reports:
  > *"AI models are not installed yet. The application is connected to the backend, but dental analysis is not available until the trained models are added."*

### Adding Trained Models
Place trained model weights in:
1. `backend/models/condition/best.pt` — for intraoral condition detection.
2. `backend/models/alignment/best.pt` — for alignment analysis.

Once both weights and validated inference adapters in `app/inference.py` are configured, `GET /api/v1/model-status` will report `screening_ready: true`.

---

## Verification & Scripts

- `npm run dev` — Starts local dev server on port 8080.
- `npx tsc --noEmit` — Type-checks TypeScript code.
- `npm test` — Runs Vitest test suite.
- `npm run build` — Compiles production client and server bundles.
