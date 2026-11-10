# Scanio AI — local Python backend starter

This is a **backend scaffold**, not a trained dental AI model. It provides the API boundary,
image validation, response contract, CORS configuration and explicit "model not ready" behavior.
It intentionally refuses to fabricate detections.

## Requirements
- Python 3.10–3.12 recommended
- Node.js version compatible with the exported frontend
- A trained and validated model is still required for real predictions

## Run on Windows (PowerShell)
```powershell
cd scanio-ai-python-backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
If PowerShell blocks activation, use Command Prompt:
```cmd
.venv\Scripts\activate.bat
```

Check:
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/api/v1/model-status
- http://127.0.0.1:8000/docs

## Connect the existing frontend
In the root of the React project, create/update `.env.local`:
```env
VITE_PREDICTION_API_URL=http://127.0.0.1:8000/api/v1/predict
```
Restart the frontend dev server after changing environment variables.
The existing `src/services/prediction.ts` already sends multipart fields named `upper`, `lower`, and `front`,
so this endpoint matches that contract.

Important: `VITE_*` variables are public frontend configuration. Put only the backend URL there,
never API keys or secrets.

## Current expected behavior
- `/health` works.
- `/api/v1/model-status` reports which model files are missing.
- `/api/v1/predict` validates all three images.
- Until both trained models and validated inference adapters are added, `/predict` returns HTTP 503.
  This is deliberate: no mock findings, fabricated boxes or invented oral-health scores.

## Model plan
### A. Visible condition screening
Start with a narrow, defensible scope: visible caries-like findings in standardized intraoral photographs.
A 2025 Scientific Data paper describes 6,313 annotated intraoral images for caries detection,
with dentist-verified annotations and YOLO/COCO/PASCAL formats. Check the dataset's current access
and license terms before use; do not assume it permits commercial use.
Source: https://www.nature.com/articles/s41597-025-05647-9
Dataset record: https://zenodo.org/records/14827784

For plaque, calculus/tartar and gingival findings, obtain additional appropriately licensed images
with dentist-reviewed labels. A dataset's class names and annotation quality must be checked before
training.

### B. Alignment screening
Do not infer alignment from the caries dataset. Alignment requires its own labeled data:
tooth instance masks or landmarks, consistent frontal/occlusal views, and expert-defined measurements.
Build this as a separate model and report descriptive visual observations only until clinically validated.

### C. Evaluation
- Split by patient, never randomly by image if several views belong to one patient.
- Keep a locked test set untouched until final evaluation.
- Report per-class precision, recall, F1, AP/mAP and confusion matrices for detection.
- Evaluate landmark/segmentation quality and measurement agreement for alignment.
- Validate on external devices, lighting, clinics and patient populations.
- Calibrate confidence; a raw model score is not a disease probability.
- Require dentist review and a documented intended-use / safety review before any clinical deployment.

## Before deployment
Add authentication, rate limiting, secure storage, retention/deletion controls, audit logs and encryption.
Do not store real patient images in browser localStorage. Obtain appropriate consent and approvals.
