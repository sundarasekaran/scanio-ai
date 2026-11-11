# Clinical Validation Protocol (Dentist-Labelled Reference Dataset)

To avoid unverified accuracy claims, benchmark evaluation uses [`app/evaluation.py`](../app/evaluation.py) with a strictly isolated **`test`** dataset separated from prompt development (`dev`/`train`).

## Metrics Computed Per View (`upper`, `lower`, `front`) & Overall
1. **Tooth Localization:** Precision, Recall, F1, mean IoU, and AP@IoU.
2. **Tooth-Surface Condition Classification:** Precision, Recall, F1, and AP@IoU requiring both label match and `IoU >= 0.5`.
3. **Dental Alignment Detection:** Precision, Recall, F1, and AP@IoU for diastema, crowding, rotation, and displacement.

## Split Isolation
`evaluate_dataset(..., allowed_splits=("test",))` rejects any manifest containing `"train"` or `"dev"` records so prompt-tuning images never contaminate test benchmarks.

See [`sample_manifest.template.json`](./sample_manifest.template.json) for the dentist-annotation JSON format.
