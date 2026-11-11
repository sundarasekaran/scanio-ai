"""
Offline validation and benchmarking module for Scanio AI using dentist-labelled
ground-truth intraoral dental datasets.

Computes metrics separately per view (`upper`, `lower`, `front`) and overall:
- Tooth localization precision, recall, F1, mean IoU, and AP/mAP@IoU.
- Tooth-surface condition classification precision, recall, and F1.
- Dental alignment detection precision, recall, and F1.
- Enforces strict train/dev/test split separation to avoid prompt leakage.
"""
from dataclasses import dataclass


def compute_iou(box_a: dict, box_b: dict) -> float:
    """Computes Intersection over Union between two normalized {x, y, w, h} boxes."""
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


def _normalize_label(value: str | None) -> str:
    return (value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _safe_prf1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_average_precision(recalls: list[float], precisions: list[float]) -> float:
    """Computes VOC-style monotonic precision-recall envelope Average Precision (AP)."""
    if not recalls or not precisions:
        return 0.0
    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for i in range(len(mpre) - 2, -1, -1):
        if mpre[i] < mpre[i + 1]:
            mpre[i] = mpre[i + 1]
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return round(ap, 4)


@dataclass
class _Bucket:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ious: list[float] | None = None
    scored_matches: list[tuple[float, bool]] | None = None
    total_gt: int = 0

    def __post_init__(self):
        if self.ious is None:
            self.ious = []
        if self.scored_matches is None:
            self.scored_matches = []

    def summary(self, include_map: bool = False) -> dict:
        out = _safe_prf1(self.tp, self.fp, self.fn)
        if include_map:
            mean_iou = sum(self.ious) / len(self.ious) if self.ious else 0.0
            out["meanIoU"] = round(mean_iou, 4)
            ranked = sorted(self.scored_matches, key=lambda x: x[0], reverse=True)
            recalls: list[float] = []
            precisions: list[float] = []
            running_tp = 0
            for idx, (_, hit) in enumerate(ranked, start=1):
                if hit:
                    running_tp += 1
                precisions.append(running_tp / idx)
                recalls.append(running_tp / self.total_gt if self.total_gt > 0 else 0.0)
            out["ap"] = compute_average_precision(recalls, precisions)
        return out


def _match_predictions(
    preds: list[dict],
    gts: list[dict],
    iou_threshold: float,
    require_label_match: bool,
    label_key: str,
) -> tuple[int, int, int, list[float], list[tuple[float, bool]]]:
    ranked = sorted(preds, key=lambda p: float(p.get("confidence", 0.0)), reverse=True)
    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    ious: list[float] = []
    scored_matches: list[tuple[float, bool]] = []

    for pred in ranked:
        pred_box = pred["box"]
        pred_label = _normalize_label(pred.get(label_key) or pred.get("name"))
        best_idx = -1
        best_iou = 0.0

        for idx, gt in enumerate(gts):
            if idx in matched_gt:
                continue
            if require_label_match:
                gt_label = _normalize_label(gt.get(label_key) or gt.get("name"))
                if pred_label != gt_label:
                    continue
            iou = compute_iou(pred_box, gt["box"])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_idx = idx

        conf = float(pred.get("confidence", 0.0))
        if best_idx >= 0:
            matched_gt.add(best_idx)
            tp += 1
            ious.append(best_iou)
            scored_matches.append((conf, True))
        else:
            fp += 1
            scored_matches.append((conf, False))

    fn = len(gts) - len(matched_gt)
    return tp, fp, fn, ious, scored_matches


def evaluate_dataset(
    predictions_by_id: dict[str, list[dict]],
    ground_truth_manifest: list[dict],
    allowed_splits: tuple[str, ...] = ("test",),
    iou_threshold: float = 0.5,
) -> dict:
    """
    Evaluates model predictions against a dentist-labelled ground-truth manifest.

    Each manifest entry requires:
      - "id": sample identifier matching predictions_by_id
      - "view": "upper" | "lower" | "front"
      - "split": "test" (samples outside allowed_splits raise ValueError)
      - "groundTruth": [{"category": "condition"|"alignment", "label": str, "box": {x, y, w, h}, ...}]
    """
    if not ground_truth_manifest:
        raise ValueError("Ground-truth manifest is empty. Provide dentist-labelled reference samples.")

    allowed = {s.strip().lower() for s in allowed_splits}
    views = ("upper", "lower", "front")

    loc_buckets = {v: _Bucket() for v in (*views, "overall")}
    cond_buckets = {v: _Bucket() for v in (*views, "overall")}
    align_buckets = {v: _Bucket() for v in (*views, "overall")}
    evaluated_counts = {v: 0 for v in views}

    for entry in ground_truth_manifest:
        split = str(entry.get("split") or "").strip().lower()
        if split not in allowed:
            raise ValueError(
                f"Split isolation violation for sample '{entry.get('id')}': "
                f"split='{split}' is not allowed during evaluation (allowed={sorted(allowed)})."
            )

        sample_id = str(entry["id"])
        view = str(entry.get("view") or "").strip().lower()
        if view not in views:
            raise ValueError(f"Unsupported view '{view}' in manifest for sample '{sample_id}'.")

        evaluated_counts[view] += 1
        gt_items = list(entry.get("groundTruth") or [])
        pred_items = list(predictions_by_id.get(sample_id) or [])

        # 1. Localization across all annotated teeth/regions in the image
        ltp, lfp, lfn, lious, lscored = _match_predictions(
            pred_items, gt_items, iou_threshold=iou_threshold, require_label_match=False, label_key="label"
        )
        for key in (view, "overall"):
            b = loc_buckets[key]
            b.tp += ltp
            b.fp += lfp
            b.fn += lfn
            b.total_gt += len(gt_items)
            b.ious.extend(lious)
            b.scored_matches.extend(lscored)

        # 2. Condition classification
        gt_cond = [g for g in gt_items if g.get("category", "condition") != "alignment"]
        pred_cond = [p for p in pred_items if p.get("category", "condition") != "alignment"]
        ctp, cfp, cfn, cious, cscored = _match_predictions(
            pred_cond, gt_cond, iou_threshold=iou_threshold, require_label_match=True, label_key="condition"
        )
        for key in (view, "overall"):
            b = cond_buckets[key]
            b.tp += ctp
            b.fp += cfp
            b.fn += cfn
            b.total_gt += len(gt_cond)
            b.ious.extend(cious)
            b.scored_matches.extend(cscored)

        # 3. Alignment detection
        gt_align = [g for g in gt_items if g.get("category") == "alignment"]
        pred_align = [p for p in pred_items if p.get("category") == "alignment"]
        atp, afp, afn, aious, ascored = _match_predictions(
            pred_align, gt_align, iou_threshold=iou_threshold, require_label_match=True, label_key="findingType"
        )
        for key in (view, "overall"):
            b = align_buckets[key]
            b.tp += atp
            b.fp += afp
            b.fn += afn
            b.total_gt += len(gt_align)
            b.ious.extend(aious)
            b.scored_matches.extend(ascored)

    per_view = {}
    for view in views:
        loc_sum = loc_buckets[view].summary(include_map=True)
        cond_sum = cond_buckets[view].summary(include_map=True)
        align_sum = align_buckets[view].summary(include_map=True)
        per_view[view] = {
            "sampleCount": evaluated_counts[view],
            "localization": loc_sum,
            "conditionClassification": cond_sum,
            "alignmentDetection": align_sum,
        }

    overall_cond = cond_buckets["overall"].summary(include_map=True)
    overall_align = align_buckets["overall"].summary(include_map=True)
    map_terms = [m["ap"] for m in (overall_cond, overall_align) if m["tp"] + m["fn"] > 0]
    overall_map = round(sum(map_terms) / len(map_terms), 4) if map_terms else 0.0

    return {
        "iouThreshold": iou_threshold,
        "allowedSplits": sorted(allowed),
        "totalSamples": sum(evaluated_counts.values()),
        "perView": per_view,
        "overall": {
            "localization": loc_buckets["overall"].summary(include_map=True),
            "conditionClassification": overall_cond,
            "alignmentDetection": overall_align,
            "mAP": overall_map,
        },
    }
