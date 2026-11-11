import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation import compute_iou, evaluate_dataset


class EvaluationTests(unittest.TestCase):
    def test_compute_iou_exact_and_partial(self):
        box = {"x": 0.1, "y": 0.2, "w": 0.2, "h": 0.2}
        self.assertAlmostEqual(compute_iou(box, box), 1.0)
        disjoint = {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}
        self.assertEqual(compute_iou(box, disjoint), 0.0)

    def test_evaluate_dataset_per_view_metrics(self):
        manifest = [
            {
                "id": "u1",
                "view": "upper",
                "split": "test",
                "groundTruth": [
                    {
                        "category": "condition",
                        "condition": "enamel_discoloration",
                        "box": {"x": 0.70, "y": 0.45, "w": 0.09, "h": 0.10},
                    },
                    {
                        "category": "alignment",
                        "findingType": "diastema",
                        "box": {"x": 0.47, "y": 0.15, "w": 0.06, "h": 0.10},
                    },
                ],
            },
            {
                "id": "l1",
                "view": "lower",
                "split": "test",
                "groundTruth": [
                    {
                        "category": "alignment",
                        "findingType": "crowding",
                        "box": {"x": 0.36, "y": 0.74, "w": 0.26, "h": 0.12},
                    }
                ],
            },
            {
                "id": "f1",
                "view": "front",
                "split": "test",
                "groundTruth": [
                    {
                        "category": "condition",
                        "condition": "caries",
                        "box": {"x": 0.40, "y": 0.30, "w": 0.08, "h": 0.14},
                    }
                ],
            },
        ]
        preds = {
            "u1": [
                {
                    "category": "condition",
                    "condition": "enamel_discoloration",
                    "confidence": 0.9,
                    "box": {"x": 0.70, "y": 0.45, "w": 0.09, "h": 0.10},
                },
                {
                    "category": "alignment",
                    "findingType": "diastema",
                    "confidence": 0.95,
                    "box": {"x": 0.47, "y": 0.15, "w": 0.06, "h": 0.10},
                },
            ],
            "l1": [
                {
                    "category": "alignment",
                    "findingType": "crowding",
                    "confidence": 0.85,
                    "box": {"x": 0.36, "y": 0.74, "w": 0.26, "h": 0.12},
                }
            ],
            "f1": [],
        }

        report = evaluate_dataset(preds, manifest, allowed_splits=("test",), iou_threshold=0.5)
        self.assertEqual(report["totalSamples"], 3)
        self.assertEqual(report["perView"]["upper"]["localization"]["f1"], 1.0)
        self.assertEqual(report["perView"]["lower"]["alignmentDetection"]["recall"], 1.0)
        self.assertEqual(report["perView"]["front"]["conditionClassification"]["fn"], 1)
        self.assertEqual(report["perView"]["front"]["conditionClassification"]["recall"], 0.0)
        self.assertGreater(report["overall"]["mAP"], 0.5)

    def test_enforces_test_split_separation(self):
        manifest = [
            {
                "id": "dev_sample",
                "view": "lower",
                "split": "dev",
                "groundTruth": [],
            }
        ]
        with self.assertRaises(ValueError):
            evaluate_dataset({}, manifest, allowed_splits=("test",))


if __name__ == "__main__":
    unittest.main()
