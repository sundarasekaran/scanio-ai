import json
import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app
from app.vision_client import (
    OpenRouterAuthError,
    OpenRouterMalformedResponseError,
    OpenRouterModelUnavailableError,
    OpenRouterPaymentError,
    OpenRouterPermissionError,
    OpenRouterProviderError,
    OpenRouterRateLimitError,
    OpenRouterTimeoutError,
    OpenRouterUnsupportedImageError,
    _extract_openrouter_text,
    _parse_json_response,
    analyze_with_openrouter,
    compute_tooth_enamel_mask,
    detect_image_mime_type,
    is_in_lower_tongue_exclusion_zone,
    is_in_palatal_exclusion_zone,
    is_valid_tooth_box,
    load_backend_env,
    normalize_image_orientation,
    sanitize_and_normalize_finding,
    sanitize_secret,
    verify_openrouter_model_availability,
)


def _make_image_bytes(fmt: str = "JPEG") -> bytes:
    buf = BytesIO()
    Image.new("RGB", (120, 120), color="white").save(buf, format=fmt)
    return buf.getvalue()


class BackendTestCase(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.dummy_img = _make_image_bytes("JPEG")

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("Scanio", data["service"])

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-v1-test-key"})
    @patch("app.inference.verify_openrouter_model_availability")
    def test_model_status_endpoint_openrouter(self, mock_verify):
        mock_verify.return_value = {
            "valid": True,
            "status": "available",
            "displayName": "Google: Gemini 2.5 Flash",
            "visionSupported": True,
            "structuredOutputSupported": True,
        }
        response = self.client.get("/api/v1/model-status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["vision_provider"], "openrouter")
        self.assertEqual(data["selected_model"], "google/gemini-2.5-flash")
        self.assertTrue(data["api_key_configured"])
        self.assertTrue(data["screening_ready"])
        self.assertEqual(data["model_availability"], "available")
        self.assertIn("OpenRouter", data["message"])
        self.assertNotIn("ollama_base_url", data)
        self.assertNotIn("ollama_running", data)
        self.assertNotIn("gemini_api_key", data)

    @patch("app.vision_client._check_openrouter_key")
    @patch("app.vision_client._fetch_openrouter_model_entry")
    def test_verify_openrouter_catalog_and_vision_support(self, mock_fetch, mock_key):
        mock_key.return_value = (True, None, None)

        # Valid vision model
        mock_fetch.return_value = {
            "id": "google/gemini-2.5-flash",
            "name": "Google: Gemini 2.5 Flash",
            "architecture": {"input_modalities": ["text", "image"]},
            "supported_parameters": ["response_format", "structured_outputs", "reasoning"],
        }
        ok = verify_openrouter_model_availability("sk-or-v1-test", "google/gemini-2.5-flash")
        self.assertTrue(ok["valid"])
        self.assertTrue(ok["visionSupported"])
        self.assertTrue(ok["structuredOutputSupported"])

        # Text-only model without image modality
        mock_fetch.return_value = {
            "id": "meta-llama/text-only",
            "architecture": {"input_modalities": ["text"]},
            "supported_parameters": [],
        }
        no_vis = verify_openrouter_model_availability("sk-or-v1-test", "meta-llama/text-only")
        self.assertFalse(no_vis["valid"])
        self.assertEqual(no_vis["status"], "unsupported_vision")

        # Missing model
        mock_fetch.return_value = None
        missing = verify_openrouter_model_availability("sk-or-v1-test", "missing/model")
        self.assertFalse(missing["valid"])
        self.assertEqual(missing["status"], "model_not_found")

    @patch("app.vision_client.urllib.request.urlopen")
    def test_analyze_with_openrouter_multimodal_payload(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "viewStatus": "assessed",
                                    "inconclusiveReason": None,
                                    "visibleToothSummary": "Upper arch visible.",
                                    "toothConditions": [
                                        {
                                            "toothNumber": "16",
                                            "condition": "Enamel Discoloration",
                                            "affectedRegion": "occlusal",
                                            "confidence": 0.85,
                                            "severity": "low",
                                            "description": "Mild fissure stain.",
                                            "recommendation": "Routine hygiene.",
                                            "box_2d": [520, 750, 660, 880],
                                        }
                                    ],
                                    "dentalAlignment": [
                                        {
                                            "findingType": "diastema",
                                            "name": "Midline Diastema",
                                            "involvedTeeth": ["11", "21"],
                                            "gapMeasurement": "visible spacing between upper central incisors",
                                            "confidence": 0.92,
                                            "severity": "moderate",
                                            "description": "Visible central gap.",
                                            "recommendation": "Orthodontic check.",
                                            "box_2d": [130, 480, 260, 520],
                                        }
                                    ],
                                }
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 80},
            }
        ).encode("utf-8")
        mock_urlopen.return_value = mock_resp

        png_bytes = _make_image_bytes("PNG")
        res = analyze_with_openrouter(
            png_bytes,
            "upper",
            "sk-or-v1-secret-test-key",
            "google/gemini-2.5-flash",
            "https://openrouter.ai/api/v1",
        )
        self.assertEqual(res["viewStatus"], "assessed")
        self.assertEqual(len(res["toothConditions"]), 1)
        self.assertEqual(len(res["dentalAlignment"]), 1)

        sent_req = mock_urlopen.call_args.args[0]
        self.assertEqual(sent_req.full_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(sent_req.get_header("Authorization"), "Bearer sk-or-v1-secret-test-key")
        sent_body = json.loads(sent_req.data.decode("utf-8"))
        self.assertEqual(sent_body["model"], "google/gemini-2.5-flash")
        self.assertNotIn("inlineData", json.dumps(sent_body))
        self.assertNotIn("generationConfig", sent_body)
        user_content = sent_body["messages"][1]["content"]
        self.assertEqual(user_content[1]["type"], "image_url")
        self.assertTrue(user_content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_valid_tooth_box(self):
        self.assertTrue(is_valid_tooth_box({"x": 0.4, "y": 0.3, "w": 0.12, "h": 0.14}))
        self.assertFalse(is_valid_tooth_box({"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.7}))
        self.assertFalse(is_valid_tooth_box({"x": -0.1, "y": 0.2, "w": 0.1, "h": 0.1}))
        self.assertFalse(is_valid_tooth_box({"x": 0.2, "y": 0.2, "w": 0.005, "h": 0.005}))

    def test_rejects_invalid_coordinates_instead_of_clamping(self):
        self.assertIsNone(sanitize_and_normalize_finding({"condition": "Caries"}, "upper"))
        self.assertIsNone(
            sanitize_and_normalize_finding(
                {"condition": "Caries", "boundingBox": {"x": -0.2, "y": 0.2, "width": 0.1, "height": 0.1}},
                "front",
            )
        )
        self.assertIsNone(
            sanitize_and_normalize_finding(
                {"condition": "Caries", "boundingBox": {"x": 0.92, "y": 0.2, "width": 0.2, "height": 0.1}},
                "front",
            )
        )
        self.assertIsNone(
            sanitize_and_normalize_finding(
                {"condition": "Caries", "boundingBox": {"x": float("nan"), "y": 0.2, "width": 0.1, "height": 0.1}},
                "front",
            )
        )

        valid_b2d = {
            "condition": "Enamel Discoloration",
            "toothNumber": "11",
            "confidence": 0.88,
            "severity": "low",
            "description": "Mild surface stain.",
            "recommendation": "Dental cleaning.",
            "box_2d": [200, 380, 560, 500],
        }
        norm_b2d = sanitize_and_normalize_finding(valid_b2d, "front", category="condition")
        self.assertIsNotNone(norm_b2d)
        self.assertEqual(norm_b2d["rawBox2d"], [200, 380, 560, 500])
        self.assertEqual(norm_b2d["box"], {"x": 0.38, "y": 0.2, "w": 0.12, "h": 0.36})

        self.assertIsNone(sanitize_and_normalize_finding({"condition": "Caries", "box_2d": [500, 200, 400, 300]}, "front"))
        self.assertIsNone(sanitize_and_normalize_finding({"condition": "Caries", "box_2d": [100, 200, 1050, 300]}, "front"))

        soft_tissue = {
            "condition": "Palatal Rugae Ridges",
            "affectedRegion": "hard palate",
            "boundingBox": {"x": 0.2, "y": 0.2, "width": 0.1, "height": 0.1},
        }
        self.assertIsNone(sanitize_and_normalize_finding(soft_tissue, "upper"))

    def test_enamel_mask_rejects_mucosa_boxes_and_accepts_teeth(self):
        im = Image.new("RGB", (200, 200), color=(190, 110, 110))
        for y in range(20, 60):
            for x in range(40, 160):
                im.putpixel((x, y), (240, 238, 225))
        buf = BytesIO()
        im.save(buf, format="JPEG")
        mask = compute_tooth_enamel_mask(buf.getvalue())
        self.assertIsNotNone(mask)

        tooth_finding = {
            "condition": "Enamel Discoloration",
            "box_2d": [100, 250, 300, 500],
            "confidence": 0.85,
            "severity": "low",
        }
        self.assertIsNotNone(
            sanitize_and_normalize_finding(tooth_finding, "upper", category="condition", enamel_mask=mask)
        )

        mucosa_finding = {
            "condition": "Enamel Discoloration",
            "box_2d": [600, 100, 850, 300],
            "confidence": 0.85,
            "severity": "low",
        }
        self.assertIsNone(
            sanitize_and_normalize_finding(mucosa_finding, "lower", category="condition", enamel_mask=mask)
        )

        palate_grabbing_align = {
            "findingType": "uneven_positioning",
            "name": "Right Posterior Uneven Positioning",
            "involvedTeeth": ["14", "15", "16"],
            "box_2d": [250, 550, 650, 850],
            "confidence": 0.75,
            "severity": "low",
        }
        self.assertIsNone(
            sanitize_and_normalize_finding(palate_grabbing_align, "upper", category="alignment", enamel_mask=mask)
        )

        self.assertTrue(is_in_lower_tongue_exclusion_zone({"x": 0.4, "y": 0.35, "w": 0.2, "h": 0.2}, "lower"))
        self.assertFalse(is_in_lower_tongue_exclusion_zone({"x": 0.4, "y": 0.78, "w": 0.2, "h": 0.12}, "lower"))

    def test_exif_orientation_normalization(self):
        im = Image.new("RGB", (120, 60), color="white")
        exif = im.getexif()
        exif[274] = 6
        buf = BytesIO()
        im.save(buf, format="JPEG", exif=exif)
        transposed = normalize_image_orientation(buf.getvalue(), "image/jpeg")
        with Image.open(BytesIO(transposed)) as out_im:
            self.assertEqual(out_im.size, (60, 120))

    def test_palatal_exclusion_zone(self):
        palate_box = {"x": 0.45, "y": 0.45, "w": 0.10, "h": 0.10}
        self.assertTrue(is_in_palatal_exclusion_zone(palate_box, "upper"))
        self.assertFalse(is_in_palatal_exclusion_zone(palate_box, "lower"))

        anterior_box = {"x": 0.46, "y": 0.14, "w": 0.08, "h": 0.10}
        self.assertFalse(is_in_palatal_exclusion_zone(anterior_box, "upper"))

        palate_finding = {"condition": "Suspected Caries", "boundingBox": palate_box}
        self.assertIsNone(sanitize_and_normalize_finding(palate_finding, "upper", category="condition"))

    def test_mime_detection_and_unsupported_upload(self):
        self.assertEqual(detect_image_mime_type(_make_image_bytes("JPEG")), "image/jpeg")
        self.assertEqual(detect_image_mime_type(_make_image_bytes("PNG")), "image/png")
        self.assertEqual(detect_image_mime_type(_make_image_bytes("WEBP")), "image/webp")

        files = {
            "upper": ("upper.txt", b"not an image", "text/plain"),
            "lower": ("lower.jpg", self.dummy_img, "image/jpeg"),
            "front": ("front.jpg", self.dummy_img, "image/jpeg"),
        }
        resp = self.client.post("/api/v1/predict", files=files)
        self.assertEqual(resp.status_code, 415)

    def test_extracts_openrouter_content_blocks_and_wrapped_json(self):
        response_payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": [
                            {
                                "type": "text",
                                "text": 'Here is the analysis:\n```json\n{"viewStatus": "assessed", "toothConditions": [], "dentalAlignment": []}\n```',
                            }
                        ]
                    },
                }
            ]
        }
        text = _extract_openrouter_text(response_payload, "front")
        parsed = _parse_json_response(text, "front")
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["viewStatus"], "assessed")

    def test_secret_sanitization(self):
        fake_key = "sk-or-v1-1234567890abcdef1234567890abcdef"
        msg = f"Authorization: Bearer {fake_key} failed for {fake_key}"
        cleaned = sanitize_secret(msg, fake_key)
        self.assertNotIn(fake_key, cleaned)

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-v1-test-key"})
    def test_env_loader(self):
        cfg = load_backend_env()
        self.assertEqual(cfg["provider"], "openrouter")
        self.assertEqual(cfg["model"], "google/gemini-2.5-flash")
        self.assertEqual(cfg["base_url"], "https://openrouter.ai/api/v1")
        self.assertTrue(cfg["has_key"])

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-v1-test-key"})
    @patch("app.inference.verify_openrouter_model_availability")
    @patch("app.inference.analyze_with_openrouter")
    def test_predict_endpoint_success(self, mock_analyze, mock_verify):
        mock_verify.return_value = {
            "valid": True,
            "status": "available",
            "displayName": "Google: Gemini 2.5 Flash",
            "visionSupported": True,
            "structuredOutputSupported": True,
        }
        mock_analyze.return_value = {
            "viewStatus": "assessed",
            "toothConditions": [
                {
                    "name": "Superficial Staining",
                    "category": "condition",
                    "condition": "Superficial Staining",
                    "confidence": 0.84,
                    "severity": "low",
                    "description": "Minor surface discoloration.",
                    "recommendation": "Clinical polish.",
                    "box": {"x": 0.4, "y": 0.2, "w": 0.1, "h": 0.1},
                    "tooth": "11",
                    "toothNumber": "11",
                    "involvedTeeth": [],
                }
            ],
            "dentalAlignment": [
                {
                    "name": "Midline Diastema (Gap)",
                    "category": "alignment",
                    "condition": "Midline Diastema (Gap)",
                    "findingType": "diastema",
                    "involvedTeeth": ["11", "21"],
                    "gapMeasurement": "visible spacing between central incisors",
                    "confidence": 0.95,
                    "severity": "low",
                    "description": "Visible interdental spacing.",
                    "recommendation": "Orthodontic consultation.",
                    "box": {"x": 0.46, "y": 0.13, "w": 0.08, "h": 0.10},
                    "tooth": "11, 21",
                    "toothNumber": "11, 21",
                }
            ],
            "all": [
                {
                    "name": "Superficial Staining",
                    "category": "condition",
                    "condition": "Superficial Staining",
                    "confidence": 0.84,
                    "severity": "low",
                    "description": "Minor surface discoloration.",
                    "recommendation": "Clinical polish.",
                    "box": {"x": 0.4, "y": 0.2, "w": 0.1, "h": 0.1},
                    "tooth": "11",
                    "toothNumber": "11",
                    "involvedTeeth": [],
                },
                {
                    "name": "Midline Diastema (Gap)",
                    "category": "alignment",
                    "condition": "Midline Diastema (Gap)",
                    "findingType": "diastema",
                    "involvedTeeth": ["11", "21"],
                    "gapMeasurement": "visible spacing between central incisors",
                    "confidence": 0.95,
                    "severity": "low",
                    "description": "Visible interdental spacing.",
                    "recommendation": "Orthodontic consultation.",
                    "box": {"x": 0.46, "y": 0.13, "w": 0.08, "h": 0.10},
                    "tooth": "11, 21",
                    "toothNumber": "11, 21",
                },
            ],
        }

        files = {
            "upper": ("upper.jpg", self.dummy_img, "image/jpeg"),
            "lower": ("lower.jpg", self.dummy_img, "image/jpeg"),
            "front": ("front.jpg", self.dummy_img, "image/jpeg"),
        }
        response = self.client.post("/api/v1/predict", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["modelProvider"], "openrouter (google/gemini-2.5-flash)")
        self.assertFalse(data["isMock"])
        self.assertFalse(data["inconclusive"])
        self.assertEqual(len(data["views"]["upper"]), 2)
        self.assertIn("modules", data)
        self.assertEqual(len(data["modules"]["toothConditions"]["upper"]), 1)
        self.assertEqual(len(data["modules"]["dentalAlignment"]["upper"]), 1)
        self.assertIn("OpenRouter", data["screeningDisclaimer"])

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-v1-test-key"})
    @patch("app.inference.verify_openrouter_model_availability")
    @patch("app.inference.analyze_with_openrouter")
    def test_predict_endpoint_inconclusive_all_views(self, mock_analyze, mock_verify):
        mock_verify.return_value = {"valid": True, "status": "available"}
        mock_analyze.return_value = {
            "viewStatus": "inconclusive",
            "inconclusiveReason": "Photo is blurry and teeth are not clearly visible.",
            "toothConditions": [],
            "dentalAlignment": [],
            "all": [],
            "rejectedCount": 0,
        }
        files = {
            "upper": ("upper.png", _make_image_bytes("PNG"), "image/png"),
            "lower": ("lower.png", _make_image_bytes("PNG"), "image/png"),
            "front": ("front.png", _make_image_bytes("PNG"), "image/png"),
        }
        response = self.client.post("/api/v1/predict", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["inconclusive"])
        self.assertEqual(data["overallScore"], 0.0)
        self.assertEqual(data["inconclusiveViews"], ["upper", "lower", "front"])

    @patch("app.main.run_screening")
    def test_predict_error_status_mapping(self, mock_run):
        files = {
            "upper": ("upper.jpg", self.dummy_img, "image/jpeg"),
            "lower": ("lower.jpg", self.dummy_img, "image/jpeg"),
            "front": ("front.jpg", self.dummy_img, "image/jpeg"),
        }
        cases = [
            (OpenRouterAuthError("Invalid OpenRouter API key"), 401),
            (OpenRouterPaymentError("Insufficient credits"), 402),
            (OpenRouterPermissionError("Permission restricted"), 403),
            (OpenRouterModelUnavailableError("Model unavailable"), 404),
            (OpenRouterRateLimitError("Rate limit exceeded"), 429),
            (OpenRouterUnsupportedImageError("Unsupported image"), 415),
            (OpenRouterTimeoutError("Timed out"), 504),
            (OpenRouterMalformedResponseError("Bad JSON"), 502),
            (OpenRouterProviderError("Upstream 5xx"), 502),
        ]
        for exc, expected_code in cases:
            mock_run.side_effect = exc
            resp = self.client.post("/api/v1/predict", files=files)
            self.assertEqual(resp.status_code, expected_code)
            self.assertIn("detail", resp.json())

    def test_cors_allows_vercel_origins(self):
        resp = self.client.options(
            "/api/v1/predict",
            headers={
                "Origin": "https://scanio-ai-preview.vercel.app",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            resp.headers.get("access-control-allow-origin"),
            "https://scanio-ai-preview.vercel.app",
        )


if __name__ == "__main__":
    unittest.main()
