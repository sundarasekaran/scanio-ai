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
    analyze_with_nvidia,
    analyze_with_openrouter,
    assess_image_quality,
    compute_tooth_enamel_mask,
    detect_image_mime_type,
    generate_arch_crops,
    is_in_lower_tongue_exclusion_zone,
    is_in_palatal_exclusion_zone,
    is_nvidia_vision_model,
    is_valid_tooth_box,
    load_backend_env,
    map_crop_box_to_original,
    normalize_image_orientation,
    prepare_image_for_provider,
    sanitize_and_normalize_finding,
    sanitize_secret,
    verify_nvidia_model_availability,
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

    @patch.dict("os.environ", {"VISION_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "sk-or-v1-test-key"})
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
        fake_or = "sk-or-v1-1234567890abcdef1234567890abcdef"
        fake_nv = "nvapi-1234567890abcdef1234567890abcdef"
        msg = f"Authorization: Bearer {fake_nv} failed alongside {fake_or}"
        cleaned = sanitize_secret(msg, fake_nv)
        self.assertNotIn(fake_nv, cleaned)
        self.assertNotIn(fake_or, cleaned)

    @patch.dict(
        "os.environ",
        {"VISION_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "sk-or-v1-test-key"},
    )
    def test_env_loader(self):
        cfg = load_backend_env()
        self.assertEqual(cfg["provider"], "openrouter")
        self.assertEqual(cfg["model"], "google/gemini-2.5-flash")
        self.assertEqual(cfg["base_url"], "https://openrouter.ai/api/v1")
        self.assertTrue(cfg["has_key"])

    @patch.dict(
        "os.environ",
        {
            "VISION_PROVIDER": "nvidia",
            "NVIDIA_API_KEY": "nvapi-test-key",
            "NVIDIA_MODEL": "nvidia/nemotron-3.5-super-vl-120b-a12b",
            "NVIDIA_BASE_URL": "https://integrate.api.nvidia.com/v1",
        },
        clear=False,
    )
    def test_env_loader_nvidia_and_stale_provider_migration(self):
        cfg = load_backend_env()
        self.assertEqual(cfg["provider"], "nvidia")
        self.assertEqual(cfg["model"], "nvidia/nemotron-3.5-super-vl-120b-a12b")
        self.assertEqual(cfg["base_url"], "https://integrate.api.nvidia.com/v1")
        self.assertTrue(cfg["has_key"])

        with patch.dict(
            "os.environ",
            {
                "VISION_PROVIDER": "nvidia",
                "NVIDIA_API_KEY ": "Bearer nvapi-clean-isolated-key",
                "NVIDIA_MODEL": "nvidia/nemotron-3.5-super-vl-120b-a12b",
                "NVIDIA_BASE_URL": "https://integrate.api.nvidia.com/v1",
            },
            clear=True,
        ):
            isolated = load_backend_env()
            self.assertEqual(isolated["provider"], "nvidia")
            self.assertEqual(isolated["api_key"], "nvapi-clean-isolated-key")
            self.assertTrue(isolated["has_key"])
            self.assertFalse(isolated["openrouter_configured"])

        with patch.dict(
            "os.environ",
            {
                "VISION_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "nvapi-misplaced",
                "NVIDIA_API_KEY": "nvapi-real-env-key",
            },
        ):
            migrated = load_backend_env()
            self.assertEqual(migrated["provider"], "nvidia")
            self.assertEqual(migrated["api_key"], "nvapi-real-env-key")

    @patch.dict(
        "os.environ",
        {
            "VISION_PROVIDER": "nvidia",
            "NVIDIA_API_KEY": "nvapi-secret-unit-test-key",
            "NVIDIA_MODEL": "nvidia/nemotron-3.5-super-vl-120b-a12b",
            "NVIDIA_BASE_URL": "https://integrate.api.nvidia.com/v1",
        },
    )
    @patch("app.inference.verify_nvidia_model_availability")
    def test_model_status_endpoint_nvidia(self, mock_verify):
        mock_verify.return_value = {
            "valid": True,
            "status": "available",
            "displayName": "NVIDIA Nemotron 3.5 Super VL 120B",
            "visionSupported": True,
            "structuredOutputSupported": True,
        }
        resp = self.client.get("/api/v1/model-status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["vision_provider"], "nvidia")
        self.assertEqual(data["selected_model"], "nvidia/nemotron-3.5-super-vl-120b-a12b")
        self.assertEqual(data["base_url"], "https://integrate.api.nvidia.com/v1")
        self.assertTrue(data["api_key_configured"])
        self.assertTrue(data["screening_ready"])
        self.assertTrue(data["vision_supported"])
        self.assertNotIn("nvapi-secret-unit-test-key", json.dumps(data))

    @patch("app.vision_client._fetch_nvidia_catalog")
    def test_verify_nvidia_catalog_and_vision_support(self, mock_catalog):
        mock_catalog.return_value = [
            {"id": "nvidia/nemotron-3.5-super-vl-120b-a12b"},
            {"id": "google/gemma-4-31b-it"},
            {"id": "nvidia/llama-3.1-nemotron-70b-instruct"},
        ]
        self.assertTrue(is_nvidia_vision_model("nvidia/nemotron-3.5-super-vl-120b-a12b"))
        self.assertTrue(is_nvidia_vision_model("google/gemma-4-31b-it"))
        self.assertFalse(is_nvidia_vision_model("nvidia/llama-3.1-nemotron-70b-instruct"))

        ok = verify_nvidia_model_availability("nvapi-test", "nvidia/nemotron-3.5-super-vl-120b-a12b", run_auth_probe=False)
        self.assertTrue(ok["valid"])
        self.assertTrue(ok["visionSupported"])

        text_only = verify_nvidia_model_availability("nvapi-test", "nvidia/llama-3.1-nemotron-70b-instruct", run_auth_probe=False)
        self.assertFalse(text_only["valid"])
        self.assertEqual(text_only["status"], "unsupported_vision")

        missing = verify_nvidia_model_availability("nvapi-test", "nvidia/unknown-model", run_auth_probe=False)
        self.assertFalse(missing["valid"])
        self.assertEqual(missing["status"], "model_not_found")

    @patch("app.vision_client.urllib.request.urlopen")
    def test_analyze_with_nvidia_multimodal_payload_and_fallback(self, mock_urlopen):
        # First request simulates 400 on json_schema, second returns valid dental findings
        err_400 = __import__("urllib.error").error.HTTPError(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            400,
            "Bad Request",
            {},
            BytesIO(b'{"error":{"message":"json_schema not supported by runtime"}}'),
        )
        ok_resp = MagicMock()
        ok_resp.__enter__.return_value = ok_resp
        ok_resp.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "viewStatus": "assessed",
                                    "conditionStatus": "assessed_with_findings",
                                    "alignmentStatus": "assessed_no_findings",
                                    "visibleToothSummary": "Mandibular molars visible",
                                    "unassessableTeethReason": None,
                                    "toothConditions": [
                                        {
                                            "toothNumber": "36",
                                            "condition": "Visible Plaque",
                                            "affectedRegion": "occlusal",
                                            "confidence": 0.86,
                                            "severity": "low",
                                            "description": "Mild plaque on lower molar occlusal surface.",
                                            "recommendation": "Brush and floss carefully.",
                                            "box_2d": [500, 800, 620, 900],
                                        }
                                    ],
                                    "dentalAlignment": [],
                                }
                            )
                        },
                    }
                ]
            }
        ).encode("utf-8")
        mock_urlopen.side_effect = [err_400, ok_resp]

        webp_bytes = _make_image_bytes("WEBP")
        res = analyze_with_nvidia(
            webp_bytes,
            "lower",
            "nvapi-test-secret",
            "google/gemma-4-31b-it",
            "https://integrate.api.nvidia.com/v1",
            capabilities={"structuredOutputSupported": True},
            enable_crop_fallback=False,
        )
        self.assertEqual(res["viewStatus"], "assessed")
        self.assertEqual(len(res["toothConditions"]), 1)
        self.assertEqual(res["toothConditions"][0]["tooth"], "36")
        second_req = mock_urlopen.call_args_list[1].args[0]
        self.assertEqual(second_req.full_url, "https://integrate.api.nvidia.com/v1/chat/completions")
        self.assertEqual(second_req.get_header("Authorization"), "Bearer nvapi-test-secret")
        sent_body = json.loads(second_req.data.decode("utf-8"))
        self.assertEqual(sent_body["response_format"], {"type": "json_object"})
        self.assertEqual(sent_body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertTrue(sent_body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/webp;base64,"))

    def test_prepare_image_for_nvidia_size_limit(self):
        small = _make_image_bytes("PNG")
        out_small, mime_small = prepare_image_for_provider(small, "image/png", provider="nvidia")
        self.assertEqual(mime_small, "image/png")
        self.assertEqual(out_small, small)

        # Build noisy 1800x1400 PNG well over 145KB -> verifies clean aspect-preserving shrinkage
        im = Image.new("RGB", (1800, 1400))
        pixels = []
        for y in range(1400):
            for x in range(1800):
                pixels.append(((x * 37 + y * 13) % 255, (x * 17 + y * 53) % 255, (x * 7 + y * 29) % 255))
        im.putdata(pixels)
        big_buf = BytesIO()
        im.save(big_buf, format="PNG")
        raw_big = big_buf.getvalue()
        self.assertGreater(len(raw_big), 145_000)

        fitted, out_mime = prepare_image_for_provider(raw_big, "image/png", provider="nvidia")
        self.assertLessEqual(len(fitted), 145_000)
        with Image.open(BytesIO(fitted)) as out_im:
            orig_ratio = 1800 / 1400
            new_ratio = out_im.size[0] / out_im.size[1]
            self.assertAlmostEqual(orig_ratio, new_ratio, delta=0.03)
        self.assertIn(out_mime, ("image/png", "image/jpeg"))

    @patch.dict("os.environ", {"VISION_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "sk-or-v1-test-key"})
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

    @patch.dict("os.environ", {"VISION_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "sk-or-v1-test-key"})
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

    def test_assess_image_quality_metrics(self):
        # Sharp patterned 300x240 image -> good quality
        sharp_im = Image.new("RGB", (300, 240), color=(235, 230, 215))
        for y in range(20, 220, 8):
            for x in range(20, 280):
                sharp_im.putpixel((x, y), (90, 80, 75))
        buf_sharp = BytesIO()
        sharp_im.save(buf_sharp, format="JPEG", quality=95)
        q_sharp = assess_image_quality(buf_sharp.getvalue())
        self.assertEqual(q_sharp["width"], 300)
        self.assertEqual(q_sharp["height"], 240)
        self.assertEqual(q_sharp["qualityStatus"], "good")
        self.assertGreater(q_sharp["laplacianVariance"], 28.0)

        # Flat/blurry image -> very_blurry
        flat_buf = BytesIO()
        Image.new("RGB", (240, 240), color=(140, 140, 140)).save(flat_buf, format="PNG")
        q_flat = assess_image_quality(flat_buf.getvalue())
        self.assertEqual(q_flat["qualityStatus"], "very_blurry")

        # Tiny thumbnail -> too_small
        tiny_buf = BytesIO()
        Image.new("RGB", (40, 40), color="white").save(tiny_buf, format="PNG")
        q_tiny = assess_image_quality(tiny_buf.getvalue())
        self.assertEqual(q_tiny["qualityStatus"], "too_small")

    @patch("app.vision_client._send_openrouter_chat_request")
    def test_crop_coordinate_translation_and_lower_fallback_recovery(self, mock_chat):
        # Build 400x400 synthetic lower arch: pink interior mucosa + ivory tooth rim at bottom/sides
        im = Image.new("RGB", (400, 400), color=(190, 110, 110))
        for y in range(290, 360):
            for x in range(80, 320):
                im.putpixel((x, y), (242, 238, 224))
        for y in range(100, 280):
            for x in range(20, 95):
                im.putpixel((x, y), (242, 238, 224))
            for x in range(305, 380):
                im.putpixel((x, y), (242, 238, 224))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=95)
        lower_bytes = buf.getvalue()

        # Verify crop coordinate mapping math
        mapped_box, mapped_b2d = map_crop_box_to_original(
            {"x": 0.5, "y": 0.5, "w": 0.25, "h": 0.2},
            (0.14, 0.48, 0.72, 0.50),
        )
        self.assertAlmostEqual(mapped_box["x"], 0.50, places=3)
        self.assertAlmostEqual(mapped_box["y"], 0.73, places=3)
        self.assertAlmostEqual(mapped_box["w"], 0.18, places=3)
        self.assertAlmostEqual(mapped_box["h"], 0.10, places=3)
        self.assertEqual(len(mapped_b2d), 4)

        crops = generate_arch_crops(lower_bytes, "image/jpeg", "lower")
        self.assertEqual([c["region"] for c in crops], ["anterior", "left_posterior", "right_posterior"])

        def _wrap(obj):
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(obj)}}]}

        # Full-frame falsely reports inconclusive; anterior crop detects mandibular anterior crowding
        mock_chat.side_effect = [
            _wrap(
                {
                    "viewStatus": "inconclusive",
                    "conditionStatus": "inconclusive",
                    "alignmentStatus": "inconclusive",
                    "inconclusiveReason": "Slightly soft mobile photo",
                    "visibleToothSummary": "Lower incisors along bottom rim",
                    "unassessableTeethReason": "Third molars out of frame",
                    "toothConditions": [],
                    "dentalAlignment": [],
                }
            ),
            _wrap(
                {
                    "viewStatus": "assessed",
                    "conditionStatus": "assessed_no_findings",
                    "alignmentStatus": "assessed_with_findings",
                    "visibleToothSummary": "Lower incisors 32, 31, 41, 42",
                    "toothConditions": [],
                    "dentalAlignment": [
                        {
                            "findingType": "crowding",
                            "name": "Lower anterior crowding",
                            "involvedTeeth": ["31", "32", "41", "42"],
                            "gapMeasurement": "mild crowding across lower incisors",
                            "confidence": 0.88,
                            "severity": "moderate",
                            "description": "Mild crowding of lower anterior incisors.",
                            "recommendation": "Orthodontic consultation.",
                            "box_2d": [540, 320, 780, 680],
                        }
                    ],
                }
            ),
            _wrap(
                {
                    "viewStatus": "assessed",
                    "conditionStatus": "assessed_no_findings",
                    "alignmentStatus": "assessed_no_findings",
                    "toothConditions": [],
                    "dentalAlignment": [],
                }
            ),
            _wrap(
                {
                    "viewStatus": "assessed",
                    "conditionStatus": "assessed_no_findings",
                    "alignmentStatus": "assessed_no_findings",
                    "toothConditions": [],
                    "dentalAlignment": [],
                }
            ),
        ]

        res = analyze_with_openrouter(
            lower_bytes,
            "lower",
            "sk-or-v1-test",
            "google/gemini-2.5-flash",
            "https://openrouter.ai/api/v1",
            mime_type="image/jpeg",
        )
        self.assertTrue(res["cropPassUsed"])
        self.assertEqual(res["cropsAnalyzed"], ["anterior", "left_posterior", "right_posterior"])
        self.assertEqual(res["viewStatus"], "assessed")
        self.assertEqual(res["explicitViewStatus"], "assessed_with_findings")
        self.assertEqual(res["conditionStatus"], "assessed_no_findings")
        self.assertEqual(res["alignmentStatus"], "assessed_with_findings")
        self.assertEqual(len(res["dentalAlignment"]), 1)
        box = res["dentalAlignment"][0]["box"]
        self.assertGreater(box["y"], 0.70)
        self.assertEqual(res["unassessableTeethReason"], "Third molars out of frame")

    @patch.dict("os.environ", {"VISION_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "sk-or-v1-test-key"})
    @patch("app.inference.verify_openrouter_model_availability")
    @patch("app.inference.analyze_with_openrouter")
    def test_predict_returns_explicit_per_module_view_statuses(self, mock_analyze, mock_verify):
        mock_verify.return_value = {"valid": True, "status": "available"}
        mock_analyze.side_effect = [
            {
                "viewStatus": "assessed",
                "explicitViewStatus": "assessed_with_findings",
                "conditionStatus": "assessed_with_findings",
                "alignmentStatus": "assessed_no_findings",
                "toothConditions": [
                    {
                        "name": "Enamel Discoloration",
                        "category": "condition",
                        "confidence": 0.85,
                        "severity": "low",
                        "description": "Mild enamel stain.",
                        "recommendation": "Routine hygiene.",
                        "box": {"x": 0.70, "y": 0.45, "w": 0.09, "h": 0.10},
                        "tooth": "26",
                        "toothNumber": "26",
                    }
                ],
                "dentalAlignment": [],
                "all": [
                    {
                        "name": "Enamel Discoloration",
                        "category": "condition",
                        "confidence": 0.85,
                        "severity": "low",
                        "description": "Mild enamel stain.",
                        "recommendation": "Routine hygiene.",
                        "box": {"x": 0.70, "y": 0.45, "w": 0.09, "h": 0.10},
                        "tooth": "26",
                        "toothNumber": "26",
                    }
                ],
                "rejectedCount": 0,
            },
            {
                "viewStatus": "assessed",
                "explicitViewStatus": "assessed_no_findings",
                "conditionStatus": "assessed_no_findings",
                "alignmentStatus": "assessed_no_findings",
                "toothConditions": [],
                "dentalAlignment": [],
                "all": [],
                "rejectedCount": 0,
                "unassessableTeethReason": "Third molars outside frame",
                "cropPassUsed": True,
                "cropsAnalyzed": ["anterior", "left_posterior", "right_posterior"],
            },
            {
                "viewStatus": "inconclusive",
                "explicitViewStatus": "inconclusive",
                "conditionStatus": "inconclusive",
                "alignmentStatus": "inconclusive",
                "inconclusiveReason": "Motion blur obscures front teeth.",
                "toothConditions": [],
                "dentalAlignment": [],
                "all": [],
                "rejectedCount": 0,
            },
        ]

        files = {
            "upper": ("upper.jpg", self.dummy_img, "image/jpeg"),
            "lower": ("lower.jpg", self.dummy_img, "image/jpeg"),
            "front": ("front.jpg", self.dummy_img, "image/jpeg"),
        }
        resp = self.client.post("/api/v1/predict", files=files)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(
            data["viewStatuses"],
            {
                "upper": "assessed_with_findings",
                "lower": "assessed_no_findings",
                "front": "inconclusive",
            },
        )
        self.assertEqual(
            data["moduleStatuses"]["toothConditions"],
            {
                "upper": "assessed_with_findings",
                "lower": "assessed_no_findings",
                "front": "inconclusive",
            },
        )
        self.assertEqual(
            data["moduleStatuses"]["dentalAlignment"],
            {
                "upper": "assessed_no_findings",
                "lower": "assessed_no_findings",
                "front": "inconclusive",
            },
        )
        self.assertIn("lower", data["imageQuality"])
        self.assertTrue(data["viewDiagnostics"]["lower"]["cropPassUsed"])
        self.assertEqual(data["viewDiagnostics"]["lower"]["unassessableTeethReason"], "Third molars outside frame")


if __name__ == "__main__":
    unittest.main()
