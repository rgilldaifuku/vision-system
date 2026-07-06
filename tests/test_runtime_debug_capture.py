import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import app.runtime.detector_service as detector_service
from app.runtime.inference_engine import (
    ModelInputSizeMismatchError,
    read_model_input_size,
    resolve_runtime_input_size,
)


def create_profile(models_dir):
    profile_dir = Path(models_dir) / "test_profile"
    profile_dir.mkdir(parents=True)
    (profile_dir / "best.pt").write_bytes(b"model")
    (profile_dir / "classes.txt").write_text("part\n", encoding="utf-8")
    (profile_dir / "config.json").write_text(
        '{"model_file":"best.pt","target_classes":["part"],"confidence":0.35}',
        encoding="utf-8",
    )
    return profile_dir


class RuntimeDebugCaptureTests(unittest.TestCase):
    def test_parser_accepts_model_override_and_debug_capture_arguments(self):
        args = detector_service.create_parser().parse_args(
            [
                "--model-path",
                "models/test/candidate_ncnn_model",
                "--debug-capture-on-detection",
                "--debug-dir",
                "data/debug_frames/test",
                "--debug-max-captures",
                "3",
            ]
        )
        self.assertEqual(args.model_path, "models/test/candidate_ncnn_model")
        self.assertTrue(args.debug_capture_on_detection)
        self.assertEqual(args.debug_max_captures, 3)

    def test_model_path_override_is_actual_status_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            create_profile(models_dir)
            candidate = models_dir / "test_profile" / "candidates" / "candidate_ncnn_model"
            candidate.mkdir(parents=True)
            (candidate / "model.ncnn.param").write_text("param", encoding="utf-8")
            (candidate / "model.ncnn.bin").write_bytes(b"bin")
            (candidate / "metadata.yaml").write_text("imgsz: [320, 320]\n", encoding="utf-8")

            with (
                mock.patch.object(detector_service, "MODELS_DIR", models_dir),
                mock.patch.object(
                    detector_service,
                    "InferenceEngine",
                    lambda path, model_format="auto": object(),
                ),
            ):
                service = detector_service.RuntimeDetectorService(
                    profile_name="test_profile",
                    model_path=str(candidate),
                    model_format="auto",
                )
                status = service.get_status()
                mismatch_service = detector_service.RuntimeDetectorService(
                    profile_name="test_profile",
                    model_path=str(candidate),
                    model_format="auto",
                    imgsz=256,
                    allow_model_input_size_mismatch=True,
                )
                mismatch_status = mismatch_service.get_status()

            self.assertEqual(service.model_path, candidate)
            self.assertEqual(status["model"]["path"], str(candidate))
            self.assertTrue(status["model"]["override"])
            self.assertEqual(status["imgsz"], 320)
            self.assertEqual(status["input_size_source"], "model_metadata")
            self.assertEqual(status["model_input_size"], {"width": 320, "height": 320})
            self.assertFalse(status["model_input_size_mismatch"])
            self.assertEqual(mismatch_status["runtime_input_size"], {"width": 256, "height": 256})
            self.assertTrue(mismatch_status["model_input_size_mismatch"])
            self.assertTrue(mismatch_status["model_input_size_mismatch_allowed"])

    def test_debug_capture_disabled_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            debug_dir = Path(tmpdir) / "captures"
            service = self._service(tmpdir, debug_dir, enabled=False)
            result = service._maybe_capture_detection_debug(
                np.zeros((20, 30, 3), dtype=np.uint8),
                np.zeros((16, 16, 3), dtype=np.uint8),
                [{"class_id": 0, "class_name": "part", "confidence": 1.0, "bbox": [1, 2, 3, 4]}],
                {"raw_detected": True, "stable_detected": False},
                {"quality_status": "GOOD"},
                {"letterbox": True},
            )
            self.assertIsNone(result)
            self.assertFalse(debug_dir.exists())

    def test_debug_capture_writes_exact_array_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            debug_dir = Path(tmpdir) / "captures"
            service = self._service(tmpdir, debug_dir, enabled=True)
            raw = np.full((20, 30, 3), 25, dtype=np.uint8)
            inference = np.full((16, 16, 3), 50, dtype=np.uint8)
            detections = [
                {
                    "class_id": 0,
                    "class_name": "part",
                    "confidence": 1.0,
                    "bbox": [1.0, 2.0, 12.0, 14.0],
                }
            ]
            metadata_path = service._maybe_capture_detection_debug(
                raw,
                inference,
                detections,
                {
                    "raw_detected": True,
                    "stable_detected": False,
                    "inspection_result": "NO_PART",
                    "class_name": "part",
                    "confidence": 1.0,
                    "result_message": "Pending stability.",
                },
                {"quality_status": "GOOD", "brightness_mean": 25.0},
                {"letterbox": True, "output_shape": {"height": 16, "width": 16, "channels": 3}},
            )

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            saved_array = np.load(metadata["inference_input"]["array_path"], allow_pickle=False)
            self.assertTrue(np.array_equal(saved_array, inference))
            self.assertEqual(metadata["inference_input"]["shape"], [16, 16, 3])
            self.assertEqual(metadata["inference_input"]["dtype"], "uint8")
            self.assertEqual(metadata["inference_input"]["in_memory_color_space"], "BGR")
            self.assertEqual(metadata["primary_detection"]["confidence"], 1.0)
            self.assertTrue(Path(metadata["inference_input"]["image_path"]).is_file())

    def test_ncnn_metadata_size_mismatch_fails_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "metadata.yaml").write_text("imgsz: [320, 320]\n", encoding="utf-8")
            self.assertEqual(read_model_input_size(model_dir), (320, 320))
            with self.assertRaisesRegex(ModelInputSizeMismatchError, "320x320"):
                resolve_runtime_input_size(
                    model_dir,
                    requested_imgsz=256,
                    model_format="ncnn",
                )

    def test_explicit_mismatch_escape_hatch_is_reported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "metadata.yaml").write_text("imgsz: [320, 320]\n", encoding="utf-8")
            resolution = resolve_runtime_input_size(
                model_dir,
                requested_imgsz=256,
                model_format="ncnn",
                allow_mismatch=True,
            )
            self.assertEqual(resolution["runtime_imgsz"], 256)
            self.assertTrue(resolution["mismatch"])
            self.assertTrue(resolution["mismatch_allowed"])
            self.assertIn("MISMATCH ALLOWED", resolution["warning"])

    def test_size_resolution_uses_legacy_fallback_without_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolution = resolve_runtime_input_size(
                Path(tmpdir),
                model_format="ncnn",
            )
            self.assertEqual(resolution["runtime_imgsz"], 256)
            self.assertEqual(resolution["source"], "legacy_fallback")

    def test_explicit_cli_size_has_priority_over_environment(self):
        with mock.patch.dict(os.environ, {"IMGSZ": "320"}):
            environment_args = detector_service.create_parser().parse_args([])
            explicit_args = detector_service.create_parser().parse_args(["--imgsz", "640"])
        self.assertEqual(environment_args.imgsz, 320)
        self.assertEqual(explicit_args.imgsz, 640)

    @staticmethod
    def _service(tmpdir, debug_dir, enabled):
        models_dir = Path(tmpdir) / "models"
        create_profile(models_dir)
        with (
            mock.patch.object(detector_service, "MODELS_DIR", models_dir),
            mock.patch.object(
                detector_service,
                "InferenceEngine",
                lambda path, model_format="auto": object(),
            ),
        ):
            return detector_service.RuntimeDetectorService(
                profile_name="test_profile",
                debug_capture_on_detection=enabled,
                debug_dir=debug_dir,
                debug_max_captures=2,
            )


if __name__ == "__main__":
    unittest.main()
