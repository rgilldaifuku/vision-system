import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import app.runtime.detector_service as detector_service
import app.runtime.health_check as health_check
from app.runtime.action_manager import ActionManager
from app.runtime.camera_sources import SimulatedCameraSource
from app.runtime.inspection_logic import InspectionLogic
from app.runtime.output_manager import OutputManager
from app.runtime.picamera2_manager import Picamera2CameraManager


class DummyOutputManager:
    last_error = ""
    last_payload = {}

    def handle_detection(
        self,
        active_profile,
        detection,
        camera_status,
        model_status="Loaded",
        simulation_mode=False,
    ):
        self.last_payload = OutputManager.build_output_payload(
            active_profile=active_profile,
            detection=detection,
            camera_status=camera_status,
            model_status=model_status,
            simulation_mode=simulation_mode,
        )
        return self.last_payload

    def log_startup(self, active_profile, details):
        return None

    def log_fault(self, active_profile, fault_type, message, details=None, cooldown_seconds=10.0):
        return None


class DummyCamera:
    status = "Failed"
    last_error = ""

    def open(self):
        return False

    def read_frame(self):
        return None

    def release(self):
        return None


def create_profile(root, profile_name="test_profile"):
    profile_dir = root / profile_name
    latest_dir = profile_dir / "latest"
    latest_dir.mkdir(parents=True)
    (latest_dir / "best.pt").write_bytes(b"placeholder model")
    (profile_dir / "classes.txt").write_text("part\n", encoding="utf-8")
    (profile_dir / "config.json").write_text(
        '{"profile_name":"test_profile","model_file":"latest/best.pt",'
        '"target_classes":["part"],"confidence":0.35}',
        encoding="utf-8",
    )
    return profile_dir


def write_test_image(path, width=64, height=48, value=90):
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), frame)
    return frame


class RuntimeTests(unittest.TestCase):
    def test_simulated_camera_source_reads_single_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.jpg"
            write_test_image(image_path)

            source = SimulatedCameraSource(image_path, frame_interval_seconds=0)
            try:
                self.assertTrue(source.open(), source.last_error)
                frame = source.read_frame()
                self.assertIsNotNone(frame)
                self.assertEqual(frame.shape[:2], (48, 64))
                self.assertEqual(source.status, "Connected")
            finally:
                source.release()

    def test_simulated_camera_source_cycles_folder_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir) / "images"
            folder.mkdir()
            write_test_image(folder / "a.jpg", value=40)
            write_test_image(folder / "b.png", value=120)

            source = SimulatedCameraSource(folder, frame_interval_seconds=0)
            try:
                self.assertTrue(source.open(), source.last_error)
                first = source.read_frame()
                second = source.read_frame()
                self.assertIsNotNone(first)
                self.assertIsNotNone(second)
                self.assertNotEqual(int(first.mean()), int(second.mean()))
            finally:
                source.release()

    def test_inspection_logic_smoothing(self):
        logic = InspectionLogic(["part"], detection_required_frames=3, miss_required_frames=3)
        detection = {"class_name": "part", "confidence": 0.9, "bbox": [0, 0, 10, 10]}
        shape = (20, 20, 3)

        self.assertFalse(logic.update([detection], shape)["stable_detected"])
        self.assertFalse(logic.update([detection], shape)["stable_detected"])
        self.assertTrue(logic.update([detection], shape)["stable_detected"])
        self.assertTrue(logic.update([], shape)["stable_detected"])
        self.assertTrue(logic.update([], shape)["stable_detected"])
        self.assertFalse(logic.update([], shape)["stable_detected"])

    def test_inspection_logic_handles_bad_confidence_and_roi_edges(self):
        logic = InspectionLogic(
            ["part"],
            detection_required_frames=1,
            miss_required_frames=1,
            roi_enabled=True,
            roi_x1=0.25,
            roi_y1=0.25,
            roi_x2=0.75,
            roi_y2=0.75,
        )
        shape = (100, 100, 3)

        outside = {"class_name": "part", "confidence": 0.95, "bbox": [0, 0, 10, 10]}
        self.assertFalse(logic.update([outside], shape)["raw_detected"])

        inside_bad_confidence = {"class_name": "part", "confidence": "bad", "bbox": [40, 40, 60, 60]}
        snapshot = logic.update([inside_bad_confidence], shape)
        self.assertTrue(snapshot["raw_detected"])
        self.assertEqual(snapshot["confidence"], "bad")

    def test_pass_class_produces_pass(self):
        logic = InspectionLogic(
            acceptable_classes=["good"],
            reject_classes=["bad"],
            minimum_confidence=0.7,
            detection_required_frames=1,
            miss_required_frames=1,
        )
        detection = {"class_name": "good", "confidence": 0.91, "bbox": [0, 0, 10, 10]}
        snapshot = logic.update([detection], (20, 20, 3))
        self.assertEqual(snapshot["inspection_result"], "PASS")
        self.assertTrue(snapshot["pass_fail_bool"])

    def test_class_name_normalization_matches_spaces_and_underscores(self):
        logic = InspectionLogic(
            acceptable_classes=["yellow_daifuku"],
            minimum_confidence=0.7,
            detection_required_frames=1,
            miss_required_frames=1,
        )
        detection = {"class_name": "yellow daifuku", "confidence": 0.91, "bbox": [0, 0, 10, 10]}
        snapshot = logic.update([detection], (20, 20, 3))
        self.assertEqual(snapshot["inspection_result"], "PASS")
        self.assertEqual(snapshot["class_name"], "yellow daifuku")

    def test_reject_class_produces_fail(self):
        logic = InspectionLogic(
            acceptable_classes=["good"],
            reject_classes=["bad"],
            minimum_confidence=0.7,
            detection_required_frames=1,
            miss_required_frames=1,
        )
        detection = {"class_name": "bad", "confidence": 0.91, "bbox": [0, 0, 10, 10]}
        snapshot = logic.update([detection], (20, 20, 3))
        self.assertEqual(snapshot["inspection_result"], "FAIL")
        self.assertFalse(snapshot["pass_fail_bool"])

    def test_no_detections_produce_no_part_after_threshold(self):
        logic = InspectionLogic(["part"], detection_required_frames=1, miss_required_frames=2)
        detection = {"class_name": "part", "confidence": 0.91, "bbox": [0, 0, 10, 10]}
        self.assertEqual(logic.update([detection], (20, 20, 3))["inspection_result"], "PASS")
        self.assertEqual(logic.update([], (20, 20, 3))["inspection_result"], "PASS")
        snapshot = logic.update([], (20, 20, 3))
        self.assertEqual(snapshot["inspection_result"], "NO_PART")
        self.assertIsNone(snapshot["pass_fail_bool"])

    def test_low_confidence_produces_low_confidence(self):
        logic = InspectionLogic(
            acceptable_classes=["good"],
            minimum_confidence=0.8,
            detection_required_frames=1,
            miss_required_frames=1,
        )
        detection = {"class_name": "good", "confidence": 0.5, "bbox": [0, 0, 10, 10]}
        snapshot = logic.update([detection], (20, 20, 3))
        self.assertEqual(snapshot["inspection_result"], "LOW_CONFIDENCE")
        self.assertFalse(snapshot["pass_fail_bool"])

    def test_camera_error_produces_camera_error(self):
        logic = InspectionLogic(["part"], detection_required_frames=1, miss_required_frames=1)
        snapshot = logic.update([], (20, 20, 3), camera_status="Failed")
        self.assertEqual(snapshot["inspection_result"], "CAMERA_ERROR")
        self.assertFalse(snapshot["pass_fail_bool"])

    def test_camera_error_is_not_hidden_by_simulation(self):
        logic = InspectionLogic(["part"], detection_required_frames=1, miss_required_frames=1)
        snapshot = logic.update([], (20, 20, 3), camera_status="Failed", simulation_mode=True)
        self.assertEqual(snapshot["inspection_result"], "CAMERA_ERROR")

    def test_dry_run_produces_simulation_result(self):
        logic = InspectionLogic(
            ["simulated_object"],
            detection_required_frames=1,
            miss_required_frames=1,
        )
        detection = {"class_name": "simulated_object", "confidence": 0.91, "bbox": [0, 0, 10, 10]}
        snapshot = logic.update([detection], (20, 20, 3), simulation_mode=True)
        self.assertEqual(snapshot["inspection_result"], "SIMULATION")
        self.assertIsNone(snapshot["pass_fail_bool"])

    def test_output_payload_shape_is_stable(self):
        detection = {
            "inspection_result": "PASS",
            "pass_fail_bool": True,
            "active_class": "good",
            "confidence": 0.91,
            "timestamp": "2026-05-27T10:00:00",
            "result_message": "Accepted class 'good' detected.",
        }
        payload = OutputManager.build_output_payload(
            active_profile="profile",
            detection=detection,
            camera_status="Connected",
            model_status="Loaded",
            simulation_mode=False,
        )
        self.assertEqual(
            set(payload),
            {
                "inspection_result",
                "pass_fail_bool",
                "active_class",
                "confidence",
                "timestamp",
                "profile",
                "camera_status",
                "model_status",
                "simulation_mode",
                "reason",
                "message",
            },
        )

    def test_missing_profile_has_useful_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(detector_service, "MODELS_DIR", Path(tmpdir)):
                service = detector_service.RuntimeDetectorService(profile_name="missing_profile")
                self.assertEqual(service.model_status, "Error")
                self.assertIn("Model profile not found", service.model_error)

    def test_missing_model_produces_model_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = Path(tmpdir) / "test_profile"
            profile_dir.mkdir()
            (profile_dir / "classes.txt").write_text("part\n", encoding="utf-8")
            (profile_dir / "config.json").write_text(
                '{"model_file":"latest/best.pt","target_classes":["part"]}',
                encoding="utf-8",
            )

            with mock.patch.object(detector_service, "MODELS_DIR", Path(tmpdir)):
                service = detector_service.RuntimeDetectorService(profile_name="test_profile")
                detection = service.inspection.update(
                    [],
                    (20, 20, 3),
                    camera_status="Connected",
                    model_status=service.model_status,
                    simulation_mode=False,
                )
                self.assertEqual(service.model_status, "Error")
                self.assertIn("Runtime model not found", service.model_error)
                self.assertEqual(detection["inspection_result"], "MODEL_ERROR")

    def test_auto_backend_preserves_explicit_opencv_camera(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_profile(Path(tmpdir))

            with (
                mock.patch.object(detector_service, "MODELS_DIR", Path(tmpdir)),
                mock.patch.object(detector_service, "InferenceEngine", lambda path, model_format="auto": object()),
                mock.patch.object(detector_service.Picamera2CameraManager, "is_available", return_value=True),
            ):
                explicit_camera = detector_service.RuntimeDetectorService(
                    profile_name="test_profile",
                    camera_index=0,
                    camera_backend="auto",
                )
                auto_camera = detector_service.RuntimeDetectorService(
                    profile_name="test_profile",
                    camera_index=None,
                    camera_backend="auto",
                )

            self.assertEqual(explicit_camera.camera.backend, "opencv")
            self.assertEqual(auto_camera.camera.backend, "picamera2")

    def test_prefer_edge_model_selects_ncnn_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = create_profile(Path(tmpdir))
            ncnn_dir = profile_dir / "best_ncnn_model"
            ncnn_dir.mkdir()
            (ncnn_dir / "model.ncnn.param").write_text("param", encoding="utf-8")
            (ncnn_dir / "model.ncnn.bin").write_bytes(b"bin")

            with (
                mock.patch.object(detector_service, "MODELS_DIR", Path(tmpdir)),
                mock.patch.object(detector_service, "InferenceEngine", lambda path, model_format="auto": object()),
            ):
                service = detector_service.RuntimeDetectorService(
                    profile_name="test_profile",
                    prefer_edge_model=True,
                    model_format="auto",
                )

            self.assertEqual(service.model_path, ncnn_dir)
            self.assertEqual(service.model_format, "ncnn")

    def test_invalid_profile_config_raises_readable_error(self):
        with tempfile.TemporaryDirectory() as models_tmp, tempfile.TemporaryDirectory() as profiles_tmp:
            create_profile(Path(models_tmp))
            rules_dir = Path(profiles_tmp) / "test_profile"
            rules_dir.mkdir(parents=True)
            (rules_dir / "config.yaml").write_text(
                "inspection:\n  minimum_confidence: not-a-number\n",
                encoding="utf-8",
            )

            with (
                mock.patch.object(detector_service, "MODELS_DIR", Path(models_tmp)),
                mock.patch.object(detector_service, "PROFILE_CONFIGS_DIR", Path(profiles_tmp)),
                mock.patch.object(detector_service, "InferenceEngine", lambda path, model_format="auto": object()),
            ):
                with self.assertRaisesRegex(detector_service.ProfileConfigError, "minimum_confidence"):
                    detector_service.RuntimeDetectorService(profile_name="test_profile")

    def test_dry_run_starts_without_model_profile(self):
        with (
            tempfile.TemporaryDirectory() as models_tmp,
            tempfile.TemporaryDirectory() as source_tmp,
            tempfile.TemporaryDirectory() as review_tmp,
        ):
            image_path = Path(source_tmp) / "sample.jpg"
            write_test_image(image_path)

            with (
                mock.patch.object(detector_service, "MODELS_DIR", Path(models_tmp)),
                mock.patch.object(detector_service, "REVIEW_IMAGES_DIR", Path(review_tmp)),
            ):
                service = detector_service.RuntimeDetectorService(
                    profile_name="missing_profile",
                    camera_source=str(image_path),
                    dry_run=True,
                    detection_required_frames=1,
                    miss_required_frames=1,
                    inference_interval_ms=20,
                )
                service.output_manager = DummyOutputManager()

                service.start()
                time.sleep(0.18)
                service.stop()

                status = service.get_status()
                latest = service.get_latest_detection()
                self.assertTrue(status["dry_run"])
                self.assertTrue(status["simulation_mode"])
                self.assertEqual(status["camera_status"], "Connected")
                self.assertGreater(status["frame_count"], 0)
                self.assertIn("DRY_RUN_NO_MODEL", status["model_path"])
                self.assertTrue(latest["simulation_mode"])

    def test_runtime_starts_without_camera_in_safe_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            create_profile(Path(tmpdir))
            with (
                mock.patch.object(detector_service, "MODELS_DIR", Path(tmpdir)),
                mock.patch.object(detector_service, "InferenceEngine", lambda path, model_format="auto": object()),
            ):
                service = detector_service.RuntimeDetectorService(
                    profile_name="test_profile",
                    inference_interval_ms=50,
                )
                service.camera = DummyCamera()
                service.output_manager = DummyOutputManager()

                service.start()
                time.sleep(0.15)
                service.stop()

                status = service.get_status()
                latest = service.get_latest_detection()
                self.assertEqual(status["camera_status"], "Failed")
                self.assertFalse(status["running"])
                self.assertFalse(latest["raw_detected"])
                self.assertFalse(latest["stable_detected"])

    def test_status_payload_shape_in_dry_run(self):
        with tempfile.TemporaryDirectory() as models_tmp:
            with mock.patch.object(detector_service, "MODELS_DIR", Path(models_tmp)):
                service = detector_service.RuntimeDetectorService(
                    profile_name="missing_profile",
                    dry_run=True,
                )
                status = service.get_status()

        for key in (
            "camera_status",
            "profile_name",
            "model_path",
            "model_status",
            "camera_source",
            "camera_backend",
            "camera_connected",
            "dry_run",
            "runtime_mode",
            "simulation_mode",
            "inspection_rules",
            "camera_fps",
            "inference_fps",
            "last_inference_ms",
            "inspection_result",
            "pass_fail_bool",
            "result_message",
            "total_detections",
            "total_images_saved",
            "low_confidence_count",
            "no_detection_count",
            "latest_detection",
            "output_payload",
        ):
            self.assertIn(key, status)
        self.assertIn("inspection_result", status["latest_detection"])
        self.assertIn("runtime", status)
        self.assertIn("camera", status)
        self.assertIn("model", status)
        self.assertIn("inspection", status)
        self.assertIn("performance", status)
        self.assertIn("counters", status)
        self.assertIn("output_payload", status)

    def test_health_check_mode_parsing(self):
        parser = health_check.create_parser()
        laptop_args = parser.parse_args(["--mode", "laptop", "--profile", "yellow_daifuku"])
        pi_args = parser.parse_args([
            "--mode",
            "pi",
            "--profile",
            "yellow_daifuku",
            "--camera-backend",
            "picamera2",
        ])

        self.assertEqual(laptop_args.mode, "laptop")
        self.assertEqual(pi_args.mode, "pi")
        self.assertEqual(pi_args.camera_backend, "picamera2")

    def test_picamera2_manager_missing_package_is_safe(self):
        with mock.patch.object(
            Picamera2CameraManager,
            "_import_picamera2",
            side_effect=RuntimeError("Picamera2 is not available"),
        ):
            camera = Picamera2CameraManager(frame_width=64, frame_height=48, warmup_seconds=0)
            self.assertFalse(camera.open())
            self.assertEqual(camera.backend, "picamera2")
            self.assertEqual(camera.status, "Failed")
            self.assertIn("Picamera2", camera.last_error)

    def test_picamera2_manager_uses_background_cached_frames(self):
        class FakeRequest:
            def __init__(self, value):
                self.value = value
                self.released = False

            def make_array(self, stream_name):
                self.stream_name = stream_name
                frame = np.zeros((8, 10, 3), dtype=np.uint8)
                frame[:, :, 0] = self.value
                frame[:, :, 1] = 2
                frame[:, :, 2] = 3
                return frame

            def release(self):
                self.released = True

        class FakePicamera2:
            requests = []

            def __init__(self):
                self.started = False
                self.closed = False
                self.count = 0

            def create_video_configuration(self, **kwargs):
                self.config_kwargs = kwargs
                return {"config": kwargs}

            def configure(self, config):
                self.config = config

            def start(self):
                self.started = True

            def capture_request(self):
                self.count += 1
                request = FakeRequest(self.count)
                FakePicamera2.requests.append(request)
                time.sleep(0.005)
                return request

            def stop(self):
                self.started = False

            def close(self):
                self.closed = True

        FakePicamera2.requests = []
        with mock.patch.object(Picamera2CameraManager, "_import_picamera2", return_value=FakePicamera2):
            camera = Picamera2CameraManager(
                frame_width=10,
                frame_height=8,
                warmup_seconds=0,
                max_stale_seconds=1.0,
            )
            try:
                self.assertTrue(camera.open(), camera.last_error)
                frame = None
                for _ in range(50):
                    frame = camera.read_frame()
                    if frame is not None and camera.camera_fps > 0:
                        break
                    time.sleep(0.02)

                self.assertIsNotNone(frame)
                self.assertEqual(frame.shape, (8, 10, 3))
                self.assertEqual(camera.status, "Connected")
                self.assertEqual(camera.last_error, "")
                self.assertGreater(camera.camera_fps, 0)
            finally:
                camera.release()
            self.assertTrue(all(request.released for request in FakePicamera2.requests))

    def test_action_manager_writes_latest_status_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = Path(tmpdir) / "latest_status.json"
            manager = ActionManager(
                action_config={"actions_by_result": {"PASS": ["write_latest_status_json", "increment_counter"]}},
                status_path=status_path,
                event_cooldown_seconds=0,
            )
            result = manager.handle(
                {
                    "timestamp": "2026-06-15T12:00:00",
                    "profile": "test_profile",
                    "runtime_mode": "PRODUCTION",
                    "simulation_mode": False,
                    "camera_backend": "opencv",
                    "camera_connected": True,
                    "model_loaded": True,
                    "inspection_result": "PASS",
                    "pass_fail_bool": True,
                    "active_class": "part",
                    "confidence": 0.9,
                    "message": "Accepted.",
                    "saved_image_path": None,
                    "counters": {},
                    "inference_ms": 25.0,
                    "inference_fps": 4.0,
                    "camera_fps": 20.0,
                }
            )

            self.assertTrue(status_path.exists())
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["inspection_result"], "PASS")
            self.assertEqual(payload["camera_backend"], "opencv")
            self.assertEqual(payload["counters"]["pass"], 1)
            self.assertEqual(result["placeholder_outputs"]["gpio"], "disabled")

    def test_health_check_reports_invalid_runtime_rules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = Path(tmpdir) / "profiles" / "bad_profile"
            profile_dir.mkdir(parents=True)
            (profile_dir / "config.yaml").write_text(
                "inspection:\n  required_consecutive_detections: 0\n",
                encoding="utf-8",
            )
            check = health_check.HealthCheck()

            with mock.patch.object(health_check, "PROJECT_ROOT", Path(tmpdir)):
                with redirect_stdout(StringIO()):
                    health_check.check_runtime_profile_rules(check, "bad_profile")

            self.assertEqual(check.failures, 1)

    def test_demo_command_can_start_dry_run_safely(self):
        repo_root = Path(__file__).resolve().parent.parent
        process = subprocess.Popen(
            ["scripts/run_demo.sh"],
            cwd=repo_root,
            env={
                **dict(os.environ),
                "VISION_FORCE_DRY_RUN": "1",
                "VISION_PORT": "8768",
                "VISION_CAMERA_SOURCE": "assets/test.jpg",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            import json
            import urllib.request

            status = None
            for _ in range(40):
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8768/status", timeout=1) as response:
                        status = json.loads(response.read().decode("utf-8"))
                    break
                except Exception:
                    time.sleep(0.25)

            self.assertIsNotNone(status)
            self.assertTrue(status["simulation_mode"])
            self.assertIn("inspection_result", status["latest_detection"])
        finally:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            if process.stdout:
                process.stdout.close()

    def test_profile_yaml_loads_inspection_rules(self):
        with tempfile.TemporaryDirectory() as models_tmp, tempfile.TemporaryDirectory() as profiles_tmp:
            create_profile(Path(models_tmp))
            rules_dir = Path(profiles_tmp) / "test_profile"
            rules_dir.mkdir(parents=True)
            (rules_dir / "config.yaml").write_text(
                "inspection:\n"
                "  acceptable_classes: [pass_part]\n"
                "  reject_classes: [fail_part]\n"
                "  minimum_confidence: 0.8\n"
                "  required_consecutive_detections: 2\n"
                "  allowed_no_detection_frames: 4\n",
                encoding="utf-8",
            )

            with (
                mock.patch.object(detector_service, "MODELS_DIR", Path(models_tmp)),
                mock.patch.object(detector_service, "PROFILE_CONFIGS_DIR", Path(profiles_tmp)),
                mock.patch.object(detector_service, "InferenceEngine", lambda path, model_format="auto": object()),
            ):
                service = detector_service.RuntimeDetectorService(profile_name="test_profile")

            self.assertEqual(service.inspection_rules["acceptable_classes"], ["pass_part"])
            self.assertEqual(service.inspection_rules["reject_classes"], ["fail_part"])
            self.assertEqual(service.inspection_rules["minimum_confidence"], 0.8)
            self.assertEqual(service.inspection_rules["detection_required_frames"], 2)
            self.assertEqual(service.inspection_rules["miss_required_frames"], 4)

    def test_review_images_are_saved_and_counted(self):
        frame = np.zeros((60, 80, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as models_tmp, tempfile.TemporaryDirectory() as review_tmp:
            create_profile(Path(models_tmp))
            with (
                mock.patch.object(detector_service, "MODELS_DIR", Path(models_tmp)),
                mock.patch.object(detector_service, "REVIEW_IMAGES_DIR", Path(review_tmp)),
                mock.patch.object(detector_service, "InferenceEngine", lambda path, model_format="auto": object()),
            ):
                service = detector_service.RuntimeDetectorService(profile_name="test_profile")

                stable_low = {
                    "stable_detected": True,
                    "raw_detected": True,
                    "class_name": "part",
                    "confidence": 0.5,
                    "stable_detection_count": 1,
                }
                result = service._handle_review_images(frame, stable_low)
                self.assertTrue(result["saved_image_path"])

                no_detection = {
                    "stable_detected": False,
                    "raw_detected": False,
                    "class_name": None,
                    "confidence": None,
                    "stable_detection_count": 1,
                }
                service._handle_review_images(frame, no_detection)

                status = service.get_status()
                root = Path(review_tmp) / "test_profile"
                self.assertEqual(status["total_detections"], 1)
                self.assertEqual(status["total_images_saved"], 3)
                self.assertEqual(status["low_confidence_count"], 1)
                self.assertEqual(status["no_detection_count"], 1)
                self.assertTrue((root / "detections").is_dir())
                self.assertTrue((root / "low_confidence").is_dir())
                self.assertTrue((root / "no_detection").is_dir())

    def test_dashboard_and_snapshot_routes(self):
        class Service:
            enable_snapshot = False
            snapshot_interval_ms = 1000

            def get_status(self):
                return {"camera_status": "Failed", "snapshot_enabled": False}

            def get_latest_detection(self):
                return {"stable_detected": False, "raw_detected": False}

            def get_snapshot_jpeg(self):
                return b""

        client = detector_service.create_app(Service()).test_client()
        self.assertEqual(client.get("/").status_code, 200)
        self.assertTrue(client.get("/status").is_json)
        self.assertTrue(client.get("/latest_detection").is_json)
        self.assertEqual(client.get("/snapshot.jpg").status_code, 404)

    def test_readme_runtime_command_matches_cli_arguments(self):
        repo_root = Path(__file__).resolve().parent.parent
        readme = (repo_root / "README.md").read_text(encoding="utf-8")
        help_result = subprocess.run(
            [sys.executable, "-m", "app.runtime.detector_service", "--help"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=True,
        )
        health_help_result = subprocess.run(
            [sys.executable, "-m", "app.runtime.health_check", "--help"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=True,
        )

        for argument in (
            "--profile",
            "--camera",
            "--camera-backend",
            "--host",
            "--port",
            "--imgsz",
            "--frame-width",
            "--frame-height",
            "--inference-interval-ms",
            "--camera-source",
            "--dry-run",
        ):
            self.assertIn(argument, readme)
            self.assertIn(argument, help_result.stdout)

        self.assertIn("python -m app.runtime.health_check", readme)
        self.assertIn("scripts/setup_local.sh", readme)
        self.assertIn("scripts/run_demo.sh", readme)
        for argument in ("--mode", "--profile", "--camera-source", "--camera-backend"):
            self.assertIn(argument, health_help_result.stdout)


if __name__ == "__main__":
    unittest.main()
