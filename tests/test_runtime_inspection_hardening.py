import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import app.runtime.detector_service as detector_service
from app.runtime.event_manager import DuplicateSuppressor, EventManager, EventSeverity
from app.runtime.health_monitor import DEGRADED, HEALTHY, UNHEALTHY, HealthMonitor
from app.runtime.image_quality import BLURRY, INVALID_FRAME, TOO_BRIGHT, TOO_DARK, compute_image_quality
from app.runtime.inspection_logic import InspectionLogic
from app.runtime.inspection_result import (
    InspectionState,
    build_decision,
    canonical_state_from_result,
    generate_inspection_id,
)
from app.runtime.notification_manager import NotificationManager
from app.runtime.action_manager import ActionManager


class RuntimeInspectionHardeningTests(unittest.TestCase):
    def test_existing_acceptable_class_behavior_still_passes(self):
        logic = InspectionLogic(
            acceptable_classes=["yellow_daifuku"],
            minimum_confidence=0.35,
            detection_required_frames=1,
            miss_required_frames=1,
        )
        snapshot = logic.update(
            [{"class_name": "yellow daifuku", "confidence": 0.91, "bbox": [1, 1, 10, 10]}],
            (20, 20, 3),
        )
        self.assertEqual(snapshot["inspection_result"], "PASS")
        self.assertEqual(snapshot["inspection_state"], "PASS")
        self.assertTrue(snapshot["inspection_id"].startswith("INS-"))

    def test_low_confidence_maps_to_review_canonical_state(self):
        logic = InspectionLogic(
            acceptable_classes=["part"],
            minimum_confidence=0.8,
            detection_required_frames=1,
            miss_required_frames=1,
        )
        snapshot = logic.update(
            [{"class_name": "part", "confidence": 0.5, "bbox": [1, 1, 10, 10]}],
            (20, 20, 3),
        )
        self.assertEqual(snapshot["inspection_result"], "LOW_CONFIDENCE")
        self.assertEqual(snapshot["inspection_state"], "REVIEW")

    def test_system_errors_map_to_system_error(self):
        self.assertEqual(canonical_state_from_result("CAMERA_ERROR"), InspectionState.SYSTEM_ERROR)
        self.assertEqual(canonical_state_from_result("MODEL_ERROR"), InspectionState.SYSTEM_ERROR)

    def test_unique_inspection_ids(self):
        first = generate_inspection_id()
        second = generate_inspection_id()
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("INS-"))

    def test_decision_serializes_timezone_aware_timestamp(self):
        decision = build_decision(
            state=InspectionState.PASS,
            reason="Accepted.",
            profile="yellow_daifuku",
            detected_class="yellow daifuku",
            confidence=0.9,
        ).to_dict()
        self.assertEqual(decision["state"], "PASS")
        self.assertIn("+00:00", decision["timestamp"])

    def test_rolling_window_passes_after_agreement(self):
        logic = InspectionLogic(
            acceptable_classes=["part"],
            minimum_confidence=0.7,
            decision_mode="rolling_window",
            rolling_window_size=4,
            rolling_min_agreeing=3,
            rolling_min_agreement_ratio=0.75,
            detection_required_frames=1,
            miss_required_frames=1,
        )
        good = {"class_name": "part", "confidence": 0.9, "bbox": [1, 1, 10, 10]}
        snapshots = [logic.update([good], (20, 20, 3)) for _ in range(3)]
        self.assertEqual(snapshots[-1]["inspection_result"], "PASS")
        self.assertAlmostEqual(snapshots[-1]["agreement_ratio"], 1.0)

    def test_rolling_window_conflict_returns_review(self):
        logic = InspectionLogic(
            acceptable_classes=["good"],
            reject_classes=["bad"],
            minimum_confidence=0.7,
            decision_mode="rolling_window",
            rolling_window_size=4,
            rolling_min_agreeing=3,
            rolling_min_agreement_ratio=0.75,
            detection_required_frames=1,
            miss_required_frames=1,
        )
        shape = (20, 20, 3)
        logic.update([{"class_name": "good", "confidence": 0.9, "bbox": [1, 1, 10, 10]}], shape)
        logic.update([{"class_name": "bad", "confidence": 0.9, "bbox": [1, 1, 10, 10]}], shape)
        logic.update([{"class_name": "good", "confidence": 0.9, "bbox": [1, 1, 10, 10]}], shape)
        snapshot = logic.update(
            [{"class_name": "bad", "confidence": 0.9, "bbox": [1, 1, 10, 10]}],
            shape,
        )
        self.assertEqual(snapshot["inspection_result"], "REVIEW")
        self.assertEqual(snapshot["inspection_state"], "REVIEW")

    def test_image_quality_core_states(self):
        dark = np.zeros((40, 40, 3), dtype=np.uint8)
        bright = np.full((40, 40, 3), 255, dtype=np.uint8)
        flat = np.full((40, 40, 3), 120, dtype=np.uint8)
        self.assertEqual(compute_image_quality(None)["quality_status"], INVALID_FRAME)
        self.assertEqual(compute_image_quality(dark)["quality_status"], TOO_DARK)
        self.assertEqual(compute_image_quality(bright)["quality_status"], TOO_BRIGHT)
        self.assertIn(compute_image_quality(flat)["quality_status"], {BLURRY, "LOW_CONTRAST"})

    def test_event_persistence_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            event_path = Path(tmpdir) / "events.jsonl"
            manager = EventManager(event_path=event_path)
            try:
                event = manager.record(
                    "INSPECTION_FAILED",
                    severity=EventSeverity.WARNING,
                    profile="yellow_daifuku",
                    inspection_id="INS-1",
                    message="Failed.",
                    details={"path": Path(tmpdir)},
                )
                manager.queue.join()
                rows = event_path.read_text(encoding="utf-8").splitlines()
                payload = json.loads(rows[0])
            finally:
                manager.stop()
        self.assertEqual(payload["event_id"], event.event_id)
        self.assertEqual(payload["details"]["path"], tmpdir)

    def test_event_write_failure_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = EventManager(event_path=Path(tmpdir))
            try:
                manager.record("RUNTIME_STARTED", profile="yellow_daifuku")
                manager.queue.join()
                self.assertTrue(manager.snapshot()["last_error"])
            finally:
                manager.stop()

    def test_duplicate_suppression_and_repeat_threshold(self):
        suppressor = DuplicateSuppressor(cooldown_seconds=10, repeat_threshold=3)
        self.assertTrue(suppressor.should_emit("camera"))
        self.assertFalse(suppressor.should_emit("camera"))
        self.assertFalse(suppressor.repeated("fail", now=1.0))
        self.assertFalse(suppressor.repeated("fail", now=2.0))
        self.assertTrue(suppressor.repeated("fail", now=3.0))

    def test_notifications_disabled_do_nothing(self):
        manager = NotificationManager(enabled=False)
        event = build_decision(
            state=InspectionState.FAIL,
            reason="Failed.",
            profile="yellow_daifuku",
        )
        result = manager.notify(type("Event", (), {"to_dict": lambda self: event.to_dict()})())
        self.assertFalse(result["sent"])
        self.assertEqual(result["reason"], "notifications_disabled")

    def test_simulation_action_counter_increments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ActionManager(status_path=Path(tmpdir) / "latest_status.json")
            result = manager.handle(
                {
                    "inspection_result": "SIMULATION",
                    "profile": "yellow_daifuku",
                    "message": "Simulation mode is active.",
                }
            )
        self.assertEqual(result["counters"]["simulation"], 1)

    def test_missing_notification_credentials_do_not_crash(self):
        manager = NotificationManager(enabled=True)
        manager.email_enabled = True
        result = manager.notify(
            type(
                "Event",
                (),
                {
                    "to_dict": lambda self: {
                        "event_id": "EVT-1",
                        "event_type": "TEST",
                        "severity": "INFO",
                        "profile": "yellow_daifuku",
                        "message": "test",
                    }
                },
            )()
        )
        self.assertFalse(result["sent"])
        self.assertTrue(result["errors"])

    def test_health_transitions(self):
        monitor = HealthMonitor(startup_time=time.monotonic(), min_free_disk_pct=0.0)
        healthy = monitor.snapshot(
            camera_connected=True,
            model_loaded=True,
            latest_frame_time=time.perf_counter(),
            data_path=Path.cwd(),
        )
        degraded = monitor.snapshot(
            camera_connected=True,
            model_loaded=True,
            latest_frame_time=time.perf_counter(),
            image_quality_status="BLURRY",
            data_path=Path.cwd(),
        )
        unhealthy = monitor.snapshot(
            camera_connected=False,
            model_loaded=True,
            latest_frame_time=time.perf_counter(),
            data_path=Path.cwd(),
        )
        self.assertEqual(healthy["status"], HEALTHY)
        self.assertEqual(degraded["status"], DEGRADED)
        self.assertEqual(unhealthy["status"], UNHEALTHY)

    def test_disk_free_pct_uses_percent_units(self):
        usage = shutil._ntuple_diskusage(total=200, used=198, free=2)
        with mock.patch("app.runtime.health_monitor.shutil.disk_usage", return_value=usage):
            monitor = HealthMonitor(startup_time=time.monotonic(), min_free_disk_pct=5.0)
            snapshot = monitor.snapshot(
                camera_connected=True,
                model_loaded=True,
                latest_frame_time=time.perf_counter(),
                data_path=Path.cwd(),
            )
        self.assertEqual(snapshot["disk"]["free_pct"], 1.0)
        self.assertEqual(snapshot["status"], DEGRADED)
        self.assertIn("Free disk space is low.", snapshot["warnings"])

    def test_status_preserves_old_fields_and_adds_new_sections(self):
        service = detector_service.RuntimeDetectorService(
            profile_name="missing_profile",
            camera_only=True,
            camera_backend="opencv",
        )
        try:
            status = service.get_status()
            latest = service.get_latest_detection()
        finally:
            service.event_manager.stop()
        self.assertIn("inspection_result", status)
        self.assertIn("latest_detection", status)
        self.assertIn("output_payload", status)
        self.assertIn("health", status)
        self.assertIn("events", status)
        self.assertIn("notifications", status)
        self.assertIn("inspection_id", latest)
        self.assertIn("inspection_state", latest)


if __name__ == "__main__":
    unittest.main()
