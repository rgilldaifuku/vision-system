import time
from datetime import datetime

from app.logging import log_detection_event, log_runtime_event


class OutputManager:
    """Publishes runtime inspection output to logs and future hardware hooks."""

    def __init__(self, log_cooldown_seconds=2.0):
        self.log_cooldown_seconds = log_cooldown_seconds
        self.last_log_time = 0.0
        self.last_fault_log_times = {}
        self.last_error = ""
        self.last_payload = {}

    def handle_detection(
        self,
        active_profile,
        detection,
        camera_status,
        model_status="Loaded",
        simulation_mode=False,
    ):
        payload = self.build_output_payload(
            active_profile=active_profile,
            detection=detection,
            camera_status=camera_status,
            model_status=model_status,
            simulation_mode=simulation_mode,
        )
        self.last_payload = payload
        self._maybe_log_detection(active_profile, detection, camera_status)
        return payload

    @staticmethod
    def build_output_payload(
        active_profile,
        detection,
        camera_status,
        model_status="Loaded",
        simulation_mode=False,
    ):
        return {
            "inspection_id": detection.get("inspection_id"),
            "inspection_state": detection.get("inspection_state"),
            "inspection_result": detection.get("inspection_result", "NO_PART"),
            "pass_fail_bool": detection.get("pass_fail_bool"),
            "active_class": detection.get("active_class") or detection.get("class_name"),
            "confidence": detection.get("confidence"),
            "average_confidence": detection.get("average_confidence"),
            "agreement_ratio": detection.get("agreement_ratio"),
            "image_quality_status": (
                (detection.get("image_quality") or {}).get("quality_status")
                if isinstance(detection.get("image_quality"), dict)
                else None
            ),
            "timestamp": detection.get("timestamp") or datetime.now().isoformat(timespec="seconds"),
            "profile": active_profile,
            "camera_status": camera_status,
            "model_status": model_status,
            "simulation_mode": bool(simulation_mode),
            "reason": detection.get("result_message", ""),
            "message": detection.get("result_message", ""),
        }

    def _maybe_log_detection(self, active_profile, detection, camera_status):
        now = time.time()
        if now - self.last_log_time < self.log_cooldown_seconds:
            return

        self.last_log_time = now
        try:
            log_detection_event(
                active_profile=active_profile,
                stable_detected=detection.get("stable_detected", False),
                raw_detected=detection.get("raw_detected", False),
                class_name=detection.get("class_name") or "",
                confidence=detection.get("confidence"),
                camera_status=camera_status,
                roi_enabled=detection.get("roi_enabled", False),
                saved_image_path=detection.get("saved_image_path", ""),
            )
            self.last_error = ""
        except Exception as exc:
            self.last_error = str(exc)

    def log_startup(self, active_profile, details):
        try:
            log_runtime_event(
                event_type="startup",
                profile=active_profile,
                message="Runtime service starting.",
                details=details,
            )
            self.last_error = ""
        except Exception as exc:
            self.last_error = str(exc)

    def log_fault(self, active_profile, fault_type, message, details=None, cooldown_seconds=10.0):
        now = time.time()
        last_logged = self.last_fault_log_times.get(fault_type, 0.0)
        if now - last_logged < cooldown_seconds:
            return

        self.last_fault_log_times[fault_type] = now
        try:
            log_runtime_event(
                event_type=fault_type,
                profile=active_profile,
                message=message,
                details=details or {},
            )
            self.last_error = ""
        except Exception as exc:
            self.last_error = str(exc)
