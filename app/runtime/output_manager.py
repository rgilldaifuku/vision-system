import time

from app.logging import log_detection_event


class OutputManager:
    """Publishes runtime inspection output to logs and future hardware hooks."""

    def __init__(self, log_cooldown_seconds=2.0):
        self.log_cooldown_seconds = log_cooldown_seconds
        self.last_log_time = 0.0

    def handle_detection(self, active_profile, detection, camera_status):
        self._maybe_log_detection(active_profile, detection, camera_status)

    def _maybe_log_detection(self, active_profile, detection, camera_status):
        now = time.time()
        if now - self.last_log_time < self.log_cooldown_seconds:
            return

        self.last_log_time = now
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

