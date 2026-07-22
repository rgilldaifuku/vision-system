from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field


HEALTHY = "HEALTHY"
DEGRADED = "DEGRADED"
UNHEALTHY = "UNHEALTHY"


@dataclass
class HealthMonitor:
    startup_time: float = field(default_factory=time.monotonic)
    max_frame_age_seconds: float = 3.0
    high_latency_ms: float = 1000.0
    min_free_disk_pct: float = 5.0
    last_system_error: str = ""

    def snapshot(
        self,
        *,
        camera_connected,
        model_loaded,
        inference_disabled=False,
        latest_frame_time=None,
        last_inference_time=None,
        inference_latency_ms=None,
        image_quality_status=None,
        notification_status=None,
        data_path=None,
    ):
        now = time.perf_counter()
        reasons = []
        warnings = []

        if not camera_connected:
            reasons.append("Camera is not connected.")
        frame_age = None
        if latest_frame_time is not None:
            frame_age = max(0.0, now - float(latest_frame_time))
            if frame_age > self.max_frame_age_seconds:
                reasons.append("Latest camera frame is stale.")
        else:
            warnings.append("No camera frame has been processed yet.")

        if not model_loaded and not inference_disabled:
            reasons.append("Model is not loaded.")

        if inference_latency_ms is not None and inference_latency_ms > self.high_latency_ms:
            warnings.append("Inference latency is high.")

        disk = self._disk_snapshot(data_path)
        if disk and disk["free_pct"] < self.min_free_disk_pct:
            warnings.append("Free disk space is low.")

        if image_quality_status and image_quality_status not in {"GOOD", "DISABLED", "NO_FRAME"}:
            warnings.append(f"Image quality is {image_quality_status}.")

        if self.last_system_error:
            warnings.append(self.last_system_error)

        status = HEALTHY
        if warnings:
            status = DEGRADED
        if reasons:
            status = UNHEALTHY

        return {
            "status": status,
            "reasons": reasons,
            "warnings": warnings,
            "uptime_seconds": round(now - self.startup_time, 3),
            "frame_age_seconds": round(frame_age, 3) if frame_age is not None else None,
            "latest_inference_age_seconds": (
                round(now - float(last_inference_time), 3)
                if last_inference_time is not None
                else None
            ),
            "disk": disk,
            "notification_status": notification_status or {},
            "last_system_error": self.last_system_error or None,
        }

    @staticmethod
    def _disk_snapshot(path):
        if not path:
            return None
        try:
            usage = shutil.disk_usage(path)
        except Exception:
            return None
        total = float(usage.total) or 1.0
        return {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "free_pct": round((usage.free / total) * 100.0, 3),
        }
