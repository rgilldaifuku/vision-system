from dataclasses import dataclass

from app.config import ROI_ENABLED, ROI_X1, ROI_Y1, ROI_X2, ROI_Y2


@dataclass
class InspectionState:
    stable_detected: bool = False
    raw_detected: bool = False
    class_name: str | None = None
    confidence: float | None = None
    stable_detection_count: int = 0
    detection_frame_count: int = 0
    miss_frame_count: int = 0


class InspectionLogic:
    """Applies target filtering, ROI checks, and stable detection smoothing."""

    def __init__(
        self,
        target_classes,
        detection_required_frames=3,
        miss_required_frames=3,
        roi_enabled=ROI_ENABLED,
        roi_x1=ROI_X1,
        roi_y1=ROI_Y1,
        roi_x2=ROI_X2,
        roi_y2=ROI_Y2,
    ):
        self.target_classes = {str(name).strip() for name in target_classes if str(name).strip()}
        self.detection_required_frames = detection_required_frames
        self.miss_required_frames = miss_required_frames
        self.roi_enabled = bool(roi_enabled)
        self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = self._normalized_roi(
            roi_x1, roi_y1, roi_x2, roi_y2
        )
        self.state = InspectionState()

    def update(self, detections, frame_shape):
        raw_detected, best_detection = self._find_best_target_detection(detections, frame_shape)
        was_stable = self.state.stable_detected

        self.state.raw_detected = raw_detected
        if raw_detected:
            self.state.detection_frame_count += 1
            self.state.miss_frame_count = 0
            self.state.class_name = best_detection["class_name"]
            self.state.confidence = best_detection["confidence"]

            if self.state.detection_frame_count >= self.detection_required_frames:
                self.state.stable_detected = True
        else:
            self.state.miss_frame_count += 1
            self.state.detection_frame_count = 0

            if self.state.miss_frame_count >= self.miss_required_frames:
                self.state.stable_detected = False

        if self.state.stable_detected and not was_stable:
            self.state.stable_detection_count += 1

        return self.snapshot()

    def snapshot(self):
        return {
            "stable_detected": self.state.stable_detected,
            "raw_detected": self.state.raw_detected,
            "class_name": self.state.class_name,
            "confidence": self.state.confidence,
            "stable_detection_count": self.state.stable_detection_count,
            "detection_frame_count": self.state.detection_frame_count,
            "miss_frame_count": self.state.miss_frame_count,
            "target_classes": sorted(self.target_classes),
            "roi_enabled": self.roi_enabled,
            "roi": {
                "x1": self.roi_x1,
                "y1": self.roi_y1,
                "x2": self.roi_x2,
                "y2": self.roi_y2,
            },
        }

    def _find_best_target_detection(self, detections, frame_shape):
        best_detection = None

        for detection in detections:
            class_name = detection.get("class_name")
            if class_name not in self.target_classes:
                continue

            if not self._bbox_center_in_roi(detection.get("bbox"), frame_shape):
                continue

            confidence = detection.get("confidence")
            if best_detection is None or confidence > best_detection["confidence"]:
                best_detection = {
                    "class_name": class_name,
                    "confidence": confidence,
                }

        return best_detection is not None, best_detection

    def _bbox_center_in_roi(self, bbox, frame_shape):
        if not self.roi_enabled:
            return True

        if not bbox or len(bbox) != 4:
            return False

        height, width = frame_shape[:2]
        if width <= 0 or height <= 0:
            return False

        center_x = ((bbox[0] + bbox[2]) / 2) / width
        center_y = ((bbox[1] + bbox[3]) / 2) / height
        return self.roi_x1 <= center_x <= self.roi_x2 and self.roi_y1 <= center_y <= self.roi_y2

    @staticmethod
    def _normalized_roi(x1, y1, x2, y2):
        left = max(0.0, min(1.0, float(x1)))
        top = max(0.0, min(1.0, float(y1)))
        right = max(0.0, min(1.0, float(x2)))
        bottom = max(0.0, min(1.0, float(y2)))
        return min(left, right), min(top, bottom), max(left, right), max(top, bottom)

