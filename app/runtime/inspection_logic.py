from dataclasses import dataclass

from app.config import DEFAULT_CONFIDENCE, ROI_ENABLED, ROI_X1, ROI_Y1, ROI_X2, ROI_Y2


PASS = "PASS"
FAIL = "FAIL"
NO_PART = "NO_PART"
LOW_CONFIDENCE = "LOW_CONFIDENCE"
CAMERA_ERROR = "CAMERA_ERROR"
MODEL_ERROR = "MODEL_ERROR"
SIMULATION = "SIMULATION"


@dataclass
class InspectionState:
    stable_detected: bool = False
    raw_detected: bool = False
    class_name: str | None = None
    confidence: float | str | None = None
    inspection_result: str = NO_PART
    pass_fail_bool: bool | None = None
    result_message: str = "No part detected."
    stable_detection_count: int = 0
    detection_frame_count: int = 0
    miss_frame_count: int = 0
    candidate_result: str | None = None
    candidate_class: str | None = None


class InspectionLogic:
    """Converts raw detections into stable operator-facing inspection decisions."""

    def __init__(
        self,
        target_classes=None,
        detection_required_frames=3,
        miss_required_frames=3,
        roi_enabled=ROI_ENABLED,
        roi_x1=ROI_X1,
        roi_y1=ROI_Y1,
        roi_x2=ROI_X2,
        roi_y2=ROI_Y2,
        acceptable_classes=None,
        reject_classes=None,
        minimum_confidence=DEFAULT_CONFIDENCE,
        allow_simulation=True,
    ):
        target_classes = target_classes or []
        acceptable_classes = acceptable_classes or target_classes
        reject_classes = reject_classes or []

        self.acceptable_classes = self._clean_class_set(acceptable_classes)
        self.reject_classes = self._clean_class_set(reject_classes)
        self.target_classes = self.acceptable_classes or self._clean_class_set(target_classes)
        self.minimum_confidence = self._safe_float(minimum_confidence, DEFAULT_CONFIDENCE)
        self.detection_required_frames = max(1, int(detection_required_frames))
        self.miss_required_frames = max(1, int(miss_required_frames))
        self.allow_simulation = bool(allow_simulation)
        self.roi_enabled = bool(roi_enabled)
        self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = self._normalized_roi(
            roi_x1, roi_y1, roi_x2, roi_y2
        )
        self.state = InspectionState()

    def update(
        self,
        detections,
        frame_shape,
        camera_status="Connected",
        model_status="Loaded",
        simulation_mode=False,
    ):
        previous_result = self.state.inspection_result

        if camera_status != "Connected":
            self._set_system_result(CAMERA_ERROR, False, f"Camera status is {camera_status}.")
            return self.snapshot()

        if model_status not in {"Loaded", "Simulation"}:
            self._set_system_result(MODEL_ERROR, False, f"Model status is {model_status}.")
            return self.snapshot()

        raw_detected, best_detection = self._find_best_detection(detections or [], frame_shape)
        self.state.raw_detected = raw_detected

        if not raw_detected:
            self._handle_no_detection(previous_result)
        else:
            self._handle_detection(best_detection)

        if simulation_mode:
            message = "Simulation mode is active."
            if not self.allow_simulation:
                message = "Simulation mode is active and is not allowed by this profile."
            self.state.inspection_result = SIMULATION
            self.state.pass_fail_bool = None
            self.state.result_message = message

        return self.snapshot()

    def snapshot(self):
        return {
            "inspection_result": self.state.inspection_result,
            "pass_fail_bool": self.state.pass_fail_bool,
            "result_message": self.state.result_message,
            "stable_detected": self.state.stable_detected,
            "raw_detected": self.state.raw_detected,
            "class_name": self.state.class_name,
            "active_class": self.state.class_name,
            "confidence": self.state.confidence,
            "stable_detection_count": self.state.stable_detection_count,
            "detection_frame_count": self.state.detection_frame_count,
            "miss_frame_count": self.state.miss_frame_count,
            "target_classes": sorted(self.target_classes),
            "acceptable_classes": sorted(self.acceptable_classes),
            "reject_classes": sorted(self.reject_classes),
            "minimum_confidence": self.minimum_confidence,
            "allow_simulation": self.allow_simulation,
            "roi_enabled": self.roi_enabled,
            "roi": {
                "x1": self.roi_x1,
                "y1": self.roi_y1,
                "x2": self.roi_x2,
                "y2": self.roi_y2,
            },
        }

    def _handle_detection(self, detection):
        class_name = detection["class_name"]
        confidence = detection["confidence"]
        confidence_value = self._confidence_value(confidence)

        self.state.class_name = class_name
        self.state.confidence = confidence
        self.state.miss_frame_count = 0

        if confidence_value < self.minimum_confidence:
            self.state.detection_frame_count = 0
            self.state.stable_detected = False
            self.state.inspection_result = LOW_CONFIDENCE
            self.state.pass_fail_bool = False
            self.state.result_message = (
                f"Detection confidence is below threshold ({confidence_value:.3f} < "
                f"{self.minimum_confidence:.3f})."
            )
            return

        candidate_result, pass_fail_bool, message = self._class_result(class_name)
        if (
            self.state.candidate_result == candidate_result
            and self.state.candidate_class == class_name
        ):
            self.state.detection_frame_count += 1
        else:
            self.state.candidate_result = candidate_result
            self.state.candidate_class = class_name
            self.state.detection_frame_count = 1

        if self.state.detection_frame_count < self.detection_required_frames:
            self.state.result_message = "Detection observed; waiting for stable confirmation."
            return

        was_stable = self.state.stable_detected
        self.state.stable_detected = True
        self.state.inspection_result = candidate_result
        self.state.pass_fail_bool = pass_fail_bool
        self.state.result_message = message

        if not was_stable:
            self.state.stable_detection_count += 1

    def _handle_no_detection(self, previous_result):
        self.state.miss_frame_count += 1
        self.state.detection_frame_count = 0
        self.state.candidate_result = None
        self.state.candidate_class = None

        if self.state.miss_frame_count >= self.miss_required_frames:
            self.state.stable_detected = False
            self.state.class_name = None
            self.state.confidence = None
            self.state.inspection_result = NO_PART
            self.state.pass_fail_bool = None
            self.state.result_message = "No part detected."
        else:
            self.state.inspection_result = previous_result
            self.state.result_message = "No detection; waiting before clearing inspection state."

    def _set_system_result(self, result, pass_fail_bool, message):
        self.state.raw_detected = False
        self.state.stable_detected = False
        self.state.detection_frame_count = 0
        self.state.miss_frame_count += 1
        self.state.inspection_result = result
        self.state.pass_fail_bool = pass_fail_bool
        self.state.result_message = message

    def _class_result(self, class_name):
        if class_name in self.acceptable_classes:
            return PASS, True, f"Accepted class '{class_name}' detected."

        if class_name in self.reject_classes:
            return FAIL, False, f"Reject class '{class_name}' detected."

        return FAIL, False, f"Class '{class_name}' is not acceptable."

    def _find_best_detection(self, detections, frame_shape):
        best_detection = None

        for detection in detections:
            if not isinstance(detection, dict):
                continue

            class_name = detection.get("class_name")
            if not class_name:
                continue

            if not self._bbox_center_in_roi(detection.get("bbox"), frame_shape):
                continue

            confidence = detection.get("confidence")
            if best_detection is None or self._confidence_value(confidence) > self._confidence_value(
                best_detection["confidence"]
            ):
                best_detection = {
                    "class_name": str(class_name),
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

    @staticmethod
    def _clean_class_set(values):
        return {str(name).strip() for name in values if str(name).strip()}

    @staticmethod
    def _safe_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _confidence_value(confidence):
        try:
            return float(confidence)
        except (TypeError, ValueError):
            return -1.0
