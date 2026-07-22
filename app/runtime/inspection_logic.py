from collections import deque
from dataclasses import dataclass

from app.config import DEFAULT_CONFIDENCE, ROI_ENABLED, ROI_X1, ROI_Y1, ROI_X2, ROI_Y2
from app.runtime.inspection_result import canonical_state_from_result, generate_inspection_id


PASS = "PASS"
FAIL = "FAIL"
NO_PART = "NO_PART"
LOW_CONFIDENCE = "LOW_CONFIDENCE"
CAMERA_ERROR = "CAMERA_ERROR"
MODEL_ERROR = "MODEL_ERROR"
SIMULATION = "SIMULATION"


def normalize_class_name(name):
    return str(name).strip().lower().replace(" ", "_")


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
    inspection_id: str = ""
    inspection_state: str = NO_PART
    average_confidence: float | None = None
    agreement_ratio: float | None = None


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
        decision_mode="consecutive",
        rolling_window_size=8,
        rolling_min_agreeing=6,
        rolling_min_agreement_ratio=0.75,
        rolling_min_average_confidence=None,
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
        self.decision_mode = str(decision_mode or "consecutive").strip().lower()
        if self.decision_mode not in {"consecutive", "rolling_window"}:
            self.decision_mode = "consecutive"
        self.rolling_window_size = max(1, int(rolling_window_size))
        self.rolling_min_agreeing = max(1, int(rolling_min_agreeing))
        self.rolling_min_agreement_ratio = max(0.0, min(1.0, float(rolling_min_agreement_ratio)))
        self.rolling_min_average_confidence = self._safe_float(
            rolling_min_average_confidence,
            self.minimum_confidence,
        )
        self.rolling_window = deque(maxlen=self.rolling_window_size)
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
            self._finalize_canonical_fields()
            return self.snapshot()

        if model_status not in {"Loaded", "Simulation"}:
            self._set_system_result(MODEL_ERROR, False, f"Model status is {model_status}.")
            self._finalize_canonical_fields()
            return self.snapshot()

        raw_detected, best_detection = self._find_best_detection(detections or [], frame_shape)
        self.state.raw_detected = raw_detected

        if self.decision_mode == "rolling_window":
            self._handle_rolling(raw_detected, best_detection, previous_result)
        elif not raw_detected:
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

        self._finalize_canonical_fields()
        return self.snapshot()

    def snapshot(self):
        if not self.state.inspection_id:
            self._finalize_canonical_fields()
        return {
            "inspection_result": self.state.inspection_result,
            "inspection_id": self.state.inspection_id,
            "inspection_state": self.state.inspection_state,
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
            "decision_mode": self.decision_mode,
            "average_confidence": self.state.average_confidence,
            "agreement_ratio": self.state.agreement_ratio,
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
            self.state.average_confidence = confidence_value if confidence_value >= 0 else None
            self.state.agreement_ratio = 1.0
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
        self.state.average_confidence = self._confidence_value(self.state.confidence)
        self.state.agreement_ratio = 1.0

    def _handle_rolling(self, raw_detected, detection, previous_result):
        if not raw_detected:
            self.state.raw_detected = False
            self.state.miss_frame_count += 1
            self.state.detection_frame_count = 0
            self.rolling_window.append(
                {
                    "result": NO_PART,
                    "pass_fail_bool": None,
                    "class_name": None,
                    "confidence": None,
                    "message": "No part detected.",
                }
            )
            if self.state.miss_frame_count < self.miss_required_frames:
                self.state.inspection_result = previous_result
                self.state.result_message = "No detection; waiting before clearing inspection state."
                return
        else:
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            confidence_value = self._confidence_value(confidence)
            self.state.raw_detected = True
            self.state.class_name = class_name
            self.state.confidence = confidence
            self.state.miss_frame_count = 0

            if confidence_value < self.minimum_confidence:
                candidate_result = LOW_CONFIDENCE
                pass_fail_bool = False
                message = (
                    f"Detection confidence is below threshold ({confidence_value:.3f} < "
                    f"{self.minimum_confidence:.3f})."
                )
            else:
                candidate_result, pass_fail_bool, message = self._class_result(class_name)

            self.rolling_window.append(
                {
                    "result": candidate_result,
                    "pass_fail_bool": pass_fail_bool,
                    "class_name": class_name,
                    "confidence": confidence,
                    "message": message,
                }
            )

        dominant = self._dominant_rolling_candidate()
        if dominant is None:
            self.state.stable_detected = False
            self.state.inspection_result = "REVIEW" if raw_detected else previous_result
            self.state.pass_fail_bool = False if raw_detected else self.state.pass_fail_bool
            self.state.result_message = "Waiting for enough rolling-window agreement."
            return

        self.state.inspection_result = dominant["result"]
        self.state.pass_fail_bool = dominant["pass_fail_bool"]
        self.state.class_name = dominant["class_name"]
        self.state.confidence = dominant["confidence"]
        self.state.average_confidence = dominant["average_confidence"]
        self.state.agreement_ratio = dominant["agreement_ratio"]
        self.state.result_message = dominant["message"]
        self.state.stable_detected = dominant["result"] in {PASS, FAIL}
        if self.state.stable_detected:
            self.state.detection_frame_count += 1
            if self.state.detection_frame_count == 1:
                self.state.stable_detection_count += 1

    def _dominant_rolling_candidate(self):
        if not self.rolling_window:
            return None

        counts = {}
        for item in self.rolling_window:
            counts[item["result"]] = counts.get(item["result"], 0) + 1

        result, count = max(counts.items(), key=lambda item: item[1])
        agreement_ratio = count / len(self.rolling_window)
        if count < self.rolling_min_agreeing or agreement_ratio < self.rolling_min_agreement_ratio:
            return None

        agreeing = [item for item in self.rolling_window if item["result"] == result]
        confidences = [
            self._confidence_value(item.get("confidence"))
            for item in agreeing
            if item.get("confidence") is not None
        ]
        valid_confidences = [value for value in confidences if value >= 0]
        average_confidence = (
            sum(valid_confidences) / len(valid_confidences) if valid_confidences else None
        )
        if (
            result in {PASS, FAIL}
            and average_confidence is not None
            and average_confidence < self.rolling_min_average_confidence
        ):
            return None

        latest = agreeing[-1]
        return {
            "result": result,
            "pass_fail_bool": latest.get("pass_fail_bool"),
            "class_name": latest.get("class_name"),
            "confidence": latest.get("confidence"),
            "message": latest.get("message"),
            "average_confidence": (
                round(float(average_confidence), 6) if average_confidence is not None else None
            ),
            "agreement_ratio": round(float(agreement_ratio), 6),
        }

    def _finalize_canonical_fields(self):
        canonical_state = canonical_state_from_result(self.state.inspection_result).value
        identity = (
            canonical_state,
            self.state.inspection_result,
            self.state.class_name,
            self.state.confidence,
            self.state.stable_detected,
        )
        previous_identity = getattr(self, "_last_decision_identity", None)
        if not self.state.inspection_id or identity != previous_identity:
            self.state.inspection_id = generate_inspection_id()
            self._last_decision_identity = identity
        self.state.inspection_state = canonical_state

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
            self.state.average_confidence = None
            self.state.agreement_ratio = 1.0
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
        self.state.average_confidence = None
        self.state.agreement_ratio = 1.0

    def _class_result(self, class_name):
        normalized_class = normalize_class_name(class_name)

        if normalized_class in self.acceptable_classes:
            return PASS, True, f"Accepted class '{class_name}' detected."

        if normalized_class in self.reject_classes:
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
        return {normalize_class_name(name) for name in values if str(name).strip()}

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
