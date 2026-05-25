import threading
import time
import json
import cv2
from ultralytics import YOLO
from PySide6.QtCore import Signal, QObject
from datetime import datetime
from app.config import (
    REVIEW_IMAGES_DIR,
    LOW_CONFIDENCE_THRESHOLD,
    ROI_ENABLED,
    ROI_X1,
    ROI_Y1,
    ROI_X2,
    ROI_Y2,
)
from pathlib import Path

# =============================
# CONFIG
# =============================
DEFAULT_CONFIDENCE = 0.35

RAINBOW = [
    (0, 0, 255), (0, 128, 255), (0, 255, 255),
    (0, 255, 0), (255, 0, 0), (130, 0, 75), (211, 0, 148)
]


def _roi_values():
    x1 = max(0.0, min(1.0, float(ROI_X1)))
    y1 = max(0.0, min(1.0, float(ROI_Y1)))
    x2 = max(0.0, min(1.0, float(ROI_X2)))
    y2 = max(0.0, min(1.0, float(ROI_Y2)))
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _bbox_center_in_roi(bbox, frame_shape):
    if not ROI_ENABLED:
        return True

    h, w = frame_shape[:2]
    if w <= 0 or h <= 0:
        return False

    x1, y1, x2, y2 = _roi_values()
    center_x = ((bbox[0] + bbox[2]) / 2) / w
    center_y = ((bbox[1] + bbox[3]) / 2) / h

    return x1 <= center_x <= x2 and y1 <= center_y <= y2


def draw_roi(frame):
    if not ROI_ENABLED:
        return frame

    output = frame.copy()
    h, w = output.shape[:2]
    x1, y1, x2, y2 = _roi_values()
    left = max(0, min(w - 1, int(round(x1 * w))))
    top = max(0, min(h - 1, int(round(y1 * h))))
    right = max(0, min(w - 1, int(round(x2 * w))))
    bottom = max(0, min(h - 1, int(round(y2 * h))))

    cv2.rectangle(output, (left, top), (right, bottom), (255, 255, 255), 2)
    return output

# =============================
# DETECTION DRAWING
# =============================
def draw_detections(frame, results, target_classes):
    output = draw_roi(frame)

    if not results or len(results[0].boxes) == 0:
        return output, False

    _, w = frame.shape[:2]

    scale = max(0.5, w / 1600)
    thickness = max(1, int(scale * 2))
    found = False

    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    sorted_indices = xyxy[:, 0].argsort()

    for i, idx in enumerate(sorted_indices):
        bbox = xyxy[idx].astype(int)
        conf = confidences[idx]
        cls = classes[idx]

        name = results[0].names.get(cls, "OBJ")
        color = RAINBOW[i % len(RAINBOW)]

        if name in target_classes and _bbox_center_in_roi(bbox, frame.shape):
            found = True

        cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

        label = f"{name} {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )

        text_y = max(text_h + baseline + 5, bbox[1])

        cv2.rectangle(
            output,
            (bbox[0], text_y - text_h - baseline - 5),
            (bbox[0] + text_w + 10, text_y),
            color,
            -1
        )

        cv2.putText(
            output,
            label,
            (bbox[0] + 5, text_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return output, found


def _resolve_profile_dir(model_path):
    path = Path(model_path)
    candidates = [path.parent]

    if path.parent.name == "latest":
        candidates.append(path.parent.parent)
    elif path.parent.parent.name == "versions":
        candidates.append(path.parent.parent.parent)

    for candidate in candidates:
        if (candidate / "config.json").exists() or (candidate / "classes.txt").exists():
            return candidate

    return path.parent


def _load_classes(profile_dir):
    classes_path = profile_dir / "classes.txt"
    if not classes_path.exists():
        return []

    return [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_model_profile(model_path):
    profile_dir = _resolve_profile_dir(model_path)
    config_path = profile_dir / "config.json"
    config = {}

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    classes = _load_classes(profile_dir)
    target_classes = config.get("target_classes") or classes
    target_classes = {str(name).strip() for name in target_classes if str(name).strip()}

    try:
        confidence = float(config.get("confidence", DEFAULT_CONFIDENCE))
    except (TypeError, ValueError):
        confidence = DEFAULT_CONFIDENCE

    return {
        "profile_dir": profile_dir,
        "classes": classes,
        "target_classes": target_classes,
        "confidence": confidence,
    }


# =============================
# AI THREAD
# =============================
class AIInferenceThread(QObject):
    result_ready = Signal(object, bool, object)

    def __init__(self, model_path):
        super().__init__()

        profile = load_model_profile(model_path)
        self.model = YOLO(model_path)
        self.frame = None
        self.confidence = profile["confidence"]
        self.target_classes = profile["target_classes"]
        self.profile_dir = profile["profile_dir"]
        self.running = True

        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            frame_copy = None

            with self.lock:
                if self.frame is not None:
                    frame_copy = self.frame.copy()
                    self.frame = None

            if frame_copy is None:
                time.sleep(0.005)
                continue

            try:
                results = self.model.predict(
                    frame_copy,
                    conf=self.confidence,
                    verbose=False
                )

                annotated, detected = draw_detections(frame_copy, results, self.target_classes)

                detections_found = False
                best_class_name = None
                best_confidence = None
                saved_image_path = ""

                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        class_name = result.names.get(cls_id, str(cls_id))

                        if class_name not in self.target_classes:
                            continue

                        bbox = box.xyxy[0].cpu().numpy()
                        if not _bbox_center_in_roi(bbox, frame_copy.shape):
                            continue

                        detections_found = True

                        confidence = float(box.conf[0])
                        if best_confidence is None or confidence > best_confidence:
                            best_confidence = confidence
                            best_class_name = class_name

                        if confidence < LOW_CONFIDENCE_THRESHOLD:
                            saved_image_path = str(save_review_image(frame_copy, "low_confidence"))

                if not detections_found:
                    saved_image_path = str(save_review_image(frame_copy, "no_detection"))

                metadata = {
                    "class_name": best_class_name,
                    "confidence": best_confidence,
                    "roi_enabled": ROI_ENABLED,
                    "saved_image_path": saved_image_path,
                }
                self.result_ready.emit(annotated, detected, metadata)

            except Exception as e:
                print(f"[ERROR] Inference failed: {e}")

    def submit_frame(self, frame):
        with self.lock:
            self.frame = frame

    def set_confidence(self, value):
        with self.lock:
            self.confidence = value

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)

def save_review_image(frame, reason):
    REVIEW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = REVIEW_IMAGES_DIR / f"{timestamp}_{reason}.jpg"

    cv2.imwrite(str(filename), frame)
    return filename
