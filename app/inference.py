import threading
import time
import json
import cv2
from ultralytics import YOLO
from PySide6.QtCore import Signal, QObject
from app.logging import log_detection
from datetime import datetime
from app.config import REVIEW_IMAGES_DIR, LOW_CONFIDENCE_THRESHOLD
from pathlib import Path

# =============================
# CONFIG
# =============================
DEFAULT_CONFIDENCE = 0.35

RAINBOW = [
    (0, 0, 255), (0, 128, 255), (0, 255, 255),
    (0, 255, 0), (255, 0, 0), (130, 0, 75), (211, 0, 148)
]

# =============================
# DETECTION DRAWING
# =============================
def draw_detections(frame, results, target_classes):
    if not results or len(results[0].boxes) == 0:
        return frame, False

    output = frame.copy()
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

        if name in target_classes:
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
    result_ready = Signal(object, bool)

    def __init__(self, model_path):
        super().__init__()

        profile = load_model_profile(model_path)
        self.model = YOLO(model_path)
        self.frame = None
        self.confidence = profile["confidence"]
        self.target_classes = profile["target_classes"]
        self.profile_dir = profile["profile_dir"]
        self.running = True

        self.last_log_time = 0
        self.log_cooldown_seconds = 2

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

                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        class_name = result.names.get(cls_id, str(cls_id))

                        if class_name not in self.target_classes:
                            continue

                        detections_found = True

                        confidence = float(box.conf[0])

                        if confidence < LOW_CONFIDENCE_THRESHOLD:
                            save_review_image(frame_copy,"low_confidence")

                if not detections_found:
                    save_review_image(frame_copy, "no_detection")

                current_time = time.time()

                if current_time - self.last_log_time >= self.log_cooldown_seconds:
                    log_detection(detected)
                    self.last_log_time = current_time

                self.result_ready.emit(annotated, detected)

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
