import argparse
import threading
import time
import cv2
from ultralytics import YOLO

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow,
    QVBoxLayout, QWidget, QDoubleSpinBox
)

# =============================
# CONFIG (REMOVE HARDCODING)
# =============================
TARGET_CLASS_ID = 0   # Replace with your actual class index
DEFAULT_CONFIDENCE = 0.35

RAINBOW = [
    (0, 0, 255), (0, 128, 255), (0, 255, 255),
    (0, 255, 0), (255, 0, 0), (130, 0, 75), (211, 0, 148)
]


# =============================
# DETECTION LOGIC
# =============================
def draw_detections(frame, results):
    if not results or len(results[0].boxes) == 0:
        return frame, False

    output = frame.copy()
    h, w = frame.shape[:2]

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

        # ✅ Fix: Use class ID instead of string match
        if cls == TARGET_CLASS_ID:
            found = True

        # Draw box
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


# =============================
# AI THREAD
# =============================
class AIInferenceThread(QObject):
    result_ready = Signal(object, bool)

    def __init__(self, model_path):
        super().__init__()

        self.model = YOLO(model_path)
        self.frame = None
        self.confidence = DEFAULT_CONFIDENCE
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

                annotated, detected = draw_detections(frame_copy, results)
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


# =============================
# MAIN WINDOW
# =============================
class MainWindow(QMainWindow):

    def __init__(self, model_path, camera_index):
        super().__init__()
        self.setWindowTitle("YOLO Detector")

        # Camera setup
        self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.camera.isOpened():
            raise RuntimeError("Camera failed to open")

        # AI thread
        self.ai = AIInferenceThread(model_path)
        self.ai.result_ready.connect(self.on_result_ready)

        self.frame_count = 0
        self.start_time = time.time()

        self._setup_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS cap

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.display = QLabel("Starting camera...")
        self.display.setMinimumSize(960, 540)
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setStyleSheet("background:#111; border:1px solid #444;")
        layout.addWidget(self.display)

        control = QHBoxLayout()

        self.fps_label = QLabel("FPS: 0")
        control.addWidget(self.fps_label)

        control.addStretch()

        conf_label = QLabel("Confidence")
        control.addWidget(conf_label)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.05, 0.99)
        self.conf_spin.setValue(DEFAULT_CONFIDENCE)
        self.conf_spin.valueChanged.connect(self.ai.set_confidence)
        control.addWidget(self.conf_spin)

        control.addStretch()

        self.status_label = QLabel("Not Detected")
        self.status_label.setFixedSize(150, 50)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("background:#222; color:#888;")
        control.addWidget(self.status_label)

        layout.addLayout(control)

    def update_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            print("[WARNING] Camera frame failed")
            return

        self.ai.submit_frame(frame)

        # FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time

        if elapsed >= 1.0:
            fps = int(self.frame_count / elapsed)
            self.fps_label.setText(f"FPS: {fps}")
            self.frame_count = 0
            self.start_time = time.time()

    def on_result_ready(self, frame, detected):
        if detected:
            self.status_label.setText("Detected")
            self.status_label.setStyleSheet("background:#0f0; color:#000;")
        else:
            self.status_label.setText("Not Detected")
            self.status_label.setStyleSheet("background:#222; color:#888;")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.display.width(),
            self.display.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.display.setPixmap(pix)

    def closeEvent(self, event):
        self.ai.stop()

        if self.camera.isOpened():
            self.camera.release()

        super().closeEvent(event)


# =============================
# ENTRY POINT
# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--camera", type=int, default=0)

    args = parser.parse_args()

    app = QApplication([])
    window = MainWindow(args.model, args.camera)
    window.show()
    app.exec()