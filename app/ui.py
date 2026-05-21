import time
import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow,
    QVBoxLayout, QWidget, QDoubleSpinBox, QComboBox
)

from app.config import MODELS_DIR, get_available_model_profiles
from app.inference import AIInferenceThread, DEFAULT_CONFIDENCE


class MainWindow(QMainWindow):

    def __init__(self, model_path, camera_index):
        super().__init__()
        self.setWindowTitle("YOLO Detector")

        # Camera
        self.camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.camera.isOpened():
            raise RuntimeError("Camera failed to open")

        # AI
        try: 
            self.ai = AIInferenceThread(model_path)
        except Exception as e:
            raise RuntimeError(f"AI failed to initialize: {e}")
        self.ai.result_ready.connect(self.on_result_ready)

        self.current_model = model_path

        self.frame_count = 0
        self.start_time = time.time()

        self._setup_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def _setup_ui(self):
        central = QWidget()
        self.model_selector = QComboBox()
        self.model_selector.addItems(get_available_model_profiles())
        self.model_selector.currentTextChanged.connect(self.change_model)

        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(self.model_selector)

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

    def change_model(self, profile_name):
        if not profile_name:
            return

        model_path = MODELS_DIR / profile_name / "best.pt" 

        if not model_path.exists():
            print(f"Model not found: {model_path}")
            return

        print(f"Switching to model: {profile_name}")

        self.current_model = str(model_path)

        try:
            self.ai.stop()
        except Exception:
            pass 

        self.ai = AIInferenceThread(self.current_model)
        self.ai.result_ready.connect(self.on_result_ready)

    def closeEvent(self, event):
        self.ai.stop()

        if self.camera.isOpened():
            self.camera.release()

        super().closeEvent(event)