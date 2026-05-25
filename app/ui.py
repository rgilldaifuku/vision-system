import time
import cv2
import json
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow,
    QVBoxLayout, QWidget, QDoubleSpinBox, QComboBox, QTabWidget,
    QPushButton, QTextEdit, QLineEdit
)

import subprocess

from app.config import MODELS_DIR, get_available_model_profiles, PROJECT_ROOT
from app.inference import AIInferenceThread, DEFAULT_CONFIDENCE

from pathlib import Path
from datetime import datetime


class TrainingWorker(QThread):
    log = Signal(str)
    finished = Signal(bool)

    def __init__(self, model_name, project_root):
        super().__init__()
        self.model_name = model_name
        self.project_root = project_root

    def run(self):
        process = subprocess.Popen(
            ["python", "training/train_pipeline.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(self.project_root),
            bufsize=1,
        )

        process.stdin.write(self.model_name + "\n")
        process.stdin.flush()
        process.stdin.close()

        for line in process.stdout:
            self.log.emit(line.rstrip())

        process.wait()
        self.finished.emit(process.returncode == 0)                                                                

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

    def _setup_train_tab(self):
        layout = QVBoxLayout(self.train_tab)

        self.train_model_input = QLineEdit()
        self.train_model_input.setPlaceholderText("Enter model/dataset name, example: mouse")

        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.run_training_pipeline)

        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)

        layout.addWidget(self.train_model_input)
        layout.addWidget(self.train_button)
        layout.addWidget(self.train_log)

    def run_training_pipeline(self):
        model_name = self.train_model_input.text().strip()

        if not model_name:
            self.train_log.append("Please enter a model/dataset name.")
            return

        self.train_button.setEnabled(False)
        self.train_log.append(f"Starting training for: {model_name}")

        self.training_worker = TrainingWorker(model_name, PROJECT_ROOT)
        self.training_worker.log.connect(self.train_log.append)
        self.training_worker.finished.connect(self.on_training_finished)
        self.training_worker.start()

    def on_training_finished(self, success):
        self.train_button.setEnabled(True)

        if success:
            self.train_log.append("Training complete")
        else:
            self.train_log.append("Trainign failed")

    def _setup_reports_tab(self):
        layout = QVBoxLayout(self.reports_tab)

        self.report_model_selector = QComboBox()
        self.report_model_selector.addItems(get_available_model_profiles())

        self.refresh_report_button = QPushButton("Load Report")
        self.refresh_report_button.clicked.connect(self.load_model_report)

        self.report_output = QTextEdit()
        self.report_output.setReadOnly(True)

        layout.addWidget(self.report_model_selector)
        layout.addWidget(self.refresh_report_button)
        layout.addWidget(self.report_output)

    def load_model_report(self):
        model_name = self.report_model_selector.currentText()

        if not model_name:
            self.report_output.setText("No model selected.")
            return

        profile_dir = MODELS_DIR / model_name
        config_path = profile_dir / "config.json"
        report_path = profile_dir / "training_report.json"
        versions_dir = profile_dir / "versions"

        output = []
        output.append(f"Model: {model_name}")
        output.append("=" * 40)

        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)

            output.append(f"Latest version: {config.get('latest_version', 'N/A')}")
            output.append(f"Model file: {config.get('model_file', 'N/A')}")
            output.append(f"Target classes: {config.get('target_classes', [])}")
            output.append(f"Confidence: {config.get('confidence', 'N/A')}")
            output.append(f"Updated at: {config.get('updated_at', 'N/A')}")
        else:
            output.append("No config.json found.")

        output.append("")

        if report_path.exists():
            with open(report_path, "r") as f:
                report = json.load(f)

            dataset = report.get("dataset", {})
            training = report.get("training", {})

            output.append("Training Report")
            output.append("-" * 40)
            output.append(f"Trained at: {report.get('trained_at', 'N/A')}")
            output.append(f"Version: {report.get('version', 'N/A')}")
            output.append(f"Train images: {dataset.get('train_images', 'N/A')}")
            output.append(f"Val images: {dataset.get('val_images', 'N/A')}")
            output.append(f"Train labels: {dataset.get('train_labels', 'N/A')}")
            output.append(f"Val labels: {dataset.get('val_labels', 'N/A')}")
            output.append(f"Epochs: {training.get('epochs', 'N/A')}")
            output.append(f"Image size: {training.get('image_size', 'N/A')}")
            output.append(f"Run folder: {training.get('run_dir', 'N/A')}")
        else:
            output.append("No training_report.json found.")

        output.append("")

        if versions_dir.exists():
            versions = [folder.name for folder in versions_dir.iterdir() if folder.is_dir()]
            output.append("Available Versions")
            output.append("-" * 40)
            output.extend(versions if versions else ["No versions found."])

        self.report_output.setText("\n".join(output))

        
    def _setup_collect_tab(self):
        layout = QVBoxLayout(self.collect_tab)

        self.collect_name_input = QLineEdit()
        self.collect_name_input.setPlaceholderText("Dataset/object name, example: mouse")

        self.capture_button = QPushButton("Capture Training Image")
        self.capture_button.clicked.connect(self.capture_training_image)

        self.collect_status = QLabel("Images saved: 0")

        layout.addWidget(self.collect_name_input)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.collect_status)

        self.captured_count = 0

    def capture_training_image(self):
        dataset_name = self.collect_name_input.text().strip().lower().replace(" ", "_")

        if not dataset_name:
            self.collect_status.setText("Please enter a dataset/object name.")
            return

        ret, frame = self.camera.read()

        if not ret or frame is None:
            self.collect_status.setText("Camara frame failed.")
            return
        
        save_dir = PROJECT_ROOT / "data" / "datasets" / dataset_name / "images" / "train"
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = save_dir / f"{dataset_name}_{timestamp}_{self.captured_count:04d}.jpg"

        cv2.imwrite(str(filename), frame)

        self.captured_count += 1
        self.collect_status.setText(f"Images saved: {self.captured_count} | Last: {filename.name}")

    def _setup_ui(self):
        central = QWidget()
        self.model_selector = QComboBox()
        self.model_selector.addItems(get_available_model_profiles())
        self.model_selector.currentTextChanged.connect(self.change_model)

        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.detect_tab = QWidget()
        self.train_tab = QWidget()

        self.tabs.addTab(self.detect_tab, "Detect")
        self.tabs.addTab(self.train_tab, "Train Model")

        self.collect_tab = QWidget()
        self.tabs.addTab(self.collect_tab, "Collect Images")
        self._setup_collect_tab()

        self.reports_tab = QWidget()
        self.tabs.addTab(self.reports_tab, "Reports / Versions")
        self._setup_reports_tab()

    

        self._setup_train_tab()

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