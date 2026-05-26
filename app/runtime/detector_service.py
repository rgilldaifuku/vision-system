import argparse
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify
from ultralytics import YOLO

from app.config import (
    ACTIVE_MODEL_PROFILE,
    CAMERA_INDEX,
    DEFAULT_CONFIDENCE,
    MODELS_DIR,
    PROJECT_ROOT,
)
from app.runtime.camera_manager import CameraManager
from app.runtime.inspection_logic import InspectionLogic
from app.runtime.output_manager import OutputManager


BOX_COLORS = {
    "target": (24, 178, 107),
    "other": (240, 180, 41),
}
SNAPSHOT_REFRESH_MS = 2500
SNAPSHOT_UPDATE_INTERVAL_SECONDS = SNAPSHOT_REFRESH_MS / 1000


DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vision System Runtime</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101418;
      --panel: #1b2229;
      --panel-strong: #242d36;
      --text: #f4f7fb;
      --muted: #9eabb8;
      --ok: #18b26b;
      --warn: #f0b429;
      --bad: #e55353;
      --line: #33404d;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
    }

    .dashboard {
      width: min(1100px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0;
    }

    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 22px;
    }

    h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 46px);
      font-weight: 700;
      letter-spacing: 0;
    }

    .subtitle {
      color: var(--muted);
      font-size: 16px;
      margin-top: 6px;
    }

    .heartbeat {
      color: var(--muted);
      text-align: right;
      font-size: 15px;
    }

    .status-card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 18px;
    }

    .primary-status {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }

    .status-box {
      background: var(--panel-strong);
      border-radius: 8px;
      padding: 22px;
      min-height: 140px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      border-left: 8px solid var(--line);
    }

    .status-box.ok {
      border-left-color: var(--ok);
    }

    .status-box.bad {
      border-left-color: var(--bad);
    }

    .status-box.warn {
      border-left-color: var(--warn);
    }

    .label {
      color: var(--muted);
      font-size: 15px;
      text-transform: uppercase;
      letter-spacing: 0;
      margin-bottom: 10px;
    }

    .value {
      font-size: clamp(32px, 6vw, 62px);
      font-weight: 700;
      line-height: 1;
      word-break: break-word;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }

    .detail {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      min-height: 94px;
    }

    .detail .value {
      font-size: 24px;
      line-height: 1.2;
    }

    .model-path {
      font-size: 18px;
      color: var(--muted);
      word-break: break-all;
      margin-top: 8px;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 120px;
      border-radius: 999px;
      padding: 8px 14px;
      font-size: 20px;
      font-weight: 700;
      background: #3a4652;
    }

    .badge.ok {
      background: var(--ok);
      color: #04140c;
    }

    .badge.bad {
      background: var(--bad);
      color: #210606;
    }

    .badge.warn {
      background: var(--warn);
      color: #1f1502;
    }

    .image-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 18px;
    }

    .image-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }

    .snapshot {
      display: block;
      width: 100%;
      max-height: 58vh;
      object-fit: contain;
      background: #05080b;
      border: 1px solid var(--line);
      border-radius: 8px;
    }

    .snapshot-status {
      color: var(--muted);
      font-size: 15px;
      text-align: right;
    }

    @media (max-width: 760px) {
      header,
      .primary-status,
      .grid {
        grid-template-columns: 1fr;
      }

      header {
        display: block;
      }

      .heartbeat {
        text-align: left;
        margin-top: 12px;
      }

      .image-header {
        display: block;
      }

      .snapshot-status {
        text-align: left;
        margin-top: 8px;
      }
    }
  </style>
</head>
<body>
  <main class="dashboard">
    <header>
      <div>
        <h1>Vision System</h1>
        <div class="subtitle">Raspberry Pi runtime dashboard</div>
      </div>
      <div class="heartbeat">
        Status refresh: 1 second<br>
        Image refresh: 2.5 seconds<br>
        Browser updated: <span id="browser-updated">--</span>
      </div>
    </header>

    <section class="status-card primary-status">
      <div id="camera-box" class="status-box warn">
        <div class="label">Camera</div>
        <div id="camera-status" class="value">Loading</div>
      </div>
      <div id="stable-box" class="status-box bad">
        <div class="label">Stable Detection</div>
        <div id="stable-status" class="value">Loading</div>
      </div>
    </section>

    <section class="image-panel">
      <div class="image-header">
        <div class="label">Live Annotated Camera Frame</div>
        <div id="snapshot-status" class="snapshot-status">Waiting for frame</div>
      </div>
      <img id="snapshot" class="snapshot" alt="Latest annotated camera frame">
    </section>

    <section class="grid">
      <div class="detail">
        <div class="label">Raw Detection</div>
        <div id="raw-status" class="badge warn">Loading</div>
      </div>
      <div class="detail">
        <div class="label">Last Class</div>
        <div id="class-name" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Confidence</div>
        <div id="confidence" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Timestamp</div>
        <div id="timestamp" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Runtime FPS</div>
        <div id="runtime-fps" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Last Inference</div>
        <div id="inference-ms" class="value">--</div>
      </div>
    </section>

    <section class="status-card">
      <div class="label">Active Profile / Model</div>
      <div id="profile-name" class="value">--</div>
      <div id="model-path" class="model-path">--</div>
    </section>
  </main>

  <script>
    function setText(id, value) {
      document.getElementById(id).textContent = value || "--";
    }

    function boolText(value) {
      return value ? "Detected" : "Not Detected";
    }

    function yesNo(value) {
      return value ? "Yes" : "No";
    }

    function setBoxState(id, state) {
      var el = document.getElementById(id);
      el.classList.remove("ok", "bad", "warn");
      el.classList.add(state);
    }

    function setBadgeState(id, value) {
      var el = document.getElementById(id);
      el.classList.remove("ok", "bad", "warn");
      el.classList.add(value ? "ok" : "bad");
    }

    function formatConfidence(value) {
      if (value === null || value === undefined || value === "") {
        return "--";
      }
      var numberValue = Number(value);
      if (Number.isNaN(numberValue)) {
        return String(value);
      }
      return numberValue.toFixed(3);
    }

    function formatNumber(value, digits, suffix) {
      if (value === null || value === undefined || value === "") {
        return "--";
      }
      var numberValue = Number(value);
      if (Number.isNaN(numberValue)) {
        return String(value);
      }
      return numberValue.toFixed(digits) + suffix;
    }

    function cameraState(status) {
      if (status === "Connected") {
        return "ok";
      }
      if (status === "Reconnecting") {
        return "warn";
      }
      return "bad";
    }

    function refreshSnapshot() {
      var image = document.getElementById("snapshot");
      image.onload = function() {
        setText("snapshot-status", "Live image updated");
      };
      image.onerror = function() {
        setText("snapshot-status", "Waiting for camera frame");
      };
      image.src = "/snapshot.jpg?t=" + Date.now();
    }

    async function refreshDashboard() {
      try {
        var responses = await Promise.all([
          fetch("/status", { cache: "no-store" }),
          fetch("/latest_detection", { cache: "no-store" })
        ]);

        if (!responses[0].ok || !responses[1].ok) {
          throw new Error("API request failed");
        }

        var status = await responses[0].json();
        var latest = await responses[1].json();

        var cameraStatus = latest.camera_status || status.camera_status || "Unknown";
        var stableDetected = Boolean(latest.stable_detected);
        var rawDetected = Boolean(latest.raw_detected);

        setText("camera-status", cameraStatus);
        setBoxState("camera-box", cameraState(cameraStatus));

        setText("stable-status", boolText(stableDetected));
        setBoxState("stable-box", stableDetected ? "ok" : "bad");

        setText("raw-status", "Raw: " + yesNo(rawDetected));
        setBadgeState("raw-status", rawDetected);

        setText("class-name", latest.class_name || "--");
        setText("confidence", formatConfidence(latest.confidence));
        setText("timestamp", latest.timestamp || "--");
        setText("runtime-fps", formatNumber(status.runtime_fps, 1, ""));
        setText("inference-ms", formatNumber(status.last_inference_ms, 0, " ms"));
        setText("profile-name", status.profile_name || latest.profile_name || "--");
        setText("model-path", status.model_path || latest.model_path || "--");
        setText("browser-updated", new Date().toLocaleTimeString());
      } catch (error) {
        setText("camera-status", "API Error");
        setBoxState("camera-box", "bad");
        setText("browser-updated", new Date().toLocaleTimeString());
      }
    }

    refreshDashboard();
    refreshSnapshot();
    window.setInterval(refreshDashboard, 1000);
    window.setInterval(refreshSnapshot, 2500);
  </script>
</body>
</html>
"""


class RuntimeDetectorService:
    """Continuous detection service for Raspberry Pi deployment."""

    def __init__(
        self,
        profile_name=ACTIVE_MODEL_PROFILE,
        model_path=None,
        camera_index=CAMERA_INDEX,
        confidence=None,
        detection_required_frames=3,
        miss_required_frames=3,
    ):
        self.profile_name = profile_name
        self.model_path, self.profile_config, self.classes = self._resolve_model(profile_name, model_path)
        self.target_classes = self._load_target_classes()
        self.confidence = self._load_confidence(confidence)

        self.model = YOLO(str(self.model_path))
        self.camera = CameraManager(camera_index=camera_index)
        self.inspection = InspectionLogic(
            target_classes=self.target_classes,
            detection_required_frames=detection_required_frames,
            miss_required_frames=miss_required_frames,
        )
        self.output_manager = OutputManager()

        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.frame_count = 0
        self.last_error = ""
        self.started_at = None
        self.latest_detection = self._empty_detection()
        self.latest_snapshot_jpeg = None
        self.latest_snapshot_at = None
        self.last_snapshot_update_time = 0.0
        self.last_frame_time = None
        self.runtime_fps = 0.0
        self.last_inference_ms = None

    def start(self):
        if self.running:
            return

        self.camera.open()
        self.running = True
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.camera.release()

    def get_status(self):
        with self.lock:
            return {
                "running": self.running,
                "started_at": self.started_at,
                "profile_name": self.profile_name,
                "model_path": str(self.model_path),
                "classes": self.classes,
                "target_classes": sorted(self.target_classes),
                "confidence": self.confidence,
                "camera_status": self.camera.status,
                "frame_count": self.frame_count,
                "runtime_fps": self.runtime_fps,
                "last_inference_ms": self.last_inference_ms,
                "latest_snapshot_at": self.latest_snapshot_at,
                "last_error": self.last_error,
                "latest_detection": self.latest_detection,
            }

    def get_latest_detection(self):
        with self.lock:
            return dict(self.latest_detection)

    def get_snapshot_jpeg(self):
        with self.lock:
            return self.latest_snapshot_jpeg

    def _run_loop(self):
        while self.running:
            frame = self.camera.read_frame()

            if frame is None:
                detection = self.inspection.update([], (1, 1, 3))
                self._update_latest(detection)
                self.output_manager.handle_detection(
                    active_profile=self.profile_name,
                    detection=detection,
                    camera_status=self.camera.status,
                )
                time.sleep(0.1)
                continue

            try:
                inference_started = time.perf_counter()
                results = self.model.predict(frame, conf=self.confidence, verbose=False)
                inference_ms = (time.perf_counter() - inference_started) * 1000
                detections = self._extract_detections(results)
                detection = self.inspection.update(detections, frame.shape)
                self._maybe_update_snapshot(frame, detections, detection)
                self._update_runtime_timing(inference_ms)
                self.last_error = ""
                self._update_latest(detection)
                self.output_manager.handle_detection(
                    active_profile=self.profile_name,
                    detection=detection,
                    camera_status=self.camera.status,
                )
            except Exception as exc:
                self.last_error = str(exc)
                self._update_latest(self.inspection.snapshot())
                time.sleep(0.1)

    def _update_latest(self, detection):
        latest = {
            **detection,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "profile_name": self.profile_name,
            "model_path": str(self.model_path),
            "camera_status": self.camera.status,
            "saved_image_path": detection.get("saved_image_path", ""),
        }

        with self.lock:
            self.latest_detection = latest

    def _update_runtime_timing(self, inference_ms):
        now = time.perf_counter()

        with self.lock:
            if self.last_frame_time is not None:
                elapsed = now - self.last_frame_time
                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    if self.runtime_fps > 0:
                        self.runtime_fps = (self.runtime_fps * 0.85) + (instant_fps * 0.15)
                    else:
                        self.runtime_fps = instant_fps

            self.last_frame_time = now
            self.last_inference_ms = inference_ms
            self.frame_count += 1

    def _maybe_update_snapshot(self, frame, detections, detection):
        now = time.monotonic()
        if now - self.last_snapshot_update_time < SNAPSHOT_UPDATE_INTERVAL_SECONDS:
            return

        self.last_snapshot_update_time = now
        self._update_snapshot(self._annotate_frame(frame, detections, detection))

    def _update_snapshot(self, frame):
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return

        with self.lock:
            self.latest_snapshot_jpeg = buffer.tobytes()
            self.latest_snapshot_at = datetime.now().isoformat(timespec="seconds")

    def _empty_detection(self):
        detection = self.inspection.snapshot()
        detection.update(
            {
                "timestamp": None,
                "profile_name": self.profile_name,
                "model_path": str(self.model_path),
                "camera_status": self.camera.status,
                "saved_image_path": "",
            }
        )
        return detection

    def _extract_detections(self, results):
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = result.names.get(cls_id, str(cls_id))
                bbox = box.xyxy[0].cpu().numpy().tolist()
                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": float(box.conf[0]),
                        "bbox": bbox,
                    }
                )
        return detections

    def _annotate_frame(self, frame, detections, detection):
        output = frame.copy()
        height, width = output.shape[:2]

        self._draw_roi(output, width, height)

        for item in detections:
            bbox = item.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [int(round(value)) for value in bbox]
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            class_name = item.get("class_name", "object")
            confidence = item.get("confidence")
            is_target = class_name in self.target_classes
            color = BOX_COLORS["target"] if is_target else BOX_COLORS["other"]
            label = f"{class_name} {confidence:.2f}" if confidence is not None else class_name

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            self._draw_label(output, label, x1, y1, color)

        status = "Detected" if detection.get("stable_detected") else "Not Detected"
        raw = "Raw: Yes" if detection.get("raw_detected") else "Raw: No"
        cv2.putText(output, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(output, raw, (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return output

    def _draw_roi(self, frame, width, height):
        if not self.inspection.roi_enabled:
            return

        x1 = int(round(self.inspection.roi_x1 * width))
        y1 = int(round(self.inspection.roi_y1 * height))
        x2 = int(round(self.inspection.roi_x2 * width))
        y2 = int(round(self.inspection.roi_y2 * height))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    @staticmethod
    def _draw_label(frame, label, x, y, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
        top = max(0, y - text_height - baseline - 8)
        bottom = top + text_height + baseline + 8
        right = min(frame.shape[1] - 1, x + text_width + 10)

        cv2.rectangle(frame, (x, top), (right, bottom), color, -1)
        cv2.putText(
            frame,
            label,
            (x + 5, bottom - baseline - 4),
            font,
            scale,
            (255, 255, 255),
            thickness,
        )

    def _load_target_classes(self):
        target_classes = self.profile_config.get("target_classes") or self.classes
        return {str(name).strip() for name in target_classes if str(name).strip()}

    def _load_confidence(self, override):
        if override is not None:
            return float(override)

        try:
            return float(self.profile_config.get("confidence", DEFAULT_CONFIDENCE))
        except (TypeError, ValueError):
            return DEFAULT_CONFIDENCE

    def _resolve_model(self, profile_name, model_path):
        profile_dir = MODELS_DIR / profile_name
        config = self._load_profile_config(profile_dir)
        classes = self._load_classes(profile_dir)

        if model_path:
            path = Path(model_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
        else:
            configured_model = config.get("model_file")
            if configured_model:
                path = profile_dir / configured_model
            else:
                path = profile_dir / "latest" / "best.pt"
                if not path.exists():
                    path = profile_dir / "best.pt"

        if not path.exists():
            raise FileNotFoundError(f"Runtime model not found: {path}")

        return path, config, classes

    @staticmethod
    def _load_profile_config(profile_dir):
        config_path = profile_dir / "config.json"
        if not config_path.exists():
            return {}

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_classes(profile_dir):
        classes_path = profile_dir / "classes.txt"
        if not classes_path.exists():
            return []

        return [
            line.strip()
            for line in classes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def create_app(service):
    app = Flask(__name__)

    @app.get("/")
    def dashboard():
        return DASHBOARD_HTML

    @app.get("/status")
    def status():
        return jsonify(service.get_status())

    @app.get("/latest_detection")
    def latest_detection():
        return jsonify(service.get_latest_detection())

    @app.get("/snapshot.jpg")
    def snapshot():
        snapshot_jpeg = service.get_snapshot_jpeg()
        if snapshot_jpeg is None:
            return "No snapshot available", 503

        return Response(
            snapshot_jpeg,
            mimetype="image/jpeg",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi runtime detection service")
    parser.add_argument("--profile", default=os.getenv("VISION_MODEL_PROFILE", ACTIVE_MODEL_PROFILE))
    parser.add_argument("--model", default=os.getenv("VISION_MODEL_PATH"))
    parser.add_argument("--camera", type=int, default=int(os.getenv("VISION_CAMERA_INDEX", CAMERA_INDEX)))
    parser.add_argument("--host", default=os.getenv("VISION_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("VISION_PORT", "8000")))
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--detection-required-frames", type=int, default=3)
    parser.add_argument("--miss-required-frames", type=int, default=3)
    args = parser.parse_args()

    service = RuntimeDetectorService(
        profile_name=args.profile,
        model_path=args.model,
        camera_index=args.camera,
        confidence=args.confidence,
        detection_required_frames=args.detection_required_frames,
        miss_required_frames=args.miss_required_frames,
    )
    service.start()

    try:
        create_app(service).run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        service.stop()


if __name__ == "__main__":
    main()
