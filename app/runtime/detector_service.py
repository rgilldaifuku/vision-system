import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml
from flask import Flask, Response, jsonify

from app.config import (
    ACTIVE_MODEL_PROFILE,
    CAMERA_INDEX,
    DEFAULT_CONFIDENCE,
    LOW_CONFIDENCE_THRESHOLD,
    MODELS_DIR,
    PROJECT_ROOT,
    REVIEW_IMAGES_DIR,
    DATA_DIR,
    ROI_ENABLED,
    ROI_X1,
    ROI_X2,
    ROI_Y1,
    ROI_Y2,
)
from app.runtime.camera_manager import CameraManager
from app.runtime.camera_sources import SimulatedCameraSource
from app.runtime.inference_engine import (
    InferenceEngine,
    MODEL_FORMAT_AUTO,
    MODEL_FORMAT_NCNN,
    MODEL_FORMAT_PT,
    resolve_model_path,
)
from app.runtime.inspection_logic import InspectionLogic, normalize_class_name
from app.runtime.action_manager import ActionManager
from app.runtime.output_manager import OutputManager
from app.runtime.picamera2_manager import Picamera2CameraManager


BOX_COLORS = {
    "target": (24, 178, 107),
    "other": (240, 180, 41),
}
DEFAULT_IMGSZ = 320
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
DEFAULT_INFERENCE_INTERVAL_MS = 200
DEFAULT_SNAPSHOT_INTERVAL_MS = 1000
DEFAULT_REVIEW_IMAGE_INTERVAL_SECONDS = 5.0
DEBUG_FRAMES_DIR = DATA_DIR / "debug_frames"
PROFILE_CONFIGS_DIR = PROJECT_ROOT / "profiles"
MODEL_STATUS_LOADED = "Loaded"
MODEL_STATUS_ERROR = "Error"
MODEL_STATUS_SIMULATION = "Simulation"
CAMERA_BACKEND_AUTO = "auto"
CAMERA_BACKEND_PICAMERA2 = "picamera2"
CAMERA_BACKEND_OPENCV = "opencv"
CAMERA_BACKEND_SIMULATED = "simulated"


class ProfileConfigError(ValueError):
    """Raised when runtime profile configuration is present but invalid."""


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
        __SNAPSHOT_REFRESH_TEXT__<br>
        Browser updated: <span id="browser-updated">--</span>
      </div>
    </header>

    <section class="status-card primary-status">
      <div id="camera-box" class="status-box warn">
        <div class="label">Camera</div>
        <div id="camera-status" class="value">Loading</div>
      </div>
      <div id="stable-box" class="status-box bad">
        <div class="label">Inspection Result</div>
        <div id="stable-status" class="value">Loading</div>
      </div>
    </section>

    <section id="image-panel" class="image-panel">
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
        <div class="label">Active Class</div>
        <div id="class-name" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Model Status</div>
        <div id="model-status" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Confidence</div>
        <div id="confidence" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Result Message</div>
        <div id="result-message" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Timestamp</div>
        <div id="timestamp" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Camera FPS</div>
        <div id="camera-fps" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Camera Backend</div>
        <div id="camera-backend" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Inference FPS</div>
        <div id="inference-fps" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Last Inference</div>
        <div id="inference-ms" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Runtime Config</div>
        <div id="runtime-config" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Stable Detections</div>
        <div id="total-detections" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Review Images</div>
        <div id="review-images" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Snapshot Mode</div>
        <div id="snapshot-mode" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Runtime Mode</div>
        <div id="runtime-mode" class="value">--</div>
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

    function countValue(value) {
      return value === null || value === undefined || value === "" ? 0 : value;
    }

    function formatRuntimeConfig(status) {
      if (!status.frame_width || !status.frame_height || !status.imgsz) {
        return "--";
      }
      return status.frame_width + "x" + status.frame_height + " / imgsz " + status.imgsz;
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

    function resultState(result) {
      if (result === "PASS") {
        return "ok";
      }
      if (result === "NO_PART" || result === "LOW_CONFIDENCE" || result === "SIMULATION") {
        return "warn";
      }
      return "bad";
    }

    function refreshSnapshot() {
      if (!SNAPSHOT_ENABLED) {
        return;
      }

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
        var rawDetected = Boolean(latest.raw_detected);
        var inspectionResult = latest.inspection_result || "NO_PART";

        setText("camera-status", cameraStatus);
        setBoxState("camera-box", cameraState(cameraStatus));

        setText("stable-status", inspectionResult);
        setBoxState("stable-box", resultState(inspectionResult));

        setText("raw-status", "Raw: " + yesNo(rawDetected));
        setBadgeState("raw-status", rawDetected);

        setText("class-name", latest.active_class || latest.class_name || "--");
        setText("model-status", latest.model_status || status.model_status || "--");
        setText("confidence", formatConfidence(latest.confidence));
        setText("result-message", latest.result_message || "--");
        setText("timestamp", latest.timestamp || "--");
        setText("camera-fps", formatNumber(status.camera_fps, 1, ""));
        setText("camera-backend", status.camera_backend || (status.camera && status.camera.backend) || "--");
        setText("inference-fps", formatNumber(status.inference_fps, 1, ""));
        setText("inference-ms", formatNumber(status.last_inference_ms, 0, " ms"));
        setText("runtime-config", formatRuntimeConfig(status));
        setText("total-detections", countValue(status.total_detections));
        setText(
          "review-images",
          countValue(status.total_images_saved) + " saved / " +
          countValue(status.low_confidence_count) + " low / " +
          countValue(status.no_detection_count) + " none"
        );
        setText("snapshot-mode", status.snapshot_enabled ? "Debug On" : "Off");
        setText("runtime-mode", status.simulation_mode ? "SIMULATION" : "Production");
        setText("profile-name", status.profile_name || latest.profile_name || "--");
        setText("model-path", status.model_path || latest.model_path || "--");
        setText("browser-updated", new Date().toLocaleTimeString());
      } catch (error) {
        setText("camera-status", "API Error");
        setBoxState("camera-box", "bad");
        setText("browser-updated", new Date().toLocaleTimeString());
      }
    }

    var SNAPSHOT_ENABLED = __SNAPSHOT_ENABLED__;
    if (!SNAPSHOT_ENABLED) {
      document.getElementById("image-panel").style.display = "none";
    }

    refreshDashboard();
    if (SNAPSHOT_ENABLED) {
      refreshSnapshot();
    }
    window.setInterval(refreshDashboard, 1000);
    if (SNAPSHOT_ENABLED) {
      window.setInterval(refreshSnapshot, __SNAPSHOT_REFRESH_MS__);
    }
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
        model_format=MODEL_FORMAT_AUTO,
        prefer_edge_model=False,
        camera_index=CAMERA_INDEX,
        camera_source=None,
        camera_backend=CAMERA_BACKEND_AUTO,
        dry_run=False,
        confidence=None,
        detection_required_frames=3,
        miss_required_frames=3,
        imgsz=DEFAULT_IMGSZ,
        frame_width=DEFAULT_FRAME_WIDTH,
        frame_height=DEFAULT_FRAME_HEIGHT,
        inference_interval_ms=DEFAULT_INFERENCE_INTERVAL_MS,
        snapshot_interval_ms=DEFAULT_SNAPSHOT_INTERVAL_MS,
        enable_snapshot=False,
        debug_detections=False,
        save_debug_frames=False,
        debug_frame_limit=20,
    ):
        self.profile_name = profile_name
        self.model_format_requested = model_format or MODEL_FORMAT_AUTO
        self.prefer_edge_model = bool(prefer_edge_model)
        self.camera_source = str(camera_source) if camera_source else ""
        self.camera_backend_requested = camera_backend or CAMERA_BACKEND_AUTO
        self.dry_run = bool(dry_run)
        self.simulation_mode = self.dry_run or bool(self.camera_source)
        self.runtime_mode = "SIMULATION" if self.simulation_mode else "PRODUCTION"
        self.model_status = MODEL_STATUS_LOADED
        self.model_error = ""
        if self.dry_run:
            (
                self.model_path,
                self.model_format,
                self.model_warning,
                self.profile_config,
                self.classes,
            ) = self._resolve_dry_run_profile(profile_name, model_path)
            self.model_status = MODEL_STATUS_SIMULATION
        else:
            try:
                (
                    self.model_path,
                    self.model_format,
                    self.model_warning,
                    self.profile_config,
                    self.classes,
                ) = self._resolve_model(profile_name, model_path)
            except ProfileConfigError:
                raise
            except Exception as exc:
                (
                    self.model_path,
                    self.model_format,
                    self.model_warning,
                    self.profile_config,
                    self.classes,
                ) = self._resolve_model_error_profile(profile_name, model_path)
                self.model_status = MODEL_STATUS_ERROR
                self.model_error = str(exc)
        self.target_classes = self._load_target_classes()
        self.confidence = self._load_confidence(confidence)
        self.inspection_rules = self._load_inspection_rules(
            detection_required_frames=detection_required_frames,
            miss_required_frames=miss_required_frames,
        )
        self.imgsz = max(1, int(imgsz))
        self.frame_width = max(1, int(frame_width))
        self.frame_height = max(1, int(frame_height))
        self.inference_interval_ms = max(0, int(inference_interval_ms))
        self.snapshot_interval_ms = max(0, int(snapshot_interval_ms))
        self.enable_snapshot = bool(enable_snapshot)
        self.debug_detections = bool(debug_detections)
        self.save_debug_frames = bool(save_debug_frames or debug_detections)
        self.debug_frame_limit = max(0, int(debug_frame_limit))
        self.debug_frame_count = 0
        self.inference_interval_seconds = max(0.0, self.inference_interval_ms / 1000)
        self.snapshot_interval_seconds = max(0.0, self.snapshot_interval_ms / 1000)

        self.model = None
        if not self.dry_run and self.model_status != MODEL_STATUS_ERROR:
            try:
                self.model = self._load_yolo_model()
            except Exception as exc:
                self.model_status = MODEL_STATUS_ERROR
                self.model_error = str(exc)
        self.camera = self._create_camera(camera_index, camera_source, self.camera_backend_requested)
        self.inspection = InspectionLogic(
            target_classes=self.target_classes,
            **self.inspection_rules,
        )
        self.output_manager = OutputManager()
        self.action_manager = ActionManager(self.profile_config.get("actions") or {})

        self.running = False
        self.camera_thread = None
        self.inference_thread = None
        self.lock = threading.Lock()
        self.frame_count = 0
        self.camera_frame_count = 0
        self.last_error = ""
        self.started_at = None
        self.latest_detection = self._empty_detection()
        self.latest_snapshot_jpeg = None
        self.latest_snapshot_at = None
        self.last_snapshot_update_time = 0.0
        self.last_camera_frame_time = None
        self.last_inference_time = None
        self.last_inference_run_time = 0.0
        self.latest_camera_frame = None
        self.camera_fps = 0.0
        self.inference_fps = 0.0
        self.runtime_fps = 0.0
        self.last_inference_ms = None
        self.total_detections = 0
        self.total_images_saved = 0
        self.low_confidence_count = 0
        self.no_detection_count = 0
        self.dry_run_inference_count = 0
        self.last_review_image_times = {
            "detections": 0.0,
            "low_confidence": 0.0,
            "no_detection": 0.0,
        }

    def start(self):
        if self.running:
            return

        self.camera.open()
        self.output_manager.log_startup(self.profile_name, self._startup_details())
        self._log_startup_faults()
        self.running = True
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.camera_thread.start()
        self.inference_thread.start()

    def stop(self):
        self.running = False
        for thread in (self.camera_thread, self.inference_thread):
            if thread is not None:
                thread.join(timeout=2.0)
        self.camera.release()

    def _create_camera(self, camera_index, camera_source, camera_backend):
        if camera_source:
            return SimulatedCameraSource(camera_source)

        if camera_backend == CAMERA_BACKEND_PICAMERA2:
            return Picamera2CameraManager(
                frame_width=self.frame_width,
                frame_height=self.frame_height,
            )

        if (
            camera_backend == CAMERA_BACKEND_AUTO
            and camera_index is None
            and Picamera2CameraManager.is_available()
        ):
            return Picamera2CameraManager(
                frame_width=self.frame_width,
                frame_height=self.frame_height,
            )

        return CameraManager(
            camera_index=CAMERA_INDEX if camera_index is None else camera_index,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
        )

    def _load_yolo_model(self):
        return InferenceEngine(self.model_path, self.model_format)

    def _startup_details(self):
        return {
            "profile": self.profile_name,
            "model_path": str(self.model_path),
            "model_format": self.model_format,
            "model_format_requested": self.model_format_requested,
            "prefer_edge_model": self.prefer_edge_model,
            "model_warning": self.model_warning,
            "model_status": self.model_status,
            "camera_source": self.camera_source,
            "camera_backend_requested": self.camera_backend_requested,
            "camera_backend": getattr(self.camera, "backend", self.camera_backend_requested),
            "dry_run": self.dry_run,
            "simulation_mode": self.simulation_mode,
            "runtime_mode": self.runtime_mode,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "imgsz": self.imgsz,
            "inference_interval_ms": self.inference_interval_ms,
            "snapshot_enabled": self.enable_snapshot,
            "debug_detections": self.debug_detections,
            "save_debug_frames": self.save_debug_frames,
            "debug_frame_limit": self.debug_frame_limit,
            "inspection_rules": self.inspection_rules,
        }

    def _log_startup_faults(self):
        if self.model_status == MODEL_STATUS_ERROR:
            self.output_manager.log_fault(
                self.profile_name,
                "model_error",
                self.model_error or "Runtime model is not available.",
                self._startup_details(),
                cooldown_seconds=0,
            )

        if self.camera.status != "Connected":
            self.output_manager.log_fault(
                self.profile_name,
                "camera_error",
                getattr(self.camera, "last_error", "") or f"Camera status is {self.camera.status}.",
                {"camera_status": self.camera.status, "camera_source": self.camera_source},
                cooldown_seconds=0,
            )

    def _camera_connected(self):
        return bool(getattr(self.camera, "connected", self.camera.status == "Connected"))

    def _counter_snapshot(self):
        action_counters = getattr(self.action_manager, "counters", {})
        return {
            "pass": action_counters.get("pass", 0),
            "fail": action_counters.get("fail", 0),
            "low_confidence": action_counters.get("low_confidence", 0),
            "no_part": action_counters.get("no_part", 0),
            "camera_error": action_counters.get("camera_error", 0),
            "model_error": action_counters.get("model_error", 0),
            "simulation": action_counters.get("simulation", 0),
            "stable_detections": self.total_detections,
            "review_images": self.total_images_saved,
            "low_confidence_images": self.low_confidence_count,
            "no_detection_images": self.no_detection_count,
        }

    def get_status(self):
        with self.lock:
            latest = dict(self.latest_detection)
            camera_connected = self._camera_connected()
            camera_backend = getattr(self.camera, "backend", self.camera_backend_requested)
            camera_status = (
                self.camera.get_status()
                if hasattr(self.camera, "get_status")
                else {}
            )
            camera_fps = camera_status.get("fps", self.camera_fps)
            model_loaded = self.model_status == MODEL_STATUS_LOADED
            counters = self._counter_snapshot()
            output_payload = latest.get("output_payload") or getattr(self.output_manager, "last_payload", {})
            return {
                "running": self.running,
                "started_at": self.started_at,
                "profile_name": self.profile_name,
                "model_path": str(self.model_path),
                "model_status": self.model_status,
                "model_error": self.model_error,
                "model_format": self.model_format,
                "model_warning": self.model_warning,
                "model_loaded": model_loaded,
                "camera_source": self.camera_source,
                "camera_backend": camera_backend,
                "camera_connected": camera_connected,
                "dry_run": self.dry_run,
                "runtime_mode": self.runtime_mode,
                "simulation_mode": self.simulation_mode,
                "classes": self.classes,
                "target_classes": sorted(self.target_classes),
                "inspection_rules": self.inspection_rules,
                "confidence": self.confidence,
                "camera_status": self.camera.status,
                "camera_last_error": getattr(self.camera, "last_error", ""),
                "frame_count": self.frame_count,
                "camera_frame_count": self.camera_frame_count,
                "camera_fps": camera_fps,
                "inference_fps": self.inference_fps,
                "runtime_fps": self.runtime_fps,
                "last_inference_ms": self.last_inference_ms,
                "frame_width": self.frame_width,
                "frame_height": self.frame_height,
                "imgsz": self.imgsz,
                "inference_interval_ms": self.inference_interval_ms,
                "snapshot_interval_ms": self.snapshot_interval_ms,
                "snapshot_enabled": self.enable_snapshot,
                "latest_snapshot_at": self.latest_snapshot_at,
                "debug_detections": self.debug_detections,
                "save_debug_frames": self.save_debug_frames,
                "debug_frame_limit": self.debug_frame_limit,
                "debug_frame_count": self.debug_frame_count,
                "review_image_interval_seconds": DEFAULT_REVIEW_IMAGE_INTERVAL_SECONDS,
                "total_detections": self.total_detections,
                "total_images_saved": self.total_images_saved,
                "low_confidence_count": self.low_confidence_count,
                "no_detection_count": self.no_detection_count,
                "logging_last_error": self.output_manager.last_error,
                "last_error": self.last_error,
                "inspection_result": latest.get("inspection_result"),
                "pass_fail_bool": latest.get("pass_fail_bool"),
                "result_message": latest.get("result_message"),
                "class_name": latest.get("class_name"),
                "active_class": latest.get("active_class") or latest.get("class_name"),
                "latest_detection": latest,
                "output_payload": output_payload,
                "runtime": {
                    "profile": self.profile_name,
                    "runtime_mode": self.runtime_mode,
                    "simulation_mode": self.simulation_mode,
                    "running": self.running,
                },
                "camera": {
                    "connected": camera_connected,
                    "backend": camera_backend,
                    "width": self.frame_width,
                    "height": self.frame_height,
                    "fps": camera_fps,
                    "error": camera_status.get("error") or getattr(self.camera, "last_error", "") or None,
                    "last_frame_time": camera_status.get(
                        "last_frame_time",
                        getattr(self.camera, "last_frame_time", None),
                    ),
                },
                "model": {
                    "loaded": model_loaded,
                    "path": str(self.model_path),
                    "imgsz": self.imgsz,
                    "format": self.model_format,
                    "warning": self.model_warning or None,
                    "error": self.model_error or None,
                    "status": self.model_status,
                },
                "inspection": {
                    "result": latest.get("inspection_result"),
                    "message": latest.get("result_message"),
                    "active_class": latest.get("active_class") or latest.get("class_name"),
                    "confidence": latest.get("confidence"),
                    "raw_detection": latest.get("raw_detected"),
                    "stable_detection": latest.get("stable_detected"),
                    "saved_image_path": latest.get("saved_image_path") or None,
                },
                "performance": {
                    "inference_ms": self.last_inference_ms,
                    "inference_fps": self.inference_fps,
                    "camera_fps": camera_fps,
                },
                "counters": counters,
            }

    def get_latest_detection(self):
        with self.lock:
            return dict(self.latest_detection)

    def get_snapshot_jpeg(self):
        if not self.enable_snapshot:
            return None

        with self.lock:
            return self.latest_snapshot_jpeg

    def _camera_loop(self):
        while self.running:
            frame = self.camera.read_frame()

            if frame is None:
                with self.lock:
                    self.latest_camera_frame = None
                time.sleep(0.02)
                continue

            self._update_camera_timing()
            with self.lock:
                self.latest_camera_frame = frame

    def _inference_loop(self):
        while self.running:
            frame = self._get_latest_camera_frame()

            if frame is None:
                if self._inference_due():
                    self._mark_inference_started()
                    detection = self.inspection.update(
                        [],
                        (1, 1, 3),
                        camera_status=self.camera.status,
                        model_status=self.model_status,
                        simulation_mode=self.simulation_mode,
                    )
                    output_payload = self.output_manager.handle_detection(
                        active_profile=self.profile_name,
                        detection=detection,
                        camera_status=self.camera.status,
                        model_status=self.model_status,
                        simulation_mode=self.simulation_mode,
                    )
                    action_result = self._handle_actions(detection, output_payload)
                    output_payload["action_result"] = action_result
                    self._update_latest(detection, output_payload)
                    if detection.get("inspection_result") == "CAMERA_ERROR":
                        self.output_manager.log_fault(
                            self.profile_name,
                            "camera_error",
                            detection.get("result_message", "Camera frame unavailable."),
                            {
                                "camera_status": self.camera.status,
                                "camera_last_error": getattr(self.camera, "last_error", ""),
                            },
                        )
                time.sleep(0.02)
                continue

            if not self._inference_due():
                time.sleep(0.005)
                continue

            try:
                self._mark_inference_started()
                inference_started = time.perf_counter()
                if self.dry_run:
                    detections = self._fake_detections(frame)
                elif self.model_status == MODEL_STATUS_ERROR or self.model is None:
                    detections = []
                    self.output_manager.log_fault(
                        self.profile_name,
                        "model_error",
                        self.model_error or "Runtime model is not available.",
                        {"model_path": str(self.model_path), "model_status": self.model_status},
                    )
                else:
                    results = self.model.predict(
                        frame,
                        confidence=self.confidence,
                        imgsz=self.imgsz,
                    )
                    results, detections = results
                inference_ms = (time.perf_counter() - inference_started) * 1000
                detection = self.inspection.update(
                    detections,
                    frame.shape,
                    camera_status=self.camera.status,
                    model_status=self.model_status,
                    simulation_mode=self.simulation_mode,
                )
                detection = self._handle_review_images(frame, detection)
                self._maybe_write_debug_frame(frame, detections, detection)
                self._maybe_update_snapshot(frame, detections, detection)
                self._update_runtime_timing(inference_ms)
                self.last_error = ""
                output_payload = self.output_manager.handle_detection(
                    active_profile=self.profile_name,
                    detection=detection,
                    camera_status=self.camera.status,
                    model_status=self.model_status,
                    simulation_mode=self.simulation_mode,
                )
                action_result = self._handle_actions(detection, output_payload)
                output_payload["action_result"] = action_result
                self._update_latest(detection, output_payload)
            except Exception as exc:
                self.last_error = str(exc)
                self._update_latest(self.inspection.snapshot())
                self.output_manager.log_fault(
                    self.profile_name,
                    "runtime_exception",
                    str(exc),
                    {"model_status": self.model_status, "camera_status": self.camera.status},
                )
                time.sleep(0.1)

    def _get_latest_camera_frame(self):
        with self.lock:
            if self.latest_camera_frame is None:
                return None
            return self.latest_camera_frame.copy()

    def _inference_due(self):
        if self.inference_interval_seconds <= 0:
            return True
        return time.monotonic() - self.last_inference_run_time >= self.inference_interval_seconds

    def _mark_inference_started(self):
        self.last_inference_run_time = time.monotonic()

    def _handle_actions(self, detection, output_payload):
        status_document = self._build_latest_status_document(detection, output_payload)
        return self.action_manager.handle(status_document)

    def _build_latest_status_document(self, detection, output_payload):
        camera_backend = getattr(self.camera, "backend", self.camera_backend_requested)
        return {
            "timestamp": output_payload.get("timestamp") or datetime.now().isoformat(timespec="seconds"),
            "profile": self.profile_name,
            "runtime_mode": self.runtime_mode,
            "simulation_mode": self.simulation_mode,
            "camera_backend": camera_backend,
            "camera_connected": self._camera_connected(),
            "camera_status": self.camera.status,
            "camera_error": getattr(self.camera, "last_error", "") or None,
            "model_loaded": self.model_status == MODEL_STATUS_LOADED,
            "model_status": self.model_status,
            "model_path": str(self.model_path),
            "model_format": self.model_format,
            "model_warning": self.model_warning or None,
            "model_error": self.model_error or None,
            "inspection_result": detection.get("inspection_result", "NO_PART"),
            "pass_fail_bool": detection.get("pass_fail_bool"),
            "active_class": detection.get("active_class") or detection.get("class_name"),
            "confidence": detection.get("confidence"),
            "message": detection.get("result_message", ""),
            "saved_image_path": detection.get("saved_image_path") or None,
            "counters": self._counter_snapshot(),
            "inference_ms": self.last_inference_ms,
            "inference_fps": self.inference_fps,
            "camera_fps": self.camera_fps,
            "output_payload": output_payload,
        }

    def _update_latest(self, detection, output_payload=None):
        latest = {
            **detection,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "profile_name": self.profile_name,
            "model_path": str(self.model_path),
            "model_status": self.model_status,
            "model_error": self.model_error,
            "model_format": self.model_format,
            "model_warning": self.model_warning,
            "camera_source": self.camera_source,
            "camera_backend": getattr(self.camera, "backend", self.camera_backend_requested),
            "camera_connected": self._camera_connected(),
            "dry_run": self.dry_run,
            "runtime_mode": self.runtime_mode,
            "simulation_mode": self.simulation_mode,
            "camera_status": self.camera.status,
            "saved_image_path": detection.get("saved_image_path", ""),
        }
        latest["output_payload"] = output_payload or OutputManager.build_output_payload(
            active_profile=self.profile_name,
            detection=latest,
            camera_status=self.camera.status,
            model_status=self.model_status,
            simulation_mode=self.simulation_mode,
        )

        with self.lock:
            self.latest_detection = latest

    def _update_camera_timing(self):
        now = time.perf_counter()

        with self.lock:
            if self.last_camera_frame_time is not None:
                elapsed = now - self.last_camera_frame_time
                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    if self.camera_fps > 0:
                        self.camera_fps = (self.camera_fps * 0.85) + (instant_fps * 0.15)
                    else:
                        self.camera_fps = instant_fps

            self.last_camera_frame_time = now
            self.camera_frame_count += 1

    def _update_runtime_timing(self, inference_ms):
        now = time.perf_counter()

        with self.lock:
            if self.last_inference_time is not None:
                elapsed = now - self.last_inference_time
                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    if self.inference_fps > 0:
                        self.inference_fps = (self.inference_fps * 0.85) + (instant_fps * 0.15)
                    else:
                        self.inference_fps = instant_fps

            self.last_inference_time = now
            self.last_inference_ms = inference_ms
            self.runtime_fps = self.inference_fps
            self.frame_count += 1

    def _handle_review_images(self, frame, detection):
        detection = dict(detection)
        detection["saved_image_path"] = ""
        self._update_detection_count(detection)

        saved_paths = []

        if detection.get("stable_detected") and detection.get("raw_detected"):
            saved_paths.append(
                self._save_review_image(
                    frame,
                    "detections",
                    detection.get("class_name"),
                    detection.get("confidence"),
                )
            )

        if self._is_low_confidence_detection(detection):
            saved_paths.append(
                self._save_review_image(
                    frame,
                    "low_confidence",
                    detection.get("class_name"),
                    detection.get("confidence"),
                )
            )

        if (
            not detection.get("raw_detected")
            and detection.get("inspection_result", "NO_PART") == "NO_PART"
        ):
            saved_paths.append(self._save_review_image(frame, "no_detection"))

        saved_paths = [str(path) for path in saved_paths if path]
        if saved_paths:
            detection["saved_image_path"] = saved_paths[-1]

        return detection

    def _update_detection_count(self, detection):
        try:
            stable_count = int(detection.get("stable_detection_count", 0))
        except (TypeError, ValueError):
            stable_count = 0

        with self.lock:
            self.total_detections = max(self.total_detections, stable_count)

    def _is_low_confidence_detection(self, detection):
        if not detection.get("raw_detected"):
            return False

        confidence = detection.get("confidence")
        if confidence is None:
            return False

        try:
            return float(confidence) < LOW_CONFIDENCE_THRESHOLD
        except (TypeError, ValueError):
            return False

    def _save_review_image(self, frame, category, class_name=None, confidence=None):
        if not self._review_image_due(category):
            return None

        output_dir = REVIEW_IMAGES_DIR / self.profile_name / category
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = self._review_image_filename(category, class_name, confidence)
        output_path = output_dir / filename

        if not cv2.imwrite(str(output_path), frame):
            return None

        with self.lock:
            self.total_images_saved += 1
            if category == "low_confidence":
                self.low_confidence_count += 1
            elif category == "no_detection":
                self.no_detection_count += 1

        return output_path

    def _review_image_due(self, category):
        now = time.monotonic()
        last_saved = self.last_review_image_times.get(category, 0.0)

        if now - last_saved < DEFAULT_REVIEW_IMAGE_INTERVAL_SECONDS:
            return False

        self.last_review_image_times[category] = now
        return True

    def _maybe_write_debug_frame(self, frame, detections, detection):
        if not self.debug_detections and not self.save_debug_frames:
            return

        if self.debug_frame_limit and self.debug_frame_count >= self.debug_frame_limit:
            return

        self.debug_frame_count += 1
        timestamp = datetime.now()
        stem = timestamp.strftime("%Y%m%d_%H%M%S_%f")

        raw_path = ""
        annotated_path = ""
        if self.save_debug_frames:
            raw_dir = DEBUG_FRAMES_DIR / "raw"
            annotated_dir = DEBUG_FRAMES_DIR / "annotated"
            raw_dir.mkdir(parents=True, exist_ok=True)
            annotated_dir.mkdir(parents=True, exist_ok=True)

            raw_output = raw_dir / f"{stem}.jpg"
            annotated_output = annotated_dir / f"{stem}.jpg"
            if cv2.imwrite(str(raw_output), frame):
                raw_path = str(raw_output)
            if cv2.imwrite(str(annotated_output), self._annotate_frame(frame, detections, detection)):
                annotated_path = str(annotated_output)

        if self.debug_detections:
            DEBUG_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": timestamp.isoformat(timespec="seconds"),
                "frame_path": raw_path,
                "annotated_frame_path": annotated_path,
                "model_path": str(self.model_path),
                "imgsz": self.imgsz,
                "confidence_threshold": self.confidence,
                "raw_detections": self._json_safe(detections),
                "inspection_result": detection.get("inspection_result"),
                "roi_config": {
                    "enabled": self.inspection.roi_enabled,
                    "x1": self.inspection.roi_x1,
                    "y1": self.inspection.roi_y1,
                    "x2": self.inspection.roi_x2,
                    "y2": self.inspection.roi_y2,
                },
                "accepted_classes": sorted(self.inspection.acceptable_classes),
                "reject_classes": sorted(self.inspection.reject_classes),
                "target_classes": sorted(self.target_classes),
                "active_class": detection.get("active_class") or detection.get("class_name"),
                "message": detection.get("result_message", ""),
            }
            debug_log = DEBUG_FRAMES_DIR / "detections_debug.jsonl"
            with open(debug_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {key: RuntimeDetectorService._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [RuntimeDetectorService._json_safe(item) for item in value]
        if hasattr(value, "item"):
            return value.item()
        return value

    @staticmethod
    def _review_image_filename(category, class_name=None, confidence=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        parts = [timestamp, category]

        if class_name:
            parts.append(RuntimeDetectorService._safe_filename_part(class_name))

        if confidence is not None:
            try:
                parts.append(f"conf{float(confidence):.3f}")
            except (TypeError, ValueError):
                pass

        return "_".join(parts) + ".jpg"

    @staticmethod
    def _safe_filename_part(value):
        cleaned = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in str(value).strip()
        )
        return cleaned or "unknown"

    def _maybe_update_snapshot(self, frame, detections, detection):
        if not self.enable_snapshot:
            return

        now = time.monotonic()
        if now - self.last_snapshot_update_time < self.snapshot_interval_seconds:
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
                "model_status": self.model_status,
                "model_error": self.model_error,
                "model_format": self.model_format,
                "model_warning": self.model_warning,
                "camera_source": self.camera_source,
                "camera_backend": getattr(self.camera, "backend", self.camera_backend_requested),
                "camera_connected": self._camera_connected(),
                "dry_run": self.dry_run,
                "runtime_mode": self.runtime_mode,
                "simulation_mode": self.simulation_mode,
                "camera_status": self.camera.status,
                "saved_image_path": "",
            }
        )
        detection["output_payload"] = self.output_manager.build_output_payload(
            active_profile=self.profile_name,
            detection=detection,
            camera_status=self.camera.status,
            model_status=self.model_status,
            simulation_mode=self.simulation_mode,
        )
        return detection

    def _fake_detections(self, frame):
        self.dry_run_inference_count += 1
        if self.dry_run_inference_count % 10 == 0:
            return []

        height, width = frame.shape[:2]
        box_width = max(8, int(width * 0.35))
        box_height = max(8, int(height * 0.35))
        center_x = width // 2
        center_y = height // 2
        x1 = max(0, center_x - box_width // 2)
        y1 = max(0, center_y - box_height // 2)
        x2 = min(width - 1, center_x + box_width // 2)
        y2 = min(height - 1, center_y + box_height // 2)
        class_name = sorted(self.target_classes)[0] if self.target_classes else "simulated_object"

        return [
            {
                "class_id": self.classes.index(class_name) if class_name in self.classes else 0,
                "class_name": class_name,
                "confidence": 0.55,
                "bbox": [x1, y1, x2, y2],
            }
        ]

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
            is_target = normalize_class_name(class_name) in self.target_classes
            color = BOX_COLORS["target"] if is_target else BOX_COLORS["other"]
            label = f"{class_name} {confidence:.2f}" if confidence is not None else class_name

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            self._draw_label(output, label, x1, y1, color)

        status = detection.get("inspection_result") or (
            "Detected" if detection.get("stable_detected") else "Not Detected"
        )
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
        return {normalize_class_name(name) for name in target_classes if str(name).strip()}

    def _load_confidence(self, override):
        if override is not None:
            return float(override)

        try:
            return float(self.profile_config.get("confidence", DEFAULT_CONFIDENCE))
        except (TypeError, ValueError):
            return DEFAULT_CONFIDENCE

    def _load_inspection_rules(self, detection_required_frames, miss_required_frames):
        inspection_config = (
            self.profile_config.get("inspection")
            or self.profile_config.get("inspection_rules")
            or {}
        )
        if not isinstance(inspection_config, dict):
            raise ProfileConfigError(
                f"Invalid inspection config for profile '{self.profile_name}': inspection must be an object."
            )

        roi_config = inspection_config.get("roi") or {}
        if not isinstance(roi_config, dict):
            raise ProfileConfigError(
                f"Invalid inspection config for profile '{self.profile_name}': roi must be an object."
            )

        acceptable_classes = (
            inspection_config.get("acceptable_classes")
            or inspection_config.get("pass_classes")
            or self.profile_config.get("acceptable_classes")
            or self.target_classes
            or self.classes
        )
        reject_classes = (
            inspection_config.get("reject_classes")
            or inspection_config.get("fail_classes")
            or self.profile_config.get("reject_classes")
            or []
        )
        minimum_confidence = (
            inspection_config.get("minimum_confidence")
            or inspection_config.get("min_confidence")
            or self.confidence
        )
        required_frames = (
            inspection_config.get("required_consecutive_detections")
            or inspection_config.get("detection_required_frames")
            or detection_required_frames
        )
        allowed_no_detection = (
            inspection_config.get("allowed_no_detection_frames")
            or inspection_config.get("miss_required_frames")
            or miss_required_frames
        )

        try:
            return {
                "acceptable_classes": self._as_list(acceptable_classes),
                "reject_classes": self._as_list(reject_classes),
                "minimum_confidence": self._as_float(minimum_confidence, "minimum_confidence"),
                "detection_required_frames": self._as_positive_int(
                    required_frames,
                    "required_consecutive_detections",
                ),
                "miss_required_frames": self._as_positive_int(
                    allowed_no_detection,
                    "allowed_no_detection_frames",
                ),
                "roi_enabled": self._as_bool(
                    roi_config.get("enabled", roi_config.get("roi_enabled", ROI_ENABLED))
                ),
                "roi_x1": self._as_float(roi_config.get("x1", roi_config.get("roi_x1", ROI_X1)), "roi.x1"),
                "roi_y1": self._as_float(roi_config.get("y1", roi_config.get("roi_y1", ROI_Y1)), "roi.y1"),
                "roi_x2": self._as_float(roi_config.get("x2", roi_config.get("roi_x2", ROI_X2)), "roi.x2"),
                "roi_y2": self._as_float(roi_config.get("y2", roi_config.get("roi_y2", ROI_Y2)), "roi.y2"),
                "allow_simulation": self._as_bool(inspection_config.get("allow_simulation", True)),
            }
        except (TypeError, ValueError) as exc:
            raise ProfileConfigError(
                f"Invalid inspection config for profile '{self.profile_name}': {exc}"
            ) from exc

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    @staticmethod
    def _as_bool(value):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _as_float(value, name):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc

    @staticmethod
    def _as_positive_int(value, name):
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an integer") from exc

        if number < 1:
            raise ValueError(f"{name} must be 1 or greater")
        return number

    def _resolve_model(self, profile_name, model_path):
        profile_dir = MODELS_DIR / profile_name

        if not profile_dir.exists():
            raise FileNotFoundError(f"Model profile not found: {profile_dir}")

        config = self._load_profile_config(profile_dir)
        classes = self._load_classes(profile_dir)

        path, model_format, warning = resolve_model_path(
            profile_dir,
            config=config,
            model_override=model_path,
            model_format=self.model_format_requested,
            prefer_edge_model=self.prefer_edge_model,
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Runtime model not found for profile '{profile_name}': {path}"
            )

        return path, model_format, warning, config, classes

    def _resolve_model_error_profile(self, profile_name, model_path):
        profile_dir = MODELS_DIR / profile_name
        config = self._load_profile_config(profile_dir) if profile_dir.exists() else {}
        classes = self._load_classes(profile_dir) if profile_dir.exists() else []

        if model_path:
            path = Path(model_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
        else:
            path = profile_dir / "MODEL_ERROR"

        return path, self.model_format_requested, "", config, classes

    def _resolve_dry_run_profile(self, profile_name, model_path):
        profile_dir = MODELS_DIR / profile_name
        config = {}
        classes = []

        if profile_dir.exists():
            config = self._load_profile_config(profile_dir)
            classes = self._load_classes(profile_dir)

        if not classes:
            classes = ["simulated_object"]

        if not config.get("target_classes"):
            config["target_classes"] = [classes[0]]
        config.setdefault("confidence", 0.5)

        if model_path:
            path = Path(model_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
        else:
            configured_model = config.get("model_file")
            if configured_model and profile_dir.exists():
                path = profile_dir / configured_model
            else:
                path = profile_dir / "latest" / "best.pt"
                if profile_dir.exists() and not path.exists():
                    path = profile_dir / "best.pt"

        if not path.exists():
            path = profile_dir / "DRY_RUN_NO_MODEL"

        return path, self.model_format_requested, "", config, classes

    @staticmethod
    def _load_profile_config(profile_dir):
        config = {}
        config_path = profile_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except json.JSONDecodeError as exc:
                raise ProfileConfigError(f"Invalid profile config JSON: {config_path}: {exc}") from exc

        yaml_path = PROFILE_CONFIGS_DIR / profile_dir.name / "config.yaml"
        if yaml_path.exists():
            try:
                yaml_config = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                raise ProfileConfigError(f"Invalid runtime profile YAML: {yaml_path}: {exc}") from exc

            if not isinstance(yaml_config, dict):
                raise ProfileConfigError(f"Runtime profile YAML must be an object: {yaml_path}")
            config = RuntimeDetectorService._deep_merge(config, yaml_config)

        return config

    @staticmethod
    def _deep_merge(base, override):
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = RuntimeDetectorService._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

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
        snapshot_interval_ms = getattr(service, "snapshot_interval_ms", DEFAULT_SNAPSHOT_INTERVAL_MS)
        snapshot_interval_seconds = snapshot_interval_ms / 1000
        snapshot_enabled = bool(getattr(service, "enable_snapshot", False))
        snapshot_refresh_text = (
            f"Image refresh: {snapshot_interval_seconds:g} seconds"
            if snapshot_enabled
            else "Live image: disabled"
        )
        return (
            DASHBOARD_HTML
            .replace("__SNAPSHOT_REFRESH_MS__", str(snapshot_interval_ms))
            .replace("__SNAPSHOT_REFRESH_TEXT__", snapshot_refresh_text)
            .replace("__SNAPSHOT_ENABLED__", "true" if snapshot_enabled else "false")
        )

    @app.get("/status")
    def status():
        return jsonify(service.get_status())

    @app.get("/latest_detection")
    def latest_detection():
        return jsonify(service.get_latest_detection())

    @app.get("/snapshot.jpg")
    def snapshot():
        if not getattr(service, "enable_snapshot", False):
            return "Snapshot disabled", 404

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
    camera_index_env = os.getenv("VISION_CAMERA_INDEX")
    parser.add_argument("--profile", default=os.getenv("VISION_MODEL_PROFILE", ACTIVE_MODEL_PROFILE))
    parser.add_argument("--model", default=os.getenv("VISION_MODEL_PATH"))
    parser.add_argument(
        "--model-format",
        choices=(MODEL_FORMAT_AUTO, MODEL_FORMAT_PT, MODEL_FORMAT_NCNN),
        default=os.getenv("VISION_MODEL_FORMAT", MODEL_FORMAT_AUTO),
        help="Model format to load. Use auto for .pt files or exported NCNN model folders.",
    )
    parser.add_argument(
        "--prefer-edge-model",
        action="store_true",
        default=os.getenv("VISION_PREFER_EDGE_MODEL", "").lower() in {"1", "true", "yes", "on"},
        help="Prefer exported edge models such as best_ncnn_model/ when available.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=int(camera_index_env) if camera_index_env else None,
        help="OpenCV/USB camera index. Passing this keeps legacy USB camera behavior.",
    )
    parser.add_argument(
        "--camera-source",
        default=os.getenv("VISION_CAMERA_SOURCE"),
        help="Use an image, image folder, or video file instead of a physical camera.",
    )
    parser.add_argument(
        "--camera-backend",
        choices=(CAMERA_BACKEND_AUTO, CAMERA_BACKEND_PICAMERA2, CAMERA_BACKEND_OPENCV),
        default=os.getenv("VISION_CAMERA_BACKEND", CAMERA_BACKEND_AUTO),
        help=(
            "Physical camera backend. auto uses --camera-source first, then Picamera2 if "
            "available, then OpenCV."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.getenv("VISION_DRY_RUN", "").lower() in {"1", "true", "yes", "on"},
        help="Run without a YOLO model using simulated detections for dashboard/runtime testing.",
    )
    parser.add_argument("--host", default=os.getenv("VISION_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("VISION_PORT", "8000")))
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument(
        "--confidence-threshold-override",
        type=float,
        default=None,
        help="Temporarily override YOLO confidence for low-threshold debugging.",
    )
    parser.add_argument("--detection-required-frames", type=int, default=3)
    parser.add_argument("--miss-required-frames", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("VISION_IMGSZ", DEFAULT_IMGSZ)))
    parser.add_argument(
        "--frame-width",
        type=int,
        default=int(os.getenv("VISION_FRAME_WIDTH", DEFAULT_FRAME_WIDTH)),
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=int(os.getenv("VISION_FRAME_HEIGHT", DEFAULT_FRAME_HEIGHT)),
    )
    parser.add_argument(
        "--inference-interval-ms",
        type=int,
        default=int(os.getenv("VISION_INFERENCE_INTERVAL_MS", DEFAULT_INFERENCE_INTERVAL_MS)),
    )
    parser.add_argument(
        "--snapshot-interval-ms",
        type=int,
        default=int(os.getenv("VISION_SNAPSHOT_INTERVAL_MS", DEFAULT_SNAPSHOT_INTERVAL_MS)),
    )
    parser.add_argument(
        "--enable-snapshot",
        action="store_true",
        default=os.getenv("VISION_ENABLE_SNAPSHOT", "").lower() in {"1", "true", "yes", "on"},
    )
    parser.add_argument(
        "--debug-detections",
        action="store_true",
        default=os.getenv("VISION_DEBUG_DETECTIONS", "").lower() in {"1", "true", "yes", "on"},
        help="Write raw YOLO detection debug JSONL records under data/debug_frames/.",
    )
    parser.add_argument(
        "--save-debug-frames",
        action="store_true",
        default=os.getenv("VISION_SAVE_DEBUG_FRAMES", "").lower() in {"1", "true", "yes", "on"},
        help="Save raw and annotated debug frames under data/debug_frames/.",
    )
    parser.add_argument(
        "--debug-frame-limit",
        type=int,
        default=int(os.getenv("VISION_DEBUG_FRAME_LIMIT", "20")),
        help="Maximum number of debug frames/records to write; 0 means unlimited.",
    )
    args = parser.parse_args()
    confidence = (
        args.confidence_threshold_override
        if args.confidence_threshold_override is not None
        else args.confidence
    )

    try:
        service = RuntimeDetectorService(
            profile_name=args.profile,
            model_path=args.model,
            model_format=args.model_format,
            prefer_edge_model=args.prefer_edge_model,
            camera_index=args.camera,
            camera_source=args.camera_source,
            camera_backend=args.camera_backend,
            dry_run=args.dry_run,
            confidence=confidence,
            detection_required_frames=args.detection_required_frames,
            miss_required_frames=args.miss_required_frames,
            imgsz=args.imgsz,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            inference_interval_ms=args.inference_interval_ms,
            snapshot_interval_ms=args.snapshot_interval_ms,
            enable_snapshot=args.enable_snapshot,
            debug_detections=args.debug_detections,
            save_debug_frames=args.save_debug_frames,
            debug_frame_limit=args.debug_frame_limit,
        )
    except Exception as exc:
        print(f"Runtime failed to initialize: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    service.start()

    try:
        create_app(service).run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        service.stop()


if __name__ == "__main__":
    main()
