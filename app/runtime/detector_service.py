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
from app.runtime.camera_profile import CameraProfileError, load_camera_profile
from app.runtime.camera_sources import SimulatedCameraSource
from app.runtime.inference_engine import (
    InferenceEngine,
    MODEL_FORMAT_AUTO,
    MODEL_FORMAT_NCNN,
    MODEL_FORMAT_PT,
    model_input_size_warning,
    resolve_model_path,
)
from app.runtime.image_quality import (
    GOOD as IMAGE_QUALITY_GOOD,
    QUALITY_CHECK_ERROR,
    compute_image_quality,
    quality_thresholds_from_config,
)
from app.runtime.inspection_logic import InspectionLogic, normalize_class_name
from app.runtime.preprocessing import apply_camera_transforms, apply_roi, standardize_frame
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
MODEL_STATUS_DISABLED = "Disabled"
INSPECTION_RESULT_CAMERA_ONLY = "CAMERA_ONLY"
INSPECTION_RESULT_INFERENCE_DISABLED = "INFERENCE_DISABLED"
INSPECTION_RESULT_IMAGE_QUALITY_ERROR = "IMAGE_QUALITY_ERROR"
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
      <div class="detail">
        <div class="label">Image Quality</div>
        <div id="quality-status" class="badge warn">--</div>
      </div>
      <div class="detail">
        <div class="label">Brightness</div>
        <div id="quality-brightness" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Blur Score</div>
        <div id="quality-blur" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Contrast</div>
        <div id="quality-contrast" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Exposure</div>
        <div id="quality-exposure" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">ROI</div>
        <div id="roi-status" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Camera Profile</div>
        <div id="camera-profile-ui" class="value">--</div>
      </div>
      <div class="detail">
        <div class="label">Preprocessing</div>
        <div id="preprocessing-status" class="value">--</div>
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
      if (
        result === "NO_PART" ||
        result === "LOW_CONFIDENCE" ||
        result === "SIMULATION" ||
        result === "CAMERA_ONLY" ||
        result === "INFERENCE_DISABLED"
      ) {
        return "warn";
      }
      return "bad";
    }

    function resultDisplay(result) {
      if (result === "CAMERA_ONLY") {
        return "CAMERA ONLY MODE";
      }
      if (result === "INFERENCE_DISABLED") {
        return "INFERENCE DISABLED";
      }
      return result || "--";
    }

    function qualityState(status) {
      if (status === "GOOD" || status === "DISABLED") {
        return "ok";
      }
      if (status === "NO_FRAME") {
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

        setText("stable-status", resultDisplay(inspectionResult));
        setBoxState("stable-box", resultState(inspectionResult));

        setText("raw-status", "Raw: " + yesNo(rawDetected));
        setBadgeState("raw-status", rawDetected);

        setText("class-name", latest.active_class || latest.class_name || "--");
        var modelText = latest.model_status || status.model_status || "--";
        if (status.inference_disabled || (status.model && status.model.inference_disabled)) {
          modelText = "Disabled";
        }
        setText("model-status", modelText);
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
        setText("runtime-mode", status.runtime_mode || (status.simulation_mode ? "SIMULATION" : "Production"));
        var quality = status.image_quality || latest.image_quality || {};
        var preprocessing = status.preprocessing || latest.preprocessing || {};
        var cameraProfile = status.camera_profile_details || {};
        var qualityStatus = quality.quality_status || "--";
        setText("quality-status", qualityStatus);
        setBoxState("quality-status", qualityState(qualityStatus));
        setText("quality-brightness", formatNumber(quality.brightness_mean, 1, ""));
        setText("quality-blur", formatNumber(quality.blur_score, 1, ""));
        setText("quality-contrast", formatNumber(quality.contrast_score, 1, ""));
        setText(
          "quality-exposure",
          formatNumber(quality.overexposed_pct, 1, "% over") + " / " +
          formatNumber(quality.underexposed_pct, 1, "% under")
        );
        setText("roi-status", preprocessing.roi_enabled ? "Enabled" : "Disabled");
        setText("camera-profile-ui", cameraProfile.name || status.camera_profile || "--");
        setText("preprocessing-status", preprocessing.preprocessing_enabled ? "Enabled" : "Disabled");
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
        camera_profile=None,
        camera_backend=CAMERA_BACKEND_AUTO,
        camera_only=False,
        disable_inference=False,
        dry_run=False,
        confidence=None,
        detection_required_frames=3,
        miss_required_frames=3,
        imgsz=DEFAULT_IMGSZ,
        frame_width=None,
        frame_height=None,
        inference_interval_ms=DEFAULT_INFERENCE_INTERVAL_MS,
        snapshot_interval_ms=DEFAULT_SNAPSHOT_INTERVAL_MS,
        enable_snapshot=False,
        debug_detections=False,
        save_debug_frames=False,
        debug_frame_limit=20,
        debug_capture_on_detection=False,
        debug_dir=None,
        debug_max_captures=5,
    ):
        self.profile_name = profile_name
        self.model_override_path = str(model_path) if model_path else ""
        self.model_format_requested = model_format or MODEL_FORMAT_AUTO
        self.prefer_edge_model = bool(prefer_edge_model)
        self.camera_source = str(camera_source) if camera_source else ""
        self.camera_profile_name = str(camera_profile or "")
        self.camera_profile = self._load_camera_profile(camera_profile)
        self.camera_profile_error = ""
        profile_backend = self.camera_profile.backend if self.camera_profile else CAMERA_BACKEND_AUTO
        requested_backend = camera_backend or CAMERA_BACKEND_AUTO
        if requested_backend == CAMERA_BACKEND_AUTO and profile_backend != CAMERA_BACKEND_AUTO:
            requested_backend = profile_backend
        self.camera_backend_requested = requested_backend
        self.camera_only = bool(camera_only)
        self.disable_inference = bool(disable_inference)
        self.inference_disabled = self.camera_only or self.disable_inference
        self.dry_run = bool(dry_run)
        self.simulation_mode = self.dry_run or bool(self.camera_source)
        if self.camera_only:
            self.runtime_mode = "CAMERA_ONLY"
        elif self.disable_inference:
            self.runtime_mode = "INFERENCE_DISABLED"
        elif self.simulation_mode:
            self.runtime_mode = "SIMULATION"
        else:
            self.runtime_mode = "PRODUCTION"
        self.model_status = MODEL_STATUS_LOADED
        self.model_error = ""
        if self.inference_disabled:
            (
                self.model_path,
                self.model_format,
                self.model_warning,
                self.profile_config,
                self.classes,
            ) = self._resolve_inference_disabled_profile(profile_name, model_path)
            self.model_status = MODEL_STATUS_DISABLED
            self.model_error = "Inference disabled"
        elif self.dry_run:
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
        size_warning = model_input_size_warning(self.model_path, self.imgsz)
        if size_warning:
            self.model_warning = " ".join(
                warning for warning in (self.model_warning, size_warning) if warning
            )
        self.frame_width = max(
            1,
            int(frame_width or (self.camera_profile.width if self.camera_profile else DEFAULT_FRAME_WIDTH)),
        )
        self.frame_height = max(
            1,
            int(frame_height or (self.camera_profile.height if self.camera_profile else DEFAULT_FRAME_HEIGHT)),
        )
        self.inference_interval_ms = max(0, int(inference_interval_ms))
        self.snapshot_interval_ms = max(0, int(snapshot_interval_ms))
        self.enable_snapshot = bool(enable_snapshot)
        self.debug_detections = bool(debug_detections)
        self.save_debug_frames = bool(save_debug_frames or debug_detections)
        self.debug_frame_limit = max(0, int(debug_frame_limit))
        self.debug_frame_count = 0
        self.debug_capture_on_detection = bool(debug_capture_on_detection)
        debug_root = Path(debug_dir) if debug_dir else DEBUG_FRAMES_DIR / "live_detections"
        self.debug_dir = debug_root if debug_root.is_absolute() else PROJECT_ROOT / debug_root
        self.debug_max_captures = max(0, int(debug_max_captures))
        self.debug_capture_count = 0
        self.inference_interval_seconds = max(0.0, self.inference_interval_ms / 1000)
        self.snapshot_interval_seconds = max(0.0, self.snapshot_interval_ms / 1000)
        self.quality_thresholds = quality_thresholds_from_config(
            self.camera_profile.quality if self.camera_profile else None
        )
        self.quality_enabled = bool(
            self.camera_profile.quality.enabled if self.camera_profile else True
        )
        self.skip_inference_on_bad_quality = bool(
            self.camera_profile.quality.skip_inference_on_bad_quality
            if self.camera_profile
            else False
        )

        self.model = None
        if (
            not self.inference_disabled
            and not self.dry_run
            and self.model_status != MODEL_STATUS_ERROR
        ):
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
        self.latest_image_quality = self._empty_image_quality()
        self.latest_preprocessing = self._empty_preprocessing_metadata()
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
        if self.model_warning:
            print(f"WARNING: {self.model_warning}", file=sys.stderr, flush=True)
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
                target_fps=self.camera_profile.fps if self.camera_profile else 30,
            )

        if (
            camera_backend == CAMERA_BACKEND_AUTO
            and camera_index is None
            and Picamera2CameraManager.is_available()
        ):
            return Picamera2CameraManager(
                frame_width=self.frame_width,
                frame_height=self.frame_height,
                target_fps=self.camera_profile.fps if self.camera_profile else 30,
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
            "model_override_path": self.model_override_path or None,
            "model_format": self.model_format,
            "model_format_requested": self.model_format_requested,
            "prefer_edge_model": self.prefer_edge_model,
            "model_warning": self.model_warning,
            "model_status": self.model_status,
            "camera_source": self.camera_source,
            "camera_profile": self.camera_profile_name,
            "camera_profile_config": self.camera_profile.to_dict() if self.camera_profile else None,
            "camera_backend_requested": self.camera_backend_requested,
            "camera_backend": getattr(self.camera, "backend", self.camera_backend_requested),
            "camera_only": self.camera_only,
            "disable_inference": self.disable_inference,
            "inference_disabled": self.inference_disabled,
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
            "debug_capture_on_detection": self.debug_capture_on_detection,
            "debug_dir": str(self.debug_dir),
            "debug_max_captures": self.debug_max_captures,
            "image_quality_enabled": self.quality_enabled,
            "quality_thresholds": self.quality_thresholds,
            "skip_inference_on_bad_quality": self.skip_inference_on_bad_quality,
            "preprocessing": (
                self.camera_profile.to_dict().get("preprocessing") if self.camera_profile else None
            ),
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
            "image_quality_error": action_counters.get("image_quality_error", 0),
            "quality_check_error": action_counters.get("quality_check_error", 0),
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
            image_quality = dict(self.latest_image_quality)
            preprocessing = dict(self.latest_preprocessing)
            camera_profile_payload = self.camera_profile.to_dict() if self.camera_profile else None
            return {
                "running": self.running,
                "started_at": self.started_at,
                "profile_name": self.profile_name,
                "model_path": str(self.model_path),
                "model_override_path": self.model_override_path or None,
                "model_status": self.model_status,
                "model_error": self.model_error,
                "model_format": self.model_format,
                "model_warning": self.model_warning,
                "model_loaded": model_loaded,
                "camera_source": self.camera_source,
                "camera_profile": self.camera_profile_name,
                "camera_profile_config": camera_profile_payload,
                "camera_backend": camera_backend,
                "camera_connected": camera_connected,
                "camera_only": self.camera_only,
                "disable_inference": self.disable_inference,
                "inference_disabled": self.inference_disabled,
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
                "debug_capture_on_detection": self.debug_capture_on_detection,
                "debug_dir": str(self.debug_dir),
                "debug_max_captures": self.debug_max_captures,
                "debug_capture_count": self.debug_capture_count,
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
                "image_quality": image_quality,
                "preprocessing": preprocessing,
                "runtime": {
                    "profile": self.profile_name,
                    "runtime_mode": self.runtime_mode,
                    "simulation_mode": self.simulation_mode,
                    "running": self.running,
                    "camera_only": self.camera_only,
                    "inference_disabled": self.inference_disabled,
                },
                "camera": {
                    "connected": camera_connected,
                    "backend": camera_backend,
                    "profile": self.camera_profile_name or None,
                    "width": self.frame_width,
                    "height": self.frame_height,
                    "fps": camera_fps,
                    "error": camera_status.get("error") or getattr(self.camera, "last_error", "") or None,
                    "last_frame_time": camera_status.get(
                        "last_frame_time",
                        getattr(self.camera, "last_frame_time", None),
                    ),
                },
                "camera_profile_details": {
                    "name": self.camera_profile.name if self.camera_profile else None,
                    "backend": self.camera_profile.backend if self.camera_profile else camera_backend,
                    "width": self.camera_profile.width if self.camera_profile else self.frame_width,
                    "height": self.camera_profile.height if self.camera_profile else self.frame_height,
                    "fps": self.camera_profile.fps if self.camera_profile else None,
                    "quality": (
                        camera_profile_payload.get("quality") if camera_profile_payload else None
                    ),
                    "preprocessing": (
                        camera_profile_payload.get("preprocessing") if camera_profile_payload else None
                    ),
                },
                "model": {
                    "loaded": model_loaded,
                    "path": str(self.model_path),
                    "override": bool(self.model_override_path),
                    "imgsz": self.imgsz,
                    "format": self.model_format,
                    "warning": self.model_warning or None,
                    "error": self.model_error or None,
                    "status": self.model_status,
                    "inference_disabled": self.inference_disabled,
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
                    if self.inference_disabled and self.camera.status == "Connected":
                        detection = self._disabled_inference_detection()
                    else:
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
                processed_frame, preprocessing_metadata, image_quality = self._prepare_runtime_frame(frame)
                if self.inference_disabled:
                    detections = []
                    detection = self._disabled_inference_detection()
                    inference_ms = 0.0
                    detection = self._attach_frame_metadata(detection, image_quality, preprocessing_metadata)
                    self._maybe_update_snapshot(processed_frame, detections, detection)
                    self._update_runtime_timing(inference_ms)
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
                    continue

                if self._should_skip_for_quality(image_quality):
                    detections = []
                    inference_ms = (time.perf_counter() - inference_started) * 1000
                    detection = self._image_quality_error_detection(image_quality)
                    detection = self._attach_frame_metadata(detection, image_quality, preprocessing_metadata)
                    self._maybe_update_snapshot(processed_frame, detections, detection)
                    self._update_runtime_timing(inference_ms)
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
                    continue

                if self.dry_run:
                    detections = self._fake_detections(processed_frame)
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
                        processed_frame,
                        confidence=self.confidence,
                        imgsz=self.imgsz,
                    )
                    results, detections = results
                inference_ms = (time.perf_counter() - inference_started) * 1000
                detection = self.inspection.update(
                    detections,
                    processed_frame.shape,
                    camera_status=self.camera.status,
                    model_status=self.model_status,
                    simulation_mode=self.simulation_mode,
                )
                detection = self._attach_frame_metadata(detection, image_quality, preprocessing_metadata)
                detection = self._handle_review_images(processed_frame, detection, detections)
                self._maybe_capture_detection_debug(
                    raw_frame=frame,
                    inference_frame=processed_frame,
                    detections=detections,
                    detection=detection,
                    image_quality=image_quality,
                    preprocessing_metadata=preprocessing_metadata,
                )
                self._maybe_write_debug_frame(processed_frame, detections, detection)
                self._maybe_update_snapshot(processed_frame, detections, detection)
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

    def _prepare_runtime_frame(self, frame):
        transformed_frame, transform_meta = apply_camera_transforms(frame, self.camera_profile)
        roi_frame, roi_meta = apply_roi(
            transformed_frame,
            getattr(self.camera_profile, "roi", None),
        )
        image_quality = self._compute_quality(roi_frame)
        preprocessing = getattr(self.camera_profile, "preprocessing", None)
        preprocessing_enabled = bool(getattr(preprocessing, "enabled", True))
        color_normalization = bool(getattr(preprocessing, "color_normalization", False))
        if preprocessing_enabled:
            processed_frame, standardize_meta = standardize_frame(
                roi_frame,
                target_size=self.imgsz,
                color_normalization=color_normalization,
            )
        else:
            processed_frame = roi_frame
            standardize_meta = {
                "applied": False,
                "resized_to": None,
                "letterbox": False,
                "scale": 1.0,
                "pad": {"left": 0, "top": 0, "right": 0, "bottom": 0},
                "color_normalization": False,
            }
        preprocessing_metadata = {
            "applied": bool(
                transform_meta.get("transformed")
                or roi_meta.get("applied")
                or standardize_meta.get("applied")
            ),
            "transforms": transform_meta,
            "roi_enabled": bool(roi_meta.get("roi_enabled")),
            "roi_pixels": roi_meta.get("roi_pixels"),
            "roi_normalized": roi_meta.get("roi_normalized"),
            "roi_applied": bool(roi_meta.get("applied")),
            "preprocessing_enabled": preprocessing_enabled,
            "resized_to": standardize_meta.get("resized_to"),
            "letterbox": standardize_meta.get("letterbox"),
            "scale": standardize_meta.get("scale"),
            "pad": standardize_meta.get("pad"),
            "color_normalization": color_normalization,
            "input_shape": self._frame_shape(frame),
            "quality_shape": self._frame_shape(roi_frame),
            "output_shape": self._frame_shape(processed_frame),
        }
        with self.lock:
            self.latest_preprocessing = preprocessing_metadata
            self.latest_image_quality = image_quality
        return processed_frame, preprocessing_metadata, image_quality

    @staticmethod
    def _frame_shape(frame):
        if frame is None or not hasattr(frame, "shape"):
            return None
        height, width = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) > 2 else 1
        return {"height": int(height), "width": int(width), "channels": int(channels)}

    def _compute_quality(self, frame):
        if not self.quality_enabled:
            quality = compute_image_quality(frame, self.quality_thresholds)
            quality["quality_status"] = "DISABLED"
            quality["message"] = "Image quality checks are disabled."
            return quality
        return compute_image_quality(frame, self.quality_thresholds)

    def _should_skip_for_quality(self, image_quality):
        if not self.skip_inference_on_bad_quality:
            return False
        status = (image_quality or {}).get("quality_status")
        return status not in {IMAGE_QUALITY_GOOD, "DISABLED"}

    def _image_quality_error_detection(self, image_quality):
        status = (image_quality or {}).get("quality_status") or QUALITY_CHECK_ERROR
        message = (image_quality or {}).get("message") or f"Image quality status is {status}."
        result = (
            QUALITY_CHECK_ERROR
            if status == QUALITY_CHECK_ERROR
            else INSPECTION_RESULT_IMAGE_QUALITY_ERROR
        )
        return {
            "inspection_result": result,
            "pass_fail_bool": False,
            "result_message": message,
            "stable_detected": False,
            "raw_detected": False,
            "class_name": None,
            "active_class": None,
            "confidence": None,
            "stable_detection_count": 0,
            "detection_frame_count": 0,
            "miss_frame_count": 0,
            "target_classes": sorted(self.target_classes),
            "acceptable_classes": sorted(self.inspection.acceptable_classes),
            "reject_classes": sorted(self.inspection.reject_classes),
            "minimum_confidence": self.inspection.minimum_confidence,
            "allow_simulation": self.inspection.allow_simulation,
            "roi_enabled": self.inspection.roi_enabled,
            "roi": {
                "x1": self.inspection.roi_x1,
                "y1": self.inspection.roi_y1,
                "x2": self.inspection.roi_x2,
                "y2": self.inspection.roi_y2,
            },
            "saved_image_path": "",
        }

    @staticmethod
    def _attach_frame_metadata(detection, image_quality, preprocessing_metadata):
        detection = dict(detection)
        detection["image_quality"] = image_quality or {}
        detection["preprocessing"] = preprocessing_metadata or {}
        return detection

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
            "camera_profile": self.camera_profile_name or None,
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
            "image_quality": detection.get("image_quality") or self.latest_image_quality,
            "preprocessing": detection.get("preprocessing") or self.latest_preprocessing,
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
            "camera_profile": self.camera_profile_name,
            "camera_backend": getattr(self.camera, "backend", self.camera_backend_requested),
            "camera_connected": self._camera_connected(),
            "camera_only": self.camera_only,
            "disable_inference": self.disable_inference,
            "inference_disabled": self.inference_disabled,
            "dry_run": self.dry_run,
            "runtime_mode": self.runtime_mode,
            "simulation_mode": self.simulation_mode,
            "camera_status": self.camera.status,
            "saved_image_path": detection.get("saved_image_path", ""),
            "image_quality": detection.get("image_quality") or dict(self.latest_image_quality),
            "preprocessing": detection.get("preprocessing") or dict(self.latest_preprocessing),
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

    def _handle_review_images(self, frame, detection, raw_detections=None):
        detection = dict(detection)
        detection["saved_image_path"] = ""
        self._update_detection_count(detection)

        saved_paths = []

        if detection.get("stable_detected") and detection.get("raw_detected"):
            saved_paths.append(
                self._save_review_image(
                    frame,
                    "detections",
                    detection,
                    raw_detections,
                    detection.get("class_name"),
                    detection.get("confidence"),
                )
            )

        if self._is_low_confidence_detection(detection):
            saved_paths.append(
                self._save_review_image(
                    frame,
                    "low_confidence",
                    detection,
                    raw_detections,
                    detection.get("class_name"),
                    detection.get("confidence"),
                )
            )

        if (
            not detection.get("raw_detected")
            and detection.get("inspection_result", "NO_PART") == "NO_PART"
        ):
            saved_paths.append(self._save_review_image(frame, "no_detection", detection, raw_detections))

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

    def _save_review_image(
        self,
        frame,
        category,
        detection=None,
        raw_detections=None,
        class_name=None,
        confidence=None,
    ):
        if not self._review_image_due(category):
            return None

        output_dir = REVIEW_IMAGES_DIR / self.profile_name / category
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = self._review_image_filename(category, class_name, confidence)
        output_path = output_dir / filename

        if not cv2.imwrite(str(output_path), frame):
            return None
        self._write_image_sidecar(
            output_path,
            category,
            detection=detection or {},
            raw_detections=raw_detections or [],
        )

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
                self._write_image_sidecar(
                    raw_output,
                    "debug_raw",
                    detection=detection,
                    raw_detections=detections,
                )
            if cv2.imwrite(str(annotated_output), self._annotate_frame(frame, detections, detection)):
                annotated_path = str(annotated_output)
                self._write_image_sidecar(
                    annotated_output,
                    "debug_annotated",
                    detection=detection,
                    raw_detections=detections,
                )

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

    def _maybe_capture_detection_debug(
        self,
        raw_frame,
        inference_frame,
        detections,
        detection,
        image_quality,
        preprocessing_metadata,
    ):
        if not self.debug_capture_on_detection or not detection.get("raw_detected"):
            return None
        if self.debug_max_captures and self.debug_capture_count >= self.debug_max_captures:
            return None

        capture_number = self.debug_capture_count + 1
        timestamp = datetime.now()
        capture_dir = self.debug_dir / (
            f"capture_{capture_number:03d}_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
        )
        try:
            capture_dir.mkdir(parents=True, exist_ok=False)
            raw_path = capture_dir / "raw_camera.png"
            inference_path = capture_dir / "inference_input.png"
            inference_array_path = capture_dir / "inference_input.npy"
            if not cv2.imwrite(str(raw_path), raw_frame):
                raise RuntimeError(f"Could not encode raw camera frame: {raw_path}")
            if not cv2.imwrite(str(inference_path), inference_frame):
                raise RuntimeError(f"Could not encode inference input: {inference_path}")

            import numpy as np

            np.save(inference_array_path, inference_frame, allow_pickle=False)
            metadata = self._build_detection_debug_metadata(
                timestamp=timestamp,
                raw_frame=raw_frame,
                inference_frame=inference_frame,
                raw_path=raw_path,
                inference_path=inference_path,
                inference_array_path=inference_array_path,
                detections=detections,
                detection=detection,
                image_quality=image_quality,
                preprocessing_metadata=preprocessing_metadata,
            )
            metadata_path = capture_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self.debug_capture_count = capture_number
            return metadata_path
        except Exception as exc:
            self.last_error = f"Detection debug capture failed: {exc}"
            return None

    def _build_detection_debug_metadata(
        self,
        timestamp,
        raw_frame,
        inference_frame,
        raw_path,
        inference_path,
        inference_array_path,
        detections,
        detection,
        image_quality,
        preprocessing_metadata,
    ):
        primary = detections[0] if detections else {}
        return {
            "timestamp": timestamp.isoformat(timespec="microseconds"),
            "profile": self.profile_name,
            "model_path": str(self.model_path),
            "model_override_path": self.model_override_path or None,
            "model_format": self.model_format,
            "configured_imgsz": self.imgsz,
            "model_warning": self.model_warning or None,
            "raw_camera_frame": {
                "path": str(raw_path),
                "shape": list(raw_frame.shape),
                "dtype": str(raw_frame.dtype),
                "in_memory_color_space": "BGR",
                "saved_file_color_encoding": "PNG written by OpenCV from BGR array",
            },
            "inference_input": {
                "image_path": str(inference_path),
                "array_path": str(inference_array_path),
                "shape": list(inference_frame.shape),
                "dtype": str(inference_frame.dtype),
                "in_memory_color_space": "BGR",
                "engine_boundary": (
                    "Exact array passed to InferenceEngine.predict before Ultralytics "
                    "backend preprocessing."
                ),
                "image_representation": (
                    "Lossless PNG; scripts/validate_model.py can use this image directly. "
                    "The NPY file preserves the exact in-memory array."
                ),
            },
            "preprocessing": self._json_safe(preprocessing_metadata),
            "image_quality": self._json_safe(image_quality),
            "raw_detections": self._json_safe(detections),
            "primary_detection": {
                "class_id": primary.get("class_id"),
                "class_name": primary.get("class_name"),
                "confidence": primary.get("confidence"),
                "bbox": primary.get("bbox"),
            },
            "inspection": {
                "raw_detected": bool(detection.get("raw_detected")),
                "stable_detected": bool(detection.get("stable_detected")),
                "inspection_result": detection.get("inspection_result"),
                "class_name": detection.get("class_name"),
                "confidence": detection.get("confidence"),
                "message": detection.get("result_message"),
            },
        }

    def _write_image_sidecar(self, image_path, category, detection=None, raw_detections=None):
        detection = detection or {}
        image_quality = detection.get("image_quality") or self.latest_image_quality
        preprocessing = detection.get("preprocessing") or self.latest_preprocessing
        sidecar_path = Path(image_path).with_suffix(".json")
        metadata = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "profile": self.profile_name,
            "camera_profile": self.camera_profile_name or None,
            "model_path": str(self.model_path),
            "model_format": self.model_format,
            "runtime_mode": self.runtime_mode,
            "inspection_result": detection.get("inspection_result"),
            "result_message": detection.get("result_message"),
            "raw_detections": self._json_safe(raw_detections or []),
            "stable_detection": detection.get("stable_detected"),
            "confidence": detection.get("confidence"),
            "class_name": detection.get("class_name"),
            "image_quality": self._json_safe(image_quality),
            "preprocessing": self._json_safe(preprocessing),
            "roi_used": self._json_safe(
                preprocessing.get("roi_normalized")
                if isinstance(preprocessing, dict)
                else None
            ),
            "frame_width": image_quality.get("width") if isinstance(image_quality, dict) else None,
            "frame_height": image_quality.get("height") if isinstance(image_quality, dict) else None,
            "saved_image_category": category,
            "image_path": str(image_path),
        }
        try:
            sidecar_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception as exc:
            self.last_error = f"Could not write image sidecar {sidecar_path}: {exc}"

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

    @staticmethod
    def _empty_image_quality():
        return {
            "brightness_mean": None,
            "brightness_std": None,
            "contrast_score": None,
            "blur_score": None,
            "overexposed_pct": None,
            "underexposed_pct": None,
            "width": 0,
            "height": 0,
            "timestamp": None,
            "quality_status": "NO_FRAME",
            "message": "No frame has been processed yet.",
        }

    @staticmethod
    def _empty_preprocessing_metadata():
        return {
            "applied": False,
            "transforms": {
                "rotation": 0,
                "flip_horizontal": False,
                "flip_vertical": False,
                "transformed": False,
            },
            "roi_enabled": False,
            "roi_pixels": None,
            "roi_normalized": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
            "roi_applied": False,
            "preprocessing_enabled": False,
            "resized_to": None,
            "letterbox": False,
            "scale": 1.0,
            "pad": {"left": 0, "top": 0, "right": 0, "bottom": 0},
            "color_normalization": False,
            "input_shape": None,
            "output_shape": None,
        }

    def _empty_detection(self):
        if self.inference_disabled:
            detection = self._disabled_inference_detection()
        else:
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
                "camera_profile": self.camera_profile_name,
                "camera_backend": getattr(self.camera, "backend", self.camera_backend_requested),
                "camera_connected": self._camera_connected(),
                "camera_only": self.camera_only,
                "disable_inference": self.disable_inference,
                "inference_disabled": self.inference_disabled,
                "dry_run": self.dry_run,
                "runtime_mode": self.runtime_mode,
                "simulation_mode": self.simulation_mode,
                "camera_status": self.camera.status,
                "saved_image_path": "",
                "image_quality": dict(self.latest_image_quality),
                "preprocessing": dict(self.latest_preprocessing),
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

    def _disabled_inference_detection(self):
        if self.camera_only:
            result = INSPECTION_RESULT_CAMERA_ONLY
            message = "Camera-only mode active. Inference disabled."
        else:
            result = INSPECTION_RESULT_INFERENCE_DISABLED
            message = "Inference disabled."

        return {
            "inspection_result": result,
            "pass_fail_bool": None,
            "result_message": message,
            "stable_detected": False,
            "raw_detected": False,
            "class_name": None,
            "active_class": None,
            "confidence": None,
            "stable_detection_count": 0,
            "detection_frame_count": 0,
            "miss_frame_count": 0,
            "target_classes": sorted(self.target_classes),
            "acceptable_classes": sorted(self.inspection.acceptable_classes),
            "reject_classes": sorted(self.inspection.reject_classes),
            "minimum_confidence": self.inspection.minimum_confidence,
            "allow_simulation": self.inspection.allow_simulation,
            "roi_enabled": self.inspection.roi_enabled,
            "roi": {
                "x1": self.inspection.roi_x1,
                "y1": self.inspection.roi_y1,
                "x2": self.inspection.roi_x2,
                "y2": self.inspection.roi_y2,
            },
            "saved_image_path": "",
        }

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

    @staticmethod
    def _load_camera_profile(camera_profile):
        if not camera_profile:
            return None

        try:
            return load_camera_profile(camera_profile)
        except CameraProfileError:
            raise
        except Exception as exc:
            raise CameraProfileError(f"Could not load camera profile '{camera_profile}': {exc}") from exc

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
        camera_roi = self.camera_profile.roi if self.camera_profile else None

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
                    roi_config.get(
                        "enabled",
                        roi_config.get(
                            "roi_enabled",
                            camera_roi.enabled if camera_roi is not None else ROI_ENABLED,
                        ),
                    )
                ),
                "roi_x1": self._as_float(
                    roi_config.get(
                        "x1",
                        roi_config.get("roi_x1", camera_roi.x1 if camera_roi is not None else ROI_X1),
                    ),
                    "roi.x1",
                ),
                "roi_y1": self._as_float(
                    roi_config.get(
                        "y1",
                        roi_config.get("roi_y1", camera_roi.y1 if camera_roi is not None else ROI_Y1),
                    ),
                    "roi.y1",
                ),
                "roi_x2": self._as_float(
                    roi_config.get(
                        "x2",
                        roi_config.get("roi_x2", camera_roi.x2 if camera_roi is not None else ROI_X2),
                    ),
                    "roi.x2",
                ),
                "roi_y2": self._as_float(
                    roi_config.get(
                        "y2",
                        roi_config.get("roi_y2", camera_roi.y2 if camera_roi is not None else ROI_Y2),
                    ),
                    "roi.y2",
                ),
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

    def _resolve_inference_disabled_profile(self, profile_name, model_path):
        profile_dir = MODELS_DIR / profile_name
        config = self._load_profile_config(profile_dir) if profile_dir.exists() else {}
        classes = self._load_classes(profile_dir) if profile_dir.exists() else []

        if not classes:
            classes = ["camera_only"]
        if not config.get("target_classes"):
            config["target_classes"] = [classes[0]]
        config.setdefault("confidence", DEFAULT_CONFIDENCE)

        if model_path:
            path = Path(model_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
        else:
            path = profile_dir / "INFERENCE_DISABLED"

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


def create_parser():
    parser = argparse.ArgumentParser(description="Raspberry Pi runtime detection service")
    camera_index_env = os.getenv("VISION_CAMERA_INDEX")
    parser.add_argument("--profile", default=os.getenv("VISION_MODEL_PROFILE", ACTIVE_MODEL_PROFILE))
    parser.add_argument(
        "--model",
        "--model-path",
        dest="model_path",
        default=os.getenv("VISION_MODEL_PATH"),
        help="Explicit model file or exported model folder. --model is retained as an alias.",
    )
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
        "--camera-profile",
        default=os.getenv("VISION_CAMERA_PROFILE"),
        help="Camera profile name from cameras/ or path to a camera YAML profile.",
    )
    parser.add_argument(
        "--camera-only",
        action="store_true",
        default=os.getenv("VISION_CAMERA_ONLY", "").lower() in {"1", "true", "yes", "on"},
        help="Start camera, dashboard, API, and optional snapshots without loading or running inference.",
    )
    parser.add_argument(
        "--disable-inference",
        action="store_true",
        default=os.getenv("VISION_DISABLE_INFERENCE", "").lower() in {"1", "true", "yes", "on"},
        help="Keep runtime structure alive but skip model loading and prediction.",
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
        default=int(os.getenv("VISION_FRAME_WIDTH")) if os.getenv("VISION_FRAME_WIDTH") else None,
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=int(os.getenv("VISION_FRAME_HEIGHT")) if os.getenv("VISION_FRAME_HEIGHT") else None,
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
    parser.add_argument(
        "--debug-capture-on-detection",
        action="store_true",
        default=os.getenv("VISION_DEBUG_CAPTURE_ON_DETECTION", "").lower()
        in {"1", "true", "yes", "on"},
        help="Save raw and exact pre-engine inference frames only when raw detection is true.",
    )
    parser.add_argument(
        "--debug-dir",
        default=os.getenv("VISION_DEBUG_DIR"),
        help="Detection capture output directory; defaults to data/debug_frames/live_detections/.",
    )
    parser.add_argument(
        "--debug-max-captures",
        type=int,
        default=int(os.getenv("VISION_DEBUG_MAX_CAPTURES", "5")),
        help="Maximum detection-triggered captures; 0 means unlimited.",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    confidence = (
        args.confidence_threshold_override
        if args.confidence_threshold_override is not None
        else args.confidence
    )

    try:
        service = RuntimeDetectorService(
            profile_name=args.profile,
            model_path=args.model_path,
            model_format=args.model_format,
            prefer_edge_model=args.prefer_edge_model,
            camera_index=args.camera,
            camera_source=args.camera_source,
            camera_profile=args.camera_profile,
            camera_backend=args.camera_backend,
            camera_only=args.camera_only,
            disable_inference=args.disable_inference,
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
            debug_capture_on_detection=args.debug_capture_on_detection,
            debug_dir=args.debug_dir,
            debug_max_captures=args.debug_max_captures,
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
