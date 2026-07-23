# Industrial AI Vision System Prototype

This repository is a prototype industrial vision system for detecting objects or defects with a mounted camera. It has two main modes:

- **Desktop engineering app** for image collection, dataset validation, YOLO training, model reports, and live engineering checks.
- **Runtime service** for Raspberry Pi 5 / industrial PC deployment with Pi Camera 3, USB/OpenCV cameras, or simulated image sources, plus a lightweight browser/HMI dashboard, stable detection state, logs, and review-image capture.

Training is intended to happen on a desktop/laptop. The Raspberry Pi runtime is intended to run an already-trained model reliably.

## Current Prototype Functionality

- Collect training images from a USB webcam.
- Label images with the simple local tool or label externally and import YOLO-format labels.
- Validate YOLO datasets before training.
- Train YOLO models and package them into `models/<profile>/`.
- Switch model profiles in the desktop app.
- Run a lightweight runtime service with `/status`, `/latest_detection`, `/snapshot.jpg`, and `/`.
- Keep production dashboard status-first: PASS/FAIL style status, confidence, counters, timings, camera health, and review-image capture.
- Report image quality metrics such as brightness, blur, contrast, overexposure, and underexposure.
- Apply camera-profile preprocessing such as rotation, flips, ROI crop, and optional letterbox standardization before runtime inference.
- Save structured detection logs and rate-limited review images for retraining.

Smooth browser video is intentionally **not** the production goal. Snapshot mode is optional/debug only.

## Folder Structure

```text
app/
  main.py                  # Desktop PySide engineering app entrypoint
  ui.py                    # Desktop UI
  inference.py             # Desktop inference worker
  logging.py               # CSV detection logging
  runtime/
    detector_service.py    # Raspberry Pi / industrial PC runtime service
    camera_manager.py      # Reconnecting OpenCV camera wrapper
    camera_sources.py      # Simulated image/folder/video camera source
    picamera2_manager.py   # Picamera2/libcamera camera wrapper
    image_quality.py       # Brightness/blur/contrast/exposure checks
    preprocessing.py       # Camera transforms, ROI crop, frame standardization
    health_check.py        # Local/Pi runtime readiness checks
    inspection_logic.py    # Target filtering, ROI, stable detection state
    output_manager.py      # Runtime CSV output hook
    action_manager.py      # Safe local action/status JSON layer

training/
  train_pipeline.py        # Dataset validation, YOLO training, model packaging
  collect_images.py        # Simple camera image collection utility
  label_images.py          # Simple local bounding-box label tool

models/
  <profile>/               # best.pt, latest/best.pt, config.json, classes.txt

profiles/
  <profile>/config.yaml     # Optional runtime inspection/pass-fail rules

data/
  datasets/                # YOLO datasets
  logs/                    # detection CSV logs
  review_images/           # runtime/desktop review images

deploy/
  install_pi.sh            # Raspberry Pi / Linux installer
  start_service.sh         # systemd service start helper
  vision.service           # systemd unit template

samples/
  README.md                # Instructions for laptop simulation images
  test_images/             # Put local test images here
```

## Hardware Assumptions

Prototype target:

- Raspberry Pi 5, 8 GB RAM.
- External SSD recommended for models, logs, and review images.
- Raspberry Pi Camera Module 3 through Picamera2/libcamera for the first Pi demo.
- USB webcam at `/dev/video0` remains supported through OpenCV.
- Later production camera may be industrial USB or GigE.
- Desktop/laptop is used for training and model management.

## Install Dependencies

Use Python 3.10+.

**Raspberry Pi runtime warning:** do not run `pip install -r requirements.txt` on the Pi runtime. That file is for desktop/training and can install pip NumPy/OpenCV wheels that break apt-installed Picamera2/libcamera. On Raspberry Pi use:

```bash
deploy/install_pi_runtime.sh
```

See [README_PI_RUNTIME_SETUP.md](README_PI_RUNTIME_SETUP.md) for the Pi-specific install, validation, runtime, and detection-debug workflow.

For Raspberry Pi 4 inference, export an NCNN model on the desktop with:

```bash
python scripts/export_profile_to_ncnn.py --profile yellow_daifuku --imgsz 320
```

The Pi launcher prefers `models/<profile>/best_ncnn_model/` when it exists.

Recommended local setup:

```bash
scripts/setup_local.sh
source .venv/bin/activate
```

Manual setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bat
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Local Demo

After setup, launch the laptop-safe dashboard demo:

```bash
scripts/run_demo.sh
```

Open:

```text
http://127.0.0.1:8000
```

The demo uses a simulated image source when available. It automatically enables `--dry-run` if the selected profile does not have model weights.

## Run Desktop App

```bash
python -m app.main
```

Optional:

```bash
python -m app.main --model models/yellow_daifuku/best.pt --camera 0
```

The desktop app supports live detection, model profile switching, training image collection, dataset training, training reports, camera reconnect status, ROI filtering, and structured logs.

## Desktop Training Workflow

1. Collect images into `data/datasets/<profile>/images/train`.
2. Label images externally or with `training/label_images.py`.
3. Ensure the dataset has:
   - `images/train`
   - `images/val`
   - `labels/train`
   - `labels/val`
   - `data.yaml`
4. Train from the desktop app or command line:

```bash
python training/train_pipeline.py
```

The training pipeline validates labels before starting YOLO training. Successful training writes:

- `models/<profile>/best.pt`
- `models/<profile>/latest/best.pt`
- `models/<profile>/versions/vN/best.pt`
- `models/<profile>/config.json`
- `models/<profile>/classes.txt`
- `models/<profile>/training_report.json`

## Runtime / Raspberry Pi Workflow

Recommended Raspberry Pi 5 + Pi Camera Module 3 runtime command:

```bash
python -m app.runtime.detector_service \
  --profile yellow_daifuku \
  --camera-profile pi_camera3 \
  --camera-backend picamera2 \
  --prefer-edge-model \
  --model-format auto \
  --host 0.0.0.0 \
  --port 8000 \
  --frame-width 640 \
  --frame-height 480 \
  --inference-interval-ms 300
```

When `--imgsz` is omitted, exported NCNN model metadata is used first. That prevents a Pi launcher from silently running a 320x320 model at a mismatched legacy size.

Camera-only dashboard for validating the camera, focus, lighting, and dashboard without loading YOLO/PyTorch/NCNN:

```bash
scripts/run_camera_dashboard.sh
```

USB/OpenCV cameras still use:

```bash
python -m app.runtime.detector_service \
  --profile yellow_daifuku \
  --camera 0 \
  --host 0.0.0.0 \
  --port 8000
```

Open the dashboard:

```text
http://<pi-ip>:8000
```

The runtime:

- Loads `models/<profile>/config.json`, `classes.txt`, and the configured model file.
- Reads camera frames continuously.
- Applies camera-profile transforms, ROI, and preprocessing in a standard pipeline.
- Computes image quality metrics before inference so lighting/focus/setup problems are visible.
- Runs YOLO on a throttled interval, not every camera frame.
- Maintains stable detection state.
- Converts raw model output into a canonical inspection state: `PASS`, `FAIL`, `REVIEW`, `NO_PART`, or `SYSTEM_ERROR`.
- Reconnects the camera after failures.
- Exposes JSON APIs for HMI/supervisor integration.
- Logs structured detection events to `data/logs/detections.csv`.
- Logs startup configuration and major runtime faults to `data/logs/runtime_events.csv`.
- Logs structured inspection/system events to `data/logs/events.jsonl`.
- Saves rate-limited review images for retraining.

### Snapshot Mode

`--enable-snapshot` is optional debug mode:

```bash
python -m app.runtime.detector_service --profile yellow_daifuku --camera 0 --enable-snapshot
```

Leave snapshot mode **off** for production/HMI use. When disabled, the dashboard hides the image panel and `/snapshot.jpg` returns `404`, keeping status updates lightweight.

## Runtime API

- `GET /` - browser/HMI dashboard.
- `GET /status` - service status, camera/model health, inspection result, timings, counters, runtime config, and output payload.
- `GET /latest_detection` - most recent inspection result plus raw/stable detection detail and saved image path.
- `GET /snapshot.jpg` - latest cached annotated JPEG only when `--enable-snapshot` is enabled.
- `GET /health` - lightweight health summary derived from cached runtime state.

## Inspection Result Logic

The runtime separates raw YOLO detections from the stable inspection result that an operator, HMI, or future PLC output should use.

- Raw detections are direct model observations from one inference cycle.
- Stable inspection results are produced by `app/runtime/inspection_logic.py` after confidence checks, class mapping, ROI filtering, and consecutive-frame smoothing.
- `/status`, `/latest_detection`, and the dashboard expose both debug details and the operator-facing `inspection_result`.

Current result states:

- `PASS` - an acceptable class was detected with enough confidence for the required number of frames.
- `FAIL` - a reject class or non-acceptable class was detected with enough confidence.
- `NO_PART` - no part/detection is present after the allowed no-detection threshold.
- `LOW_CONFIDENCE` - a detection exists, but confidence is below the inspection threshold.
- `CAMERA_ERROR` - the camera source is failed or reconnecting.
- `MODEL_ERROR` - the requested model/profile cannot be loaded or used.
- `SIMULATION` - dry-run or simulated camera-source mode is active, so the result is not a production decision.
- `CAMERA_ONLY` - camera/dashboard mode is active and inference is intentionally disabled.
- `INFERENCE_DISABLED` - the runtime shell is alive but model prediction is intentionally skipped.
- `IMAGE_QUALITY_ERROR` - image quality failed and the active camera profile is configured to skip inference on bad quality.
- `QUALITY_CHECK_ERROR` - image quality analysis failed unexpectedly.

The runtime also exposes a canonical `inspection_state` for downstream integrations:

- `PASS` - acceptable finalized inspection.
- `FAIL` - unacceptable finalized inspection.
- `REVIEW` - low confidence, simulation, camera-only, disabled inference, image-quality warning, or conflicting evidence.
- `NO_PART` - no part detected after the configured miss threshold.
- `SYSTEM_ERROR` - camera/model/runtime fault.

Every finalized decision gets a readable `inspection_id` such as `INS-20260722-073142-000184`. Existing `inspection_result` values are preserved for backward compatibility.

Decision mode defaults to the existing consecutive-frame behavior. Profiles may opt into rolling-window decisions:

```yaml
inspection:
  decision_mode: rolling_window
  rolling_window_size: 8
  rolling_min_agreeing: 6
  rolling_min_agreement_ratio: 0.75
```

External notifications are disabled by default. Optional test/staging values are read from environment variables:

```text
VISION_NOTIFICATIONS_ENABLED
VISION_EMAIL_ENABLED
VISION_SMTP_HOST
VISION_SMTP_PORT
VISION_SMTP_USERNAME
VISION_SMTP_PASSWORD
VISION_EMAIL_FROM
VISION_EMAIL_TO
VISION_TEAMS_WEBHOOK_URL
```

Do not store real credentials in the repo. Missing credentials are reported without stopping the runtime.

## Image Quality And Preprocessing

Most industrial vision failures come from lighting, focus, camera position, reflections, dirty lenses, weak datasets, or ROI mismatch. Phase 2 adds lightweight diagnostics for those setup issues.

Camera profiles in `cameras/` can define:

- rotation and horizontal/vertical flips
- normalized ROI crop
- preprocessing enablement and color normalization
- quality thresholds for brightness, contrast, blur, overexposure, and underexposure
- whether bad quality should skip inference

Runtime `/status` now includes:

- `image_quality`
- `preprocessing`
- `camera_profile_details`
- `inspection`
- `health`
- `events`
- `notifications`

`/health` disk usage reports `free_pct` as a human percent from `0` to `100`. For example, `0.879` means `0.879%` free, and the default low-disk warning threshold is `5.0` percent free.

Review/debug/model-test images get a matching `.json` sidecar with model/profile, detections, image quality, preprocessing, ROI, and result metadata. These sidecars make the retraining loop easier: collect evidence, label hard cases, retrain on desktop, export NCNN, redeploy to Pi.

Validation commands:

```bash
python scripts/validate_camera.py --camera-profile pi_camera3 --duration 10
python scripts/validate_model.py --profile yellow_daifuku --image data/debug_frames/camera_validation/<image>.jpg --prefer-edge-model --model-format auto
python scripts/validate_runtime.py --profile yellow_daifuku --camera-profile pi_camera3 --duration 15 --prefer-edge-model --model-format auto
```

Quality warnings mean:

- `TOO_DARK` / `TOO_BRIGHT`: adjust lighting, exposure, aperture, or object position.
- `BLURRY`: check focus, vibration, motion blur, lens cleanliness, and mounting.
- `LOW_CONTRAST`: improve lighting angle/background or revisit the inspection ROI.
- high over/underexposed percentage: reduce glare, reflections, shadows, or saturation.

Runtime inspection rules can be added in:

```text
profiles/<profile>/config.yaml
```

Example:

```yaml
profile_name: yellow_daifuku
inspection:
  acceptable_classes:
    - yellow_daifuku
  reject_classes: []
  minimum_confidence: 0.35
  required_consecutive_detections: 3
  allowed_no_detection_frames: 3
  allow_simulation: true
  roi:
    enabled: false
    x1: 0.0
    y1: 0.0
    x2: 1.0
    y2: 1.0
```

The output payload prepared by `app/runtime/output_manager.py` includes `inspection_id`, `inspection_state`, `inspection_result`, `pass_fail_bool`, `active_class`, `confidence`, `average_confidence`, `agreement_ratio`, `image_quality_status`, `timestamp`, `profile`, `camera_status`, `model_status`, `simulation_mode`, and a human-readable message. A future PLC/HMI adapter should consume this payload instead of raw YOLO class names.

Optional inspection evidence images can be enabled under:

```text
data/inspections/YYYY-MM-DD/fail/
data/inspections/YYYY-MM-DD/review/
data/inspections/YYYY-MM-DD/system_error/
```

Inspection evidence saving is off by default to preserve current storage behavior. Existing review-image folders remain unchanged and remain the primary retraining evidence path.

## Testing Without Raspberry Pi Hardware

You can test most of the runtime loop on a laptop by replacing the physical camera with an image or folder source.

Prepare sample images:

```bash
mkdir -p samples/test_images
cp assets/test.jpg samples/test.jpg
cp assets/*.jpg samples/test_images/
```

Run a readiness check against a simulated image source:

```bash
python -m app.runtime.health_check \
  --mode laptop \
  --profile yellow_daifuku \
  --camera-source samples/test.jpg
```

Run the runtime with a single image source and simulated detections:

```bash
python -m app.runtime.detector_service \
  --profile yellow_daifuku \
  --camera-source samples/test.jpg \
  --dry-run \
  --host 127.0.0.1 \
  --port 8000
```

Run the runtime while cycling through a folder of images:

```bash
python -m app.runtime.detector_service \
  --profile yellow_daifuku \
  --camera-source samples/test_images/ \
  --dry-run \
  --host 127.0.0.1 \
  --port 8000
```

Open the local dashboard:

```text
http://127.0.0.1:8000
```

Validate API endpoints:

```bash
curl http://127.0.0.1:8000/status
curl http://127.0.0.1:8000/latest_detection
curl -I http://127.0.0.1:8000/snapshot.jpg
```

`--dry-run` skips YOLO model loading and creates fake detections, so it is for simulation only. The dashboard and API report `simulation_mode: true` and show `Runtime Mode: SIMULATION`. To test a real trained model without a webcam, remove `--dry-run` and keep `--camera-source`.

Still requires real hardware later:

- USB camera enumeration and disconnect/reconnect behavior.
- Raspberry Pi CPU performance, thermals, boot/service behavior, and storage performance.
- Real production lighting, camera mount, field of view, lens/focus, and line-speed repeatability.

## Review-Image Feedback Loop

The runtime saves rate-limited evidence under:

```text
data/review_images/<profile>/detections/
data/review_images/<profile>/low_confidence/
data/review_images/<profile>/no_detection/
```

Recommended loop:

1. Run the runtime on the line or bench.
2. Collect review images from hard cases.
3. Label useful images externally.
4. Add them back into `data/datasets/<profile>/`.
5. Retrain on desktop.
6. Deploy the updated `models/<profile>/` to the runtime machine.
7. Repeat until the model is stable enough for the inspection task.

## Raspberry Pi Service Deployment

On the Pi or industrial Linux machine:

```bash
deploy/install_pi.sh
deploy/start_service.sh
```

Edit [deploy/vision.service](deploy/vision.service) if your repository path, user, model profile, camera backend, or runtime parameters differ.

Useful checks:

```bash
python -m app.runtime.health_check --mode pi --profile yellow_daifuku --camera-profile pi_camera3 --camera-backend picamera2
systemctl status vision.service
journalctl -u vision.service -f
curl http://127.0.0.1:8000/status
curl http://127.0.0.1:8000/latest_detection
```

See [README_PI_CAMERA3_RUNTIME.md](README_PI_CAMERA3_RUNTIME.md) for Pi Camera 3 setup, camera tests, service notes, and troubleshooting.

## Future PLC/HMI Integration

The current runtime is ready for basic HMI polling over HTTP. Future PLC integration should add a small output adapter layer for industrial protocols or hardware I/O, such as:

- Modbus TCP
- OPC UA
- digital GPIO through an isolated I/O module
- vendor-specific PLC gateway

The runtime should keep inference and inspection logic separate from output integration so the vision system remains testable without hardware.

## Validation Commands

```bash
python -m py_compile app/main.py app/ui.py app/inference.py app/logging.py app/runtime/detector_service.py app/runtime/camera_manager.py app/runtime/camera_sources.py app/runtime/picamera2_manager.py app/runtime/action_manager.py app/runtime/health_check.py app/runtime/inspection_logic.py app/runtime/output_manager.py training/train_pipeline.py
python -m unittest discover -s tests
python -m app.runtime.health_check --mode laptop --profile yellow_daifuku --camera-source assets/test.jpg
```

## Safe Cleanup Commands

Generated runtime/debug files are ignored by Git. To clean local caches and generated outputs without deleting datasets or models:

```bash
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete
find data/datasets -type f -name '*.cache' -delete
rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov
rm -rf data/debug_frames/*
rm -f data/logs/*.csv data/logs/*.json
```

Review images may be useful retraining evidence. Delete them only after saving anything useful:

```bash
rm -rf data/review_images/*
```

See [REPOSITORY_CLEANUP_REPORT.md](REPOSITORY_CLEANUP_REPORT.md) for the current audit notes and files intentionally left for manual review.

## Notes

- Existing `archive/` and older `docs/` files are retained for reference and are not the primary workflow.
- Runtime image snapshots are debug-only and intentionally disabled by default.
- Model weights are large; keep only deployable models under `models/<profile>/`.
