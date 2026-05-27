# Functionality Report

This report describes the prototype as it exists now. It is intentionally practical: what works, what is ready for bench testing, and what still needs engineering before production use.

## Current Desktop App Functionality

- Launches with `python -m app.main`.
- Opens a PySide6 desktop engineering UI.
- Displays live camera frames with YOLO annotations.
- Loads model profiles from `models/<profile>/`.
- Reads profile `config.json` and `classes.txt`.
- Uses profile-specific target classes and confidence.
- Applies stable detection smoothing in the displayed UI state.
- Shows raw detection and stable detection separately.
- Shows active profile, last class, last confidence, camera status, and stable detection count.
- Supports image collection into `data/datasets/<profile>/images/train`.
- Runs the training pipeline from the UI.
- Shows training validation errors in the log.
- Shows model/training report information.
- Writes structured detection CSV logs through `app/logging.py`.

## Current Runtime Functionality

- Launches with `python -m app.runtime.detector_service`.
- Loads model profiles from `models/<profile>/`.
- Reads `config.json`, `classes.txt`, and model file paths.
- Opens a USB camera through OpenCV.
- Can use a simulated camera source from an image, image folder, or video file.
- Can run in dry-run simulation mode without loading a trained YOLO model.
- Sets requested camera width/height.
- Reconnects camera after repeated failures.
- Keeps reading camera frames continuously.
- Runs YOLO inference on a throttled interval.
- Converts raw detections into operator-facing inspection results.
- Applies pass/fail class mapping, confidence checks, optional ROI checks, and stable detection smoothing.
- Tracks camera FPS, inference FPS, last inference time, frame counts, and runtime config.
- Reports whether the runtime is in production mode or simulation mode.
- Keeps debug snapshots disabled by default.
- Saves rate-limited review images for detections, low-confidence detections, and no-detection events.
- Logs structured detection events.
- Provides `python -m app.runtime.health_check` for local or Pi readiness checks.

## Current API Endpoints

- `GET /`
  - Browser/HMI dashboard.
- `GET /status`
  - Runtime status, camera health, model health, timing, counters, profile/model info, latest inspection summary, and output payload.
- `GET /latest_detection`
  - Latest inspection result, pass/fail boolean, raw/stable detection detail, class, confidence, timestamp, ROI state, model/camera status, and saved image path.
- `GET /snapshot.jpg`
  - Returns cached annotated JPEG only when `--enable-snapshot` is active.
  - Returns `404` when snapshot mode is disabled.

## Current Dashboard Behavior

- Status-first display intended for operators/HMI use.
- Shows camera status.
- Shows a large operator-facing inspection result.
- Shows raw detection as a smaller debug detail.
- Shows model status.
- Shows active profile/model.
- Shows last class and confidence.
- Shows the result message/reason.
- Shows timestamp.
- Shows camera FPS, inference FPS, and last inference timing.
- Shows stable detection count and review-image counters.
- Shows whether debug snapshot mode is enabled.
- Shows whether runtime mode is production or simulation.
- Hides the image panel unless snapshot mode is enabled.
- Polls JSON status without requiring smooth video streaming.

## Current Model/Profile Behavior

- Profiles live under `models/<profile>/`.
- Expected files:
  - `best.pt` or `latest/best.pt`
  - `config.json`
  - `classes.txt`
- `config.json` can define:
  - `profile_name`
  - `model_file`
  - `target_classes`
  - `confidence`
  - `latest_version`
- Runtime and desktop detection do not assume class id `0`.
- Runtime reports `MODEL_ERROR` when the requested model profile or model file is missing.
- Dry-run mode is explicitly marked as simulation and can start without a model/profile for dashboard and runtime-loop testing.

## Current Inspection Result States

- `PASS`
  - Acceptable class detected above the inspection confidence threshold for the required number of frames.
- `FAIL`
  - Reject class or non-acceptable class detected above the inspection confidence threshold.
- `NO_PART`
  - No detection remains after the allowed no-detection frame threshold.
- `LOW_CONFIDENCE`
  - A detection is present but below the inspection confidence threshold.
- `CAMERA_ERROR`
  - Camera source is failed or reconnecting.
- `MODEL_ERROR`
  - Model/profile is missing or failed to load.
- `SIMULATION`
  - Dry-run mode or simulated camera-source mode is active.

## Current Pass/Fail Decision Logic

- Runtime rules can be defined in `profiles/<profile>/config.yaml`.
- Model artifacts still live in `models/<profile>/`.
- The YAML rules can define:
  - `acceptable_classes`
  - `reject_classes`
  - `minimum_confidence`
  - `required_consecutive_detections`
  - `allowed_no_detection_frames`
  - `allow_simulation`
  - optional normalized ROI fields
- If no YAML rule exists, the runtime falls back to existing profile `target_classes` and confidence settings.
- Raw detections remain available for debugging, but downstream pass/fail consumers should use `inspection_result` and `pass_fail_bool`.

## Output Payload Format

`app/runtime/output_manager.py` prepares a stable payload for future PLC/HMI adapters:

```json
{
  "inspection_result": "PASS",
  "pass_fail_bool": true,
  "active_class": "yellow_daifuku",
  "confidence": 0.91,
  "timestamp": "2026-05-27T10:00:00",
  "profile": "yellow_daifuku",
  "camera_status": "Connected",
  "model_status": "Loaded",
  "simulation_mode": false,
  "reason": "Accepted class 'yellow_daifuku' detected.",
  "message": "Accepted class 'yellow_daifuku' detected."
}
```

## PLC/HMI Readiness

- The runtime now has an output-ready payload separate from raw YOLO detections.
- The dashboard is centered on machine/operator states rather than model internals.
- Future PLC/HMI integration can consume the output payload and map:
  - `PASS` to pass/accept output.
  - `FAIL`, `LOW_CONFIDENCE`, `CAMERA_ERROR`, and `MODEL_ERROR` to reject/fault outputs.
  - `NO_PART` to no-part/no-trigger state.
  - `SIMULATION` to a non-production/safe state.

## Current Logging/Review-Image Behavior

- Structured detection logs write to `data/logs/detections.csv`.
- Runtime startup and major fault events write to `data/logs/runtime_events.csv`.
- Review images write to:
  - `data/review_images/<profile>/detections/`
  - `data/review_images/<profile>/low_confidence/`
  - `data/review_images/<profile>/no_detection/`
- Review-image filenames include timestamp and include class/confidence when available.
- Runtime review-image saving is rate-limited per category.
- `/latest_detection` includes `saved_image_path` when an image was saved for that result.
- `/status` includes:
  - `total_detections`
  - `total_images_saved`
  - `low_confidence_count`
  - `no_detection_count`

## Current Deployment Readiness

- `scripts/setup_local.sh` prepares a local venv, dependencies, data folders, and sample folders.
- `scripts/run_demo.sh` starts a safe laptop dashboard demo on `127.0.0.1:8000`.
- `deploy/vision.service` provides a systemd unit template.
- `deploy/install_pi.sh` creates a virtual environment, installs dependencies, prepares folders, and installs the service.
- `deploy/start_service.sh` enables and restarts the service.
- Dockerfile and docker-compose are present for runtime-oriented container testing.
- `python -m app.runtime.health_check --mode laptop` checks Python, imports, profile/model paths, writable folders, config values, and simulated source access.
- `python -m app.runtime.health_check --mode pi` requires a camera index and checks hardware capture readiness.
- Recommended Pi runtime defaults are low-resolution and CPU-friendly:
  - `imgsz=256`
  - `frame-width=424`
  - `frame-height=240`
  - `inference-interval-ms=300`
  - snapshots disabled

## Current Demo Workflow

```bash
scripts/setup_local.sh
source .venv/bin/activate
scripts/run_demo.sh
```

Then open:

```text
http://127.0.0.1:8000
```

The demo uses `assets/test.jpg` or `samples/test.jpg` as a simulated camera source. It automatically uses dry-run mode if the selected profile does not have model weights.

## Exact Laptop Validation Checklist

```bash
source .venv/bin/activate
python -m unittest discover -s tests
python -m app.runtime.health_check --mode laptop --profile yellow_daifuku --camera-source assets/test.jpg
scripts/run_demo.sh
curl http://127.0.0.1:8000/status
curl http://127.0.0.1:8000/latest_detection
curl -I http://127.0.0.1:8000/snapshot.jpg
```

Expected laptop behavior:

- Dashboard opens at `http://127.0.0.1:8000`.
- `/status` includes `inspection_result`, `model_status`, `camera_status`, and `output_payload`.
- `SIMULATION` is shown when dry-run or simulated-source mode is active.
- Snapshot remains disabled unless `--enable-snapshot` is explicitly used.

## Exact Pi Validation Checklist

```bash
source .venv/bin/activate
python -m app.runtime.health_check --mode pi --profile yellow_daifuku --camera 0
python -m app.runtime.detector_service \
  --profile yellow_daifuku \
  --camera 0 \
  --host 0.0.0.0 \
  --port 8000 \
  --imgsz 256 \
  --frame-width 424 \
  --frame-height 240 \
  --inference-interval-ms 300
```

Then from the Pi or another machine:

```bash
curl http://127.0.0.1:8000/status
curl http://127.0.0.1:8000/latest_detection
```

For systemd deployment:

```bash
deploy/install_pi.sh
deploy/start_service.sh
systemctl status vision.service
journalctl -u vision.service -f
```

## What Is Demo-Ready

- Local one-command dashboard demo.
- Laptop simulated-camera and dry-run testing.
- Explicit operator inspection states.
- Output-ready payload for future PLC/HMI work.
- Health checks for laptop and Pi modes.
- Startup/fault event logging under `data/logs/runtime_events.csv`.
- Review-image capture folders and counters.

## What Remains Hardware-Dependent

- Real camera compatibility and stable frame rate.
- Raspberry Pi CPU/thermal performance under real YOLO inference.
- USB cable/camera disconnect behavior.
- Lighting, fixture, lens, focus, and field-of-view repeatability.
- Real pass/fail rule tuning using production samples.

## What Can Be Tested Today on a Laptop

- Runtime Flask service startup and shutdown.
- `/`, `/status`, `/latest_detection`, and `/snapshot.jpg` route behavior.
- Status-first dashboard polling.
- Simulated camera input from `--camera-source samples/test.jpg`.
- Simulated camera input from `--camera-source samples/test_images/`.
- Dry-run/no-model mode with fake detections.
- Stable detection smoothing and raw/stable status reporting.
- Runtime counters, inference timing fields, and simulation-mode reporting.
- Review-image directory creation and rate-limited image saving.
- Structured logging path creation.
- Health-check command for package imports, profile/model paths, write permissions, and simulated sources.
- Real YOLO inference against a still image/folder source when a trained model exists.

## What Requires a USB Camera

- Real webcam capture from `--camera 0`.
- Camera resolution negotiation at the requested width/height.
- Physical disconnect/reconnect behavior.
- Exposure, focus, motion blur, and camera FPS under real capture conditions.

## What Requires Raspberry Pi

- Pi CPU inference speed and thermal behavior.
- systemd service installation/startup behavior on the target OS.
- Boot-time camera availability and service restart behavior.
- External SSD write throughput for logs and review images.
- Network access from the actual HMI/browser device.

## What Requires Production Hardware/Lighting

- Final camera mount, lens, focus, distance, and field of view.
- Production lighting repeatability and glare control.
- Inspection-zone/ROI tuning against real fixture geometry.
- Real line speed, vibration, trigger timing, and part presentation variation.
- PASS/FAIL thresholds validated against labeled production samples.

## Known Limitations

- No PLC protocol adapter is implemented yet.
- Pass/fail rules are intentionally simple and profile-local.
- No external trigger/part-present sensor integration exists yet.
- No authentication is implemented on the runtime dashboard/API.
- Snapshot debug mode is JPEG polling, not video streaming.
- Training still uses Ultralytics defaults and is not tuned per product.
- Local label tool is basic; external labeling is recommended for serious datasets.
- Dataset split creation and dataset versioning are still manual.
- No model rollback UI exists.
- No automatic disk quota or cleanup policy exists for review images.
- Industrial USB/GigE camera SDKs are not integrated yet.
- Current runtime process is single-model at startup; model switching is not exposed in the dashboard.

## Recommended Next Engineering Steps

1. Add disk retention controls for `data/review_images/` and `data/logs/`.
2. Add explicit runtime config file support for production deployments.
3. Add a PLC/HMI output adapter abstraction with a simulated test adapter first.
4. Add dataset import/export tooling for externally labeled datasets.
5. Add model package validation before deployment.
6. Add service health checks and startup diagnostics.
7. Add a small hardware bench test checklist for camera, lighting, trigger, and mounting repeatability.
8. Add CI or a local validation script that runs compile checks and runtime unit tests.

## Hardware-Readiness Checklist

- [ ] `python -m app.runtime.health_check --mode laptop --profile <profile> --camera-source <image>` passes on laptop.
- [ ] `python -m app.runtime.health_check --mode pi --profile <profile> --camera 0` passes on the deployment machine.
- [ ] Laptop simulation works with `--camera-source` and `--dry-run`.
- [ ] Laptop real-model test works with `--camera-source` and no `--dry-run`.
- [ ] Raspberry Pi 5 boots from or writes to reliable external storage.
- [ ] USB camera appears as `/dev/video0` or configured camera index.
- [ ] Camera resolution works at the configured width/height.
- [ ] Runtime starts without snapshots and dashboard remains responsive.
- [ ] `/status` reports camera FPS and inference FPS.
- [ ] Review images are saved under the expected profile folders.
- [ ] Logs are written under `data/logs/`.
- [ ] Model profile exists under `models/<profile>/`.
- [ ] Lighting and camera mount are repeatable.
- [ ] Operators can interpret stable/raw detection and counters.
- [ ] Future PLC/HMI integration path is selected and tested with a simulator first.
