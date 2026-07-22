# Raspberry Pi 5 + Pi Camera 3 Runtime

This guide is for running the edge runtime on a Raspberry Pi 5 with a Raspberry Pi Camera Module 3 and an existing trained model profile such as `yellow_daifuku`.

The runtime is status-first. Smooth browser video is not the production goal. `--enable-snapshot` is optional debug mode and should stay off for normal HMI/operator use.

## Hardware

- Raspberry Pi 5.
- Raspberry Pi Camera Module 3 connected through the CSI camera connector.
- Raspberry Pi OS with libcamera/Picamera2 support.
- Trained model profile under `models/yellow_daifuku/`.
- External SSD recommended for model files, logs, and review images.

## Setup

Preferred one-command runtime setup:

```bash
deploy/install_pi_runtime.sh
```

Do not install the desktop/training `requirements.txt` on the Pi runtime. See [README_PI_RUNTIME_SETUP.md](README_PI_RUNTIME_SETUP.md) for the reason and the full repeatable workflow.

```bash
ssh pi@<pi-ip>
cd <repo>
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements-pi-runtime.txt
```

Picamera2 is normally installed from Raspberry Pi OS packages, not plain pip:

```bash
sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-libcamera libcamera-apps
```

Using `--system-site-packages` lets the virtual environment see those system-installed camera packages.

## Camera Test

Quick libcamera test:

```bash
libcamera-hello --timeout 3000
```

Quick Picamera2 import/capture test:

```bash
python - <<'PY'
from picamera2 import Picamera2
cam = Picamera2()
config = cam.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
cam.configure(config)
cam.start()
frame = cam.capture_array()
cam.stop()
cam.close()
print("Captured frame:", frame.shape)
PY
```

Runtime health check:

```bash
python -m app.runtime.health_check \
  --mode pi \
  --profile yellow_daifuku \
  --camera-profile pi_camera3 \
  --camera-backend picamera2
```

Direct runtime camera-manager smoke test:

```bash
python scripts/test_picamera2_manager.py
```

## NCNN Runtime Model

Keep `.pt` files for desktop training/testing. For Raspberry Pi 4 runtime, export NCNN on the desktop and copy the exported folder to the Pi:

```bash
python scripts/export_profile_to_ncnn.py --profile yellow_daifuku --imgsz 320
```

Expected Pi folder:

```text
models/yellow_daifuku/best_ncnn_model/
```

The Pi runtime launcher prefers this folder automatically. If only `best.pt` exists, the Pi may crash during inference with `Illegal instruction`.

## Camera Quality And Preprocessing

The runtime now checks image quality before inference and reports the result in `/status` and on the dashboard. Metrics include brightness, blur score, contrast, overexposed percentage, and underexposed percentage.

Run camera validation:

```bash
python scripts/validate_camera.py --camera-profile pi_camera3 --duration 10
```

Run model validation on a captured frame:

```bash
python scripts/validate_model.py \
  --profile yellow_daifuku \
  --image data/debug_frames/camera_validation/<image>.jpg \
  --prefer-edge-model \
  --model-format auto
```

Run full pipeline validation without requiring Flask:

```bash
python scripts/validate_runtime.py \
  --profile yellow_daifuku \
  --camera-profile pi_camera3 \
  --duration 15 \
  --prefer-edge-model \
  --model-format auto
```

Changing cameras can change color response, lens distortion, focus, exposure, and field of view. Camera profiles plus preprocessing reduce unnecessary retraining by keeping orientation, ROI, and frame preparation consistent.

Quality warning meaning:

- `TOO_DARK` or `TOO_BRIGHT`: adjust lighting/exposure or reduce shadows/glare.
- `BLURRY`: check focus, vibration, lens cleanliness, and object motion.
- `LOW_CONTRAST`: improve background/lighting separation or refine ROI.
- high over/underexposed percentage: reduce reflections, saturation, or deep shadows.

Every saved review/debug/model-test image gets a `.json` sidecar with image quality, preprocessing, ROI, model, and inspection metadata.

## Dataset Image Capture

Use the Pi camera setup to collect raw training images without loading YOLO, NCNN, PyTorch, or Flask:

```bash
python scripts/capture_dataset_images.py \
  --profile yellow_daifuku \
  --camera-profile pi_camera3 \
  --label positive \
  --session pi_demo_v1 \
  --count 30 \
  --interval-seconds 2 \
  --save-quality-warnings
```

Collect negative examples in the same session:

```bash
python scripts/capture_dataset_images.py \
  --profile yellow_daifuku \
  --camera-profile pi_camera3 \
  --label negative \
  --session pi_demo_v1 \
  --count 20 \
  --interval-seconds 2 \
  --save-quality-warnings
```

Captured images are stored under:

```text
data/collections/<profile>/<camera_profile>/<session>/
```

Each image gets a `.json` sidecar with image-quality metrics and camera-profile details. The session also gets `manifest.jsonl` and `session_summary.json`. These are raw collection images only. Label them externally, then build or update the YOLO dataset under `data/datasets/` as a separate step.

## Runtime Command

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

Leave `--imgsz` unset for normal NCNN deployment so the runtime uses the exported model size from `metadata.yaml`.

Open the dashboard from another machine:

```text
http://<pi-ip>:8000
```

## Camera-Only Dashboard

Use this first when validating a new Pi/camera install. It starts the camera, browser dashboard, API, and `/snapshot.jpg` without loading YOLO, PyTorch, or NCNN:

```bash
scripts/run_camera_dashboard.sh
```

Equivalent command:

```bash
python -m app.runtime.detector_service \
  --profile yellow_daifuku \
  --camera-profile pi_camera3 \
  --camera-only \
  --enable-snapshot \
  --host 0.0.0.0 \
  --port 8000
```

The dashboard will show `CAMERA_ONLY`, model disabled, inference disabled, camera FPS, and the last frame timestamp. Use it to verify focus, lighting, framing, and camera stability before testing NCNN inference.

## Camera Profiles

Camera profiles live in `cameras/`:

```text
cameras/pi_camera3.yaml
cameras/usb_webcam.yaml
```

Profiles define backend, resolution, FPS, rotation, flips, optional ROI, and preprocessing flags. `--camera-profile pi_camera3` selects the Pi Camera 3 defaults. CLI values such as `--frame-width`, `--frame-height`, and `--camera-backend` still override the profile when provided.

Runtime mode summary:

- `--camera-only`: real camera plus dashboard/API/snapshot, no inference engine loaded.
- `--disable-inference`: normal runtime shell with model prediction intentionally skipped.
- `--dry-run`: no model loading, simulated detections for dashboard/API testing.
- Real Pi inference: use `--prefer-edge-model --model-format auto` with `best_ncnn_model/`.

## API Tests

```bash
curl http://127.0.0.1:8000/status
curl http://127.0.0.1:8000/latest_detection
curl -I http://127.0.0.1:8000/snapshot.jpg
```

`/snapshot.jpg` returns `404` unless `--enable-snapshot` is used. That is expected for production-style status dashboards.

## Saved Files

```text
data/logs/detections.csv
data/logs/runtime_events.csv
data/logs/latest_status.json
data/review_images/<profile>/detections/
data/review_images/<profile>/low_confidence/
data/review_images/<profile>/no_detection/
```

`latest_status.json` is the local machine-readable output for the newest stable inspection result. It is safe for future HMI/PLC/cloud adapters to read without controlling machinery directly.

## systemd Service

Install and start the service:

```bash
deploy/install_pi.sh
deploy/start_service.sh
```

Check logs:

```bash
systemctl status vision.service
journalctl -u vision.service -f
```

The service template uses Picamera2 by default:

```text
--profile yellow_daifuku --camera-profile pi_camera3 --camera-backend picamera2 --prefer-edge-model --model-format auto --host 0.0.0.0 --port 8000 --frame-width 640 --frame-height 480 --inference-interval-ms 300
```

## Inspection Events, Health, And Notifications

The Pi runtime preserves the existing `inspection_result` values and adds canonical integration states: `PASS`, `FAIL`, `REVIEW`, `NO_PART`, and `SYSTEM_ERROR`. `/status`, `/latest_detection`, `latest_status.json`, and the dashboard include the current `inspection_id`, canonical state, agreement ratio, image-quality status, health, recent events, and notification status.

Optional inspection evidence files, when enabled:

```text
data/logs/events.jsonl
data/inspections/YYYY-MM-DD/<state>/
```

Notifications are off unless `VISION_NOTIFICATIONS_ENABLED` is set. Email and Teams credentials are read from environment variables such as `VISION_SMTP_HOST`, `VISION_EMAIL_TO`, and `VISION_TEAMS_WEBHOOK_URL`; missing values never stop the runtime.

## Troubleshooting

Picamera2 import fails:

- Install `python3-picamera2 python3-libcamera libcamera-apps`.
- Recreate the venv with `python3 -m venv --system-site-packages .venv`.
- Run the Picamera2 import/capture test above inside the venv.

Camera not detected:

- Power down and reseat the CSI ribbon cable.
- Confirm the camera works with `libcamera-hello --timeout 3000`.
- Run the health check with `--camera-backend picamera2`.

Model path missing:

- Confirm `models/yellow_daifuku/classes.txt` exists.
- Confirm `models/yellow_daifuku/latest/best.pt` or the configured model file exists.
- Run `python -m app.runtime.health_check --mode pi --profile yellow_daifuku --camera-backend picamera2`.

Dashboard not reachable:

- Confirm the service is running.
- Confirm the command uses `--host 0.0.0.0`.
- Check firewall/network routing on the Pi.

Low FPS:

- Lower `--imgsz`, for example `--imgsz 224`.
- Increase `--inference-interval-ms`, for example `500`.
- Keep `--enable-snapshot` off.
- Use lower frame dimensions if acceptable.

No detections:

- Confirm the trained profile is the intended one.
- Confirm `target_classes`, `acceptable_classes`, and confidence thresholds match the model classes.
- Check lighting, focus, camera angle, and object distance.

Wrong color format:

- The Picamera2 backend captures RGB and converts frames to BGR for the existing OpenCV/YOLO pipeline.
- If colors look wrong in debug snapshots, verify the camera test output and report the exact backend/status payload.

Service starts but camera fails:

- Check `data/logs/runtime_events.csv`.
- Check `data/logs/latest_status.json`.
- Check `/status` for `camera.backend`, `camera.error`, and `inspection.result`.
