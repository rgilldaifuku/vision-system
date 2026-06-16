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

```bash
ssh pi@<pi-ip>
cd <repo>
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
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
  --camera-backend picamera2
```

Direct runtime camera-manager smoke test:

```bash
python scripts/test_picamera2_manager.py
```

## Runtime Command

```bash
python -m app.runtime.detector_service \
  --profile yellow_daifuku \
  --camera-backend picamera2 \
  --host 0.0.0.0 \
  --port 8000 \
  --imgsz 256 \
  --frame-width 640 \
  --frame-height 480 \
  --inference-interval-ms 300
```

Open the dashboard from another machine:

```text
http://<pi-ip>:8000
```

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
--profile yellow_daifuku --camera-backend picamera2 --host 0.0.0.0 --port 8000 --imgsz 256 --frame-width 640 --frame-height 480 --inference-interval-ms 300
```

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
