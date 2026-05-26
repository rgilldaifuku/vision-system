# Vision System

Desktop industrial vision application for collecting images, validating YOLO datasets, training YOLO models, and running live detection.

## Install

Use Python 3.10+ and install dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with:

```bat
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run The App

```bash
python -m app.main
```

Optional arguments:

```bash
python -m app.main --model models/mouse/best.pt --camera 0
```

The app opens the live detection UI, supports model profile switching, image collection, training, reports, camera reconnect status, detection logs, and optional ROI filtering.

## Run The Raspberry Pi Runtime

The Pi runtime is separate from the desktop UI. It loads a model profile, runs camera inference continuously, applies stable detection smoothing, reconnects the camera when needed, logs detection state, and exposes JSON endpoints for an HMI or supervisor process.

```bash
python -m app.runtime.detector_service --profile mouse --camera 0 --host 0.0.0.0 --port 8000
```

Runtime endpoints:

- `GET /status` - service, camera, model, and latest inspection state.
- `GET /latest_detection` - most recent raw/stable detection result.

By default the runtime uses `models/<profile>/config.json`, `models/<profile>/classes.txt`, and the profile model path from `config.json`. If no model path is configured, it falls back to `models/<profile>/latest/best.pt`, then `models/<profile>/best.pt`.

Useful environment variables for deployment:

- `VISION_MODEL_PROFILE` - model profile name, default `mouse`.
- `VISION_MODEL_PATH` - optional explicit model path.
- `VISION_CAMERA_INDEX` - camera index, default `0`.
- `VISION_HOST` - API bind host, default `0.0.0.0`.
- `VISION_PORT` - API port, default `8000`.

## Train A Model

From the app, open the **Train Model** tab, enter the dataset/profile name, and click **Train Model**.

From the command line:

```bash
python training/train_pipeline.py
```

Enter the dataset/profile name when prompted. Training validates the dataset first. If validation fails, it prints a short summary and detailed errors. After successful training, the latest model is packaged under `models/<profile>/` and the app loads that latest model automatically.

## Storage Layout

- `data/datasets/<profile>/` - YOLO datasets.
- `data/datasets/<profile>/images/train` - training images.
- `data/datasets/<profile>/images/val` - validation images.
- `data/datasets/<profile>/labels/train` - training labels.
- `data/datasets/<profile>/labels/val` - validation labels.
- `data/datasets/<profile>/data.yaml` - YOLO dataset config and class names.
- `models/<profile>/best.pt` - compatibility copy of the latest trained model.
- `models/<profile>/latest/best.pt` - latest packaged model.
- `models/<profile>/versions/vN/best.pt` - versioned model weights.
- `models/<profile>/config.json` - active profile metadata.
- `models/<profile>/training_report.json` - dataset counts, version, run folder, class list, and YOLO metrics when available.
- `data/logs/detections.csv` - structured detection event log.
- `data/review_images/` - review images saved by inference.
- `app/runtime/` - Raspberry Pi runtime service modules.
- `deploy/` - systemd service and Pi install/start scripts.

## Raspberry Pi Service Deployment

On the Raspberry Pi, clone or copy this repository, then run:

```bash
deploy/install_pi.sh
deploy/start_service.sh
```

The install script creates a local virtual environment, installs Python dependencies, prepares data folders, and installs `vision.service` for systemd. The service runs:

```bash
python -m app.runtime.detector_service
```

After startup, check the runtime locally:

```bash
curl http://127.0.0.1:8000/status
curl http://127.0.0.1:8000/latest_detection
```

## Dataset Expectations

Datasets use YOLO detection format. Each image must have a matching `.txt` label with lines like:

```text
class_id x_center y_center width height
```

All coordinates must be normalized from `0` to `1`. Class ids must match the classes listed in `data.yaml`.
