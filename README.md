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

## Dataset Expectations

Datasets use YOLO detection format. Each image must have a matching `.txt` label with lines like:

```text
class_id x_center y_center width height
```

All coordinates must be normalized from `0` to `1`. Class ids must match the classes listed in `data.yaml`.
