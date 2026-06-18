# Raspberry Pi Runtime Setup

This is the recommended setup path for Raspberry Pi 4/5, Raspberry Pi Camera Module 3, Raspberry Pi OS Bookworm, Picamera2/libcamera, and the `yellow_daifuku` YOLO profile.

## Why The Pi Uses A Separate Install

Do not run the full desktop/training install on the Pi runtime:

```bash
pip install -r requirements.txt
```

That file is for the desktop engineering/training app. It includes desktop packages and can install pip `numpy` or `opencv-python` into the venv. On Raspberry Pi OS, Picamera2 and simplejpeg are apt/system packages and can break when mixed with pip NumPy/OpenCV binary wheels.

The Pi runtime should use apt packages for camera-critical dependencies:

```text
python3-picamera2
python3-libcamera
libcamera-apps
python3-opencv
python3-numpy
python3-flask
python3-yaml
python3-requests
```

The venv must be created with:

```bash
/usr/bin/python3 -m venv --system-site-packages .venv
```

## Install

From the repository root on the Pi:

```bash
deploy/install_pi_runtime.sh
```

The installer:

- Installs runtime apt packages.
- Creates `.venv` with `--system-site-packages`.
- Installs minimal pip runtime dependencies from `requirements-pi-runtime.txt`.
- Removes pip `numpy`, `opencv-python`, and `opencv-contrib-python` if they were pulled in.
- Prints the file paths for NumPy and OpenCV so you can confirm they are not loading from `.venv`.

## Validate

```bash
scripts/pi_validate_runtime.sh
```

This runs:

- `rpicam-hello --list-cameras` or `libcamera-hello --list-cameras`
- `python -m app.runtime.health_check --mode pi --profile yellow_daifuku --camera-backend picamera2`
- `python scripts/test_picamera2_manager.py`

## Run

```bash
scripts/run_pi_runtime.sh
```

Override defaults with environment variables:

```bash
PROFILE=yellow_daifuku IMGSZ=320 scripts/run_pi_runtime.sh
```

The default command is equivalent to:

```bash
python -m app.runtime.detector_service \
  --profile yellow_daifuku \
  --camera-backend picamera2 \
  --prefer-edge-model \
  --model-format auto \
  --host 0.0.0.0 \
  --port 8000 \
  --imgsz 256 \
  --frame-width 640 \
  --frame-height 480 \
  --inference-interval-ms 300
```

## Model Format On Raspberry Pi

Desktop/training should keep using `.pt` model files. Raspberry Pi runtime should use an exported NCNN model when possible.

Expected profile layout:

```text
models/yellow_daifuku/
  best.pt
  classes.txt
  config.json
  best_ncnn_model/
    model.ncnn.param
    model.ncnn.bin
```

Export NCNN on the desktop/work computer, not on the Pi:

```bash
python scripts/export_profile_to_ncnn.py --profile yellow_daifuku --imgsz 320
```

Then copy this folder to the Pi:

```text
models/yellow_daifuku/best_ncnn_model/
```

The Pi launcher uses `--prefer-edge-model --model-format auto`, so it will prefer `best_ncnn_model/` when present and only fall back to `.pt` if no NCNN export exists. On Raspberry Pi 4, `.pt` inference may fail with `Illegal instruction`; NCNN is the recommended deployment format.

## Debugging Object Not Detected

1. Capture a manual camera frame:

```bash
python scripts/capture_picamera2_frame.py
```

Inspect the saved image under:

```text
data/debug_frames/manual/
```

Check focus, lighting, object size, angle, and whether the object is actually in frame.

2. Run the model on that image:

```bash
python scripts/test_model_on_image.py \
  --profile yellow_daifuku \
  --image data/debug_frames/manual/<image>.jpg \
  --imgsz 320 \
  --conf 0.10 \
  --prefer-edge-model \
  --model-format auto
```

This prints raw detections, model class names, profile classes, and writes an annotated image under `data/debug_frames/model_tests/`.

3. Run the detector with debug capture:

```bash
EXTRA_ARGS="--debug-detections --save-debug-frames --debug-frame-limit 20 --confidence-threshold-override 0.15" \
  scripts/run_pi_runtime.sh
```

Debug output is written to:

```text
data/debug_frames/raw/
data/debug_frames/annotated/
data/debug_frames/detections_debug.jsonl
```

Use `detections_debug.jsonl` to determine whether:

- YOLO detects nothing.
- YOLO detects the wrong class.
- YOLO detects below the confidence threshold.
- ROI filtering removes the detection.
- Profile class names do not match the model.
- The model path/profile is wrong.
- Camera color/framing/focus is wrong.

## Dashboard

Open:

```text
http://<pi-ip>:8000
```

Snapshot mode stays off by default. The production runtime is status-first, not smooth-video-first.
