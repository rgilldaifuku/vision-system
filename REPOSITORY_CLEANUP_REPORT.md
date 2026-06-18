# Repository Cleanup Report

This report documents the current repository shape and cleanup decisions for the industrial vision prototype.

## Essential Runtime Files

- `app/runtime/detector_service.py` - Flask dashboard/API and runtime loop.
- `app/runtime/inference_engine.py` - `.pt` and NCNN model selection/inference adapter.
- `app/runtime/camera_profile.py` - camera profile loading and validation.
- `app/runtime/camera_manager.py` - OpenCV/USB camera wrapper.
- `app/runtime/picamera2_manager.py` - Picamera2/libcamera camera wrapper.
- `app/runtime/camera_sources.py` - image/folder/video simulation source.
- `app/runtime/inspection_logic.py` - stable PASS/FAIL/NO_PART/LOW_CONFIDENCE/CAMERA_ERROR/MODEL_ERROR/SIMULATION decisions.
- `app/runtime/output_manager.py` and `app/runtime/action_manager.py` - CSV/status JSON output and safe local actions.
- `cameras/pi_camera3.yaml` and `cameras/usb_webcam.yaml` - camera defaults.
- `scripts/run_pi_runtime.sh`, `scripts/run_camera_dashboard.sh`, `scripts/pi_validate_runtime.sh` - Pi runtime launch and validation.
- `deploy/install_pi_runtime.sh` and `requirements-pi-runtime.txt` - Pi-safe dependency workflow.

## Essential Desktop/Training Files

- `app/main.py`, `app/ui.py`, `app/inference.py` - desktop engineering app.
- `training/train_pipeline.py`, `training/collect_images.py`, `training/label_images.py`, `training/verify_dataset.py` - training/data utilities.
- `requirements.txt` - desktop/training dependencies only.
- `data/datasets/` - YOLO training datasets.

## Essential Model/Profile Files

- `models/yellow_daifuku/best.pt` - desktop/training model artifact.
- `models/yellow_daifuku/best_ncnn_model/` - Raspberry Pi edge model artifact.
- `models/yellow_daifuku/classes.txt` and `models/yellow_daifuku/config.json`.
- `profiles/yellow_daifuku/config.yaml` - runtime inspection rules.

## Generated Or Ignored Files

These should not be committed:

- Python caches: `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`.
- Runtime outputs: `data/logs/`, `data/review_images/`, `data/debug_frames/`, `data/runtime/`, `data/calibration/`.
- Training/runtime cache files: `data/**/*.cache`.
- Local environments: `.venv/`, `venv/`, `.env*`.
- Build outputs: `build/`, `dist/`, `*.egg-info/`.

## Files Kept But Worth Reviewing Later

- `archive/` contains old reference code. It is not part of the main workflow.
- `app/camara.py` appears to be a legacy/typo-named camera helper. It was not removed because it may still be referenced manually.
- `models/mouse/` contains multiple historical `.pt` versions. They are not part of the `yellow_daifuku` Pi workflow and may be candidates for manual archival.
- `models/yolov8n.pt` is a generic base model. Keep it only if still used for training/bootstrap.
- `ansible/`, `Dockerfile`, `docker-compose.yml`, and `hardware/` are deployment/hardware experiments and are not required for the current Pi Camera 3 MVP path.
- `docs/New Text Document.txt` looks like an accidental placeholder, but was left untouched.

## Model Artifact Notes

Preserved intentionally:

- `models/yellow_daifuku/best.pt`
- `models/yellow_daifuku/best_ncnn_model/model.ncnn.param`
- `models/yellow_daifuku/best_ncnn_model/model.ncnn.bin`
- `models/yellow_daifuku/best_ncnn_model/metadata.yaml`
- `models/yellow_daifuku/classes.txt`
- `models/yellow_daifuku/config.json`

Potential duplicate/legacy model files found but not deleted:

- `models/mouse/best.pt`
- `models/mouse/latest/best.pt`
- `models/mouse/versions/v1` through `v8`
- `models/yolov8n.pt`

## Safe Local Cleanup Commands

These commands remove generated local files only. They do not delete datasets or models.

```bash
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete
find data/datasets -type f -name '*.cache' -delete
rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov
rm -rf data/debug_frames/*
rm -f data/logs/*.csv data/logs/*.json
```

Review images can be useful retraining evidence. Delete them only after copying anything useful:

```bash
rm -rf data/review_images/*
```
