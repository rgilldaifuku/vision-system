#!/usr/bin/env bash
set -euo pipefail

# Raspberry Pi runtime installer for Picamera2 + YOLO inference.
# This intentionally does NOT install the desktop/training requirements file.
# NumPy/OpenCV/Picamera2 should come from Raspberry Pi OS apt packages.
# Export NCNN models on the desktop/work computer and copy best_ncnn_model/
# to the Pi. This installer does not export models on the Pi.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${VISION_PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
VENV_DIR="$PROJECT_DIR/.venv"

echo "== Industrial Vision Pi Runtime Installer =="
echo "Project: $PROJECT_DIR"
echo

echo "== Installing apt runtime packages =="
sudo apt-get update
sudo apt-get install -y \
  python3-venv \
  python3-pip \
  python3-picamera2 \
  python3-libcamera \
  libcamera-apps \
  python3-opencv \
  python3-numpy \
  python3-flask \
  python3-yaml \
  python3-requests \
  libgl1 \
  libglib2.0-0 \
  libjpeg62-turbo \
  libopenblas0

if ! sudo apt-get install -y python3-psutil; then
  echo "WARN: python3-psutil apt package was unavailable; pip psutil may be used instead."
fi

echo
echo "== Creating venv with system site packages =="
/usr/bin/python3 -m venv --system-site-packages "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo
echo "== Installing minimal pip runtime dependencies =="
python -m pip install --upgrade pip
python -m pip install -r "$PROJECT_DIR/requirements-pi-runtime.txt"

echo
echo "== Removing pip NumPy/OpenCV if pip installed them =="
python -m pip uninstall -y numpy opencv-python opencv-contrib-python || true

echo
echo "== Creating runtime folders =="
mkdir -p \
  "$PROJECT_DIR/data/datasets" \
  "$PROJECT_DIR/data/logs" \
  "$PROJECT_DIR/data/review_images" \
  "$PROJECT_DIR/data/debug_frames/raw" \
  "$PROJECT_DIR/data/debug_frames/annotated" \
  "$PROJECT_DIR/data/debug_frames/manual" \
  "$PROJECT_DIR/models"

echo
echo "== Verifying camera/runtime imports and package origins =="
python - <<'PY'
from pathlib import Path
from picamera2 import Picamera2
import cv2
import flask
import numpy
import yaml

print("Picamera2 OK")
print("NumPy:", numpy.__version__, numpy.__file__)
print("OpenCV:", cv2.__version__, cv2.__file__)
print("Flask:", getattr(flask, "__version__", "unknown"))
print("YAML OK:", yaml.__file__)

venv = Path(".venv").resolve()
for name, module in (("NumPy", numpy), ("OpenCV", cv2)):
    module_path = Path(module.__file__).resolve()
    if venv in module_path.parents:
        print(f"WARNING: {name} is loading from the venv: {module_path}")
        print("         On Raspberry Pi runtime this should come from /usr/lib/python3/dist-packages.")
PY

echo
echo "== Next commands =="
echo "  source \"$VENV_DIR/bin/activate\""
echo "  scripts/pi_validate_runtime.sh"
echo "  scripts/run_pi_runtime.sh"
