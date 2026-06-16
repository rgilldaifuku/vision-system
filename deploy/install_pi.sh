#!/usr/bin/env bash
set -euo pipefail

# Raspberry Pi / industrial Linux installer for the runtime service.
# Target: Raspberry Pi 5 + Raspberry Pi Camera Module 3 using Picamera2/libcamera.
# Run from the repository root, or set VISION_PROJECT_DIR explicitly.
# Optional:
#   VISION_PROJECT_DIR=/opt/vision-system
#   VISION_SERVICE_USER=pi
#   VISION_SERVICE_GROUP=pi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${VISION_PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SERVICE_USER="${VISION_SERVICE_USER:-$(id -un)}"
SERVICE_GROUP="${VISION_SERVICE_GROUP:-$(id -gn)}"
SERVICE_FILE="/etc/systemd/system/vision.service"

echo "Installing Vision System runtime from: $PROJECT_DIR"
echo "Service user/group: $SERVICE_USER:$SERVICE_GROUP"

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

sudo apt-get install -y python3-psutil || true

# Picamera2 is usually provided by Raspberry Pi OS packages, not plain pip.
# --system-site-packages lets the venv see python3-picamera2/python3-libcamera.
python3 -m venv --system-site-packages "$PROJECT_DIR/.venv"
"$PROJECT_DIR/.venv/bin/python" -m pip install --upgrade pip
"$PROJECT_DIR/.venv/bin/python" -m pip install -r "$PROJECT_DIR/requirements-pi-runtime.txt"
"$PROJECT_DIR/.venv/bin/python" -m pip uninstall -y numpy opencv-python opencv-contrib-python || true

mkdir -p \
  "$PROJECT_DIR/data/datasets" \
  "$PROJECT_DIR/data/logs" \
  "$PROJECT_DIR/data/review_images" \
  "$PROJECT_DIR/data/debug_frames/raw" \
  "$PROJECT_DIR/data/debug_frames/annotated" \
  "$PROJECT_DIR/data/debug_frames/manual" \
  "$PROJECT_DIR/models"

sudo install -m 0644 "$SCRIPT_DIR/vision.service" "$SERVICE_FILE"
sudo sed -i "s|^WorkingDirectory=.*|WorkingDirectory=$PROJECT_DIR|" "$SERVICE_FILE"
sudo sed -i "s|^ExecStart=.*|ExecStart=$PROJECT_DIR/.venv/bin/python -m app.runtime.detector_service --profile yellow_daifuku --camera-backend picamera2 --host 0.0.0.0 --port 8000 --imgsz 256 --frame-width 640 --frame-height 480 --inference-interval-ms 300|" "$SERVICE_FILE"
sudo sed -i "s|^User=.*|User=$SERVICE_USER|" "$SERVICE_FILE"
sudo sed -i "s|^Group=.*|Group=$SERVICE_GROUP|" "$SERVICE_FILE"

sudo systemctl daemon-reload

echo "Installed $SERVICE_FILE"
echo
echo "Next commands:"
echo "  source \"$PROJECT_DIR/.venv/bin/activate\""
echo "  python -m app.runtime.health_check --mode pi --profile yellow_daifuku --camera-backend picamera2"
echo "  python -m app.runtime.detector_service --profile yellow_daifuku --camera-backend picamera2 --host 0.0.0.0 --port 8000 --imgsz 256 --frame-width 640 --frame-height 480 --inference-interval-ms 300"
echo "  deploy/start_service.sh"
