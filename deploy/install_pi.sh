#!/usr/bin/env bash
set -euo pipefail

# Raspberry Pi / industrial Linux installer for the runtime service.
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
  libgl1 \
  libglib2.0-0 \
  libjpeg62-turbo \
  libopenblas0

python3 -m venv "$PROJECT_DIR/.venv"
"$PROJECT_DIR/.venv/bin/python" -m pip install --upgrade pip
"$PROJECT_DIR/.venv/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"

mkdir -p \
  "$PROJECT_DIR/data/datasets" \
  "$PROJECT_DIR/data/logs" \
  "$PROJECT_DIR/data/review_images" \
  "$PROJECT_DIR/models"

sudo install -m 0644 "$SCRIPT_DIR/vision.service" "$SERVICE_FILE"
sudo sed -i "s|^WorkingDirectory=.*|WorkingDirectory=$PROJECT_DIR|" "$SERVICE_FILE"
sudo sed -i "s|^ExecStart=.*|ExecStart=$PROJECT_DIR/.venv/bin/python -m app.runtime.detector_service|" "$SERVICE_FILE"
sudo sed -i "s|^User=.*|User=$SERVICE_USER|" "$SERVICE_FILE"
sudo sed -i "s|^Group=.*|Group=$SERVICE_GROUP|" "$SERVICE_FILE"

sudo systemctl daemon-reload

echo "Installed $SERVICE_FILE"
echo "Review deploy/vision.service environment values before production use."
echo "Start with: deploy/start_service.sh"
