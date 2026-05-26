#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${VISION_PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SERVICE_USER="${VISION_SERVICE_USER:-$(id -un)}"
SERVICE_GROUP="${VISION_SERVICE_GROUP:-$(id -gn)}"
SERVICE_FILE="/etc/systemd/system/vision.service"

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libgl1 libglib2.0-0

python3 -m venv "$PROJECT_DIR/.venv"
"$PROJECT_DIR/.venv/bin/python" -m pip install --upgrade pip
"$PROJECT_DIR/.venv/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"

mkdir -p "$PROJECT_DIR/data/logs" "$PROJECT_DIR/data/review_images"

sudo install -m 0644 "$SCRIPT_DIR/vision.service" "$SERVICE_FILE"
sudo sed -i "s|^WorkingDirectory=.*|WorkingDirectory=$PROJECT_DIR|" "$SERVICE_FILE"
sudo sed -i "s|^ExecStart=.*|ExecStart=$PROJECT_DIR/.venv/bin/python -m app.runtime.detector_service --profile \${VISION_MODEL_PROFILE} --camera \${VISION_CAMERA_INDEX} --host \${VISION_HOST} --port \${VISION_PORT}|" "$SERVICE_FILE"
sudo sed -i "s|^User=.*|User=$SERVICE_USER|" "$SERVICE_FILE"
sudo sed -i "s|^Group=.*|Group=$SERVICE_GROUP|" "$SERVICE_FILE"

sudo systemctl daemon-reload

echo "Installed vision.service for $PROJECT_DIR"
echo "Start it with: deploy/start_service.sh"

