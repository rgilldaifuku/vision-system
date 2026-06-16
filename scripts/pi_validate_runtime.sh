#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PROFILE="${PROFILE:-yellow_daifuku}"
CAMERA_BACKEND="${CAMERA_BACKEND:-picamera2}"

pass() {
  echo "[PASS] $*"
}

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

echo "== Raspberry Pi runtime validation =="
echo "Project: $PROJECT_DIR"
echo "Profile: $PROFILE"
echo "Camera backend: $CAMERA_BACKEND"
echo

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  fail "Missing venv at $VENV_DIR. Run deploy/install_pi_runtime.sh first."
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

if command -v rpicam-hello >/dev/null 2>&1; then
  rpicam-hello --list-cameras
  pass "rpicam-hello camera listing completed"
elif command -v libcamera-hello >/dev/null 2>&1; then
  libcamera-hello --list-cameras
  pass "libcamera-hello camera listing completed"
else
  fail "Neither rpicam-hello nor libcamera-hello was found"
fi

python -m app.runtime.health_check \
  --mode pi \
  --profile "$PROFILE" \
  --camera-backend "$CAMERA_BACKEND"
pass "Runtime health check passed"

python scripts/test_picamera2_manager.py
pass "Picamera2 manager smoke test passed"

echo
pass "Pi runtime validation complete"
