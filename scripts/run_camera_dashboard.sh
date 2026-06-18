#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"

PROFILE="${PROFILE:-yellow_daifuku}"
CAMERA_PROFILE="${CAMERA_PROFILE:-pi_camera3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "[FAIL] Missing venv at $VENV_DIR. Run deploy/install_pi_runtime.sh first." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "Starting camera-only dashboard:"
echo "  profile=$PROFILE camera_profile=$CAMERA_PROFILE host=$HOST port=$PORT"

# shellcheck disable=SC2086
python -m app.runtime.detector_service \
  --profile "$PROFILE" \
  --camera-profile "$CAMERA_PROFILE" \
  --camera-only \
  --enable-snapshot \
  --host "$HOST" \
  --port "$PORT" \
  $EXTRA_ARGS
