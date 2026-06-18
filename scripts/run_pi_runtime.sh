#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"

PROFILE="${PROFILE:-yellow_daifuku}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
IMGSZ="${IMGSZ:-256}"
FRAME_WIDTH="${FRAME_WIDTH:-640}"
FRAME_HEIGHT="${FRAME_HEIGHT:-480}"
INTERVAL_MS="${INTERVAL_MS:-300}"
CAMERA_BACKEND="${CAMERA_BACKEND:-picamera2}"
CAMERA_PROFILE="${CAMERA_PROFILE:-pi_camera3}"
CAMERA_ONLY="${CAMERA_ONLY:-0}"
DISABLE_INFERENCE="${DISABLE_INFERENCE:-0}"
MODEL_FORMAT="${MODEL_FORMAT:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "[FAIL] Missing venv at $VENV_DIR. Run deploy/install_pi_runtime.sh first." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "Starting Pi runtime:"
echo "  profile=$PROFILE backend=$CAMERA_BACKEND host=$HOST port=$PORT"
echo "  imgsz=$IMGSZ frame=${FRAME_WIDTH}x${FRAME_HEIGHT} interval_ms=$INTERVAL_MS"
if [ -n "$CAMERA_PROFILE" ]; then
  echo "  camera_profile=$CAMERA_PROFILE"
fi

ARGS=(
  --profile "$PROFILE"
  --camera-backend "$CAMERA_BACKEND"
  --host "$HOST"
  --port "$PORT"
  --imgsz "$IMGSZ"
  --frame-width "$FRAME_WIDTH"
  --frame-height "$FRAME_HEIGHT"
  --inference-interval-ms "$INTERVAL_MS"
  --prefer-edge-model
  --model-format "$MODEL_FORMAT"
)

if [ -n "$CAMERA_PROFILE" ]; then
  ARGS+=(--camera-profile "$CAMERA_PROFILE")
fi

if [ "$CAMERA_ONLY" = "1" ] || [ "$CAMERA_ONLY" = "true" ]; then
  ARGS+=(--camera-only --enable-snapshot)
fi

if [ "$DISABLE_INFERENCE" = "1" ] || [ "$DISABLE_INFERENCE" = "true" ]; then
  ARGS+=(--disable-inference)
fi

# shellcheck disable=SC2086
python -m app.runtime.detector_service "${ARGS[@]}" $EXTRA_ARGS
