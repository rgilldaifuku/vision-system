#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -x ".venv/bin/python" ]; then
  echo "Missing .venv. Run scripts/setup_local.sh first."
  exit 1
fi

PROFILE="${VISION_MODEL_PROFILE:-yellow_daifuku}"
HOST="${VISION_HOST:-127.0.0.1}"
PORT="${VISION_PORT:-8000}"
CAMERA_SOURCE="${VISION_CAMERA_SOURCE:-}"

if [ -z "$CAMERA_SOURCE" ]; then
  if [ -f "samples/test.jpg" ]; then
    CAMERA_SOURCE="samples/test.jpg"
  elif [ -f "assets/test.jpg" ]; then
    CAMERA_SOURCE="assets/test.jpg"
  else
    first_sample="$(find samples/test_images -type f \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) 2>/dev/null | head -n 1 || true)"
    CAMERA_SOURCE="$first_sample"
  fi
fi

if [ -z "$CAMERA_SOURCE" ]; then
  echo "No simulated camera source found."
  echo "Run scripts/setup_local.sh or place an image at samples/test.jpg."
  exit 1
fi

ARGS=(
  .venv/bin/python -m app.runtime.detector_service
  --profile "$PROFILE"
  --camera-source "$CAMERA_SOURCE"
  --host "$HOST"
  --port "$PORT"
  --imgsz "${VISION_IMGSZ:-256}"
  --frame-width "${VISION_FRAME_WIDTH:-424}"
  --frame-height "${VISION_FRAME_HEIGHT:-240}"
  --inference-interval-ms "${VISION_INFERENCE_INTERVAL_MS:-300}"
)

if [ "${VISION_FORCE_DRY_RUN:-}" = "1" ] || [ "${VISION_FORCE_DRY_RUN:-}" = "true" ]; then
  ARGS+=(--dry-run)
elif [ ! -f "models/$PROFILE/best.pt" ] && [ ! -f "models/$PROFILE/latest/best.pt" ]; then
  ARGS+=(--dry-run)
fi

echo "Starting local vision demo"
echo "  Profile: $PROFILE"
echo "  Camera source: $CAMERA_SOURCE"
echo "  Dashboard: http://$HOST:$PORT"
if [[ " ${ARGS[*]} " == *" --dry-run "* ]]; then
  echo "  Mode: dry-run simulation"
else
  echo "  Mode: simulated camera with real model"
fi
echo

exec "${ARGS[@]}"
