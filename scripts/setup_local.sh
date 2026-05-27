#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python command not found: $PYTHON_BIN"
  echo "Set PYTHON_BIN=/path/to/python3 and run this script again."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating .venv with $PYTHON_BIN"
  "$PYTHON_BIN" -m venv .venv
else
  echo "Using existing .venv"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p \
  data/datasets \
  data/logs \
  data/review_images \
  samples/test_images \
  profiles

if [ ! -f "samples/test.jpg" ] && [ -f "assets/test.jpg" ]; then
  cp assets/test.jpg samples/test.jpg
  echo "Copied assets/test.jpg to samples/test.jpg"
fi

echo
echo "Local setup complete."
echo
echo "Next commands:"
echo "  source .venv/bin/activate"
echo "  python -m app.runtime.health_check --mode laptop --profile yellow_daifuku --camera-source assets/test.jpg"
echo "  scripts/run_demo.sh"
echo
echo "Dashboard after demo starts:"
echo "  http://127.0.0.1:8000"
