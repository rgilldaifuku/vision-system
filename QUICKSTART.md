# Quickstart

Use this when you want the shortest path from a fresh checkout to a running local dashboard.

## Laptop Demo Today

```bash
scripts/setup_local.sh
source .venv/bin/activate
scripts/run_demo.sh
```

Open:

```text
http://127.0.0.1:8000
```

The demo uses a simulated image source and binds only to `127.0.0.1`. If model weights are missing, it automatically uses dry-run simulation mode.

## Test With A Sample Image

```bash
python -m app.runtime.detector_service \
  --profile yellow_daifuku \
  --camera-source assets/test.jpg \
  --dry-run \
  --host 127.0.0.1 \
  --port 8000
```

## Test With A Folder Of Images

```bash
mkdir -p samples/test_images
cp assets/*.jpg samples/test_images/

python -m app.runtime.detector_service \
  --profile yellow_daifuku \
  --camera-source samples/test_images/ \
  --dry-run \
  --host 127.0.0.1 \
  --port 8000
```

## Health Check

Laptop/simulation:

```bash
python -m app.runtime.health_check \
  --mode laptop \
  --profile yellow_daifuku \
  --camera-source assets/test.jpg
```

Later on Raspberry Pi with USB camera:

```bash
python -m app.runtime.health_check \
  --mode pi \
  --profile yellow_daifuku \
  --camera 0
```

## API Checks

With the runtime running:

```bash
curl http://127.0.0.1:8000/status
curl http://127.0.0.1:8000/latest_detection
curl -I http://127.0.0.1:8000/snapshot.jpg
```

## Later On Raspberry Pi

1. Copy the repo and model profile to the Pi.
2. Run `deploy/install_pi.sh`.
3. Run `python -m app.runtime.health_check --mode pi --profile yellow_daifuku --camera 0`.
4. Run `deploy/start_service.sh`.
5. Open `http://<pi-ip>:8000` from the HMI/browser.

Keep `--enable-snapshot` off for production-style demos. Snapshot mode is useful for debugging, but the operator dashboard is designed to be status-first, not video-first.
