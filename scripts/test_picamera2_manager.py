#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.runtime.picamera2_manager import Picamera2CameraManager


def main():
    parser = argparse.ArgumentParser(description="Smoke test the Picamera2 runtime camera manager.")
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    camera = Picamera2CameraManager(
        frame_width=args.frame_width,
        frame_height=args.frame_height,
    )

    try:
        if not camera.open():
            print(f"FAILED to open camera: {camera.last_error}")
            raise SystemExit(1)

        deadline = time.time() + max(0.1, args.seconds)
        frames = 0
        last_print = 0.0

        while time.time() < deadline:
            frame = camera.read_frame()
            if frame is not None:
                frames += 1

            now = time.time()
            if now - last_print >= 1.0:
                last_print = now
                status = camera.get_status()
                shape = tuple(frame.shape) if frame is not None else None
                print(
                    "status={status} connected={connected} fps={fps} "
                    "shape={shape} error={error}".format(
                        status=status["status"],
                        connected=status["connected"],
                        fps=status["fps"],
                        shape=shape,
                        error=status["error"],
                    ),
                    flush=True,
                )
            time.sleep(0.02)

        print(f"Read {frames} cached frame(s) in {args.seconds:g} second(s).")
        if frames == 0:
            raise SystemExit(2)
    finally:
        camera.release()


if __name__ == "__main__":
    main()
