#!/usr/bin/env python3
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from app.config import DATA_DIR
from app.runtime.picamera2_manager import Picamera2CameraManager


def main():
    parser = argparse.ArgumentParser(description="Capture one Picamera2 frame for focus/lighting checks.")
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    parser.add_argument("--output-dir", default=str(DATA_DIR / "debug_frames" / "manual"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = Picamera2CameraManager(
        frame_width=args.frame_width,
        frame_height=args.frame_height,
    )
    try:
        if not camera.open():
            print(f"FAILED to open camera: {camera.last_error}")
            raise SystemExit(1)

        frame = None
        deadline = time.time() + 5.0
        while time.time() < deadline:
            frame = camera.read_frame()
            if frame is not None:
                break
            time.sleep(0.05)

        if frame is None:
            print(f"FAILED to capture frame: {camera.last_error}")
            raise SystemExit(2)

        output_path = output_dir / f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        if not cv2.imwrite(str(output_path), frame):
            print(f"FAILED to write image: {output_path}")
            raise SystemExit(3)

        print(output_path)
    finally:
        camera.release()


if __name__ == "__main__":
    main()
