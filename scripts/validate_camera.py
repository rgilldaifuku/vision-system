#!/usr/bin/env python3
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from app.config import DATA_DIR
from app.runtime.camera_manager import CameraManager
from app.runtime.camera_profile import load_camera_profile
from app.runtime.image_quality import INVALID_FRAME, compute_image_quality, quality_thresholds_from_config
from app.runtime.picamera2_manager import Picamera2CameraManager
from app.runtime.preprocessing import preprocess_for_inference


def main():
    parser = argparse.ArgumentParser(description="Validate camera capture and image quality.")
    parser.add_argument("--camera-profile", default="pi_camera3")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output-dir", default=str(DATA_DIR / "debug_frames" / "camera_validation"))
    args = parser.parse_args()

    profile = load_camera_profile(args.camera_profile)
    camera = create_camera(profile, args.camera)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    frames = 0
    quality_records = []
    sample_path = output_dir / f"{profile.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    sample_saved = False

    try:
        if not camera.open():
            print(f"FAIL: camera did not open: {getattr(camera, 'last_error', '')}")
            raise SystemExit(1)

        deadline = time.time() + max(0.1, args.duration)
        last_shape = None
        while time.time() < deadline:
            frame = camera.read_frame()
            if frame is None:
                time.sleep(0.02)
                continue

            processed, preprocessing = preprocess_for_inference(
                frame,
                camera_profile=profile,
                imgsz=None,
            )
            quality = compute_image_quality(
                processed,
                quality_thresholds_from_config(profile.quality),
            )
            quality_records.append(quality)
            frames += 1
            last_shape = processed.shape
            if not sample_saved and cv2.imwrite(str(sample_path), processed):
                sample_saved = True
                write_json(
                    sample_path.with_suffix(".json"),
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "camera_profile": profile.to_dict(),
                        "image_quality": quality,
                        "preprocessing": preprocessing,
                        "sample_frame": str(sample_path),
                    },
                )

        elapsed = max(0.001, time.time() - started)
        report = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "camera_profile": profile.to_dict(),
            "duration_seconds": round(elapsed, 3),
            "frames": frames,
            "fps": round(frames / elapsed, 3),
            "frame_shape": list(last_shape) if last_shape is not None else None,
            "sample_frame": str(sample_path) if sample_saved else "",
            "quality_average": average_quality(quality_records),
            "last_camera_error": getattr(camera, "last_error", ""),
        }
        report_path = output_dir / f"{profile.name}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        write_json(report_path, report)

        print(json.dumps(report, indent=2, sort_keys=True))
        if frames <= 0:
            print("FAIL: camera opened but returned no frames")
            raise SystemExit(1)
        if report["quality_average"].get("quality_status") == INVALID_FRAME:
            print("FAIL: image quality status is INVALID_FRAME")
            raise SystemExit(1)
        print(f"PASS: camera returned {frames} frame(s), report: {report_path}")
    finally:
        camera.release()


def create_camera(profile, camera_index):
    if profile.backend == "picamera2":
        return Picamera2CameraManager(
            frame_width=profile.width,
            frame_height=profile.height,
            target_fps=profile.fps,
        )
    return CameraManager(
        camera_index=camera_index,
        frame_width=profile.width,
        frame_height=profile.height,
    )


def average_quality(records):
    if not records:
        return {"quality_status": INVALID_FRAME}

    numeric_keys = (
        "brightness_mean",
        "brightness_std",
        "contrast_score",
        "blur_score",
        "overexposed_pct",
        "underexposed_pct",
    )
    averaged = {}
    for key in numeric_keys:
        values = [record.get(key) for record in records if isinstance(record.get(key), (int, float))]
        averaged[key] = round(sum(values) / len(values), 3) if values else None
    averaged["width"] = records[-1].get("width", 0)
    averaged["height"] = records[-1].get("height", 0)
    averaged["timestamp"] = datetime.now().isoformat(timespec="seconds")
    averaged["quality_status"] = records[-1].get("quality_status", INVALID_FRAME)
    return averaged


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
