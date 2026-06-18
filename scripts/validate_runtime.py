#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import MODELS_DIR
from app.runtime.camera_manager import CameraManager
from app.runtime.camera_profile import load_camera_profile
from app.runtime.image_quality import GOOD, compute_image_quality, quality_thresholds_from_config
from app.runtime.inference_engine import (
    InferenceEngine,
    MODEL_FORMAT_AUTO,
    MODEL_FORMAT_NCNN,
    MODEL_FORMAT_PT,
    resolve_model_path,
)
from app.runtime.picamera2_manager import Picamera2CameraManager
from app.runtime.preprocessing import preprocess_for_inference
from scripts.test_model_on_image import load_classes, load_profile_config


def main():
    parser = argparse.ArgumentParser(description="Validate camera, preprocessing, quality, and model path.")
    parser.add_argument("--profile", default="yellow_daifuku")
    parser.add_argument("--camera-profile", default="pi_camera3")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--model")
    parser.add_argument(
        "--model-format",
        choices=(MODEL_FORMAT_AUTO, MODEL_FORMAT_PT, MODEL_FORMAT_NCNN),
        default=MODEL_FORMAT_AUTO,
    )
    parser.add_argument("--prefer-edge-model", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    profile_dir = MODELS_DIR / args.profile
    camera_profile = load_camera_profile(args.camera_profile)
    profile_config = load_profile_config(profile_dir)
    model_path, model_format, warning = resolve_model_path(
        profile_dir,
        config=profile_config,
        model_override=args.model,
        model_format=args.model_format,
        prefer_edge_model=args.prefer_edge_model,
    )

    report = {
        "camera": {},
        "preprocessing": {},
        "image_quality": {},
        "model": {
            "profile": args.profile,
            "path": str(model_path),
            "format": model_format,
            "exists": model_path.exists(),
            "warning": warning,
            "classes": load_classes(profile_dir),
        },
        "inference": {"attempted": False, "succeeded": False, "detections": []},
    }

    camera = create_camera(camera_profile, args.camera)
    try:
        if not camera.open():
            report["camera"] = {"status": "FAIL", "error": getattr(camera, "last_error", "")}
            print_report(report)
            raise SystemExit(1)

        frame = read_frame_until(camera, args.duration)
        if frame is None:
            report["camera"] = {"status": "FAIL", "error": "Camera returned no frames."}
            print_report(report)
            raise SystemExit(1)

        processed, preprocessing = preprocess_for_inference(frame, camera_profile, args.imgsz)
        quality = compute_image_quality(
            processed,
            quality_thresholds_from_config(camera_profile.quality),
        )
        report["camera"] = {
            "status": "PASS",
            "backend": camera_profile.backend,
            "profile": camera_profile.name,
            "frame_shape": list(frame.shape),
        }
        report["preprocessing"] = preprocessing
        report["image_quality"] = quality

        if quality.get("quality_status") != GOOD:
            report["image_quality"]["validation_warning"] = (
                "Camera works, but image quality may reduce detection reliability."
            )

        if not args.skip_inference and model_path.exists():
            report["inference"]["attempted"] = True
            engine = InferenceEngine(model_path, model_format)
            _, detections = engine.predict(processed, confidence=0.10, imgsz=args.imgsz)
            report["inference"]["succeeded"] = True
            report["inference"]["detections"] = detections

        print_report(report)
        if not model_path.exists():
            print("WARN: model path does not exist; camera/preprocessing validation still passed.")
        print("PASS: runtime validation completed.")
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


def read_frame_until(camera, duration):
    deadline = time.time() + max(0.1, duration)
    while time.time() < deadline:
        frame = camera.read_frame()
        if frame is not None:
            return frame
        time.sleep(0.02)
    return None


def print_report(report):
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
