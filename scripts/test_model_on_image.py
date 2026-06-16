#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import yaml
from ultralytics import YOLO

from app.config import DATA_DIR, MODELS_DIR


def main():
    parser = argparse.ArgumentParser(description="Run one YOLO model/profile on one image.")
    parser.add_argument("--profile", default="yellow_daifuku")
    parser.add_argument("--image", required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--model")
    parser.add_argument("--output-dir", default=str(DATA_DIR / "debug_frames" / "model_tests"))
    args = parser.parse_args()

    profile_dir = MODELS_DIR / args.profile
    model_path = resolve_model_path(profile_dir, args.model)
    classes = load_classes(profile_dir)
    profile_config = load_profile_config(profile_dir)
    expected_classes = (
        profile_config.get("target_classes")
        or profile_config.get("inspection", {}).get("acceptable_classes")
        or classes
    )

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image does not exist: {image_path}")
    if not model_path.exists():
        raise SystemExit(f"Model does not exist: {model_path}")

    model = YOLO(str(model_path))
    print(f"Profile: {args.profile}")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"classes.txt: {classes}")
    print(f"expected profile classes: {expected_classes}")
    print(f"model names: {model.names}")

    results = model.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, verbose=False)
    detections = extract_detections(results)
    print(json.dumps(detections, indent=2, sort_keys=True))

    model_class_names = {str(name) for name in model.names.values()}
    for expected in expected_classes:
        if expected not in model_class_names:
            print(f"WARNING: expected class '{expected}' not present in model.names")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    annotated = results[0].plot()
    cv2.imwrite(str(output_path), annotated)
    print(f"Annotated image: {output_path}")


def resolve_model_path(profile_dir, model_override=None):
    if model_override:
        path = Path(model_override)
        return path if path.is_absolute() else PROJECT_ROOT / path

    config_path = profile_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        configured_model = config.get("model_file")
        if configured_model:
            return profile_dir / configured_model

    latest_model = profile_dir / "latest" / "best.pt"
    if latest_model.exists():
        return latest_model
    return profile_dir / "best.pt"


def load_classes(profile_dir):
    classes_path = profile_dir / "classes.txt"
    if not classes_path.exists():
        return []
    return [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_profile_config(profile_dir):
    config = {}
    config_json = profile_dir / "config.json"
    if config_json.exists():
        config.update(json.loads(config_json.read_text(encoding="utf-8")))

    config_yaml = PROJECT_ROOT / "profiles" / profile_dir.name / "config.yaml"
    if config_yaml.exists():
        yaml_config = yaml.safe_load(config_yaml.read_text(encoding="utf-8")) or {}
        config = deep_merge(config, yaml_config)
    return config


def deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def extract_detections(results):
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": result.names.get(class_id, str(class_id)),
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].cpu().numpy().tolist(),
                }
            )
    return detections


if __name__ == "__main__":
    main()
