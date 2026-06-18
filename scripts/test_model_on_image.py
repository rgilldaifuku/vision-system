#!/usr/bin/env python3
import argparse
import json
import platform
import signal
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import yaml

from app.config import DATA_DIR, MODELS_DIR
from app.runtime.inference_engine import (
    InferenceEngine,
    MODEL_FORMAT_AUTO,
    MODEL_FORMAT_NCNN,
    MODEL_FORMAT_PT,
    resolve_model_path,
)
from app.runtime.inspection_logic import normalize_class_name
from app.runtime.image_quality import compute_image_quality


def main():
    install_sigill_message()

    parser = argparse.ArgumentParser(description="Run one YOLO model/profile on one image.")
    parser.add_argument("--profile", default="yellow_daifuku")
    parser.add_argument("--image", required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--model")
    parser.add_argument(
        "--model-format",
        choices=(MODEL_FORMAT_AUTO, MODEL_FORMAT_PT, MODEL_FORMAT_NCNN),
        default=MODEL_FORMAT_AUTO,
    )
    parser.add_argument("--prefer-edge-model", action="store_true")
    parser.add_argument("--output-dir", default=str(DATA_DIR / "debug_frames" / "model_tests"))
    args = parser.parse_args()

    profile_dir = MODELS_DIR / args.profile
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

    model_path, model_format, warning = resolve_model_path(
        profile_dir,
        config=profile_config,
        model_override=args.model,
        model_format=args.model_format,
        prefer_edge_model=args.prefer_edge_model,
    )

    print(f"Profile: {args.profile}")
    print(f"Selected model: {model_path}")
    print(f"Selected model format: {model_format}")
    print(f"Model path exists: {model_path.exists()}")
    print(f"Image: {image_path}")
    print(f"classes.txt: {classes}")
    print(f"expected profile classes: {expected_classes}")
    if warning:
        print(f"WARNING: {warning}")
    if is_pi_like() and model_format == MODEL_FORMAT_PT:
        print(
            "WARNING: PyTorch .pt inference is not recommended on Raspberry Pi. "
            "If this exits with Illegal instruction, export NCNN on desktop and deploy the NCNN folder."
        )

    if not model_path.exists():
        print_export_instructions(args.profile)
        raise SystemExit(1)

    engine = InferenceEngine(model_path, model_format)
    print(f"model names: {engine.names}")

    results, detections = engine.predict(read_image(image_path), confidence=args.conf, imgsz=args.imgsz)
    print(json.dumps(detections, indent=2, sort_keys=True))

    model_class_names = {normalize_class_name(name) for name in engine.names.values()}
    for expected in expected_classes:
        if normalize_class_name(expected) not in model_class_names:
            print(f"WARNING: expected class '{expected}' not present in model.names")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    annotated = results[0].plot()
    cv2.imwrite(str(output_path), annotated)
    sidecar_path = output_path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "profile": args.profile,
                "model_path": str(model_path),
                "model_format": model_format,
                "class_list": classes,
                "model_names": engine.names,
                "raw_detections": detections,
                "image_quality": compute_image_quality(read_image(image_path)),
                "saved_image_category": "model_test_annotated",
                "image_path": str(output_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Annotated image: {output_path}")
    print(f"Metadata JSON: {sidecar_path}")


def install_sigill_message():
    if not hasattr(signal, "SIGILL"):
        return

    def handler(signum, frame):
        print(
            "\nPyTorch .pt inference is not supported on this Raspberry Pi environment. "
            "Export to NCNN on desktop and deploy the NCNN folder.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(132)

    try:
        signal.signal(signal.SIGILL, handler)
    except Exception:
        pass


def is_pi_like():
    machine = platform.machine().lower()
    return machine.startswith("arm") or machine in {"aarch64", "arm64"}


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    return image


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


def print_export_instructions(profile):
    print("No usable model was found.")
    print("Export on desktop:")
    print(f"  yolo export model=models/{profile}/best.pt format=ncnn imgsz=320")
    print("Then copy:")
    print(f"  models/{profile}/best_ncnn_model/ to the Pi")


if __name__ == "__main__":
    main()
