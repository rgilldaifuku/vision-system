#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from app.config import MODELS_DIR


def main():
    parser = argparse.ArgumentParser(description="Export a desktop-trained profile model to NCNN.")
    parser.add_argument("--profile", default="yellow_daifuku")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--model")
    parser.add_argument("--output-name", default="best_ncnn_model")
    args = parser.parse_args()

    profile_dir = MODELS_DIR / args.profile
    model_path = resolve_pt_model(profile_dir, args.model)
    target_dir = profile_dir / args.output_name

    if not model_path.exists():
        raise SystemExit(f"PT model not found: {model_path}")

    print(f"Exporting profile: {args.profile}")
    print(f"Source PT model: {model_path}")
    print(f"Target NCNN folder: {target_dir}")

    model = YOLO(str(model_path))
    exported = Path(model.export(format="ncnn", imgsz=args.imgsz))
    print(f"Ultralytics export path: {exported}")

    if exported.resolve() != target_dir.resolve():
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(exported, target_dir)

    print()
    print("NCNN export ready:")
    print(f"  {target_dir}")
    print()
    print("Copy this folder to the Pi at the same profile path:")
    print(f"  models/{args.profile}/{target_dir.name}/")


def resolve_pt_model(profile_dir, model_override=None):
    if model_override:
        path = Path(model_override)
        return path if path.is_absolute() else PROJECT_ROOT / path

    latest_model = profile_dir / "latest" / "best.pt"
    if latest_model.exists():
        return latest_model
    return profile_dir / "best.pt"


if __name__ == "__main__":
    main()
