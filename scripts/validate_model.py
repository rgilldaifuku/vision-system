#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Validate a model/profile against one image.")
    parser.add_argument("--profile", default="yellow_daifuku")
    parser.add_argument("--image", required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--model")
    parser.add_argument("--model-format", choices=("auto", "pt", "ncnn"), default="auto")
    parser.add_argument("--prefer-edge-model", action="store_true")
    args = parser.parse_args()

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "test_model_on_image.py"),
        "--profile",
        args.profile,
        "--image",
        args.image,
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
        "--model-format",
        args.model_format,
    ]
    if args.model:
        command.extend(["--model", args.model])
    if args.prefer_edge_model:
        command.append("--prefer-edge-model")

    print("== Model Validation ==")
    print("Command:", " ".join(command))
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True)
    if result.returncode == 0:
        print("PASS: inference completed. Check detections above; no detections may still indicate dataset/setup issues.")
    else:
        print(f"FAIL: model validation exited with code {result.returncode}")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
