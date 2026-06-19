#!/usr/bin/env python3
"""Collect raw camera images for later external labeling.

This script intentionally does not import or run YOLO, NCNN, PyTorch, or Flask.
It captures raw images from the configured camera profile, records lightweight
image-quality metadata, and leaves dataset labeling/YOLO formatting as a
separate workflow.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.runtime.camera_manager import CameraManager
from app.runtime.camera_profile import CameraProfileError, load_camera_profile
from app.runtime.image_quality import compute_image_quality, quality_thresholds_from_config
from app.runtime.picamera2_manager import Picamera2CameraManager


VALID_LABELS = ("positive", "negative")
DEFAULT_OUTPUT_ROOT = Path("data/collections")


def create_parser():
    parser = argparse.ArgumentParser(
        description="Capture raw Raspberry Pi/USB camera images for later labeling."
    )
    parser.add_argument("--profile", default="yellow_daifuku")
    parser.add_argument("--camera-profile", default="pi_camera3")
    parser.add_argument("--label", choices=VALID_LABELS, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--count", type=positive_int, default=20)
    parser.add_argument("--interval-seconds", type=non_negative_float, default=2.0)
    parser.add_argument("--warmup-seconds", type=non_negative_float, default=3.0)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--min-frame-difference",
        type=non_negative_float,
        default=0.0,
        help="Skip frames whose mean pixel difference from the last saved frame is below this value.",
    )
    parser.add_argument(
        "--min-blur-score",
        type=non_negative_float,
        default=0.0,
        help="Skip frames whose Laplacian blur score is below this value.",
    )
    parser.add_argument(
        "--save-quality-warnings",
        action="store_true",
        help="Include explicit quality warning fields in metadata for non-GOOD frames.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic frames and do not open a physical camera.",
    )
    return parser


def positive_int(value):
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return number


def non_negative_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("value must be numeric") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("value must be 0 or greater")
    return number


def resolve_output_root(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def build_session_paths(output_root, profile, camera_profile_name, session, label):
    output_root = Path(output_root)
    session_dir = output_root / profile / camera_profile_name / session
    positive_dir = session_dir / "positive"
    negative_dir = session_dir / "negative"
    label_dir = session_dir / label
    return {
        "session_dir": session_dir,
        "positive_dir": positive_dir,
        "negative_dir": negative_dir,
        "label_dir": label_dir,
        "manifest_path": session_dir / "manifest.jsonl",
        "summary_path": session_dir / "session_summary.json",
    }


def ensure_output_dirs(paths):
    paths["positive_dir"].mkdir(parents=True, exist_ok=True)
    paths["negative_dir"].mkdir(parents=True, exist_ok=True)


def next_image_index(label_dir):
    max_index = 0
    for path in Path(label_dir).glob("image_*.jpg"):
        try:
            max_index = max(max_index, int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return max_index + 1


def create_camera(camera_profile):
    backend = camera_profile.backend
    if backend == "auto":
        backend = "picamera2" if Picamera2CameraManager.is_available() else "opencv"

    if backend == "picamera2":
        return Picamera2CameraManager(
            frame_width=camera_profile.width,
            frame_height=camera_profile.height,
            target_fps=camera_profile.fps,
        )
    if backend == "opencv":
        return CameraManager(
            camera_index=0,
            frame_width=camera_profile.width,
            frame_height=camera_profile.height,
        )

    raise CameraProfileError(f"Unsupported capture backend: {camera_profile.backend}")


def synthetic_frame(camera_profile, index):
    height = int(camera_profile.height)
    width = int(camera_profile.width)
    base = np.full((height, width, 3), 90 + (index % 60), dtype=np.uint8)
    cv2.putText(
        base,
        f"DRY RUN {index:04d}",
        (20, max(40, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return base


def frame_difference_score(previous_frame, current_frame):
    if previous_frame is None or current_frame is None:
        return None
    try:
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if previous_gray.shape != current_gray.shape:
            current_gray = cv2.resize(
                current_gray,
                (previous_gray.shape[1], previous_gray.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        return float(np.mean(cv2.absdiff(previous_gray, current_gray)))
    except Exception:
        return None


def make_capture_record(
    *,
    args,
    camera_profile,
    image_path,
    json_path,
    capture_index,
    saved_index,
    frame,
    image_quality,
    frame_difference,
    quality_warning,
):
    height, width = frame.shape[:2]
    timestamp = datetime.now().isoformat(timespec="seconds")
    return {
        "timestamp": timestamp,
        "profile": args.profile,
        "camera_profile": camera_profile.name,
        "session": args.session,
        "label": args.label,
        "capture_index": capture_index,
        "saved_index": saved_index,
        "image_path": str(image_path),
        "metadata_path": str(json_path),
        "frame_width": int(width),
        "frame_height": int(height),
        "image_quality": image_quality,
        "quality_warning": quality_warning,
        "camera_settings": camera_profile.to_dict(),
        "frame_difference": frame_difference,
        "dry_run": bool(args.dry_run),
    }


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_manifest(manifest_path, record):
    with Path(manifest_path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def average_quality(records):
    numeric_keys = (
        "brightness_mean",
        "brightness_std",
        "contrast_score",
        "blur_score",
        "overexposed_pct",
        "underexposed_pct",
    )
    averages = {}
    for key in numeric_keys:
        values = [
            record["image_quality"].get(key)
            for record in records
            if isinstance(record.get("image_quality"), dict)
            and isinstance(record["image_quality"].get(key), (int, float))
        ]
        averages[key] = round(float(sum(values) / len(values)), 3) if values else None
    return averages


def write_session_summary(paths, args, camera_profile, records, skips, started_at):
    finished_at = datetime.now().isoformat(timespec="seconds")
    summary = {
        "started_at": started_at,
        "finished_at": finished_at,
        "profile": args.profile,
        "camera_profile": camera_profile.name,
        "session": args.session,
        "label": args.label,
        "dry_run": bool(args.dry_run),
        "images_requested": int(args.count),
        "images_saved": len(records),
        "images_skipped": len(skips),
        "skip_reasons": skips,
        "average_image_quality": average_quality(records),
        "output_folders": {
            "session": str(paths["session_dir"]),
            "positive": str(paths["positive_dir"]),
            "negative": str(paths["negative_dir"]),
            "label": str(paths["label_dir"]),
            "manifest": str(paths["manifest_path"]),
            "summary": str(paths["summary_path"]),
        },
    }
    write_json(paths["summary_path"], summary)
    return summary


def relative_display_path(path):
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_capture(args):
    camera_profile = load_camera_profile(args.camera_profile)
    output_root = resolve_output_root(args.output_root)
    paths = build_session_paths(output_root, args.profile, camera_profile.name, args.session, args.label)
    ensure_output_dirs(paths)

    thresholds = quality_thresholds_from_config(camera_profile.quality)
    records = []
    skips = []
    previous_saved_frame = None
    next_index = next_image_index(paths["label_dir"])
    camera = None
    started_at = datetime.now().isoformat(timespec="seconds")

    print(
        f"Capturing {args.count} {args.label} image(s) for profile '{args.profile}' "
        f"using camera profile '{camera_profile.name}'."
    )
    print(f"Output: {relative_display_path(paths['session_dir'])}")

    try:
        if args.dry_run:
            print("DRY RUN: using synthetic frames; no physical camera will be opened.")
        else:
            camera = create_camera(camera_profile)
            if not camera.open():
                error = getattr(camera, "last_error", "") or "camera failed to open"
                raise RuntimeError(f"Camera open failed: {error}")
            if args.warmup_seconds:
                print(f"Warming up camera for {args.warmup_seconds:.1f}s...")
                time.sleep(args.warmup_seconds)

        for capture_index in range(1, args.count + 1):
            frame = synthetic_frame(camera_profile, capture_index) if args.dry_run else camera.read_frame()
            if frame is None:
                reason = "camera returned no frame"
                skips.append({"capture_index": capture_index, "reason": reason})
                print(f"Skipped frame: {reason}")
                time.sleep(args.interval_seconds)
                continue

            image_quality = compute_image_quality(frame, thresholds)
            blur_score = image_quality.get("blur_score")
            if args.min_blur_score > 0 and isinstance(blur_score, (int, float)):
                if blur_score < args.min_blur_score:
                    reason = "blur score below threshold"
                    skips.append(
                        {
                            "capture_index": capture_index,
                            "reason": reason,
                            "blur_score": blur_score,
                            "minimum": args.min_blur_score,
                        }
                    )
                    print(f"Skipped frame: {reason} ({blur_score:.1f} < {args.min_blur_score:.1f})")
                    time.sleep(args.interval_seconds)
                    continue

            difference = frame_difference_score(previous_saved_frame, frame)
            if args.min_frame_difference > 0 and difference is not None:
                if difference < args.min_frame_difference:
                    reason = "near duplicate"
                    skips.append(
                        {
                            "capture_index": capture_index,
                            "reason": reason,
                            "frame_difference": round(difference, 3),
                            "minimum": args.min_frame_difference,
                        }
                    )
                    print(f"Skipped frame: {reason}")
                    time.sleep(args.interval_seconds)
                    continue

            saved_index = next_index
            next_index += 1
            image_path = paths["label_dir"] / f"image_{saved_index:04d}.jpg"
            json_path = paths["label_dir"] / f"image_{saved_index:04d}.json"
            quality_warning = None
            if args.save_quality_warnings and image_quality.get("quality_status") != "GOOD":
                quality_warning = image_quality.get("message") or image_quality.get("quality_status")

            if not cv2.imwrite(str(image_path), frame):
                reason = "failed to write image"
                skips.append({"capture_index": capture_index, "reason": reason, "path": str(image_path)})
                print(f"Skipped frame: {reason}")
                time.sleep(args.interval_seconds)
                continue

            record = make_capture_record(
                args=args,
                camera_profile=camera_profile,
                image_path=image_path,
                json_path=json_path,
                capture_index=capture_index,
                saved_index=saved_index,
                frame=frame,
                image_quality=image_quality,
                frame_difference=difference,
                quality_warning=quality_warning,
            )
            write_json(json_path, record)
            append_manifest(paths["manifest_path"], record)
            records.append(record)
            previous_saved_frame = frame.copy()

            brightness = image_quality.get("brightness_mean")
            blur = image_quality.get("blur_score")
            quality_status = image_quality.get("quality_status")
            print(
                f"Saved {len(records)}/{args.count}: {args.label}/{image_path.name} | "
                f"brightness={format_metric(brightness)} | "
                f"blur={format_metric(blur)} | quality={quality_status}"
            )
            if quality_warning:
                print(f"Quality warning: {quality_warning}")

            time.sleep(args.interval_seconds)
    finally:
        if camera is not None:
            camera.release()

    summary = write_session_summary(paths, args, camera_profile, records, skips, started_at)
    print(
        f"Done. Saved {summary['images_saved']} image(s), skipped {summary['images_skipped']}."
    )
    print(f"Manifest: {relative_display_path(paths['manifest_path'])}")
    print(f"Summary: {relative_display_path(paths['summary_path'])}")
    return summary


def format_metric(value):
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return "N/A"


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        run_capture(args)
    except (CameraProfileError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCapture interrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
