import argparse
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = PROJECT_ROOT / "data" / "collections" / "yellow_daifuku" / "pi_camera3"
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_OUTPUT_NAME = "yellow_daifuku_pi_camera_v1"
CLASS_NAME = "yellow_daifuku"

POSITIVE_SESSIONS = (
    ("pi_demo_v1_pos_a", "positive"),
    ("pi_demo_v1_pos_b", "positive"),
    ("pi_demo_v1_pos_c", "positive"),
)
NEGATIVE_SESSIONS = (
    ("pi_demo_v1_negatives", "negative"),
    ("pi_demo_v1_negatives_resume", "negative"),
)
EXCLUDED_HOLDOUT_SESSIONS = (
    "pi_demo_v1_holdout_positive",
    "pi_demo_v1_holdout_negative",
)


@dataclass(frozen=True)
class SourceSample:
    session: str
    category: str
    image_path: Path
    label_path: Path

    @property
    def prefixed_stem(self):
        return f"{self.session}__{self.image_path.stem}"

    @property
    def output_image_name(self):
        return f"{self.prefixed_stem}{self.image_path.suffix.lower()}"

    @property
    def output_label_name(self):
        return f"{self.prefixed_stem}.txt"


def create_parser():
    parser = argparse.ArgumentParser(
        description="Build a clean YOLO dataset from labeled Pi camera collection sessions."
    )
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=val_ratio, default=0.2)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output dataset folder.",
    )
    return parser


def val_ratio(value):
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("--val-ratio must be numeric") from exc
    if ratio < 0 or ratio >= 1:
        raise argparse.ArgumentTypeError("--val-ratio must be >= 0 and < 1")
    return ratio


def included_source_sessions():
    return [
        {"session": session, "image_subdir": subdir, "category": "positive"}
        for session, subdir in POSITIVE_SESSIONS
    ] + [
        {"session": session, "image_subdir": subdir, "category": "negative"}
        for session, subdir in NEGATIVE_SESSIONS
    ]


def discover_samples(source_root=SOURCE_ROOT):
    source_root = Path(source_root)
    samples = []
    errors = []

    for entry in included_source_sessions():
        session = entry["session"]
        image_dir = source_root / session / entry["image_subdir"]
        label_dir = source_root / session / "labels"
        category = entry["category"]

        if not image_dir.exists():
            errors.append(f"Missing source image folder: {image_dir}")
            continue
        if not label_dir.exists():
            errors.append(f"Missing source label folder: {label_dir}")
            continue

        image_paths = [
            path
            for path in sorted(image_dir.iterdir())
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
        if not image_paths:
            errors.append(f"No valid images found directly inside: {image_dir}")
            continue

        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                errors.append(f"Missing label for {image_path}: expected {label_path}")
                continue

            label_errors = validate_label_file(label_path, category)
            errors.extend(label_errors)
            samples.append(
                SourceSample(
                    session=session,
                    category=category,
                    image_path=image_path,
                    label_path=label_path,
                )
            )

    if errors:
        raise ValueError(format_errors("Pi camera collection validation failed.", errors))
    if not samples:
        raise ValueError("Pi camera collection validation failed: no source samples found.")
    return samples


def validate_label_file(label_path, category):
    errors = []
    lines = label_path.read_text(encoding="utf-8").splitlines()
    non_empty_lines = [line.strip() for line in lines if line.strip()]

    if not non_empty_lines:
        if category == "negative":
            return []
        return [f"{label_path}: positive image label is empty."]

    for line_number, line in enumerate(non_empty_lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path}:{line_number}: YOLO label must have exactly 5 values.")
            continue

        if parts[0] != "0":
            errors.append(f"{label_path}:{line_number}: class id must be 0, got '{parts[0]}'.")

        for value_name, raw_value in zip(("x_center", "y_center", "width", "height"), parts[1:]):
            try:
                value = float(raw_value)
            except ValueError:
                errors.append(
                    f"{label_path}:{line_number}: {value_name} must be numeric, got '{raw_value}'."
                )
                continue
            if not 0.0 <= value <= 1.0:
                errors.append(
                    f"{label_path}:{line_number}: {value_name} must be between 0 and 1, got {raw_value}."
                )

    return errors


def split_balanced(samples, seed=42, val_ratio=0.2):
    positives = [sample for sample in samples if sample.category == "positive"]
    negatives = [sample for sample in samples if sample.category == "negative"]

    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    positive_train, positive_val = _split_group(positives, val_ratio)
    negative_train, negative_val = _split_group(negatives, val_ratio)
    train = positive_train + negative_train
    val = positive_val + negative_val

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _split_group(samples, val_ratio):
    if not samples:
        return [], []
    val_count = int(round(len(samples) * val_ratio))
    if val_ratio > 0 and len(samples) > 1:
        val_count = max(1, val_count)
    val_count = min(val_count, max(0, len(samples) - 1))
    return samples[val_count:], samples[:val_count]


def prepare_output_dir(output_dir, overwrite=False):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_dir}. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    for relative in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def copy_split(samples, output_dir, split_name):
    copied = []
    image_dir = output_dir / "images" / split_name
    label_dir = output_dir / "labels" / split_name

    for sample in samples:
        destination_image = image_dir / sample.output_image_name
        destination_label = label_dir / sample.output_label_name
        shutil.copy2(sample.image_path, destination_image)
        shutil.copy2(sample.label_path, destination_label)
        copied.append(
            {
                "session": sample.session,
                "category": sample.category,
                "source_image": str(sample.image_path),
                "source_label": str(sample.label_path),
                "output_image": str(destination_image),
                "output_label": str(destination_label),
            }
        )

    return copied


def write_data_yaml(output_dir):
    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                f"  0: {CLASS_NAME}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def write_dataset_report(output_dir, samples, train_samples, val_samples, copied, seed, val_ratio):
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_path": str(output_dir),
        "seed": seed,
        "validation_ratio": val_ratio,
        "included_source_sessions": included_source_sessions(),
        "excluded_holdout_sessions": list(EXCLUDED_HOLDOUT_SESSIONS),
        "positive_count": sum(1 for sample in samples if sample.category == "positive"),
        "negative_count": sum(1 for sample in samples if sample.category == "negative"),
        "train_count": len(train_samples),
        "validation_count": len(val_samples),
        "train_positive_count": sum(1 for sample in train_samples if sample.category == "positive"),
        "train_negative_count": sum(1 for sample in train_samples if sample.category == "negative"),
        "validation_positive_count": sum(1 for sample in val_samples if sample.category == "positive"),
        "validation_negative_count": sum(1 for sample in val_samples if sample.category == "negative"),
        "copied_files": copied,
    }
    report_path = output_dir / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_dataset(
    output_name=DEFAULT_OUTPUT_NAME,
    seed=42,
    val_ratio=0.2,
    overwrite=False,
    source_root=SOURCE_ROOT,
    datasets_dir=DATASETS_DIR,
):
    output_dir = Path(datasets_dir) / output_name
    samples = discover_samples(source_root)
    train_samples, val_samples = split_balanced(samples, seed=seed, val_ratio=val_ratio)

    prepare_output_dir(output_dir, overwrite=overwrite)
    copied = copy_split(train_samples, output_dir, "train")
    copied.extend(copy_split(val_samples, output_dir, "val"))
    write_data_yaml(output_dir)
    report = write_dataset_report(
        output_dir,
        samples,
        train_samples,
        val_samples,
        copied,
        seed,
        val_ratio,
    )
    return report


def format_errors(title, errors):
    shown = "\n".join(f"- {error}" for error in errors[:100])
    extra = "" if len(errors) <= 100 else f"\n- ... and {len(errors) - 100} more error(s)"
    return f"{title}\n{shown}{extra}"


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        report = build_dataset(
            output_name=args.output_name,
            seed=args.seed,
            val_ratio=args.val_ratio,
            overwrite=args.overwrite,
        )
    except (FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Built dataset: {report['output_path']}")
    print(f"Positive images: {report['positive_count']}")
    print(f"Negative images: {report['negative_count']}")
    print(f"Train images: {report['train_count']}")
    print(f"Validation images: {report['validation_count']}")
    print("Holdout sessions were excluded:")
    for session in report["excluded_holdout_sessions"]:
        print(f"- {session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
