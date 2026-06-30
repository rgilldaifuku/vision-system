import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
ORIGINAL_DATASET = DATASETS_DIR / "yellow_daifuku"
PI_CAMERA_DATASET = DATASETS_DIR / "yellow_daifuku_pi_camera_v1"
DEFAULT_OUTPUT_NAME = "yellow_daifuku_pi_camera_combined_v1"
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val")
DEFAULT_CLASS_NAME = "yellow_daifuku"


@dataclass(frozen=True)
class SourceSample:
    dataset_name: str
    split: str
    image_path: Path
    label_path: Path
    is_positive: bool

    @property
    def output_stem(self):
        return f"{self.dataset_name}__{self.split}__{self.image_path.stem}"

    @property
    def output_image_name(self):
        return f"{self.output_stem}{self.image_path.suffix.lower()}"

    @property
    def output_label_name(self):
        return f"{self.output_stem}.txt"


def create_parser():
    parser = argparse.ArgumentParser(
        description="Combine the original and Pi-camera Daifuku YOLO datasets."
    )
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output dataset folder.",
    )
    return parser


def load_class_name(data_yaml):
    data_yaml = Path(data_yaml)
    if not data_yaml.is_file():
        raise ValueError(f"Missing dataset config: {data_yaml}")

    try:
        config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid dataset config {data_yaml}: {exc}") from exc

    names = config.get("names")
    if isinstance(names, list) and names:
        return str(names[0])
    if isinstance(names, dict):
        class_name = names.get(0, names.get("0"))
        if class_name is not None:
            return str(class_name)
    raise ValueError(f"{data_yaml}: expected a class 0 entry in 'names'.")


def validate_label_file(label_path):
    errors = []
    non_empty_lines = [
        line.strip()
        for line in Path(label_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for line_number, line in enumerate(non_empty_lines, start=1):
        parts = line.split()
        is_detection_box = len(parts) == 5
        is_segmentation_polygon = len(parts) >= 7 and len(parts) % 2 == 1
        if not (is_detection_box or is_segmentation_polygon):
            errors.append(
                f"{label_path}:{line_number}: expected a 5-value YOLO box or a class id "
                "followed by at least 3 coordinate pairs."
            )
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            errors.append(f"{label_path}:{line_number}: class id must be integer 0.")
        else:
            if class_id != 0:
                errors.append(
                    f"{label_path}:{line_number}: class id must be 0, got '{parts[0]}'."
                )

        for coordinate_index, raw_value in enumerate(parts[1:], start=1):
            value_name = (
                ("x_center", "y_center", "width", "height")[coordinate_index - 1]
                if is_detection_box
                else f"polygon_coordinate_{coordinate_index}"
            )
            try:
                value = float(raw_value)
            except ValueError:
                errors.append(
                    f"{label_path}:{line_number}: {value_name} must be numeric, "
                    f"got '{raw_value}'."
                )
                continue
            if not 0.0 <= value <= 1.0:
                errors.append(
                    f"{label_path}:{line_number}: {value_name} must be between 0 and 1, "
                    f"got {raw_value}."
                )

    return errors, bool(non_empty_lines)


def discover_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    errors = []
    samples = []

    if not dataset_path.is_dir():
        raise ValueError(f"Source dataset folder does not exist: {dataset_path}")

    for split in SPLITS:
        image_dir = dataset_path / "images" / split
        label_dir = dataset_path / "labels" / split
        if not image_dir.is_dir():
            errors.append(f"Missing source image folder: {image_dir}")
            continue
        if not label_dir.is_dir():
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

        image_stems = {path.stem for path in image_paths}
        for label_path in sorted(label_dir.glob("*.txt")):
            if label_path.stem not in image_stems:
                errors.append(f"Label has no matching image: {label_path}")

        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                errors.append(f"Missing label for {image_path}: expected {label_path}")
                continue

            label_errors, is_positive = validate_label_file(label_path)
            errors.extend(label_errors)
            samples.append(
                SourceSample(
                    dataset_name=dataset_path.name,
                    split=split,
                    image_path=image_path,
                    label_path=label_path,
                    is_positive=is_positive,
                )
            )

    if errors:
        raise ValueError(format_errors(f"Dataset validation failed for {dataset_path}.", errors))
    if not samples:
        raise ValueError(f"Dataset validation failed for {dataset_path}: no samples found.")
    return samples


def validate_output_names(samples):
    image_names = set()
    label_names = set()
    errors = []
    for sample in samples:
        image_key = (sample.split, sample.output_image_name)
        label_key = (sample.split, sample.output_label_name)
        if image_key in image_names:
            errors.append(f"Output image filename collision: {sample.output_image_name}")
        if label_key in label_names:
            errors.append(f"Output label filename collision: {sample.output_label_name}")
        image_names.add(image_key)
        label_names.add(label_key)
    if errors:
        raise ValueError(format_errors("Output filename validation failed.", errors))


def prepare_output_dir(output_dir, overwrite=False):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_dir}. "
                "Re-run with --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_samples(samples, output_dir):
    copied = []
    for sample in samples:
        destination_image = output_dir / "images" / sample.split / sample.output_image_name
        destination_label = output_dir / "labels" / sample.split / sample.output_label_name
        shutil.copy2(sample.image_path, destination_image)
        shutil.copy2(sample.label_path, destination_label)
        copied.append(
            {
                "source_dataset": sample.dataset_name,
                "original_split": sample.split,
                "source_image": str(sample.image_path),
                "source_label": str(sample.label_path),
                "output_image": str(destination_image),
                "output_label": str(destination_label),
                "category": "positive" if sample.is_positive else "negative",
            }
        )
    return copied


def write_data_yaml(output_dir, class_name):
    path = output_dir / "data.yaml"
    path.write_text(
        "\n".join(
            [
                f"path: {output_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                f"  0: {class_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def count_samples(samples):
    return {
        "total": len(samples),
        "train": sum(sample.split == "train" for sample in samples),
        "validation": sum(sample.split == "val" for sample in samples),
        "positive": sum(sample.is_positive for sample in samples),
        "negative": sum(not sample.is_positive for sample in samples),
    }


def write_dataset_report(
    output_dir,
    original_path,
    pi_path,
    original_samples,
    pi_samples,
    class_name,
    copied_files,
):
    all_samples = original_samples + pi_samples
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_path": str(output_dir),
        "class_name": class_name,
        "source_paths": {
            "original_dataset": str(original_path),
            "pi_camera_dataset": str(pi_path),
        },
        "source_counts": {
            "original_dataset": count_samples(original_samples),
            "pi_camera_dataset": count_samples(pi_samples),
        },
        "total_counts": count_samples(all_samples),
        "split_policy": "Existing train/validation assignments preserved.",
        "raw_pi_collections_scanned": False,
        "copied_files": copied_files,
    }
    report_path = output_dir / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_dataset(
    output_name=DEFAULT_OUTPUT_NAME,
    overwrite=False,
    original_dataset=ORIGINAL_DATASET,
    pi_dataset=PI_CAMERA_DATASET,
    datasets_dir=DATASETS_DIR,
):
    original_dataset = Path(original_dataset)
    pi_dataset = Path(pi_dataset)
    output_dir = Path(datasets_dir) / output_name

    if output_dir.resolve() in {original_dataset.resolve(), pi_dataset.resolve()}:
        raise ValueError("Output dataset must be different from both source datasets.")

    original_class = load_class_name(original_dataset / "data.yaml")
    pi_class = load_class_name(pi_dataset / "data.yaml")
    if normalize_class_name(original_class) != normalize_class_name(pi_class):
        raise ValueError(
            "Source class names do not match: "
            f"original='{original_class}', pi_camera='{pi_class}'."
        )

    original_samples = discover_dataset(original_dataset)
    pi_samples = discover_dataset(pi_dataset)
    all_samples = original_samples + pi_samples
    validate_output_names(all_samples)

    # Validation is complete before an existing output can be removed.
    prepare_output_dir(output_dir, overwrite=overwrite)
    copied_files = copy_samples(all_samples, output_dir)
    write_data_yaml(output_dir, original_class)
    return write_dataset_report(
        output_dir,
        original_dataset,
        pi_dataset,
        original_samples,
        pi_samples,
        original_class,
        copied_files,
    )


def normalize_class_name(name):
    return str(name).strip().lower().replace(" ", "_")


def format_errors(title, errors):
    shown = "\n".join(f"- {error}" for error in errors[:100])
    extra = "" if len(errors) <= 100 else f"\n- ... and {len(errors) - 100} more error(s)"
    return f"{title}\n{shown}{extra}"


def main(argv=None):
    args = create_parser().parse_args(argv)
    try:
        report = build_dataset(
            output_name=args.output_name,
            overwrite=args.overwrite,
        )
    except (FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    totals = report["total_counts"]
    print(f"Built combined dataset: {report['output_path']}")
    print(f"Class: {report['class_name']}")
    print(f"Train images: {totals['train']}")
    print(f"Validation images: {totals['validation']}")
    print(f"Positive images: {totals['positive']}")
    print(f"Negative images: {totals['negative']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
