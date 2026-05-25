from pathlib import Path
import json
import shutil
import yaml
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = Path("C:/Temp/detection_runs")
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def clean_name(name):
    return name.strip().lower().replace(" ", "_")

def count_files(folder, extensions):
    if not folder.exists():
        return 0
    return sum(1 for file in folder.iterdir() if file.suffix.lower() in extensions)

def _read_classes_txt(classes_path):
    if not classes_path.exists():
        return []

    return [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def _load_allowed_class_ids(dataset_name, dataset_dir, data_yaml, errors):
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        errors.append(f"{data_yaml}: could not read data.yaml: {exc}")
        return set()

    names = data.get("names") if isinstance(data, dict) else None

    if isinstance(names, list):
        class_names = [str(name).strip() for name in names]
        if not class_names or any(not name for name in class_names):
            errors.append(f"{data_yaml}: names list must contain at least one non-empty class name.")
            return set()
        return set(range(len(class_names)))

    if isinstance(names, dict):
        allowed_ids = set()
        for raw_class_id, raw_name in names.items():
            try:
                class_id = int(raw_class_id)
            except (TypeError, ValueError):
                errors.append(f"{data_yaml}: class id '{raw_class_id}' in names is not an integer.")
                continue

            if class_id < 0:
                errors.append(f"{data_yaml}: class id {class_id} in names must be >= 0.")
                continue

            if not str(raw_name).strip():
                errors.append(f"{data_yaml}: class id {class_id} has an empty class name.")
                continue

            allowed_ids.add(class_id)

        if allowed_ids:
            return allowed_ids

    for classes_path in (dataset_dir / "classes.txt", MODELS_DIR / dataset_name / "classes.txt"):
        class_names = _read_classes_txt(classes_path)
        if class_names:
            return set(range(len(class_names)))

    errors.append(f"{data_yaml}: missing class names in data.yaml and no classes.txt fallback was found.")
    return set()

def _collect_images(images_dir, split_name, errors):
    images = {}

    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMG_EXTENSIONS:
            continue

        if image_path.stem in images:
            errors.append(
                f"{split_name}: duplicate image stem '{image_path.stem}' in {images_dir}; "
                "image filenames must be unique aside from extension."
            )
            continue

        images[image_path.stem] = image_path

    return images

def _collect_labels(labels_dir):
    return {
        label_path.stem: label_path
        for label_path in sorted(labels_dir.iterdir())
        if label_path.is_file() and label_path.suffix.lower() == ".txt"
    }

def _validate_label_file(label_path, allowed_class_ids, errors):
    text = label_path.read_text(encoding="utf-8", errors="replace")

    if not text.strip():
        errors.append(f"{label_path}: label file is empty.")
        return

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()

        if not stripped:
            errors.append(f"{label_path}: line {line_number}: blank label lines are not allowed.")
            continue

        if "," in stripped:
            errors.append(
                f"{label_path}: line {line_number}: contains a comma; "
                "YOLO labels must be space-separated: class_id x_center y_center width height."
            )

        parts = stripped.split()
        if len(parts) != 5:
            errors.append(
                f"{label_path}: line {line_number}: expected exactly 5 values, got {len(parts)}."
            )
            continue

        class_token = parts[0]
        try:
            class_id = int(class_token)
        except ValueError:
            errors.append(f"{label_path}: line {line_number}: class id '{class_token}' is not an integer.")
            continue

        if allowed_class_ids and class_id not in allowed_class_ids:
            allowed = ", ".join(str(class_id) for class_id in sorted(allowed_class_ids))
            errors.append(
                f"{label_path}: line {line_number}: class id {class_id} is not in configured classes [{allowed}]."
            )

        for name, value_text in zip(("x_center", "y_center", "width", "height"), parts[1:]):
            try:
                value = float(value_text)
            except ValueError:
                errors.append(
                    f"{label_path}: line {line_number}: {name} value '{value_text}' is not numeric."
                )
                continue

            if value < 0 or value > 1:
                errors.append(
                    f"{label_path}: line {line_number}: {name} value {value_text} must be between 0 and 1."
                )

def write_training_report(dataset_name, version_name, run_dir, data_yaml):
    dataset_dir = DATASETS_DIR / dataset_name 

    report = {
        "model_name": dataset_name,
        "version": version_name,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "data_yaml": str(data_yaml),
            "train_images": count_files(dataset_dir / "images" / "train", [".jpg", ".jpeg", ".png"]),
            "val_images": count_files(dataset_dir / "images" / "val", [".jpg", ".jpeg", ".png"]),
            "train_labels": count_files(dataset_dir / "labels" / "train", [".txt"]),
            "val_labels": count_files(dataset_dir / "labels" / "val", [".txt"]),
        },
        "training": {
            "epochs": 50,
            "image_size": 640,
            "run_dir": str(run_dir),
            "best_model": str(run_dir / "weights" / "best.pt"),
        },
        "deployment": {
            "latest_model": str(MODELS_DIR / dataset_name / "latest" / "best.pt"),
            "compatibility_model": str(MODELS_DIR / dataset_name / "best.pt"),
        },
    }

    report_path = MODELS_DIR/dataset_name / "training_report.json"
    with open (report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Training report saved: {report_path}")
    
def validate_dataset(dataset_name):
    dataset_dir = DATASETS_DIR / dataset_name

    images_train = dataset_dir / "images" / "train"
    images_val = dataset_dir / "images" / "val"
    labels_train = dataset_dir / "labels" / "train"
    labels_val  = dataset_dir / "labels" / "val"
    data_yaml = dataset_dir / "data.yaml"

    required = [images_train, images_val, labels_train, labels_val, data_yaml]
    errors = []

    for path in required: 
        if not path.exists():
            errors.append(f"Missing required path: {path}")

    if errors:
        raise RuntimeError("Dataset validation failed:\n" + "\n".join(f"- {error}" for error in errors))

    allowed_class_ids = _load_allowed_class_ids(dataset_name, dataset_dir, data_yaml, errors)

    images_by_split = {
        "train": _collect_images(images_train, "train", errors),
        "val": _collect_images(images_val, "val", errors),
    }

    labels_by_split = {
        "train": _collect_labels(labels_train),
        "val": _collect_labels(labels_val),
    }

    train_images = len(images_by_split["train"])
    val_images = len(images_by_split["val"])
    train_labels = len(labels_by_split["train"])
    val_labels = len(labels_by_split["val"])

    if train_images == 0:
        errors.append("No training images found.")
        
    if val_images == 0:
        errors.append("No validation images found.")

    if train_labels == 0:
        errors.append("No training labels found.")

    if val_labels == 0:
        errors.append("No validation labels found.")

    for split_name in ("train", "val"):
        image_stems = set(images_by_split[split_name])
        label_stems = set(labels_by_split[split_name])

        for stem in sorted(image_stems - label_stems):
            errors.append(
                f"{split_name}: missing label for image '{images_by_split[split_name][stem].name}'."
            )

        for stem in sorted(label_stems - image_stems):
            errors.append(
                f"{split_name}: label '{labels_by_split[split_name][stem].name}' has no matching image."
            )

        for label_path in labels_by_split[split_name].values():
            _validate_label_file(label_path, allowed_class_ids, errors)

    duplicate_stems = sorted(set(images_by_split["train"]) & set(images_by_split["val"]))
    for stem in duplicate_stems:
        errors.append(f"Duplicate image stem across train and val: '{stem}'.")

    if errors:
        shown_errors = errors[:100]
        message = "Dataset validation failed:\n" + "\n".join(f"- {error}" for error in shown_errors)
        if len(errors) > len(shown_errors):
            message += f"\n- ... and {len(errors) - len(shown_errors)} more error(s)."
        raise RuntimeError(message)

    print("Dataset validation passed.")
    print(f"Train images: {train_images}")
    print(f"Val images: {val_images}")
    print(f"Train labels: {train_labels}")
    print(f"Val labels: {val_labels}")

    return data_yaml

def train_model(dataset_name, data_yaml):
    from ultralytics import YOLO

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        project=str(RUNS_DIR),
        name=dataset_name,
        exist_ok=True,
    )

    return Path(results.save_dir)

def get_next_version(profile_dir):
    versions_dir = profile_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    existing_versions = {
        folder.name for folder in versions_dir.iterdir()
        if folder.is_dir() and folder.name.startswith("v")
    }

    version_numbers = []

    for version in existing_versions:
        try:
            version_numbers.append(int(version.replace("v", "")))
        except ValueError:
            pass

    next_number = max(version_numbers, default=0) + 1
    return f"v{next_number}"

def update_model_profile(dataset_name, run_dir):
    weights_dir = run_dir / "weights"
    best_model = weights_dir / "best.pt"

    if not best_model.exists():
        raise FileNotFoundError(f"Could not find trained model: {best_model}")

    profile_dir = MODELS_DIR / dataset_name
    profile_dir.mkdir(parents=True, exist_ok=True)

    version_name = get_next_version(profile_dir)

    version_dir = profile_dir /"versions" / version_name
    latest_dir = profile_dir / "latest"

    version_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(best_model, version_dir / "best.pt")

    shutil.copy2(best_model, latest_dir / "best.pt")

    shutil.copy2(best_model, profile_dir / "best.pt")

    with open(profile_dir / "classes.txt", "w") as f:
        f.write(dataset_name + "\n")

    config = {
        "profile_name": dataset_name,
        "model_file": "latest/best.pt",
        "latest_version": version_name,
        "target_classes": [dataset_name],
        "confidence": 0.35,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(profile_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    write_training_report(dataset_name, version_name, run_dir, DATASETS_DIR / dataset_name / "data.yaml")

    print(f"Model profile updated: {profile_dir}")
    print(f"Created version: {version_name}")
    print(f"Latest model: {latest_dir / 'best.pt'}")

def main():
    dataset_name = clean_name(input("Object/model name to train: "))

    if not dataset_name:
        print("Object name cannot be empty")
        return

    print(f"Training mode for: {dataset_name}")
    try:
        data_yaml = validate_dataset(dataset_name)
    except RuntimeError as exc:
        print(exc)
        raise SystemExit(1)

    run_dir = train_model(dataset_name, data_yaml)
    update_model_profile(dataset_name, run_dir)

    print("Training pipeline complete")
    print(f"Active Model File: models/{dataset_name}/best.pt")

if __name__ == "__main__":
    main()
