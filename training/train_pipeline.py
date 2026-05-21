from pathlib import Path
import json
import shutil
from ultralytics import YOLO
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = Path("C:/Temp/detection_runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

def clean_name(name):
    return name.strip().lower().replace(" ", "_")

def count_files(folder, extensions):
    if not folder.exists():
        return 0
    return sum(1 for file in folder.iterdir() if file.suffix.lower() in extensions)

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

    for path in required: 
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    train_images = count_files(images_train, [".jpg", ".jpeg", ".png"])
    val_images = count_files(images_val, [".jpg", ".jpeg", ".png"])
    train_labels = count_files(labels_train, [".txt"])
    val_labels = count_files(labels_val, [".txt"])

    if train_images == 0:
        raise RuntimeError("no training images found.")
        
    if val_images == 0:
        raise RuntimeError("no validation images found")

    if train_labels == 0:
        raise RuntimeError("no training labels found")

    if val_labels == 0:
        raise RuntimeError("no validation labels found")

    print("Dataset check passed.")
    print(f"Train images:{train_images}")
    print(f"Val images: {val_images}")
    print(f"Train labels: {train_labels}")
    print(f"Val labels: {val_labels}")

    return data_yaml

def train_model(dataset_name, data_yaml):
    model = YOLO("yolov8n.pt")

    run_name = f"{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}"

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
    data_yaml = validate_dataset(dataset_name)
    run_dir = train_model(dataset_name, data_yaml)
    update_model_profile(dataset_name, run_dir)

    print("Training pipeline complete")
    print(f"Active Model File: models/{dataset_name}/best.pt")

if __name__ == "__main__":
    main()