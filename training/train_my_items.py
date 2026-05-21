from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_NAME = "mouse"

DATA_YAML = PROJECT_ROOT / "data" / "datasets" / DATASET_NAME / "data.yaml"


def main() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Could not find data.yaml: {DATA_YAML}")

    model = YOLO("yolov8n.pt")  # uses your local yolov8n.pt (or downloads if missing)

    model.train(
        data=str(DATA_YAML),
        epochs=150,
        imgsz=640,
        project=str(PROJECT_ROOT / "runs"),
        name=DATASET_NAME,
    )


if __name__ == "__main__":
    main()
