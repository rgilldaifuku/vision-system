import argparse
from PySide6.QtWidgets import QApplication
from app.ui import MainWindow
from app.config import DEFAULT_MODEL_PATH
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to model")
    parser.add_argument("--camera", type=int, default=0)

    args = parser.parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[INFO] Using model: {model_path}")

    app = QApplication([])
    window = MainWindow(args.model, args.camera)
    window.show()
    app.exec()