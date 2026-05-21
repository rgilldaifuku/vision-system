import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def _default_model_path() -> str:
    candidates = [
        # Most common Ultralytics CLI output
        Path("runs/detect/train/weights/best.pt"),
        Path("runs/detect/my_items/weights/best.pt"),
        # Your terminal run showed this nested path
        Path("runs/detect/runs/my_items/weights/best.pt"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "yolov8n.pt"  # fallback (auto-downloads)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained YOLO model on a webcam.")
    parser.add_argument("--model", default=_default_model_path(), help="Path to .pt model weights.")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (0, 1, ...).")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.cam}. Try --cam 1.")

    window = f"Webcam Detection - {Path(args.model).name} (press q)"
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=args.conf, verbose=False)
        annotated = results[0].plot()
        cv2.imshow(window, annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()