import argparse
import sys
from pathlib import Path

import cv2


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_ID = 0

boxes = []
drawing = False
start_x, start_y = -1, -1


def create_parser():
    parser = argparse.ArgumentParser(
        description="Label one explicit image folder with YOLO bounding boxes."
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Exact folder containing .jpg, .jpeg, or .png images to label.",
    )
    parser.add_argument(
        "--label-dir",
        help="Destination folder for YOLO .txt labels. Defaults to <session>/labels.",
    )
    parser.add_argument(
        "--skip-labeled",
        action="store_true",
        help="Only open images that do not already have a matching .txt label file.",
    )
    return parser


def resolve_label_dir(image_dir, label_dir=None):
    image_dir = Path(image_dir).expanduser().resolve()
    if label_dir:
        return Path(label_dir).expanduser().resolve()
    return image_dir.parent / "labels"


def find_image_paths(image_dir):
    image_dir = Path(image_dir).expanduser().resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"Source image folder does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Source image path is not a folder: {image_dir}")

    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def filter_unlabeled_images(image_paths, label_dir):
    label_dir = Path(label_dir)
    return [path for path in image_paths if not (label_dir / f"{path.stem}.txt").exists()]


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y

        x1, y1 = min(start_x, end_x), min(start_y, end_y)
        x2, y2 = max(start_x, end_x), max(start_y, end_y)

        if x2 - x1 > 5 and y2 - y1 > 5:
            boxes.append((CLASS_ID, x1, y1, x2, y2))


def save_yolo_label(label_path, boxes_to_save, image_width, image_height):
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as handle:
        for _class_id, x1, y1, x2, y2 in boxes_to_save:
            center_x = ((x1 + x2) / 2) / image_width
            center_y = ((y1 + y2) / 2) / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height

            handle.write(
                f"{CLASS_ID} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
            )


def load_yolo_label(label_path, image_width, image_height):
    label_path = Path(label_path)
    if not label_path.exists():
        return []

    loaded_boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) != 5:
            print(f"Warning: skipping malformed label line {line_number} in {label_path}")
            continue

        try:
            _class_id = int(float(parts[0]))
            center_x, center_y, width, height = [float(value) for value in parts[1:]]
        except ValueError:
            print(f"Warning: skipping non-numeric label line {line_number} in {label_path}")
            continue

        x1 = int(round((center_x - width / 2) * image_width))
        y1 = int(round((center_y - height / 2) * image_height))
        x2 = int(round((center_x + width / 2) * image_width))
        y2 = int(round((center_y + height / 2) * image_height))
        loaded_boxes.append((CLASS_ID, x1, y1, x2, y2))

    return loaded_boxes


def print_startup_summary(image_dir, label_dir, image_count, skip_labeled):
    print(f"Source image directory: {image_dir}")
    print(f"Label directory: {label_dir}")
    print(f"Images found: {image_count}")
    if skip_labeled:
        print("Resume mode: skipping images that already have .txt labels.")
    print()


def label_images(image_paths, label_dir):
    global boxes

    window_name = "Daifuku Label Tool"

    print("Controls:")
    print("Drag mouse = draw box")
    print("S = save YOLO label")
    print("N = next image")
    print("U = undo last box")
    print("Q = quit")
    print("For negative images, press S with zero boxes to create an empty no-object label.")
    print()

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: could not open image, skipping: {image_path}")
            continue

        height, width = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        boxes = load_yolo_label(label_path, width, height)

        if boxes:
            print(f"Loaded {len(boxes)} existing box(es): {label_path}")
        elif label_path.exists():
            print(f"Loaded existing empty no-object label: {label_path}")

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            display = image.copy()

            for _class_id, x1, y1, x2, y2 in boxes:
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                display,
                f"{image_path.name} | boxes: {len(boxes)} | S=save N=next U=undo Q=quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("u"), ord("U")):
                if boxes:
                    boxes.pop()

            elif key in (ord("s"), ord("S")):
                save_yolo_label(label_path, boxes, width, height)
                print(f"Saved label: {label_path}")

            elif key in (ord("n"), ord("N")):
                break

            elif key in (ord("q"), ord("Q")):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("Labeling finished.")


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        image_dir = Path(args.image_dir).expanduser().resolve()
        label_dir = resolve_label_dir(image_dir, args.label_dir)
        image_paths = find_image_paths(image_dir)
        if args.skip_labeled:
            image_paths = filter_unlabeled_images(image_paths, label_dir)

        print_startup_summary(image_dir, label_dir, len(image_paths), args.skip_labeled)

        if not image_paths:
            print(
                "No valid images found to label. Expected .jpg, .jpeg, or .png files "
                "directly inside the source folder.",
                file=sys.stderr,
            )
            return 1

        label_dir.mkdir(parents=True, exist_ok=True)
        label_images(image_paths, label_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
