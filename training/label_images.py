from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"


boxes = []
drawing = False
start_x, start_y = -1, -1
class_names = []
current_class_id = 0

def clean_name(name):
    return name.strip().lower().replace(" ", "_")

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

        if x2-x1 > 5 and y2 - y1 > 5:
            boxes.append((current_class_id, x1, y1, x2, y2))

def save_yolo_label(label_path, boxes, image_width, image_height):
    with open(label_path, "w") as f:
        for class_id, x1, y1, x2, y2 in boxes:
            center_x = ((x1 + x2) / 2) / image_width
            center_y = ((y1+y2) / 2) / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height

            f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

def main():
    global boxes, current_class_id

    object_name = clean_name(input("Object/dataset name: "))

    classes_input = input("Enter class names separated by commas: ")

    global class_names
    class_names = [c.strip() for c in classes_input.split(",") if c.strip()]

    if not class_names: 
        print("At lest one class is required.")
        return

    print("Classes: ")
    for i, name in enumerate(class_names):
        print(f"{i}: {name}")

    image_dir = DATASETS_DIR / object_name /"images" / "train"
    label_dir = DATASETS_DIR / object_name / "labels" / "train"
    label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

    if not image_paths:
        print(f"No images found in: {image_dir}")
        return

    print("Controls:")
    print("Drage mouse = draw box")
    print("S = save label")
    print(" U = undo last box")
    print("Q = quit")

    for image_path in image_paths:
        boxes = []

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        window_name = "Daifuku Label Tool"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            display  = image.copy()

            for class_id, x1, y1, x2, y2 in boxes:
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

            if key == ord("u"):
                if boxes:
                    boxes.pop()

            elif key == ord("s"):
                label_path = label_dir / f"{image_path.stem}.txt"
                save_yolo_label(label_path, boxes, w, h)
                print(f"Saved Label: {label_path}")

            elif ord("0") <= key <= ord("9"):
                selected = key - ord("0")

                if selected < len(class_names):
                    current_class_id= selected
                    print(f"Selected class {selected}: {class_names[selected]}")

            elif key == ord("n"):
                break
            
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()
    print("Labeling finished.")

if __name__ =="__main__":
    main()