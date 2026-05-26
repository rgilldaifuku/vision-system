from pathlib import Path
from datetime import datetime
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT/ "data" / "datasets"

def clean_name(name):
    return name.strip().lower().replace(" ", "_")

def main():
    object_name = clean_name(input("Object name to collect images for: "))

    if not object_name:
        print("Object name cannot be empty.")
        return
    
    save_dir = DATASETS_DIR / object_name / "images" / "train"
    save_dir.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("ERROR: Camera failed to open.")
        return

    print(f"Saving images to: {save_dir}")
    print("Press Space to save image")
    print("Press Q to quit")

    count = len(list(save_dir.glob("*.jpg")))

    while True: 
        ret, frame = camera.read()

        if not ret:
            print("WARNING: failed to read camera frame.")
            continue

        preview = frame.copy()
        cv2.putText(
            preview,
            f"{object_name} | saved: {count} | SPACE= save | Q=quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Image Collection", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == 32:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_dir / f"{object_name}_{timestamp}_{count:04d}.jpg"
            cv2.imwrite(str(filename), frame)
            count += 1
            print(f"Saved: {filename}")

    camera.release()
    cv2.destroyAllWindows()
    print("Image collection finished.")

if __name__ == "__main__":
    main()
