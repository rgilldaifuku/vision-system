from __future__ import annotations

from pathlib import Path


def _pairs_ok(images_dir: Path, labels_dir: Path) -> tuple[int, int]:
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in images_dir.glob("*") if p.suffix.lower() in img_exts]
    missing = 0
    for img in images:
        lbl = labels_dir / f"{img.stem}.txt"
        if not lbl.exists():
            missing += 1
            print(f"Missing label for image: {img.name} -> expected {lbl.name}")
    return len(images), missing


def main() -> None:
    root = Path(__file__).resolve().parent / "datasets" / "my_items"
    splits = [("train", root / "images" / "train", root / "labels" / "train"),
              ("val", root / "images" / "val", root / "labels" / "val")]

    if not root.exists():
        raise SystemExit(f"Dataset folder not found: {root}")

    total_images = 0
    total_missing = 0
    for name, images_dir, labels_dir in splits:
        if not images_dir.exists():
            print(f"[{name}] missing images dir: {images_dir}")
            continue
        if not labels_dir.exists():
            print(f"[{name}] missing labels dir: {labels_dir}")
            continue

        n_images, n_missing = _pairs_ok(images_dir, labels_dir)
        total_images += n_images
        total_missing += n_missing
        print(f"[{name}] images={n_images} missing_labels={n_missing}")

    if total_images == 0:
        print("No images found yet. Put your .jpg/.png into datasets/my_items/images/train and images/val.")
        raise SystemExit(2)

    if total_missing:
        print(f"ERROR: {total_missing} images are missing labels.")
        raise SystemExit(1)

    print("OK: all images have matching label .txt files.")


if __name__ == "__main__":
    main()
