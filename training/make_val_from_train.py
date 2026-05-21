"""
Move (or copy) N random YOLO pairs from train -> val.

Expected structure (relative to --dataset):
  images/train, labels/train
Creates if missing:
  images/val,   labels/val
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _find_train_images(images_train: Path) -> list[Path]:
    return sorted([p for p in images_train.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _paired_images(images: list[Path], labels_train: Path) -> list[Path]:
    paired: list[Path] = []
    for im in images:
        lb = labels_train / f"{im.stem}.txt"
        if lb.exists():
            paired.append(im)
    return paired


def _transfer(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def main() -> None:
    ap = argparse.ArgumentParser(description="Create/refresh val split by taking N samples from train.")
    ap.add_argument(
        "--dataset",
        default="datasets/my_items",
        help="Dataset root containing images/ and labels/ (default: datasets/my_items)",
    )
    ap.add_argument("--n", type=int, required=True, help="Number of image/label pairs to put into val.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    ap.add_argument(
        "--copy",
        action="store_true",
        help="Copy into val instead of moving (keeps originals in train).",
    )
    args = ap.parse_args()

    root = Path(args.dataset)
    images_train = root / "images" / "train"
    labels_train = root / "labels" / "train"
    images_val = root / "images" / "val"
    labels_val = root / "labels" / "val"

    if not images_train.exists() or not labels_train.exists():
        raise SystemExit(
            f"Missing train folders.\n"
            f"Expected: {images_train}\n"
            f"          {labels_train}"
        )

    all_train_images = _find_train_images(images_train)
    paired = _paired_images(all_train_images, labels_train)
    if not paired:
        raise SystemExit(f"No image/label pairs found in {images_train} + {labels_train}.")

    if args.n <= 0:
        raise SystemExit("--n must be > 0")

    if args.n > len(paired):
        raise SystemExit(f"Requested n={args.n} but only {len(paired)} paired train images exist.")

    random.seed(args.seed)
    chosen = random.sample(paired, args.n)

    images_val.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)

    moved = 0
    for im in chosen:
        lb = labels_train / f"{im.stem}.txt"
        _transfer(im, images_val / im.name, copy=bool(args.copy))
        _transfer(lb, labels_val / lb.name, copy=bool(args.copy))
        moved += 1

    action = "Copied" if args.copy else "Moved"
    print(f"{action} {moved} pairs to val.")
    print(f"Train now has {len(_paired_images(_find_train_images(images_train), labels_train))} paired images.")


if __name__ == "__main__":
    main()

