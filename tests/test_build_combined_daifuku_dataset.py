import tempfile
import unittest
from pathlib import Path

import training.build_combined_daifuku_dataset as combined


def create_dataset(root, name, class_name="yellow_daifuku", invalid_label=None):
    dataset = Path(root) / name
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "images" / "val").mkdir(parents=True)
    (dataset / "labels" / "train").mkdir(parents=True)
    (dataset / "labels" / "val").mkdir(parents=True)
    (dataset / "data.yaml").write_text(
        f"train: images/train\nval: images/val\nnames:\n  0: {class_name}\n",
        encoding="utf-8",
    )

    for split in combined.SPLITS:
        (dataset / "images" / split / "same_name.jpg").write_bytes(
            f"{name}-{split}".encode("utf-8")
        )
        label = "" if split == "val" else "0 0.5 0.5 0.25 0.25\n"
        if invalid_label is not None and split == "train":
            label = invalid_label
        (dataset / "labels" / split / "same_name.txt").write_text(label, encoding="utf-8")
    return dataset


def snapshot_tree(path):
    return {
        file.relative_to(path): (file.read_bytes(), file.stat().st_mtime_ns)
        for file in Path(path).rglob("*")
        if file.is_file()
    }


class CombinedDaifukuDatasetTests(unittest.TestCase):
    def test_prefixes_names_and_preserves_empty_negative_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original = create_dataset(root, "yellow_daifuku")
            pi = create_dataset(root, "yellow_daifuku_pi_camera_v1", "yellow daifuku")

            report = combined.build_dataset(
                output_name="combined",
                original_dataset=original,
                pi_dataset=pi,
                datasets_dir=root,
            )
            output = Path(report["output_path"])
            train_names = {path.name for path in (output / "images" / "train").iterdir()}
            val_labels = list((output / "labels" / "val").glob("*.txt"))

            self.assertEqual(len(train_names), 2)
            self.assertIn("yellow_daifuku__train__same_name.jpg", train_names)
            self.assertIn("yellow_daifuku_pi_camera_v1__train__same_name.jpg", train_names)
            self.assertEqual(len(val_labels), 2)
            self.assertTrue(all(path.read_text(encoding="utf-8") == "" for path in val_labels))
            self.assertEqual(report["total_counts"]["negative"], 2)
            self.assertFalse(report["raw_pi_collections_scanned"])

    def test_sources_are_not_modified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original = create_dataset(root, "yellow_daifuku")
            pi = create_dataset(root, "yellow_daifuku_pi_camera_v1")
            before_original = snapshot_tree(original)
            before_pi = snapshot_tree(pi)

            combined.build_dataset(
                output_name="combined",
                original_dataset=original,
                pi_dataset=pi,
                datasets_dir=root,
            )

            self.assertEqual(before_original, snapshot_tree(original))
            self.assertEqual(before_pi, snapshot_tree(pi))

    def test_invalid_label_is_rejected_before_output_creation(self):
        invalid_labels = (
            "1 0.5 0.5 0.2 0.2\n",
            "0 1.1 0.5 0.2 0.2\n",
            "0 nope 0.5 0.2 0.2\n",
        )
        for invalid_label in invalid_labels:
            with self.subTest(invalid_label=invalid_label):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    original = create_dataset(root, "yellow_daifuku")
                    pi = create_dataset(
                        root,
                        "yellow_daifuku_pi_camera_v1",
                        invalid_label=invalid_label,
                    )

                    with self.assertRaises(ValueError):
                        combined.build_dataset(
                            output_name="combined",
                            original_dataset=original,
                            pi_dataset=pi,
                            datasets_dir=root,
                        )
                    self.assertFalse((root / "combined").exists())

    def test_missing_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original = create_dataset(root, "yellow_daifuku")
            pi = create_dataset(root, "yellow_daifuku_pi_camera_v1")
            (pi / "labels" / "train" / "same_name.txt").unlink()

            with self.assertRaisesRegex(ValueError, "Missing label"):
                combined.build_dataset(
                    output_name="combined",
                    original_dataset=original,
                    pi_dataset=pi,
                    datasets_dir=root,
                )

    def test_yolo_polygon_label_is_validated_and_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original = create_dataset(root, "yellow_daifuku")
            pi = create_dataset(root, "yellow_daifuku_pi_camera_v1")
            polygon = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n"
            source_label = original / "labels" / "train" / "same_name.txt"
            source_label.write_text(polygon, encoding="utf-8")

            report = combined.build_dataset(
                output_name="combined",
                original_dataset=original,
                pi_dataset=pi,
                datasets_dir=root,
            )

            copied_label = (
                Path(report["output_path"])
                / "labels"
                / "train"
                / "yellow_daifuku__train__same_name.txt"
            )
            self.assertEqual(copied_label.read_text(encoding="utf-8"), polygon)


if __name__ == "__main__":
    unittest.main()
