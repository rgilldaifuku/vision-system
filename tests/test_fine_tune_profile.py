import tempfile
import unittest
from pathlib import Path

from training import fine_tune_profile
from training.train_pipeline import validate_dataset


def create_dataset(root, empty_train_label=False):
    dataset = Path(root) / "combined"
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
        (dataset / "images" / split / f"{split}.jpg").write_bytes(b"image")
        label = "" if empty_train_label and split == "train" else "0 0.5 0.5 0.2 0.2\n"
        (dataset / "labels" / split / f"{split}.txt").write_text(label, encoding="utf-8")
    (dataset / "data.yaml").write_text(
        "train: images/train\nval: images/val\nnames:\n  0: yellow_daifuku\n",
        encoding="utf-8",
    )
    return dataset


class FineTuneProfileTests(unittest.TestCase):
    def test_empty_labels_are_accepted_as_negatives(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = create_dataset(tmpdir, empty_train_label=True)
            _, summary = validate_dataset(
                "combined",
                allow_empty_labels=True,
                allow_segmentation=True,
                return_summary=True,
                dataset_dir=dataset,
            )
            self.assertEqual(summary["negative_labels"], 1)
            self.assertEqual(summary["train_images"], 1)

    def test_base_model_resolves_to_profile_best_pt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            expected = models_dir / "yellow_daifuku" / "best.pt"
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"active")

            resolved = fine_tune_profile.resolve_base_model(
                "yellow_daifuku",
                models_dir=models_dir,
            )
            self.assertEqual(resolved, expected)

    def test_candidate_packaging_does_not_modify_active_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            profile_dir = root / "models" / "yellow_daifuku"
            active = profile_dir / "best.pt"
            latest = profile_dir / "latest" / "best.pt"
            latest.parent.mkdir(parents=True)
            active.write_bytes(b"active")
            latest.write_bytes(b"latest")
            trained = root / "run" / "weights" / "best.pt"
            trained.parent.mkdir(parents=True)
            trained.write_bytes(b"candidate")
            data_yaml = root / "data.yaml"
            data_yaml.write_text("names:\n  0: yellow_daifuku\n", encoding="utf-8")
            candidate_dir = profile_dir / "candidates" / "pi_camera_v1"

            candidate_model = fine_tune_profile.package_candidate(
                candidate_dir,
                trained,
                data_yaml,
                {"metrics": {}},
            )

            self.assertEqual(candidate_model.read_bytes(), b"candidate")
            self.assertEqual(active.read_bytes(), b"active")
            self.assertEqual(latest.read_bytes(), b"latest")

    def test_candidate_overwrite_requires_explicit_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_dir = Path(tmpdir) / "candidate"
            candidate_dir.mkdir()
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                fine_tune_profile.ensure_candidate_available(candidate_dir)


if __name__ == "__main__":
    unittest.main()
