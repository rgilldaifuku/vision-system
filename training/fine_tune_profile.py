import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_pipeline import extract_training_metrics, validate_dataset


DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "data" / "runs"


def safe_name(value):
    value = str(value).strip()
    if not value or value in {".", ".."} or Path(value).name != value:
        raise argparse.ArgumentTypeError("must be a single folder name without path separators")
    return value


def parse_batch(value):
    if value is None or str(value).strip().lower() == "auto":
        return -1
    try:
        batch = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("--batch must be 'auto' or a positive integer") from exc
    if batch <= 0:
        raise argparse.ArgumentTypeError("--batch must be 'auto' or a positive integer")
    return batch


def create_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune an existing profile model into an isolated candidate."
    )
    parser.add_argument("--profile", required=True, type=safe_name)
    parser.add_argument("--dataset", required=True, type=safe_name)
    parser.add_argument("--candidate-name", required=True, type=safe_name)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=parse_batch,
        default=-1,
        metavar="AUTO_OR_INT",
        help="Ultralytics batch size; defaults to auto.",
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing candidate after training succeeds.",
    )
    return parser


def resolve_base_model(profile, models_dir=MODELS_DIR):
    base_model = Path(models_dir) / profile / "best.pt"
    if not base_model.is_file():
        raise FileNotFoundError(
            f"Base model not found: {base_model}. "
            "Fine-tuning requires the existing deployed profile model."
        )
    return base_model


def resolve_candidate_dir(profile, candidate_name, models_dir=MODELS_DIR):
    return Path(models_dir) / profile / "candidates" / candidate_name


def ensure_candidate_available(candidate_dir, overwrite=False):
    if Path(candidate_dir).exists() and not overwrite:
        raise FileExistsError(
            f"Candidate already exists: {candidate_dir}. "
            "Re-run with --overwrite to replace it after successful training."
        )


def train_candidate(
    base_model,
    data_yaml,
    profile,
    candidate_name,
    epochs,
    imgsz,
    batch,
    device=None,
    runs_dir=RUNS_DIR,
):
    from ultralytics import YOLO

    project_dir = Path(runs_dir) / profile
    project_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(base_model))
    train_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(project_dir),
        "name": candidate_name,
        "exist_ok": False,
    }
    if device:
        train_args["device"] = device

    results = model.train(**train_args)
    run_dir = Path(results.save_dir)
    best_model = run_dir / "weights" / "best.pt"
    if not best_model.is_file():
        raise FileNotFoundError(f"Training finished but best model was not found: {best_model}")
    return best_model, run_dir, extract_training_metrics(results)


def package_candidate(
    candidate_dir,
    trained_best_model,
    data_yaml,
    metadata,
    overwrite=False,
):
    candidate_dir = Path(candidate_dir)
    ensure_candidate_available(candidate_dir, overwrite=overwrite)

    staging_dir = candidate_dir.with_name(f".{candidate_dir.name}.staging")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    try:
        shutil.copy2(trained_best_model, staging_dir / "best.pt")
        shutil.copy2(data_yaml, staging_dir / "data.yaml")
        (staging_dir / "training_report.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        staging_dir.replace(candidate_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    return candidate_dir / "best.pt"


def build_training_report(
    profile,
    dataset,
    candidate_name,
    base_model,
    dataset_dir,
    data_yaml,
    candidate_dir,
    run_dir,
    validation_summary,
    training_args,
    metrics,
):
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "profile": profile,
        "candidate_name": candidate_name,
        "base_model_path": str(base_model),
        "dataset_path": str(dataset_dir),
        "dataset_data_yaml": str(data_yaml),
        "candidate_path": str(candidate_dir / "best.pt"),
        "run_folder": str(run_dir),
        "dataset": {
            "name": dataset,
            "train_images": validation_summary["train_images"],
            "validation_images": validation_summary["validation_images"],
            "train_labels": validation_summary["train_labels"],
            "validation_labels": validation_summary["validation_labels"],
            "negative_labels": validation_summary["negative_labels"],
        },
        "training_arguments": training_args,
        "metrics": {
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "mAP50": metrics.get("mAP50"),
            "mAP50-95": metrics.get("mAP50-95"),
        },
        "deployment": {
            "active_model_modified": False,
            "candidate_only": True,
        },
    }


def format_metric(value):
    return "N/A" if value is None else str(value)


def main(argv=None):
    args = create_parser().parse_args(argv)
    dataset_dir = DATASETS_DIR / args.dataset
    candidate_dir = resolve_candidate_dir(args.profile, args.candidate_name)

    try:
        base_model = resolve_base_model(args.profile)
        ensure_candidate_available(candidate_dir, overwrite=args.overwrite)
        data_yaml, validation_summary = validate_dataset(
            args.dataset,
            allow_empty_labels=True,
            allow_segmentation=True,
            return_summary=True,
            dataset_dir=dataset_dir,
        )

        training_args = {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": "auto" if args.batch == -1 else args.batch,
            "device": args.device,
        }
        trained_best, run_dir, metrics = train_candidate(
            base_model=base_model,
            data_yaml=data_yaml,
            profile=args.profile,
            candidate_name=args.candidate_name,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
        report = build_training_report(
            profile=args.profile,
            dataset=args.dataset,
            candidate_name=args.candidate_name,
            base_model=base_model,
            dataset_dir=dataset_dir,
            data_yaml=data_yaml,
            candidate_dir=candidate_dir,
            run_dir=run_dir,
            validation_summary=validation_summary,
            training_args=training_args,
            metrics=metrics,
        )
        candidate_model = package_candidate(
            candidate_dir,
            trained_best,
            data_yaml,
            report,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("\nFine-tuning complete. Active model was not changed.")
    print(f"Candidate PT: {candidate_model}")
    print(f"Precision: {format_metric(metrics.get('precision'))}")
    print(f"Recall: {format_metric(metrics.get('recall'))}")
    print(f"mAP50: {format_metric(metrics.get('mAP50'))}")
    print(f"mAP50-95: {format_metric(metrics.get('mAP50-95'))}")
    print("\nNext holdout test:")
    print(
        f"python scripts/validate_model.py --profile {args.profile} "
        f"--model \"{candidate_model}\" --model-format pt --image <holdout-image>"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
