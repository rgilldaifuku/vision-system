from pathlib import Path
import platform

import yaml

MODEL_FORMAT_AUTO = "auto"
MODEL_FORMAT_PT = "pt"
MODEL_FORMAT_NCNN = "ncnn"


class InferenceEngineError(RuntimeError):
    """Raised when the runtime model cannot be selected or loaded."""


class ModelInputSizeMismatchError(InferenceEngineError):
    """Raised when runtime preprocessing conflicts with an exported edge model."""


class InferenceEngine:
    """Small adapter around Ultralytics for PT and exported edge model folders."""

    def __init__(self, model_path, model_format=MODEL_FORMAT_AUTO):
        self.model_path = Path(model_path)
        self.model_format = infer_model_format(self.model_path, model_format)
        self.model = self._load_model()
        self.names = getattr(self.model, "names", {}) or {}

    def predict(self, frame, confidence, imgsz):
        results = self.model.predict(
            frame,
            conf=confidence,
            imgsz=imgsz,
            verbose=False,
        )
        return results, extract_detections(results)

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise InferenceEngineError(f"Ultralytics YOLO is not available: {exc}") from exc

        try:
            return YOLO(str(self.model_path))
        except Exception as exc:
            raise InferenceEngineError(
                f"Failed to load {self.model_format} model at {self.model_path}: {exc}"
            ) from exc


def resolve_model_path(
    profile_dir,
    config=None,
    model_override=None,
    model_format=MODEL_FORMAT_AUTO,
    prefer_edge_model=False,
):
    profile_dir = Path(profile_dir)
    config = config or {}

    if model_override:
        path = Path(model_override)
        return (
            path if path.is_absolute() else profile_dir.parent.parent / path,
            infer_model_format(path, model_format),
            "",
        )

    if model_format == MODEL_FORMAT_NCNN or (
        model_format == MODEL_FORMAT_AUTO and prefer_edge_model
    ):
        edge_model = find_ncnn_model(profile_dir)
        if edge_model:
            return edge_model, MODEL_FORMAT_NCNN, ""
        if model_format == MODEL_FORMAT_NCNN:
            return (
                profile_dir / "best_ncnn_model",
                MODEL_FORMAT_NCNN,
                "NCNN model folder not found. Export NCNN on desktop and copy it to this profile.",
            )

    configured_model = config.get("model_file")
    if configured_model:
        path = profile_dir / configured_model
    else:
        path = profile_dir / "latest" / "best.pt"
        if not path.exists():
            path = profile_dir / "best.pt"

    selected_format = infer_model_format(path, model_format)
    warning = ""
    if prefer_edge_model and selected_format == MODEL_FORMAT_PT:
        warning = (
            "Only a .pt model was found. On Raspberry Pi 4, PyTorch .pt inference may fail "
            "with Illegal instruction. Export NCNN on desktop for runtime."
        )
    elif selected_format == MODEL_FORMAT_PT and is_arm_linux():
        warning = (
            "Selected .pt model on ARM Linux. PyTorch inference may fail with Illegal instruction; "
            "NCNN is recommended for Raspberry Pi runtime."
        )
    return path, selected_format, warning


def find_ncnn_model(profile_dir):
    profile_dir = Path(profile_dir)
    candidates = [profile_dir / "best_ncnn_model"]
    candidates.extend(sorted(profile_dir.glob("*_ncnn_model")))

    for candidate in candidates:
        if is_ncnn_model_path(candidate):
            return candidate
    return None


def is_ncnn_model_path(path):
    path = Path(path)
    if not path.is_dir():
        return False
    return any(path.glob("*.ncnn.param")) and any(path.glob("*.ncnn.bin"))


def infer_model_format(path, requested_format=MODEL_FORMAT_AUTO):
    if requested_format != MODEL_FORMAT_AUTO:
        return requested_format

    path = Path(path)
    if is_ncnn_model_path(path):
        return MODEL_FORMAT_NCNN
    if path.suffix.lower() == ".pt":
        return MODEL_FORMAT_PT
    if path.is_dir():
        return MODEL_FORMAT_NCNN
    return MODEL_FORMAT_PT


def is_arm_linux():
    machine = platform.machine().lower()
    return platform.system().lower() == "linux" and (
        machine.startswith("arm") or machine in {"aarch64", "arm64"}
    )


def read_model_input_size(model_path):
    """Return exported model input size as (width, height), when metadata provides it."""
    model_path = Path(model_path)
    if not model_path.is_dir():
        return None

    metadata_path = model_path / "metadata.yaml"
    if not metadata_path.is_file():
        return None
    try:
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None

    imgsz = metadata.get("imgsz")
    if isinstance(imgsz, (int, float)):
        size = int(imgsz)
        return (size, size) if size > 0 else None
    if isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
        try:
            height, width = (int(imgsz[0]), int(imgsz[1]))
        except (TypeError, ValueError):
            return None
        return (width, height) if width > 0 and height > 0 else None
    return None


def model_input_size_warning(model_path, runtime_imgsz):
    expected = read_model_input_size(model_path)
    if expected is None:
        return ""

    runtime_size = (int(runtime_imgsz), int(runtime_imgsz))
    if expected == runtime_size:
        return ""
    return (
        f"Runtime imgsz {runtime_size[0]}x{runtime_size[1]} conflicts with exported "
        f"model input {expected[0]}x{expected[1]} from {Path(model_path) / 'metadata.yaml'}. "
        "The runtime size was not changed."
    )


def resolve_runtime_input_size(
    model_path,
    requested_imgsz=None,
    model_format=MODEL_FORMAT_AUTO,
    fallback_imgsz=256,
    allow_mismatch=False,
):
    selected_format = infer_model_format(model_path, model_format)
    expected = read_model_input_size(model_path) if selected_format == MODEL_FORMAT_NCNN else None

    if requested_imgsz is not None:
        runtime_imgsz = int(requested_imgsz)
        source = "explicit"
    elif expected is not None:
        if expected[0] != expected[1]:
            raise ModelInputSizeMismatchError(
                f"NCNN model declares non-square input {expected[0]}x{expected[1]}, but the "
                "runtime currently accepts one square --imgsz value. Set an explicit size "
                "only with ALLOW_MODEL_INPUT_SIZE_MISMATCH=1."
            )
        runtime_imgsz = expected[0]
        source = "model_metadata"
    else:
        runtime_imgsz = int(fallback_imgsz)
        source = "legacy_fallback"

    if runtime_imgsz < 1:
        raise ValueError("Runtime imgsz must be 1 or greater.")

    runtime_size = (runtime_imgsz, runtime_imgsz)
    mismatch = expected is not None and expected != runtime_size
    warning = ""
    if mismatch:
        message = (
            f"Runtime input {runtime_size[0]}x{runtime_size[1]} conflicts with NCNN model "
            f"input {expected[0]}x{expected[1]} declared by "
            f"{Path(model_path) / 'metadata.yaml'}."
        )
        if not allow_mismatch:
            raise ModelInputSizeMismatchError(
                f"{message} Use the model size or set "
                "ALLOW_MODEL_INPUT_SIZE_MISMATCH=1 to accept this risk explicitly."
            )
        warning = f"MODEL INPUT SIZE MISMATCH ALLOWED: {message}"

    return {
        "runtime_imgsz": runtime_imgsz,
        "runtime_size": {"width": runtime_size[0], "height": runtime_size[1]},
        "model_size": (
            {"width": expected[0], "height": expected[1]} if expected is not None else None
        ),
        "source": source,
        "mismatch": mismatch,
        "mismatch_allowed": bool(mismatch and allow_mismatch),
        "warning": warning,
    }


def extract_detections(results):
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = result.names.get(cls_id, str(cls_id))
            bbox = box.xyxy[0].cpu().numpy().tolist()
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": float(box.conf[0]),
                    "bbox": bbox,
                }
            )
    return detections
