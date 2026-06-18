from pathlib import Path
import platform

MODEL_FORMAT_AUTO = "auto"
MODEL_FORMAT_PT = "pt"
MODEL_FORMAT_NCNN = "ncnn"


class InferenceEngineError(RuntimeError):
    """Raised when the runtime model cannot be selected or loaded."""


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
