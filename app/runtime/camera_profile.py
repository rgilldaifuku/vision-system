from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from app.config import PROJECT_ROOT


CAMERA_PROFILES_DIR = PROJECT_ROOT / "cameras"
SUPPORTED_BACKENDS = {"auto", "picamera2", "opencv"}


class CameraProfileError(ValueError):
    """Raised when a camera profile cannot be loaded or validated."""


@dataclass
class CameraRoi:
    enabled: bool = False
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 1.0


@dataclass
class CameraPreprocessing:
    enabled: bool = True
    brightness_check: bool = True
    blur_check: bool = True
    color_normalization: bool = False


@dataclass
class CameraProfile:
    name: str
    backend: str = "auto"
    width: int = 640
    height: int = 480
    fps: int = 30
    rotation: int = 0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    exposure_mode: str = "auto"
    exposure_time: float | None = None
    gain: float | None = None
    white_balance: str = "auto"
    roi: CameraRoi = field(default_factory=CameraRoi)
    preprocessing: CameraPreprocessing = field(default_factory=CameraPreprocessing)
    path: str = ""

    def to_dict(self):
        return asdict(self)


def available_camera_profiles(profiles_dir=CAMERA_PROFILES_DIR):
    profiles_path = Path(profiles_dir)
    if not profiles_path.exists():
        return []
    return sorted(path.stem for path in profiles_path.glob("*.yaml"))


def load_camera_profile(name_or_path, profiles_dir=CAMERA_PROFILES_DIR):
    profile_path = resolve_camera_profile_path(name_or_path, profiles_dir)
    if not profile_path.exists():
        raise CameraProfileError(f"Camera profile not found: {profile_path}")

    try:
        raw = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise CameraProfileError(f"Invalid camera profile YAML: {profile_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise CameraProfileError(f"Camera profile must be a YAML object: {profile_path}")

    return build_camera_profile(raw, profile_path)


def resolve_camera_profile_path(name_or_path, profiles_dir=CAMERA_PROFILES_DIR):
    value = Path(str(name_or_path))
    if value.suffix in {".yaml", ".yml"} or value.is_absolute() or value.parent != Path("."):
        return value if value.is_absolute() else PROJECT_ROOT / value
    return Path(profiles_dir) / f"{value.name}.yaml"


def build_camera_profile(raw, profile_path=None):
    name = str(raw.get("name") or (Path(profile_path).stem if profile_path else "")).strip()
    if not name:
        raise CameraProfileError("Camera profile is missing required field: name")

    backend = normalize_backend(raw.get("backend", "auto"))
    width = _positive_int(raw.get("width", 640), "width")
    height = _positive_int(raw.get("height", 480), "height")
    fps = _positive_int(raw.get("fps", 30), "fps")
    rotation = _rotation(raw.get("rotation", 0))

    roi = raw.get("roi") or {}
    if not isinstance(roi, dict):
        raise CameraProfileError("Camera profile roi must be an object")

    preprocessing = raw.get("preprocessing") or {}
    if not isinstance(preprocessing, dict):
        raise CameraProfileError("Camera profile preprocessing must be an object")

    return CameraProfile(
        name=name,
        backend=backend,
        width=width,
        height=height,
        fps=fps,
        rotation=rotation,
        flip_horizontal=_bool(raw.get("flip_horizontal", False)),
        flip_vertical=_bool(raw.get("flip_vertical", False)),
        exposure_mode=str(raw.get("exposure_mode") or "auto"),
        exposure_time=_optional_float(raw.get("exposure_time"), "exposure_time"),
        gain=_optional_float(raw.get("gain"), "gain"),
        white_balance=str(raw.get("white_balance") or "auto"),
        roi=CameraRoi(
            enabled=_bool(roi.get("enabled", False)),
            x1=_unit_float(roi.get("x1", 0.0), "roi.x1"),
            y1=_unit_float(roi.get("y1", 0.0), "roi.y1"),
            x2=_unit_float(roi.get("x2", 1.0), "roi.x2"),
            y2=_unit_float(roi.get("y2", 1.0), "roi.y2"),
        ),
        preprocessing=CameraPreprocessing(
            enabled=_bool(preprocessing.get("enabled", True)),
            brightness_check=_bool(preprocessing.get("brightness_check", True)),
            blur_check=_bool(preprocessing.get("blur_check", True)),
            color_normalization=_bool(preprocessing.get("color_normalization", False)),
        ),
        path=str(profile_path or ""),
    )


def normalize_backend(value):
    backend = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "cv2": "opencv",
        "open_cv": "opencv",
        "usb": "opencv",
        "usb_webcam": "opencv",
        "pi": "picamera2",
        "picamera": "picamera2",
        "pi_camera": "picamera2",
    }
    backend = aliases.get(backend, backend)
    if backend not in SUPPORTED_BACKENDS:
        raise CameraProfileError(
            f"Unsupported camera backend '{value}'. Use one of: {', '.join(sorted(SUPPORTED_BACKENDS))}."
        )
    return backend


def _bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _positive_int(value, name):
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise CameraProfileError(f"Camera profile {name} must be an integer") from exc
    if number < 1:
        raise CameraProfileError(f"Camera profile {name} must be greater than 0")
    return number


def _rotation(value):
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise CameraProfileError("Camera profile rotation must be an integer") from exc
    if number not in {0, 90, 180, 270}:
        raise CameraProfileError("Camera profile rotation must be one of 0, 90, 180, or 270")
    return number


def _optional_float(value, name):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CameraProfileError(f"Camera profile {name} must be numeric or null") from exc


def _unit_float(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CameraProfileError(f"Camera profile {name} must be numeric") from exc
    if not 0.0 <= number <= 1.0:
        raise CameraProfileError(f"Camera profile {name} must be between 0 and 1")
    return number
