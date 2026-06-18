import argparse
import importlib
import json
import sys
import tempfile
import time
from pathlib import Path

from app.config import DATA_DIR, LOGS_DIR, MODELS_DIR, PROJECT_ROOT, REVIEW_IMAGES_DIR


DEFAULT_IMGSZ = 320
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
DEFAULT_INFERENCE_INTERVAL_MS = 200
DEFAULT_SNAPSHOT_INTERVAL_MS = 1000
CAMERA_BACKEND_AUTO = "auto"
CAMERA_BACKEND_PICAMERA2 = "picamera2"
CAMERA_BACKEND_OPENCV = "opencv"


DESKTOP_IMPORTS = {
    "PySide6": "PySide6",
}
COMMON_RUNTIME_IMPORTS = {
    "cv2": "opencv-python",
    "flask": "Flask",
    "numpy": "numpy",
    "yaml": "PyYAML",
}
MODEL_RUNTIME_IMPORTS = {
    "torch": "torch",
    "ultralytics": "ultralytics",
}
PI_RUNTIME_IMPORTS = {
    "picamera2": "python3-picamera2",
}
RUNTIME_MODULES = (
    "app.runtime.detector_service",
    "app.runtime.camera_manager",
    "app.runtime.camera_sources",
    "app.runtime.health_check",
    "app.runtime.inspection_logic",
    "app.runtime.inference_engine",
    "app.runtime.output_manager",
    "app.runtime.action_manager",
    "app.runtime.picamera2_manager",
)


class HealthCheck:
    def __init__(self):
        self.failures = 0
        self.warnings = 0

    def ok(self, message):
        print(f"[OK] {message}")

    def warn(self, message):
        self.warnings += 1
        print(f"[WARN] {message}")

    def fail(self, message):
        self.failures += 1
        print(f"[FAIL] {message}")


def create_parser():
    parser = argparse.ArgumentParser(description="Runtime hardware/readiness health check")
    parser.add_argument(
        "--mode",
        choices=("laptop", "pi"),
        default="laptop",
        help="Use laptop for simulated/local readiness or pi for hardware readiness.",
    )
    parser.add_argument("--profile", default="yellow_daifuku")
    parser.add_argument("--model")
    parser.add_argument("--camera", type=int)
    parser.add_argument("--camera-source")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model/YOLO import checks for dashboard/runtime dry-run validation.",
    )
    parser.add_argument(
        "--camera-backend",
        choices=(CAMERA_BACKEND_AUTO, CAMERA_BACKEND_PICAMERA2, CAMERA_BACKEND_OPENCV),
        default=CAMERA_BACKEND_AUTO,
        help="Camera backend to validate when a physical camera is used.",
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--frame-width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--frame-height", type=int, default=DEFAULT_FRAME_HEIGHT)
    parser.add_argument("--inference-interval-ms", type=int, default=DEFAULT_INFERENCE_INTERVAL_MS)
    parser.add_argument("--snapshot-interval-ms", type=int, default=DEFAULT_SNAPSHOT_INTERVAL_MS)
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    check = HealthCheck()
    print(f"Health check mode: {args.mode}")
    print(f"Camera backend: {args.camera_backend}")

    check_python(check)
    check_imports(check, args)
    check_pi_package_origins(check, args)
    check_runtime_imports(check)
    check_runtime_config(check, args)
    check_profile_and_model(check, args)
    check_write_permissions(check)
    check_camera(check, args)

    print()
    if check.failures:
        print(f"Health check failed: {check.failures} failure(s), {check.warnings} warning(s)")
        raise SystemExit(1)

    print(f"Health check passed: {check.warnings} warning(s)")


def check_python(check):
    version = sys.version_info
    if version >= (3, 10):
        check.ok(f"Python {version.major}.{version.minor}.{version.micro}")
    else:
        check.fail(f"Python 3.10+ required, found {version.major}.{version.minor}.{version.micro}")


def check_imports(check, args):
    imports = dict(COMMON_RUNTIME_IMPORTS)
    if not args.dry_run:
        imports.update(MODEL_RUNTIME_IMPORTS)
    if args.mode == "laptop":
        imports.update(DESKTOP_IMPORTS)
    else:
        imports.update(PI_RUNTIME_IMPORTS)
        check.warn("Skipping PySide6 check in Pi mode; it is only required for the desktop app.")

    for module_name, package_name in imports.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            check.fail(f"Cannot import {module_name} ({package_name}): {exc}")
        else:
            check.ok(f"Imported {module_name} ({package_name})")


def check_pi_package_origins(check, args):
    if args.mode != "pi":
        return

    for module_name, label in (("numpy", "NumPy"), ("cv2", "OpenCV")):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        module_path = Path(getattr(module, "__file__", "")).resolve()
        check.ok(f"{label} path: {module_path}")
        if ".venv" in module_path.parts or "site-packages" in module_path.parts:
            check.fail(
                f"{label} is loading from a venv/pip path ({module_path}). "
                "On Raspberry Pi runtime it should come from apt/system packages."
            )
        elif "dist-packages" not in module_path.parts:
            check.warn(
                f"{label} is not clearly from /usr/lib/python3/dist-packages: {module_path}"
            )


def check_runtime_imports(check):
    for module_name in RUNTIME_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            check.fail(f"Cannot import runtime module {module_name}: {exc}")
        else:
            check.ok(f"Imported runtime module {module_name}")


def check_runtime_config(check, args):
    positive_ints = {
        "imgsz": args.imgsz,
        "frame-width": args.frame_width,
        "frame-height": args.frame_height,
    }
    non_negative_ints = {
        "inference-interval-ms": args.inference_interval_ms,
        "snapshot-interval-ms": args.snapshot_interval_ms,
    }

    for name, value in positive_ints.items():
        if value and value > 0:
            check.ok(f"{name}={value}")
        else:
            check.fail(f"{name} must be greater than 0")

    for name, value in non_negative_ints.items():
        if value is not None and value >= 0:
            check.ok(f"{name}={value}")
        else:
            check.fail(f"{name} must be 0 or greater")


def check_profile_and_model(check, args):
    from app.runtime.inference_engine import find_ncnn_model

    profile_dir = MODELS_DIR / args.profile
    if not profile_dir.exists():
        if args.dry_run:
            check.warn(f"Model profile path does not exist, but dry-run is enabled: {profile_dir}")
        else:
            check.fail(f"Model profile path does not exist: {profile_dir}")
        return

    check.ok(f"Model profile exists: {profile_dir}")

    classes_path = profile_dir / "classes.txt"
    if classes_path.exists():
        classes = [
            line.strip()
            for line in classes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if classes:
            check.ok(f"Loaded {len(classes)} class(es) from {classes_path}")
        else:
            check.fail(f"classes.txt is empty: {classes_path}")
    else:
        check.fail(f"Missing classes.txt: {classes_path}")

    config = {}
    config_path = profile_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            check.fail(f"Invalid config.json: {exc}")
        else:
            check.ok(f"Loaded profile config: {config_path}")
    else:
        check.warn(f"Missing config.json; runtime will use defaults: {config_path}")

    model_path = resolve_model_path(profile_dir, config, args.model)
    if model_path.exists():
        check.ok(f"Model file exists: {model_path}")
    elif args.dry_run:
        check.warn(f"Model file does not exist, but dry-run is enabled: {model_path}")
    else:
        check.fail(f"Model file does not exist: {model_path}")

    ncnn_model = find_ncnn_model(profile_dir)
    if ncnn_model:
        check.ok(f"NCNN edge model exists: {ncnn_model}")
    elif args.mode == "pi":
        check.warn(
            "Only .pt/no edge model found. On Raspberry Pi 4, .pt inference may fail "
            "with Illegal instruction. Export NCNN on desktop for runtime."
        )

    check_runtime_profile_rules(check, args.profile)


def resolve_model_path(profile_dir, config, model_override):
    if model_override:
        path = Path(model_override)
        return path if path.is_absolute() else PROJECT_ROOT / path

    configured_model = config.get("model_file")
    if configured_model:
        return profile_dir / configured_model

    latest_model = profile_dir / "latest" / "best.pt"
    if latest_model.exists():
        return latest_model

    return profile_dir / "best.pt"


def check_runtime_profile_rules(check, profile_name):
    rules_path = PROJECT_ROOT / "profiles" / profile_name / "config.yaml"
    if not rules_path.exists():
        check.warn(f"No runtime inspection rule YAML found: {rules_path}")
        return

    try:
        yaml = importlib.import_module("yaml")
        rules = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        check.fail(f"Invalid runtime inspection rules YAML: {exc}")
        return

    inspection = rules.get("inspection", {})
    if not isinstance(inspection, dict):
        check.fail(f"inspection must be a mapping in {rules_path}")
        return

    for key in ("acceptable_classes", "reject_classes"):
        if key in inspection and not isinstance(inspection[key], list):
            check.fail(f"inspection.{key} must be a list in {rules_path}")
            return

    for key in ("minimum_confidence", "required_consecutive_detections", "allowed_no_detection_frames"):
        if key not in inspection:
            continue
        try:
            value = float(inspection[key])
        except (TypeError, ValueError):
            check.fail(f"inspection.{key} must be numeric in {rules_path}")
            return
        if key != "minimum_confidence" and value < 1:
            check.fail(f"inspection.{key} must be 1 or greater in {rules_path}")
            return

    check.ok(f"Runtime inspection rules loaded: {rules_path}")


def check_write_permissions(check):
    for folder in (DATA_DIR, LOGS_DIR, REVIEW_IMAGES_DIR):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=folder, prefix=".health_check_", delete=True):
                pass
        except Exception as exc:
            check.fail(f"Cannot write to {folder}: {exc}")
        else:
            check.ok(f"Writable folder: {folder}")


def check_camera(check, args):
    if args.mode == "laptop" and not args.camera_source and args.camera is not None:
        check.warn("Laptop mode is usually tested with --camera-source; checking physical camera")

    if args.camera_source:
        if args.mode == "pi":
            check.warn("Pi mode received --camera-source; this checks simulation, not hardware capture")
        check_simulated_camera_source(check, args.camera_source)
        return

    if args.camera_backend == CAMERA_BACKEND_PICAMERA2:
        check_picamera2_camera(check, args, required=True)
        return

    if args.camera_backend == CAMERA_BACKEND_AUTO and args.mode == "pi":
        if check_picamera2_camera(check, args, required=False):
            return

        if args.camera is not None:
            check.warn("Picamera2 check failed; also checking OpenCV fallback camera")
            check_opencv_camera(check, args)
        else:
            check.fail("No usable camera backend found. Install Picamera2 or provide --camera for OpenCV fallback.")
        return

    if args.camera is None:
        check.warn(
            "No camera check requested; pass --camera-source <path>, "
            "--camera-backend picamera2, or --camera 0"
        )
        return

    check_opencv_camera(check, args)


def check_opencv_camera(check, args):
    try:
        cv2 = importlib.import_module("cv2")
    except Exception as exc:
        check.fail(f"Cannot check camera because OpenCV import failed: {exc}")
        return False

    capture = cv2.VideoCapture(args.camera)
    try:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.frame_width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.frame_height))
        if not capture.isOpened():
            check.fail(f"Camera {args.camera} did not open")
            return False

        ok, frame = capture.read()
        if ok and frame is not None:
            check.ok(f"Camera {args.camera} opened and returned a frame")
            return True
        else:
            check.fail(f"Camera {args.camera} opened but did not return a frame")
            return False
    finally:
        capture.release()


def check_picamera2_camera(check, args, required=True):
    try:
        from app.runtime.picamera2_manager import Picamera2CameraManager
    except Exception as exc:
        _camera_check_issue(check, f"Cannot import Picamera2 camera manager: {exc}", required)
        return False

    try:
        Picamera2CameraManager._import_picamera2()
    except Exception as exc:
        _camera_check_issue(check, str(exc), required)
        return False

    camera = Picamera2CameraManager(
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        warmup_seconds=0.2,
    )
    try:
        if not camera.open():
            _camera_check_issue(check, camera.last_error or "Picamera2 camera did not open", required)
            return False

        frame = None
        deadline = time.time() + 5.0
        while time.time() < deadline:
            frame = camera.read_frame()
            if frame is not None:
                break
            time.sleep(0.05)

        if frame is None:
            _camera_check_issue(
                check,
                camera.last_error or "Picamera2 opened but did not return a frame",
                required,
            )
            return False

        status = camera.get_status()
        check.ok(
            f"Picamera2 returned frame {frame.shape[1]}x{frame.shape[0]} "
            f"at {status.get('fps', 0.0):.2f} fps"
        )
        return True
    finally:
        camera.release()


def _camera_check_issue(check, message, required):
    if required:
        check.fail(message)
    else:
        check.warn(message)


def check_simulated_camera_source(check, source_path):
    try:
        from app.runtime.camera_sources import SimulatedCameraSource
    except Exception as exc:
        check.fail(f"Cannot import simulated camera source: {exc}")
        return

    source = SimulatedCameraSource(source_path, frame_interval_seconds=0)
    try:
        if not source.open():
            check.fail(source.last_error or f"Could not open simulated camera source: {source_path}")
            return

        frame = source.read_frame()
        if frame is None:
            check.fail(source.last_error or f"Could not read simulated camera source: {source_path}")
            return

        check.ok(f"Simulated camera source returned frame {frame.shape[1]}x{frame.shape[0]}")
    finally:
        source.release()


if __name__ == "__main__":
    main()
