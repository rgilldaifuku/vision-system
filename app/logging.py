from datetime import datetime
import csv 

from app.config import LOGS_DIR

LOG_FILE = LOGS_DIR /"detections.csv"

LOG_FIELDS = [
    "timestamp",
    "active_profile",
    "stable_detected",
    "raw_detected",
    "class_name",
    "confidence",
    "camera_status",
    "roi_enabled",
    "saved_image_path",
]


def _legacy_log_path():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    legacy_path = LOG_FILE.with_name(f"detections_legacy_{timestamp}.csv")
    counter = 1

    while legacy_path.exists():
        legacy_path = LOG_FILE.with_name(f"detections_legacy_{timestamp}_{counter}.csv")
        counter += 1

    return legacy_path


def _ensure_log_file():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if LOG_FILE.exists():
        with open(LOG_FILE, "r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)

        if header == LOG_FIELDS:
            return

        LOG_FILE.rename(_legacy_log_path())

    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()


def _format_bool(value):
    return "true" if bool(value) else "false"


def _format_confidence(confidence):
    if confidence is None or confidence == "":
        return ""

    try:
        return f"{float(confidence):.4f}"
    except (TypeError, ValueError):
        return str(confidence)


def log_detection_event(
    active_profile,
    stable_detected,
    raw_detected,
    class_name="",
    confidence=None,
    camera_status="",
    roi_enabled=False,
    saved_image_path="",
):
    _ensure_log_file()

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "active_profile": active_profile,
        "stable_detected": _format_bool(stable_detected),
        "raw_detected": _format_bool(raw_detected),
        "class_name": class_name or "",
        "confidence": _format_confidence(confidence),
        "camera_status": camera_status,
        "roi_enabled": _format_bool(roi_enabled),
        "saved_image_path": saved_image_path or "",
    }

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writerow(row)


def log_detection(detected, class_name="", confidence=0.0):
    log_detection_event(
        active_profile="",
        stable_detected=detected,
        raw_detected=detected,
        class_name=class_name,
        confidence=confidence,
    )
