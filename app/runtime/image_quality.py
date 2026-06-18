from dataclasses import asdict, dataclass
from datetime import datetime

import cv2
import numpy as np


GOOD = "GOOD"
TOO_DARK = "TOO_DARK"
TOO_BRIGHT = "TOO_BRIGHT"
BLURRY = "BLURRY"
LOW_CONTRAST = "LOW_CONTRAST"
INVALID_FRAME = "INVALID_FRAME"
QUALITY_CHECK_ERROR = "QUALITY_CHECK_ERROR"


DEFAULT_QUALITY_THRESHOLDS = {
    "min_brightness": 40.0,
    "max_brightness": 220.0,
    "min_blur_score": 50.0,
    "min_contrast": 15.0,
    "max_overexposed_pct": 20.0,
    "max_underexposed_pct": 20.0,
}


@dataclass
class ImageQualityResult:
    brightness_mean: float | None = None
    brightness_std: float | None = None
    contrast_score: float | None = None
    blur_score: float | None = None
    overexposed_pct: float | None = None
    underexposed_pct: float | None = None
    width: int = 0
    height: int = 0
    timestamp: str = ""
    quality_status: str = INVALID_FRAME
    message: str = ""

    def to_dict(self):
        return asdict(self)


def quality_thresholds_from_config(config=None):
    thresholds = dict(DEFAULT_QUALITY_THRESHOLDS)
    if config is None:
        return thresholds

    source = config
    if hasattr(config, "to_dict"):
        source = config.to_dict()
    elif hasattr(config, "__dict__"):
        source = vars(config)

    for key in thresholds:
        value = source.get(key) if isinstance(source, dict) else None
        if value is None:
            continue
        try:
            thresholds[key] = float(value)
        except (TypeError, ValueError):
            pass
    return thresholds


def compute_image_quality(frame, thresholds=None):
    timestamp = datetime.now().isoformat(timespec="seconds")
    thresholds = thresholds or DEFAULT_QUALITY_THRESHOLDS

    if frame is None or not hasattr(frame, "shape") or frame.size == 0:
        return ImageQualityResult(
            timestamp=timestamp,
            quality_status=INVALID_FRAME,
            message="Frame is empty or unavailable.",
        ).to_dict()

    try:
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            return ImageQualityResult(
                width=max(0, int(width)),
                height=max(0, int(height)),
                timestamp=timestamp,
                quality_status=INVALID_FRAME,
                message="Frame has invalid dimensions.",
            ).to_dict()

        gray = _to_gray(frame)
        brightness_mean = float(np.mean(gray))
        brightness_std = float(np.std(gray))
        contrast_score = brightness_std
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        overexposed_pct = float(np.count_nonzero(gray >= 250) * 100.0 / gray.size)
        underexposed_pct = float(np.count_nonzero(gray <= 5) * 100.0 / gray.size)

        status, message = classify_quality(
            brightness_mean=brightness_mean,
            contrast_score=contrast_score,
            blur_score=blur_score,
            overexposed_pct=overexposed_pct,
            underexposed_pct=underexposed_pct,
            thresholds=thresholds,
        )

        return ImageQualityResult(
            brightness_mean=round(brightness_mean, 3),
            brightness_std=round(brightness_std, 3),
            contrast_score=round(contrast_score, 3),
            blur_score=round(blur_score, 3),
            overexposed_pct=round(overexposed_pct, 3),
            underexposed_pct=round(underexposed_pct, 3),
            width=int(width),
            height=int(height),
            timestamp=timestamp,
            quality_status=status,
            message=message,
        ).to_dict()
    except Exception as exc:
        return ImageQualityResult(
            timestamp=timestamp,
            quality_status=QUALITY_CHECK_ERROR,
            message=f"Quality check failed: {exc}",
        ).to_dict()


def classify_quality(
    brightness_mean,
    contrast_score,
    blur_score,
    overexposed_pct,
    underexposed_pct,
    thresholds=None,
):
    thresholds = thresholds or DEFAULT_QUALITY_THRESHOLDS

    if brightness_mean < thresholds["min_brightness"]:
        return TOO_DARK, "Image is too dark."
    if brightness_mean > thresholds["max_brightness"]:
        return TOO_BRIGHT, "Image is too bright."
    if overexposed_pct > thresholds["max_overexposed_pct"]:
        return TOO_BRIGHT, "Image has too many overexposed pixels."
    if underexposed_pct > thresholds["max_underexposed_pct"]:
        return TOO_DARK, "Image has too many underexposed pixels."
    if blur_score < thresholds["min_blur_score"]:
        return BLURRY, "Image appears blurry."
    if contrast_score < thresholds["min_contrast"]:
        return LOW_CONTRAST, "Image contrast is too low."
    return GOOD, "Image quality is acceptable."


def _to_gray(frame):
    if len(frame.shape) == 2:
        return frame
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
