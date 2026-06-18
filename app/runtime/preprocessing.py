import cv2
import numpy as np


def apply_camera_transforms(frame, camera_profile=None):
    metadata = {
        "rotation": 0,
        "flip_horizontal": False,
        "flip_vertical": False,
        "transformed": False,
    }
    if frame is None or camera_profile is None:
        return frame, metadata

    rotation = int(getattr(camera_profile, "rotation", 0) or 0)
    flip_horizontal = bool(getattr(camera_profile, "flip_horizontal", False))
    flip_vertical = bool(getattr(camera_profile, "flip_vertical", False))

    output = frame
    if rotation == 90:
        output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        output = cv2.rotate(output, cv2.ROTATE_180)
    elif rotation == 270:
        output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if flip_horizontal and flip_vertical:
        output = cv2.flip(output, -1)
    elif flip_horizontal:
        output = cv2.flip(output, 1)
    elif flip_vertical:
        output = cv2.flip(output, 0)

    metadata.update(
        {
            "rotation": rotation,
            "flip_horizontal": flip_horizontal,
            "flip_vertical": flip_vertical,
            "transformed": output is not frame,
        }
    )
    return output, metadata


def apply_roi(frame, roi_config=None):
    metadata = {
        "roi_enabled": False,
        "roi_normalized": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
        "roi_pixels": None,
        "applied": False,
    }
    if frame is None:
        return frame, metadata

    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return frame, metadata

    if not _roi_enabled(roi_config):
        metadata["roi_pixels"] = {"x1": 0, "y1": 0, "x2": int(width), "y2": int(height)}
        return frame, metadata

    x1, y1, x2, y2 = _normalized_roi_values(roi_config)
    left = int(round(x1 * width))
    top = int(round(y1 * height))
    right = int(round(x2 * width))
    bottom = int(round(y2 * height))

    left = max(0, min(width - 1, left))
    top = max(0, min(height - 1, top))
    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))

    metadata.update(
        {
            "roi_enabled": True,
            "roi_normalized": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "roi_pixels": {"x1": left, "y1": top, "x2": right, "y2": bottom},
            "applied": True,
        }
    )
    return frame[top:bottom, left:right].copy(), metadata


def standardize_frame(frame, target_size, color_normalization=False):
    metadata = {
        "applied": False,
        "resized_to": None,
        "letterbox": False,
        "scale": 1.0,
        "pad": {"left": 0, "top": 0, "right": 0, "bottom": 0},
        "color_normalization": bool(color_normalization),
    }
    if frame is None:
        return frame, metadata

    size = _target_size(target_size)
    if size is None:
        output = frame.copy()
    else:
        output, resize_metadata = _letterbox(frame, size)
        metadata.update(resize_metadata)

    if color_normalization:
        output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)

    metadata["applied"] = True
    return output, metadata


def preprocess_for_inference(frame, camera_profile=None, imgsz=320):
    transformed, transform_meta = apply_camera_transforms(frame, camera_profile)
    roi_frame, roi_meta = apply_roi(transformed, getattr(camera_profile, "roi", None))
    preprocessing = getattr(camera_profile, "preprocessing", None)
    preprocessing_enabled = bool(getattr(preprocessing, "enabled", True))
    color_normalization = bool(getattr(preprocessing, "color_normalization", False))

    if preprocessing_enabled:
        processed, standardize_meta = standardize_frame(
            roi_frame,
            target_size=imgsz,
            color_normalization=color_normalization,
        )
    else:
        processed = roi_frame
        standardize_meta = {
            "applied": False,
            "resized_to": None,
            "letterbox": False,
            "scale": 1.0,
            "pad": {"left": 0, "top": 0, "right": 0, "bottom": 0},
            "color_normalization": False,
        }

    metadata = {
        "applied": bool(transform_meta.get("transformed") or roi_meta.get("applied") or standardize_meta.get("applied")),
        "transforms": transform_meta,
        "roi_enabled": bool(roi_meta.get("roi_enabled")),
        "roi_pixels": roi_meta.get("roi_pixels"),
        "roi_normalized": roi_meta.get("roi_normalized"),
        "roi_applied": bool(roi_meta.get("applied")),
        "preprocessing_enabled": preprocessing_enabled,
        "resized_to": standardize_meta.get("resized_to"),
        "letterbox": standardize_meta.get("letterbox"),
        "scale": standardize_meta.get("scale"),
        "pad": standardize_meta.get("pad"),
        "color_normalization": color_normalization,
        "input_shape": _shape(frame),
        "output_shape": _shape(processed),
    }
    return processed, metadata


def _target_size(target_size):
    if target_size is None:
        return None
    if isinstance(target_size, (tuple, list)):
        if len(target_size) != 2:
            return None
        width, height = int(target_size[0]), int(target_size[1])
    else:
        width = height = int(target_size)
    if width <= 0 or height <= 0:
        return None
    return width, height


def _letterbox(frame, size):
    target_width, target_height = size
    height, width = frame.shape[:2]
    scale = min(target_width / width, target_height / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    pad_left = (target_width - new_width) // 2
    pad_right = target_width - new_width - pad_left
    pad_top = (target_height - new_height) // 2
    pad_bottom = target_height - new_height - pad_top
    output = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return output, {
        "resized_to": {"width": target_width, "height": target_height},
        "letterbox": True,
        "scale": float(scale),
        "pad": {
            "left": int(pad_left),
            "top": int(pad_top),
            "right": int(pad_right),
            "bottom": int(pad_bottom),
        },
    }


def _roi_enabled(roi_config):
    if roi_config is None:
        return False
    if isinstance(roi_config, dict):
        return bool(roi_config.get("enabled", False))
    return bool(getattr(roi_config, "enabled", False))


def _normalized_roi_values(roi_config):
    if isinstance(roi_config, dict):
        values = (
            roi_config.get("x1", 0.0),
            roi_config.get("y1", 0.0),
            roi_config.get("x2", 1.0),
            roi_config.get("y2", 1.0),
        )
    else:
        values = (
            getattr(roi_config, "x1", 0.0),
            getattr(roi_config, "y1", 0.0),
            getattr(roi_config, "x2", 1.0),
            getattr(roi_config, "y2", 1.0),
        )

    x1, y1, x2, y2 = (_clamp_unit(value) for value in values)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _clamp_unit(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(0.0, min(1.0, number))


def _shape(frame):
    if frame is None or not hasattr(frame, "shape"):
        return None
    height, width = frame.shape[:2]
    channels = frame.shape[2] if len(frame.shape) > 2 else 1
    return {"height": int(height), "width": int(width), "channels": int(channels)}
