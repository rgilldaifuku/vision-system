from __future__ import annotations

import itertools
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class InspectionState(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"
    NO_PART = "NO_PART"
    SYSTEM_ERROR = "SYSTEM_ERROR"


_ID_COUNTER = itertools.count(1)
_ID_LOCK = threading.Lock()


def utc_now():
    return datetime.now(timezone.utc)


def iso_timestamp(value=None):
    timestamp = value or utc_now()
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.isoformat(timespec="seconds")
    return str(timestamp)


def generate_inspection_id(timestamp=None):
    stamp = timestamp or utc_now()
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    with _ID_LOCK:
        sequence = next(_ID_COUNTER)
    return f"INS-{stamp.strftime('%Y%m%d-%H%M%S')}-{sequence:06d}"


def canonical_state_from_result(result):
    value = str(result or "").upper()
    if value == "PASS":
        return InspectionState.PASS
    if value == "FAIL":
        return InspectionState.FAIL
    if value == "NO_PART":
        return InspectionState.NO_PART
    if value in {"CAMERA_ERROR", "MODEL_ERROR", "SYSTEM_ERROR"}:
        return InspectionState.SYSTEM_ERROR
    return InspectionState.REVIEW


@dataclass
class InspectionDecision:
    inspection_id: str
    state: InspectionState
    reason: str
    timestamp: datetime
    profile: str
    model_name: str | None = None
    model_version: str | None = None
    detected_class: str | None = None
    confidence: float | None = None
    average_confidence: float | None = None
    agreement_ratio: float | None = None
    image_quality_status: str | None = None
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["timestamp"] = iso_timestamp(self.timestamp)
        return payload


def build_decision(
    *,
    state,
    reason,
    profile,
    model_name=None,
    model_version=None,
    detected_class=None,
    confidence=None,
    average_confidence=None,
    agreement_ratio=None,
    image_quality_status=None,
    image_path=None,
    metadata=None,
    timestamp=None,
):
    timestamp = timestamp or utc_now()
    return InspectionDecision(
        inspection_id=generate_inspection_id(timestamp),
        state=state if isinstance(state, InspectionState) else InspectionState(str(state)),
        reason=str(reason or ""),
        timestamp=timestamp,
        profile=str(profile or ""),
        model_name=str(model_name) if model_name else None,
        model_version=str(model_version) if model_version else None,
        detected_class=str(detected_class) if detected_class else None,
        confidence=_safe_float(confidence),
        average_confidence=_safe_float(average_confidence),
        agreement_ratio=_safe_float(agreement_ratio),
        image_quality_status=str(image_quality_status) if image_quality_status else None,
        image_path=str(image_path) if image_path else None,
        metadata=dict(metadata or {}),
    )


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
