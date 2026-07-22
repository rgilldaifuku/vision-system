from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import LOGS_DIR
from app.runtime.inspection_result import iso_timestamp


EVENTS_JSONL = LOGS_DIR / "events.jsonl"


class EventSeverity:
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class RuntimeEvent:
    event_id: str
    event_type: str
    severity: str
    timestamp: datetime
    profile: str = ""
    inspection_id: str | None = None
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    image_path: str | None = None

    def to_dict(self):
        payload = asdict(self)
        payload["timestamp"] = iso_timestamp(self.timestamp)
        return payload


def build_event(
    event_type,
    severity=EventSeverity.INFO,
    profile="",
    inspection_id=None,
    message="",
    details=None,
    image_path=None,
):
    return RuntimeEvent(
        event_id=f"EVT-{uuid.uuid4().hex[:12].upper()}",
        event_type=str(event_type),
        severity=str(severity),
        timestamp=datetime.now(timezone.utc),
        profile=str(profile or ""),
        inspection_id=str(inspection_id) if inspection_id else None,
        message=str(message or ""),
        details=_json_safe(dict(details or {})),
        image_path=str(image_path) if image_path else None,
    )


class EventManager:
    """Thread-safe JSONL event recorder with bounded background persistence."""

    def __init__(self, event_path=None, max_queue_size=200, recent_limit=25):
        self.event_path = Path(event_path or EVENTS_JSONL)
        self.queue = queue.Queue(maxsize=max(1, int(max_queue_size)))
        self.recent_limit = max(1, int(recent_limit))
        self.recent_events = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.last_error = ""
        self.dropped_events = 0
        self.worker.start()

    def record(
        self,
        event_type,
        severity=EventSeverity.INFO,
        profile="",
        inspection_id=None,
        message="",
        details=None,
        image_path=None,
    ):
        event = build_event(
            event_type=event_type,
            severity=severity,
            profile=profile,
            inspection_id=inspection_id,
            message=message,
            details=details,
            image_path=image_path,
        )
        self._remember(event)
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            with self.lock:
                self.dropped_events += 1
        return event

    def snapshot(self):
        with self.lock:
            return {
                "recent": [event.to_dict() for event in self.recent_events],
                "dropped_events": self.dropped_events,
                "last_error": self.last_error,
                "event_path": str(self.event_path),
            }

    def stop(self, timeout=1.0):
        deadline = time.time() + max(0.0, float(timeout))
        while not self.queue.empty() and time.time() < deadline:
            time.sleep(0.02)
        self.stop_event.set()
        self.worker.join(timeout=timeout)

    def _remember(self, event):
        with self.lock:
            self.recent_events.append(event)
            self.recent_events = self.recent_events[-self.recent_limit :]

    def _worker_loop(self):
        while not self.stop_event.is_set():
            try:
                event = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._write_event(event)
                with self.lock:
                    self.last_error = ""
            except Exception as exc:
                with self.lock:
                    self.last_error = str(exc)
            finally:
                self.queue.task_done()

    def _write_event(self, event):
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.event_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")


class DuplicateSuppressor:
    """Cooldown and repeated-event helper for notifications and derived events."""

    def __init__(self, cooldown_seconds=60.0, repeat_threshold=3, repeat_window_seconds=300.0):
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.repeat_threshold = max(1, int(repeat_threshold))
        self.repeat_window_seconds = max(1.0, float(repeat_window_seconds))
        self.last_times = {}
        self.history = {}

    def should_emit(self, key, now=None):
        now = now or time.time()
        last_time = self.last_times.get(key, 0.0)
        if now - last_time < self.cooldown_seconds:
            return False
        self.last_times[key] = now
        return True

    def repeated(self, key, now=None):
        now = now or time.time()
        history = [item for item in self.history.get(key, []) if now - item <= self.repeat_window_seconds]
        history.append(now)
        self.history[key] = history
        return len(history) >= self.repeat_threshold


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
