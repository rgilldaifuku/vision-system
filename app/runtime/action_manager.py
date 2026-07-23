import json
import time
from pathlib import Path

from app.config import LOGS_DIR
from app.logging import log_runtime_event


DEFAULT_ACTIONS = {
    "PASS": ["log_event", "write_latest_status_json", "increment_counter"],
    "FAIL": ["log_event", "save_review_image", "write_latest_status_json", "increment_counter"],
    "REVIEW": ["log_event", "save_review_image", "write_latest_status_json", "increment_counter"],
    "LOW_CONFIDENCE": [
        "log_event",
        "save_review_image",
        "write_latest_status_json",
        "increment_counter",
    ],
    "NO_PART": ["log_event", "save_review_image", "write_latest_status_json", "increment_counter"],
    "CAMERA_ERROR": ["log_event", "write_latest_status_json", "increment_counter"],
    "MODEL_ERROR": ["log_event", "write_latest_status_json", "increment_counter"],
    "SYSTEM_ERROR": ["log_event", "write_latest_status_json", "increment_counter"],
    "SIMULATION": ["log_event", "write_latest_status_json", "increment_counter"],
    "CAMERA_ONLY": ["log_event", "write_latest_status_json"],
    "INFERENCE_DISABLED": ["log_event", "write_latest_status_json"],
    "IMAGE_QUALITY_ERROR": ["log_event", "write_latest_status_json", "increment_counter"],
    "QUALITY_CHECK_ERROR": ["log_event", "write_latest_status_json", "increment_counter"],
}


class ActionManager:
    """Runs safe local actions from the final inspection result."""

    def __init__(
        self,
        action_config=None,
        status_path=None,
        event_cooldown_seconds=2.0,
        console_print=False,
    ):
        self.action_map = self._load_action_map(action_config or {})
        self.status_path = Path(status_path or LOGS_DIR / "latest_status.json")
        self.event_cooldown_seconds = max(0.0, float(event_cooldown_seconds))
        self.console_print = bool(console_print)
        self.last_event_times = {}
        self.counters = {
            "pass": 0,
            "fail": 0,
            "review": 0,
            "low_confidence": 0,
            "no_part": 0,
            "system_error": 0,
            "camera_error": 0,
            "model_error": 0,
            "simulation": 0,
            "image_quality_error": 0,
            "quality_check_error": 0,
        }
        self.last_error = ""

    def handle(self, status_document):
        result = status_document.get("inspection_result", "NO_PART")
        actions = self.action_map.get(result, [])

        if "increment_counter" in actions:
            self._increment_counter(result)

        status_document = dict(status_document)
        status_document["counters"] = {
            **status_document.get("counters", {}),
            **self.counters,
        }

        if "write_latest_status_json" in actions:
            self._write_latest_status(status_document)

        if "log_event" in actions:
            self._log_action_event(status_document)

        if "console_print" in actions or self.console_print:
            print(
                f"[vision] {result}: {status_document.get('message', '')}",
                flush=True,
            )

        # Placeholders stay explicit and disabled. Future adapters can hook in here.
        return {
            "actions": actions,
            "counters": dict(self.counters),
            "latest_status_path": str(self.status_path),
            "placeholder_outputs": {
                "webhook": "disabled",
                "gpio": "disabled",
            },
            "last_error": self.last_error,
        }

    def _increment_counter(self, result):
        key = str(result).lower()
        if key in self.counters:
            self.counters[key] += 1

    def _write_latest_status(self, status_document):
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.status_path.with_suffix(".tmp")
            temp_path.write_text(
                json.dumps(status_document, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temp_path.replace(self.status_path)
            self.last_error = ""
        except Exception as exc:
            self.last_error = str(exc)

    def _log_action_event(self, status_document):
        result = status_document.get("inspection_result", "NO_PART")
        now = time.time()
        if now - self.last_event_times.get(result, 0.0) < self.event_cooldown_seconds:
            return

        self.last_event_times[result] = now
        try:
            log_runtime_event(
                event_type="inspection_action",
                profile=status_document.get("profile", ""),
                message=status_document.get("message", ""),
                details={
                    "inspection_result": result,
                    "pass_fail_bool": status_document.get("pass_fail_bool"),
                    "active_class": status_document.get("active_class"),
                    "confidence": status_document.get("confidence"),
                    "saved_image_path": status_document.get("saved_image_path"),
                    "actions": self.action_map.get(result, []),
                },
            )
            self.last_error = ""
        except Exception as exc:
            self.last_error = str(exc)

    @staticmethod
    def _load_action_map(config):
        configured_actions = config.get("actions_by_result") or config.get("by_result") or {}
        action_map = {key: list(value) for key, value in DEFAULT_ACTIONS.items()}

        for result, actions in configured_actions.items():
            if isinstance(actions, str):
                action_map[str(result)] = [actions]
            else:
                action_map[str(result)] = list(actions)

        return action_map
