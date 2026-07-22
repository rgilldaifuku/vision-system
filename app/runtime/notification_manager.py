from __future__ import annotations

import json
import os
import smtplib
import urllib.request
from email.message import EmailMessage

from app.runtime.event_manager import DuplicateSuppressor, EventSeverity


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class NotificationManager:
    """Optional external notification adapter. Disabled by default."""

    def __init__(self, enabled=None, cooldown_seconds=None, timeout_seconds=5.0):
        self.enabled = _env_bool("VISION_NOTIFICATIONS_ENABLED", False) if enabled is None else bool(enabled)
        self.email_enabled = _env_bool("VISION_EMAIL_ENABLED", False)
        self.teams_webhook_url = os.getenv("VISION_TEAMS_WEBHOOK_URL", "")
        self.timeout_seconds = max(0.5, float(timeout_seconds))
        cooldown = cooldown_seconds
        if cooldown is None:
            cooldown = float(os.getenv("VISION_NOTIFICATION_COOLDOWN_SECONDS", "60"))
        self.suppressor = DuplicateSuppressor(cooldown_seconds=cooldown)
        self.last_error = ""
        self.last_delivery = None

    def status(self):
        return {
            "enabled": self.enabled,
            "email_enabled": self.email_enabled,
            "teams_enabled": bool(self.teams_webhook_url),
            "last_error": self.last_error,
            "last_delivery": self.last_delivery,
        }

    def notify(self, event):
        if not self.enabled:
            return {"sent": False, "reason": "notifications_disabled"}

        event_payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        key = f"{event_payload.get('event_type')}:{event_payload.get('severity')}"
        if not self.suppressor.should_emit(key):
            return {"sent": False, "reason": "cooldown"}

        sent = []
        errors = []
        if self.email_enabled:
            result = self._send_email(event_payload)
            sent.append("email") if result["ok"] else errors.append(result["error"])
        if self.teams_webhook_url:
            result = self._send_teams(event_payload)
            sent.append("teams") if result["ok"] else errors.append(result["error"])

        self.last_error = "; ".join(errors)
        self.last_delivery = {"event_id": event_payload.get("event_id"), "channels": sent}
        return {"sent": bool(sent), "channels": sent, "errors": errors}

    def send_test_notification(self):
        class _Event:
            def to_dict(self):
                return {
                    "event_id": "TEST",
                    "event_type": "TEST_NOTIFICATION",
                    "severity": EventSeverity.INFO,
                    "profile": "test",
                    "inspection_id": None,
                    "message": "Vision notification test.",
                    "details": {},
                    "image_path": None,
                    "timestamp": "",
                }

        return self.notify(_Event())

    def _send_email(self, payload):
        required = {
            "host": os.getenv("VISION_SMTP_HOST"),
            "from": os.getenv("VISION_EMAIL_FROM"),
            "to": os.getenv("VISION_EMAIL_TO"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            return {"ok": False, "error": f"email missing: {', '.join(missing)}"}

        message = EmailMessage()
        message["Subject"] = f"Vision {payload.get('severity')} {payload.get('event_type')}"
        message["From"] = required["from"]
        message["To"] = required["to"]
        message.set_content(json.dumps(_redacted_payload(payload), indent=2, sort_keys=True))
        try:
            port = int(os.getenv("VISION_SMTP_PORT", "587"))
            username = os.getenv("VISION_SMTP_USERNAME")
            password = os.getenv("VISION_SMTP_PASSWORD")
            with smtplib.SMTP(required["host"], port, timeout=self.timeout_seconds) as smtp:
                if username and password:
                    smtp.starttls()
                    smtp.login(username, password)
                smtp.send_message(message)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": f"email delivery failed: {exc}"}

    def _send_teams(self, payload):
        card = {
            "text": (
                f"{payload.get('severity')} {payload.get('event_type')}\\n"
                f"Profile: {payload.get('profile')}\\n"
                f"Inspection: {payload.get('inspection_id')}\\n"
                f"Message: {payload.get('message')}"
            )
        }
        request = urllib.request.Request(
            self.teams_webhook_url,
            data=json.dumps(card).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response.read()
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": f"teams delivery failed: {exc}"}


def _redacted_payload(payload):
    return {
        key: ("<redacted>" if "password" in str(key).lower() or "secret" in str(key).lower() else value)
        for key, value in payload.items()
    }
