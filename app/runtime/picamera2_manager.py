import time

import cv2


class Picamera2CameraManager:
    """Picamera2/libcamera wrapper for Raspberry Pi Camera Module 3."""

    CONNECTED = "Connected"
    RECONNECTING = "Reconnecting"
    FAILED = "Failed"

    def __init__(
        self,
        frame_width=640,
        frame_height=480,
        warmup_seconds=0.5,
        reconnect_cooldown_seconds=2.0,
    ):
        self.frame_width = int(frame_width or 640)
        self.frame_height = int(frame_height or 480)
        self.warmup_seconds = max(0.0, float(warmup_seconds))
        self.reconnect_cooldown_seconds = max(0.0, float(reconnect_cooldown_seconds))

        self.camera = None
        self.status = self.FAILED
        self.last_error = ""
        self.last_frame_time = None
        self.last_reconnect_attempt = 0.0
        self.backend = "picamera2"

    @property
    def connected(self):
        return self.status == self.CONNECTED

    @classmethod
    def is_available(cls):
        try:
            cls._import_picamera2()
        except Exception:
            return False
        return True

    def open(self):
        self.release()

        try:
            Picamera2 = self._import_picamera2()
            self.camera = Picamera2()
            config = self.camera.create_video_configuration(
                main={
                    "size": (self.frame_width, self.frame_height),
                    "format": "RGB888",
                }
            )
            self.camera.configure(config)
            self.camera.start()
            if self.warmup_seconds:
                time.sleep(self.warmup_seconds)
        except Exception as exc:
            self.last_error = self._format_error(exc)
            self.status = self.FAILED
            self.release()
            return False

        self.status = self.CONNECTED
        self.last_error = ""
        return True

    def read_frame(self):
        if self.camera is None:
            self.status = self.RECONNECTING
            self._maybe_reconnect()
            return None

        try:
            frame = self.camera.capture_array()
        except Exception as exc:
            self.last_error = self._format_error(exc)
            self.status = self.RECONNECTING
            self._maybe_reconnect()
            return None

        if frame is None:
            self.last_error = "Picamera2 returned no frame"
            self.status = self.RECONNECTING
            self._maybe_reconnect()
            return None

        self.status = self.CONNECTED
        self.last_error = ""
        self.last_frame_time = time.time()
        return self._to_bgr(frame)

    def release(self):
        if self.camera is None:
            return

        try:
            self.camera.stop()
        except Exception:
            pass

        try:
            self.camera.close()
        except Exception:
            pass

        self.camera = None

    def _maybe_reconnect(self):
        now = time.time()
        if now - self.last_reconnect_attempt < self.reconnect_cooldown_seconds:
            return

        self.last_reconnect_attempt = now
        if not self.open():
            self.status = self.FAILED

    @staticmethod
    def _import_picamera2():
        try:
            from picamera2 import Picamera2
        except Exception as exc:
            raise RuntimeError(
                "Picamera2 is not available. On Raspberry Pi OS, install it with "
                "`sudo apt install python3-picamera2 python3-libcamera libcamera-apps` "
                "and create the venv with `python3 -m venv --system-site-packages .venv`."
            ) from exc

        return Picamera2

    @staticmethod
    def _to_bgr(frame):
        if len(frame.shape) != 3:
            return frame

        channels = frame.shape[2]
        if channels == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        if channels == 3:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    @staticmethod
    def _format_error(exc):
        return f"{type(exc).__name__}: {exc}"
