import time

import cv2


class CameraManager:
    """Small reconnecting wrapper around OpenCV camera capture."""

    CONNECTED = "Connected"
    RECONNECTING = "Reconnecting"
    FAILED = "Failed"

    def __init__(
        self,
        camera_index=0,
        reconnect_after_failures=5,
        reconnect_cooldown_seconds=2.0,
        frame_width=None,
        frame_height=None,
    ):
        self.camera_index = camera_index
        self.reconnect_after_failures = reconnect_after_failures
        self.reconnect_cooldown_seconds = reconnect_cooldown_seconds
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.capture = None
        self.status = self.FAILED
        self.failure_count = 0
        self.last_reconnect_attempt = 0.0
        self.last_error = ""
        self.backend = "opencv"
        self.last_frame_time = None

    @property
    def connected(self):
        return self.status == self.CONNECTED

    def open(self):
        self.release()

        try:
            self.capture = cv2.VideoCapture(self.camera_index)

            if self.frame_width:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.frame_width))
            if self.frame_height:
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.frame_height))
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as exc:
            self.last_error = str(exc)
            self.status = self.FAILED
            self.release()
            return False

        if self.capture.isOpened():
            self.status = self.CONNECTED
            self.failure_count = 0
            self.last_error = ""
            return True

        self.last_error = f"OpenCV camera {self.camera_index} did not open"
        self.release()
        self.status = self.FAILED
        return False

    def read_frame(self):
        if self.capture is None or not self.capture.isOpened():
            self.status = self.RECONNECTING
            self._maybe_reconnect()
            return None

        try:
            ok, frame = self.capture.read()
        except Exception as exc:
            self.last_error = str(exc)
            ok, frame = False, None

        if ok and frame is not None:
            self.status = self.CONNECTED
            self.failure_count = 0
            self.last_error = ""
            self.last_frame_time = time.time()
            return frame

        self.failure_count += 1
        self.last_error = f"OpenCV camera {self.camera_index} returned no frame"
        if self.failure_count >= self.reconnect_after_failures:
            self.status = self.RECONNECTING
            self._maybe_reconnect()

        return None

    def _maybe_reconnect(self):
        now = time.time()
        if now - self.last_reconnect_attempt < self.reconnect_cooldown_seconds:
            return

        self.last_reconnect_attempt = now
        if not self.open():
            self.status = self.FAILED

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
