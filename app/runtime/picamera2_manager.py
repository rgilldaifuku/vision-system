import threading
import time

import cv2


class Picamera2CameraManager:
    """Threaded Picamera2/libcamera wrapper for Raspberry Pi Camera Module 3."""

    CONNECTED = "Connected"
    RECONNECTING = "Reconnecting"
    FAILED = "Failed"

    def __init__(
        self,
        frame_width=640,
        frame_height=480,
        warmup_seconds=0.5,
        reconnect_cooldown_seconds=2.0,
        target_fps=30,
        max_stale_seconds=3.0,
        buffer_count=4,
    ):
        self.frame_width = int(frame_width or 640)
        self.frame_height = int(frame_height or 480)
        self.warmup_seconds = max(0.0, float(warmup_seconds))
        self.reconnect_cooldown_seconds = max(0.0, float(reconnect_cooldown_seconds))
        self.target_fps = max(1, int(target_fps or 30))
        self.max_stale_seconds = max(0.5, float(max_stale_seconds))
        self.buffer_count = max(1, int(buffer_count or 4))

        self.camera = None
        self.status = self.FAILED
        self.last_error = ""
        self.last_frame_time = None
        self.camera_fps = 0.0
        self.backend = "picamera2"

        self.latest_frame = None
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.capture_thread = None
        self.last_reconnect_attempt = 0.0
        self.opened_at = None

    @property
    def connected(self):
        return self.status == self.CONNECTED and not self._is_frame_stale()

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
            camera = Picamera2()
            try:
                config = camera.create_video_configuration(
                    main={
                        "size": (self.frame_width, self.frame_height),
                        "format": "RGB888",
                    },
                    buffer_count=self.buffer_count,
                )
            except TypeError:
                config = camera.create_video_configuration(
                    main={
                        "size": (self.frame_width, self.frame_height),
                        "format": "RGB888",
                    }
                )
            camera.configure(config)
            camera.start()
            if self.warmup_seconds:
                time.sleep(self.warmup_seconds)
        except Exception as exc:
            self.last_error = self._format_error(exc)
            self.status = self.FAILED
            self._close_camera_safely(locals().get("camera"))
            return False

        with self.lock:
            self.camera = camera
            self.latest_frame = None
            self.last_frame_time = None
            self.camera_fps = 0.0
            self.last_error = ""
            self.status = self.CONNECTED
            self.opened_at = time.time()

        self.stop_event.clear()
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            name="picamera2-capture",
            daemon=True,
        )
        self.capture_thread.start()
        return True

    def read_frame(self):
        if self.camera is None:
            self.status = self.RECONNECTING
            self._maybe_reconnect("Picamera2 camera is not open")
            return None

        if self._is_frame_stale():
            self.status = self.RECONNECTING
            self._maybe_reconnect(
                f"No Picamera2 frame received for {self.max_stale_seconds:.1f} seconds"
            )
            return None

        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def read(self):
        frame = self.read_frame()
        return frame is not None, frame

    def release(self):
        self.stop_event.set()

        thread = self.capture_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)

        self.capture_thread = None

        with self.lock:
            camera = self.camera
            self.camera = None
            self.latest_frame = None
            self.status = self.FAILED

        self._close_camera_safely(camera)

    def get_status(self):
        with self.lock:
            return {
                "connected": self.connected,
                "backend": self.backend,
                "status": self.status,
                "error": self.last_error or None,
                "width": self.frame_width,
                "height": self.frame_height,
                "fps": round(float(self.camera_fps), 2),
                "last_frame_time": self.last_frame_time,
            }

    def _capture_loop(self):
        while not self.stop_event.is_set():
            request = None
            try:
                camera = self.camera
                if camera is None:
                    self._mark_capture_error("Picamera2 camera closed during capture")
                    return

                request = camera.capture_request()
                frame = request.make_array("main")
                bgr_frame = self._to_bgr(frame)
                self._store_frame(bgr_frame)
            except Exception as exc:
                self._mark_capture_error(self._format_error(exc))
                time.sleep(0.1)
                return
            finally:
                if request is not None:
                    try:
                        request.release()
                    except Exception as exc:
                        self._mark_capture_error(f"Request release failed: {self._format_error(exc)}")

    def _store_frame(self, frame):
        now = time.time()
        with self.lock:
            if self.last_frame_time is not None:
                elapsed = now - self.last_frame_time
                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    if self.camera_fps > 0:
                        self.camera_fps = (self.camera_fps * 0.85) + (instant_fps * 0.15)
                    else:
                        self.camera_fps = instant_fps

            self.latest_frame = frame
            self.last_frame_time = now
            self.status = self.CONNECTED
            self.last_error = ""

    def _mark_capture_error(self, message):
        with self.lock:
            self.last_error = message
            self.status = self.RECONNECTING

    def _is_frame_stale(self):
        with self.lock:
            if self.last_frame_time is not None:
                return time.time() - self.last_frame_time > self.max_stale_seconds

            if self.opened_at is None:
                return False

            return time.time() - self.opened_at > self.max_stale_seconds

    def _maybe_reconnect(self, reason):
        now = time.time()
        if now - self.last_reconnect_attempt < self.reconnect_cooldown_seconds:
            with self.lock:
                if not self.last_error:
                    self.last_error = reason
            return

        self.last_reconnect_attempt = now
        with self.lock:
            self.last_error = reason
            self.status = self.RECONNECTING

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
    def _close_camera_safely(camera):
        if camera is None:
            return

        try:
            camera.stop()
        except Exception:
            pass

        try:
            camera.close()
        except Exception:
            pass

    @staticmethod
    def _format_error(exc):
        return f"{type(exc).__name__}: {exc}"
