import time
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4"}


class SimulatedCameraSource:
    """Camera-like source backed by an image, image folder, or video file."""

    CONNECTED = "Connected"
    RECONNECTING = "Reconnecting"
    FAILED = "Failed"

    def __init__(self, source_path, frame_interval_seconds=0.1):
        self.source_path = Path(source_path)
        self.frame_interval_seconds = max(0.0, float(frame_interval_seconds))
        self.status = self.FAILED
        self.last_error = ""

        self.mode = None
        self.image_paths = []
        self.image_index = 0
        self.image_frame = None
        self.capture = None
        self.last_frame_time = 0.0

    def open(self):
        self.release()

        if not self.source_path.exists():
            self.last_error = f"Simulated camera source not found: {self.source_path}"
            self.status = self.FAILED
            return False

        if self.source_path.is_dir():
            self.image_paths = self._collect_images(self.source_path)
            if not self.image_paths:
                self.last_error = f"No supported images found in: {self.source_path}"
                self.status = self.FAILED
                return False

            self.mode = "folder"
            self.status = self.CONNECTED
            self.last_error = ""
            return True

        suffix = self.source_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            frame = cv2.imread(str(self.source_path))
            if frame is None:
                self.last_error = f"Unable to read simulated image: {self.source_path}"
                self.status = self.FAILED
                return False

            self.mode = "image"
            self.image_frame = frame
            self.status = self.CONNECTED
            self.last_error = ""
            return True

        if suffix in VIDEO_EXTENSIONS:
            self.capture = cv2.VideoCapture(str(self.source_path))
            if self.capture.isOpened():
                self.mode = "video"
                self.status = self.CONNECTED
                self.last_error = ""
                return True

            self.last_error = f"Unable to open simulated video: {self.source_path}"
            self.status = self.FAILED
            self.release()
            return False

        self.last_error = f"Unsupported simulated camera source: {self.source_path}"
        self.status = self.FAILED
        return False

    def read_frame(self):
        if self.status != self.CONNECTED:
            self.open()
            if self.status != self.CONNECTED:
                time.sleep(self.frame_interval_seconds)
                return None

        self._pace_frames()

        if self.mode == "image":
            return self.image_frame.copy()

        if self.mode == "folder":
            return self._read_next_folder_image()

        if self.mode == "video":
            return self._read_next_video_frame()

        self.last_error = "Simulated camera source is not open"
        self.status = self.FAILED
        return None

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _read_next_folder_image(self):
        for _ in range(len(self.image_paths)):
            image_path = self.image_paths[self.image_index]
            self.image_index = (self.image_index + 1) % len(self.image_paths)

            frame = cv2.imread(str(image_path))
            if frame is not None:
                self.last_error = ""
                return frame

            self.last_error = f"Unable to read simulated image: {image_path}"

        self.status = self.FAILED
        return None

    def _read_next_video_frame(self):
        ok, frame = self.capture.read()
        if ok and frame is not None:
            self.last_error = ""
            return frame

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.capture.read()
        if ok and frame is not None:
            self.last_error = ""
            return frame

        self.last_error = f"Unable to read simulated video frame: {self.source_path}"
        self.status = self.FAILED
        return None

    def _pace_frames(self):
        if self.frame_interval_seconds <= 0:
            return

        now = time.monotonic()
        elapsed = now - self.last_frame_time
        if self.last_frame_time and elapsed < self.frame_interval_seconds:
            time.sleep(self.frame_interval_seconds - elapsed)

        self.last_frame_time = time.monotonic()

    @staticmethod
    def _collect_images(folder):
        return sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
