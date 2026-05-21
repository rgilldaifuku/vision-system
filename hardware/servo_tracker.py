"""
servo_tracker.py

Run YOLO detection on a webcam and move a servo at a fixed angular rate
whenever an object is detected.

Default behavior:
- If any detection exists in a frame, increase servo angle at +5 deg/sec.
- If no detection, hold angle.

Hardware output:
- Sends angles over a serial port (e.g. COM3) to an Arduino.
- Message format: "ANGLE:<int_degrees>\\n"

You can run without hardware using --dry-run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    import serial  # type: ignore
except Exception:
    serial = None  # pyserial is optional if using --dry-run


IMG_PREVIEW = True


def _default_model_path() -> str:
    candidates = [
        Path("runs/detect/runs/my_items2/weights/best.pt"),
        Path("runs/detect/runs/my_items2/weights/last.pt"),
        Path("runs/detect/runs/my_items/weights/best.pt"),
        Path("runs/detect/runs/my_items/weights/last.pt"),
        Path("runs/detect/train/weights/best.pt"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "yolov8n.pt"


class ServoSerial:
    def __init__(self, port: str, baud: int, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.port = port
        self.baud = baud
        self._last_sent_angle = None
        self._last_sent_at = 0.0

        if self.dry_run:
            self.ser = None
            return

        if serial is None:
            raise SystemExit("pyserial not installed. Install with: pip install pyserial")

        # Non-blocking-ish; short timeout so app keeps running even if device is slow.
        self.ser = serial.Serial(port=self.port, baudrate=self.baud, timeout=0.1)
        time.sleep(1.5)  # allow Arduino reset on open

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()

    def send_angle(self, angle_deg: int, min_interval_s: float = 0.05) -> None:
        """Send angle if changed and not too frequently."""
        now = time.time()
        if self._last_sent_angle == angle_deg and (now - self._last_sent_at) < 0.5:
            return
        if (now - self._last_sent_at) < min_interval_s:
            return

        msg = f"ANGLE:{int(angle_deg)}\n"
        if self.dry_run:
            print(msg.strip())
        else:
            assert self.ser is not None
            self.ser.write(msg.encode("utf-8"))

        self._last_sent_angle = angle_deg
        self._last_sent_at = now


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=_default_model_path(), help="Path to YOLO .pt weights")
    ap.add_argument("--cam", type=int, default=0, help="Webcam index")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    ap.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Only trigger when this class id is detected (default: any class).",
    )
    ap.add_argument("--rate", type=float, default=5.0, help="Servo speed in degrees/second when detected")
    ap.add_argument("--angle", type=float, default=90.0, help="Starting servo angle (deg)")
    ap.add_argument("--min-angle", type=float, default=0.0, help="Minimum servo angle (deg)")
    ap.add_argument("--max-angle", type=float, default=180.0, help="Maximum servo angle (deg)")
    ap.add_argument("--direction", choices=["+", "-"], default="+", help="Direction to move when detected")
    ap.add_argument("--port", default="COM3", help="Serial port (e.g. COM3)")
    ap.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    ap.add_argument("--dry-run", action="store_true", help="Do not open serial; just print commands")
    ap.add_argument("--no-preview", action="store_true", help="Disable preview window")
    args = ap.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam index {args.cam}. Try --cam 1.")

    servo = ServoSerial(port=args.port, baud=args.baud, dry_run=bool(args.dry_run))
    angle = float(args.angle)
    sign = 1.0 if args.direction == "+" else -1.0

    last_t = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t = time.perf_counter()
            dt = max(0.0, t - last_t)
            last_t = t

            results = model.predict(frame, conf=float(args.conf), verbose=False)
            r0 = results[0]

            detected = False
            if r0.boxes is not None and len(r0.boxes) > 0:
                if args.class_id is None:
                    detected = True
                else:
                    # r0.boxes.cls is a tensor of class ids
                    detected = bool((r0.boxes.cls.int() == int(args.class_id)).any().item())

            if detected:
                angle += sign * float(args.rate) * dt

            angle = max(float(args.min_angle), min(float(args.max_angle), angle))

            servo.send_angle(int(round(angle)))

            if not args.no_preview:
                annotated = r0.plot()
                status = "DETECTED" if detected else "no"
                cv2.putText(
                    annotated,
                    f"{status} | angle={angle:.1f} | dt={dt*1000:.0f}ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if detected else (0, 0, 255),
                    2,
                )
                cv2.imshow("Servo Tracker (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        servo.close()
        if not args.no_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

