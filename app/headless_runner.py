"""
Headless inference runner for Vision System on Raspberry Pi.
Reads frames from camera, runs YOLO inference, logs detections, saves review images.
No GUI - designed for Docker container deployment
"""

import argparse
import time
import cv2
import logging
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

from app.config import REVIEW_IMAGES_DIR, LOW_CONFIDENCE_THRESHOLD, TARGET_CLASSES, LOGS_DIR
from app.inference import draw_detections
from app.logging import log_detection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(), # stdout to Docker logs
        logging.FileHandler(LOGS_DIR / "inference.log")
    ]
)
logger = logging.getLogger(__name__)

def run_inference(model_path, camera_index=0, confidence=0.35, headless=True):
    """
    Run continuous inference from camera

    Args:
        model_path (str): Path to YOLO model (.pt file)
        camera_index (int): Camera device index (0 for /dev/video0)
        confidence (float): Detection confidence threshold
        headless (bool): If True, no GUI; save review images instead
    """

    # Load model
    logger.info(f"Loading model: {model_path}")
    try:
        model = YOLO(model_path)    
        logger.info(f"Model loaded. Classes: {list(model.names.values())}")
    except:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Open Camera
    logger.info(f"OPening camera device {camera_index}")
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        logger.error(f"Failed to open camera device {camera_index}")
        raise RuntimeError("Camera initialization failed")

    # Set camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    logger.info("Camera initialized: 1280x720")

    # Ensure output directories exist
    REVIEW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    detection_count = 0
    last_log_time = time.time()
    log_cooldown = 2.0 # log every N seconds

    logger.info("Starting inference loop...")

    try:
        while True:
            ret, frame = camera.read()

            if not ret or frame is None:
                logger.warning("Failed to read camera frame")
                time.sleep(0.1)
                continue

            frame_count += 1

            # Run YOLO inference
            try:
                results = model.predict(frame, conf=confidence, verbose=False)
            except:
                logger.error(f"Inference error: {e}")
                continue

            # Draw detections
            annotated_frame, target_found = draw_detections(frame, results)

            # Check for target class detections
            detections_found = False
            detected_classes = []

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = result.names.get(cls_id, "UNKNOWN")
                    confidence_score = float(box.conf[0])

                    if class_name in TARGET_CLASSES:
                        detection_found = True
                        detected_classes.append(class_name)

                        # Save low-confidence detections for review
                        if confidence_score < LOW_CONFIDENCE_THRESHOLD:
                            save_review_image(frame, f"low_confidence_{class_name}")
                            logger.info(f"Low confidence detection saved: {class_name} ({confidence_score:.2f})")

            # Save frames where target wasn't detected
            if not detections_found:
                save_review_image(frame, "no_detection")

            # Log detections periodically
            current_time = time.time()
            if current_time - last_log_time >= log_cooldown:
                if detections_found:
                    logger.info(f"Detection: {', '.join(detected_classes)}")
                    detection_cound += 1
                last_log_time = current_time
            
            # Print stats every 100 frames
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames, {detection_count} detections")

    except KeyboardInterrupt:
        logger.info("Interrupted by User")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        camera.release()
        logger.info(f"Camera released. Total: {frame_count} frames, {detection_cound} detections")

def save_review_image(frame, reason):
        """Save a frame for manual review."""
        try:
            REVIEW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = REVIEW_IMAGES_DIR / f"{timestamp}_{reason}.jpg"
            cv2.imwrite(str(filename), frame)
        except Exception as e:
            logger.error(f"Failed to save review image: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Headless YOLO inference for Raspberry Pi"
    )
    parser.add_argument(
        "--model",
        default="models/mouse/best.pt",
        help="Path to YOLO model (default: models/mouse/best.pt)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0 = dev/video0)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Detection confidence threshold (default: 0.35)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)"
    )

    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level))

    logger.info(f"Vision System Headless Inference")
    logger.info(f"Model: {args.model}")
    logger.info(f"Camera: /dev/video{args.camera}")
    logger.info(f"Confidence: {args.confidence}")

    run_inference(
        model_path=args.model,
        camera_index=args.camera,
        confidence=args.confidence,
        headless=True
    )