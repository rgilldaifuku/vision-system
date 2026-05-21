from pathlib import Path
from datetime import datetime
import csv 

from app.config import LOGS_DIR

LOG_FILE = LOGS_DIR /"detections.csv"

#ensure folder exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

#crete csv header if file doesnt exist
if not LOG_FILE.exists():
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "detected",
            "class_name",
            "confidence"
        ])

def log_detection(detected, class_name="", confidence=0.0):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            detected, 
            class_name,
            round(confidence, 4)
        ])