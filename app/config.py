from pathlib import Path 

# Project root = detection folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT /"data"
LOGS_DIR = DATA_DIR / "logs"
REVIEW_IMAGES_DIR = DATA_DIR /"review_images"

# Model
ACTIVE_MODEL_PROFILE = "mouse"
ACTIVE_MODEL_DIR = MODELS_DIR / ACTIVE_MODEL_PROFILE
DEFAULT_MODEL_PATH = ACTIVE_MODEL_DIR / "best.pt"
MODEL_CONFIG_PATH = ACTIVE_MODEL_DIR / "config.json"
CLASSES_PATH = ACTIVE_MODEL_DIR / "classes.txt"
# DEFAULT_MODEL_PATH = MODELS_DIR / ACTIVE_MODEL_PROFILE / "best.pt"

# Detection settings
DEFAULT_CONFIDENCE = 0.65
CAMERA_INDEX = 0
ROI_ENABLED = False
ROI_X1 = 0.0
ROI_Y1 = 0.0
ROI_X2 = 1.0
ROI_Y2 = 1.0

# Runtime Settings
SAVE_REVIEW_IMAGES = True
SAVE_LOW_CONFIDENCE_IMAGES = True
LOW_CONFIDENCE_THRESHOLD = 0.65

# UI

WINDOW_TITLE = "Production Image Detection System"

TARGET_CLASSES = ["mouse"]

def get_available_model_profiles():
    return [
        folder.name
        for folder in MODELS_DIR.iterdir()
        if folder.is_dir() and (folder / "best.pt").exists()
    ]
