from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TRAIN_DATA_DIR = DATA_DIR / "train"
VAL_DATA_DIR = DATA_DIR / "val"
TEST_DATA_DIR = DATA_DIR / "test"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
for p in [DATA_DIR, RAW_DATA_DIR, TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR, SAMPLE_IMAGES_DIR, MODELS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Dataset Configuration (Intel Natural Scene / Location Classification)
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}

# Image & Model Hyperparameters
IMG_HEIGHT = 150
IMG_WIDTH = 150
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 0.001

# Model artifact paths
CNN_MODEL_PATH = MODELS_DIR / "cnn_baseline.keras"
MOBILENET_MODEL_PATH = MODELS_DIR / "mobilenetv2_location.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
