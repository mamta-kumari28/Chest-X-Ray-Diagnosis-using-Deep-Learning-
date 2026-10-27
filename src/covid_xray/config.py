from pathlib import Path

# Is file se project ke main folder ka path nikalta hai.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset locations
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "covidx_cxr3a_v7"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model aur evaluation output locations
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
RESULTS_DIR = ARTIFACTS_DIR / "results"

# Dataset details
DATASET_NAME = "COVIDx CXR-3A"
DATASET_VERSION = "Kaggle Version 7"

# Provided counts; downloaded label files se verify karenge.
EXPECTED_TRAIN_IMAGES = 29_986
EXPECTED_TEST_IMAGES = 400

# Initial training settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.20
MAX_EPOCHS = 30
LEARNING_RATE = 0.001


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_NAME} — {DATASET_VERSION}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Maximum epochs: {MAX_EPOCHS}")
    
# SHA-256 of the evaluated best checkpoint.
MODEL_SHA256 = (
    "806d01759ea3f389d1b052b5f2a5ae5bb6e9bd0d3a4931cb328726c0227530c9"
)    