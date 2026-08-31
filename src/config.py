from pathlib import Path

RNG = 42
N_JOBS = -1
DATA_TAG = "medium"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / f"fma_{DATA_TAG}"
METADATA_DIR = DATA_DIR / "fma_metadata"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"

X_PATH = OUTPUTS_DIR / f"X_{DATA_TAG}.npy"
IDS_PATH = OUTPUTS_DIR / f"track_ids_{DATA_TAG}.npy"
BAD_FILES_CSV = OUTPUTS_DIR / f"bad_files_{DATA_TAG}.csv"
DATASET_CSV = OUTPUTS_DIR / f"features_with_genre_{DATA_TAG}.csv"
CV_SUMMARY_CSV = OUTPUTS_DIR / "cv_summary.csv"
TEST_SUMMARY_CSV = OUTPUTS_DIR / "test_summary.csv"
SELECTED_FEATURES_CSV = OUTPUTS_DIR / "selected_features_top10.csv"

# Audio / feature extraction
SR = 22050
DURATION = 30.0
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 20

# Modeling
KNN_K = 5
KBEST_K = 10
CV_FOLDS = 5

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
