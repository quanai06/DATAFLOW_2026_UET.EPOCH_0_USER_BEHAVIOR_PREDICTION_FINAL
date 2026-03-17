from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "model" / "combine"
LEGACY_MODEL_DIR = PROJECT_ROOT / "models" / "ensemble"
SAVED_ARTIFACTS_DIR = PROJECT_ROOT / "saved_artifacts"
DATA_DIR = PROJECT_ROOT / "data" / "data_raw"
X_TEST_PATH = DATA_DIR / "X_test.csv"
PRECOMPUTED_PARQUET_PATH = PROJECT_ROOT / "data" / "precomputed_orders.parquet"
PRECOMPUTED_CSV_PATH = PROJECT_ROOT / "data" / "precomputed_orders.csv"
PRECOMPUTED_SUMMARY_PATH = PROJECT_ROOT / "data" / "precomputed_dataset_summary.json"

CHECK_MODEL_PATH = MODEL_DIR / "check_model_lstm.keras"
ENSEMBLE_GLOB = "model_*.keras"
PREFERRED_SINGLE_MODEL = "model_0_lstm.keras"
FULL_SCALER_PATH = MODEL_DIR / "scaler_full.pkl"
SCALER_PHASE1_PATH = MODEL_DIR / "scaler_phase1.joblib"
SCALER_PHASE2_PATH = MODEL_DIR / "scaler_phase2.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_SOURCE_MODE = "precomputed"

DEFAULT_MAX_SEQUENCE_LENGTH = 66
DEFAULT_TOP_HUBS = (102, 105, 103, 606, 760, 8615, 603, 709, 685, 621)
DEFAULT_COMBINE_TOP_ACTIONS = (
    102,
    105,
    103,
    606,
    760,
    8615,
    603,
    709,
    685,
    621,
    21040,
    658,
    697,
    975,
    867,
)
DEFAULT_FEATURE_PROFILE = "combine_25"
DEFAULT_TARGETS = ("attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6")
PLANNING_TODAY_MONTH = 1
PLANNING_TODAY_DAY = 1
TARGET_MAX_VALUES = {
    "attr_1": 11,
    "attr_2": 30,
    "attr_3": 99,
    "attr_4": 11,
    "attr_5": 30,
    "attr_6": 99,
}
BUSINESS_DENOMINATORS = {
    "attr_3": 99.0,
    "attr_4": 12.0,
    "attr_5": 31.0,
    "attr_6": 99.0,
}
