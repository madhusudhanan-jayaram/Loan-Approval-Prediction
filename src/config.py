# src/config.py
from pathlib import Path

# Always points to your repo root, no matter where code runs from
ROOT = Path(__file__).resolve().parents[1]

# Common dirs
DATA_DIR = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "artifacts"

# Files
CSV_PATH = DATA_DIR / "loan_data_set.csv"   # adjust if needed
RAW_PARQUET = ARTIFACTS_DIR / "01_raw.parquet"
CLEAN_PARQUET = ARTIFACTS_DIR / "02_clean.parquet"
SCALAR_PARQUET = ARTIFACTS_DIR / "02_scalar.parquet"
X_TRAIN_PARQUET = ARTIFACTS_DIR / "02_X_train.parquet"
X_TEST_PARQUET = ARTIFACTS_DIR / "02_X_test.parquet"
Y_TRAIN_PARQUET = ARTIFACTS_DIR / "02_y_train.parquet"
Y_TEST_PARQUET = ARTIFACTS_DIR / "02_y_test.parquet"

