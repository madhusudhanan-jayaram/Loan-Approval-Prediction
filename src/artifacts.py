from pathlib import Path
import numpy as np
import os, json, pandas as pd, joblib

def _ensure_dir(path: Path):
    os.makedirs(path.parent, exist_ok=True)

def save_df(df: pd.DataFrame, path: str | Path):
    p = Path(path); _ensure_dir(p)
    df.to_parquet(p)

def load_df(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def save_json(obj, path: str | Path):
    p = Path(path); _ensure_dir(p)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_obj(obj, path: str | Path):
    p = Path(path); _ensure_dir(p)
    joblib.dump(obj, p)

def load_obj(path: str | Path):
    p = Path(path)
    return joblib.load(p)