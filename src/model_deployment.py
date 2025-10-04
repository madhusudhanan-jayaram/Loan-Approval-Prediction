# model_deployment.py (simple + robust)
import pandas as pd
import joblib
from pathlib import Path

def _artifacts_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "artifacts"

def load_model_and_columns():
    art = _artifacts_dir()
    # load best model
    model_path = next(art.glob("*_best.pkl"))
    model = joblib.load(model_path)
    # load expected columns
    expected_cols = joblib.load(art / "expected_cols.pkl")
    print(f"✅ Loaded model: {model_path.name} | columns: {len(expected_cols)}")
    return model, expected_cols

def prepare_input_like_training(data, expected_cols):
    """Make incoming data look like the training matrix."""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # one-hot encode like training did
    X = pd.get_dummies(data, drop_first=False)

    # align to expected columns (add missing as 0, drop extras)
    X = X.reindex(columns=expected_cols, fill_value=0)
    return X

def predict(input_data):
    model, expected_cols = load_model_and_columns()
    X = prepare_input_like_training(input_data, expected_cols)
    preds = model.predict(X)

    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]

    return preds, probs
