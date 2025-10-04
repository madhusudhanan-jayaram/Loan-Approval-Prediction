import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


def run_tuning_and_evaluation(df: pd.DataFrame, target: str):

    # --- Prepare data ---
    X = df.drop(columns=[target])
    y = df[target].copy()
    expected_cols = X.columns.tolist()

    # Handle Y/N to 1/0 automatically if needed
    if y.dtype == object:
        y = y.map({"Y": 1, "N": 0}).fillna(y).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- Define models and grids ---
    models_and_grids = {
        "Logistic Regression": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, solver="liblinear")),
            {
                "logisticregression__C": [0.01, 0.1, 1, 3, 10],
                "logisticregression__class_weight": [None, "balanced"],
            },
        ),
        "SVM (Linear)": (
            make_pipeline(StandardScaler(), SVC(kernel="linear")),
            {
                "svc__C": [0.01, 0.1, 1, 3, 10],
                "svc__class_weight": [None, "balanced"],
            },
        ),
        "Naive Bayes": (
            GaussianNB(),
            {
                "var_smoothing": [1e-9, 1e-8, 1e-7],
            },
        ),
    }

    results = []
    tuned_models = {}

    # --- Loop through each model ---
    for name, (estimator, grid) in models_and_grids.items():
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=grid,
            cv=3,
            scoring="f1",
            refit=True,
            n_jobs=-1,
            verbose=0,
        )

        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\n=== {name} (tuned) ===")
        print("Best Params:", gs.best_params_)
        print("Accuracy:", acc)
        print("F1 Score:", f1)
        print(classification_report(y_test, y_pred))

        ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
        plt.title(f"{name} (tuned)")
        plt.show()

        results.append([name, acc, f1, gs.best_params_])
        tuned_models[name] = best_model

    # --- Summary ---
    res_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1", "Best Params"])
    res_df = res_df.sort_values("F1", ascending=False).reset_index(drop=True)
    print("\n=== Tuned Model Comparison ===")
    print(res_df)

    # --- Identify and Save Best Model ---
    best_model_name = res_df.iloc[0]["Model"]
    best_model = tuned_models[best_model_name]

    # ✅ Go up one level from src to project root → artifacts
    save_dir = Path(__file__).resolve().parent.parent / "artifacts"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save expected columns
    expected_cols_path = save_dir / "expected_cols.pkl"
    joblib.dump(expected_cols, expected_cols_path)
    print(f"✅ Expected columns saved to: {expected_cols_path}")

    model_path = save_dir / f"{best_model_name.replace(' ', '_').lower()}_best.pkl"
    joblib.dump(best_model, model_path)
    print(f"\n✅ Best model '{best_model_name}' saved to: {model_path}")

    # --- Save summary CSV ---
    summary_path = save_dir / "tuned_model_summary.csv"
    res_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    return res_df, best_model, model_path
