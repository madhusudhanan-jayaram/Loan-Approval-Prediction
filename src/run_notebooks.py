import os
import papermill as pm

os.makedirs("runs", exist_ok=True)

NOTEBOOKS = [
    ("notebooks/01_data_collection.ipynb", "runs/01_data_collection_out.ipynb"),
    ("notebooks/02_preprocessing.ipynb",   "runs/02_preprocessing_out.ipynb"),
    ("notebooks/03_eda_univariate_bivariate.ipynb",  "runs/03_eda_univariate_bivariate.ipynb"),
]

for src, dst in NOTEBOOKS:
    print(f"▶️ Running {src} ...")
    pm.execute_notebook(src, dst, kernel_name="aiml")
    print(f"✅ Saved: {dst}")