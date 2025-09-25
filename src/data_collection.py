import os
import pandas as pd

def load_data(path):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)
        print(f"Loaded data: {df.shape}")
        return df
    except FileNotFoundError as exc:
        print(exc)
        return None
    except pd.errors.EmptyDataError:
        print("The CSV file is empty.")
        return None
    except Exception as exc:
        print(f"Unexpected error while loading data: {exc}")
        return None

def preview_data(df, n=5):
    try:
        if df is None or df.empty:
            print("No data to preview.")
            return
        print(df.head(n))
    except Exception as exc:
        print(f"Error while previewing data: {exc}")
