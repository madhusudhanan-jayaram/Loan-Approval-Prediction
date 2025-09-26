import pandas as pd

def split_columns(df: pd.DataFrame):
    quan = [col for col in df.columns if df[col].dtype != 'O']
    qual = [col for col in df.columns if df[col].dtype == 'O']
    return quan, qual

def missing_values(df: pd.DataFrame):
    return (df.isna().mean() * 100).to_frame("Missing_%")

def numeric_summary(df: pd.DataFrame, quan):
    return df[quan].describe().T  # count, mean, std, min, quartiles, max

def freq_table(df: pd.DataFrame, col: str):
    return df[col].value_counts(normalize=True).to_frame("Relative_Freq")

def target_distribution(df: pd.DataFrame, target_col="Loan_Status"):
    if target_col in df.columns:
        return df[target_col].value_counts(normalize=True).to_frame("Target_Dist")
    return None
