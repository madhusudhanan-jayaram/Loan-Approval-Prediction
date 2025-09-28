import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    return df


def convert_dependents(df: pd.DataFrame, col: str = "Dependents") -> pd.DataFrame:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("+", "", regex=False)
            .astype(float)
        )
    return df

def clean_data(df: pd.DataFrame):
    # 1) Drop id if present
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    # 2) Clean + normalize Dependents
    df_clean = handle_missing(df)
    df_clean = convert_dependents(df_clean)

    df_clean = pd.get_dummies(df, drop_first=True)

    print("After dummies:", df_clean.shape)
    print("New columns:", df_clean.columns.tolist())  # show last 10 columns
    indep_X = df_clean.drop('Loan_Status_Y', axis=1)
    dep_Y = df_clean['Loan_Status_Y']
    # Print shapes
    print("Shape of full dataset:", df_clean.shape)
    print("Shape of Features (indep_X):", indep_X.shape)
    print("Shape of Target (dep_Y):", dep_Y.shape)

    # Print first few rows of features
    print("\nIndependent Variables (X):")
    print(indep_X.head())

    # Print first few rows of target
    print("\nDependent Variable (y):")
    print(dep_Y.head())
    return df_clean
