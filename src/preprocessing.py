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

def split_scalar(indep_X, dep_Y):
    X_train, X_test, y_train, y_test = train_test_split(
        indep_X, dep_Y, test_size=0.25, random_state=0, stratify=dep_Y
    )
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test  = sc.transform(X_test)
    return X_train, X_test, y_train, y_test, sc

def clean_data(df: pd.DataFrame):
    # 1. Drop Loan_ID
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    # 2. Missing values
    df_clean = handle_missing(df)   # keep clean version

    # --- Extract target BEFORE encoding ---
    if "Loan_Status" not in df_clean.columns:
        raise KeyError("Loan_Status column is missing in input data.")
    y = df_clean["Loan_Status"]
    if y.dtype == "object":
        y = y.astype(str).str.strip().map({"Y": 1, "N": 0})

    # 3. Encode only features
    X = df_clean.drop(columns=["Loan_Status"])
    X = pd.get_dummies(X, drop_first=True)
    df_enc = X.copy()
    df_enc["Loan_Status"] = y   # keep target in encoded DF for inspection

    # 4. Split + scale
    X_train, X_test, y_train, y_test, sc = split_scalar(X, y)

    return df_clean, df_enc, X_train, X_test, y_train, y_test, sc
