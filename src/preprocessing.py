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

def clean_data(df: pd.DataFrame):

    # 1. Drop Loan_ID
    if "Loan_ID" in df.columns:
        df = df.drop("Loan_ID", axis=1)

    # 2. Missing values
    df = handle_missing(df)


    # 3. One-hot encode categorical
    df = pd.get_dummies(df, drop_first=True)

    # 4. Train-test split
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = split_scalar(X, y)


    return X_train, X_test, y_train, y_test, df

def split_scalar(indep_X, dep_Y):
    # 1. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        indep_X, dep_Y, test_size=0.25, random_state=0
    )

    # 2. Initialize scaler
    sc = StandardScaler()

    # 3. Fit on training, transform train
    X_train = sc.fit_transform(X_train)

    # 4. Transform test with same scaler
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test, sc