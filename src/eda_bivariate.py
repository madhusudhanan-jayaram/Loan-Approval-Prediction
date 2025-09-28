import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def prepare_from_df(df, target="Loan_Status_Y"):
    # Target to numeric
    y, _ = pd.factorize(df[target])

    # Features are everything else (already encoded & imputed)
    X = df.drop(columns=[target])

    # Scale to [0,1] for chi2
    X_scaled = MinMaxScaler().fit_transform(X)

    return X, X_scaled, y
