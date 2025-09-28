import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, f_classif, SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

def prepare_from_df(df, target="Loan_Status_Y"):
    # Target to numeric
    y, _ = pd.factorize(df[target])

    # Features are everything else (already encoded & imputed)
    X = df.drop(columns=[target])

    # Scale to [0,1] for chi2
    X_scaled = MinMaxScaler().fit_transform(X)

    return X, X_scaled, y

def chi2_table(X_scaled, y, feature_names):
    sel = SelectKBest(score_func=chi2, k="all").fit(X_scaled, y)
    return pd.DataFrame({
        "Feature": feature_names,
        "Chi2": sel.scores_,
        "P-Value": sel.pvalues_
    }).sort_values("Chi2", ascending=False)

def anova_table(X, y, feature_names):
    sel = SelectKBest(score_func=f_classif, k="all").fit(X, y)
    return pd.DataFrame({
        "Feature": feature_names,
        "F-Score": sel.scores_,
        "P-Value": sel.pvalues_
    }).sort_values("F-Score", ascending=False)

def print_scorecard(df_scores, top_n=5, method="ANOVA"):
    print(f"\n🔹 {method} Feature Scorecard (Top {top_n}) 🔹\n")
    for _, row in df_scores.head(top_n).iterrows():
        feat = row["Feature"]
        score = row.get("F-Score", row.get("Chi2"))
        pval = row["P-Value"]
        strength = "✔️ Strong" if pval < 0.05 else "⚠️ Weak"
        print(f"{feat:25s} | Score={score:10.3f} | P={pval:.1e} | {strength}")