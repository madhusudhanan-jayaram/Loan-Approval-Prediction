import pandas as pd
from sklearn.model_selection import train_test_split 
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
from IPython.display import HTML
from IPython.display import display, Markdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, f_classif, SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

def selectkbest(X: pd.DataFrame, y, k: int):
    X = X.copy()
    y = np.asarray(y)

    # --- split columns ---
    cat_cols, num_cols = [], []
    for c in X.columns:
        s = X[c]
        if s.dtype == bool:
            cat_cols.append(c)
        elif np.issubdtype(s.dtype, np.integer):
            if s.nunique(dropna=True) <= 10 or set(s.dropna().unique()).issubset({0,1}):
                cat_cols.append(c)
            else:
                num_cols.append(c)
        elif np.issubdtype(s.dtype, np.floating):
            num_cols.append(c)

    rows = []

    # --- chi2 for categorical (non-negative) ---
    if cat_cols:
        X_cat = X[cat_cols].fillna(0)
        if (X_cat.values < 0).any():
            X_cat[:] = MinMaxScaler().fit_transform(X_cat)
        chi_s, chi_p = chi2(X_cat, y)
        rows += [{"Feature": c, "Method": "chi2", "Score": s, "P": p}
                 for c, s, p in zip(cat_cols, chi_s, chi_p)]

    # --- ANOVA F for numeric ---
    if num_cols:
        X_num = X[num_cols].apply(lambda col: col.fillna(col.median()))
        f_s, f_p = f_classif(X_num, y)
        rows += [{"Feature": c, "Method": "f_classif", "Score": float(s), "P": float(p)}
                 for c, s, p in zip(num_cols, f_s, f_p)]

    if not rows:
        raise ValueError("No features to score.")

    scores = pd.DataFrame(rows)
    scores["Rank"] = -np.log10(np.where(scores["P"] == 0, np.nextafter(0, 1), scores["P"]))
    scores = scores.sort_values(["Rank", "Score"], ascending=False, ignore_index=True)

    selected = scores["Feature"].head(k)
    X_selected = X[selected].to_numpy()

    # quick glance
    print(scores.head(k)[["Feature", "Method", "Score", "P"]])
    return X_selected, selected, scores
