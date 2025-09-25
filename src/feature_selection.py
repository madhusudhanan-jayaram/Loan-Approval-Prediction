from sklearn.feature_selection import (
    SelectKBest, chi2, mutual_info_classif, RFE, SelectFromModel
)

from sklearn.linear_model import LogisticRegression
def select_features_all(X, y, k=10):
    results = {}

    # Filter methods
    kbest_chi2 = SelectKBest(score_func=chi2, k=min(k, X.shape[1]))
    kbest_chi2.fit(X.abs() + 1e-9, y)  # chi2 needs non-negative
    results["select_kbest_chi2"] = X.columns[kbest_chi2.get_support()].tolist()

    kbest_mi = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
    kbest_mi.fit(X, y)
    results["select_kbest_mutual_info"] = X.columns[kbest_mi.get_support()].tolist()

    # Wrapper method
    rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=min(k, X.shape[1]))
    rfe.fit(X, y)
    results["rfe_logistic"] = X.columns[rfe.support_].tolist()

    # Embedded methods
    lasso = SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000))
    lasso.fit(X, y)
    results["select_from_model_l1"] = X.columns[lasso.get_support()].tolist()

    forest = SelectFromModel(RandomForestClassifier(n_estimators=200, random_state=42))
    forest.fit(X, y)
    results["select_from_model_random_forest"] = X.columns[forest.get_support()].tolist()

    return results