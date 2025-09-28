import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import chi2_contingency
import seaborn as sns


def prepare_from_df(df, target="Loan_Status_Y"):
    # Target to numeric
    y, _ = pd.factorize(df[target])

    # Features are everything else (already encoded & imputed)
    X = df.drop(columns=[target])

    # Scale to [0,1] for chi2
    X_scaled = MinMaxScaler().fit_transform(X)

    return X, X_scaled, y
def pointbiserial_analysis(df, target="Loan_Status", num_features=None):
    
    # Convert target to binary (Y=1, N=0)
    y = df[target].map({"Y":1,"N":0}) if df[target].dtype == object else df[target]
    
    # Auto-detect numeric features
    if num_features is None:
        num_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in num_features:
            num_features.remove(target)

    results = []
    for feat in num_features:
        x = df[feat].fillna(df[feat].median())
        corr, pval = pointbiserialr(y, x)
        
        # Interpretation
        if pval < 0.05:
            if abs(corr) > 0.5:
                interp = "✔️ Strong relationship"
            elif abs(corr) > 0.2:
                interp = "⚠️ Moderate relationship"
            else:
                interp = "ℹ️ Weak but statistically significant"
        else:
            interp = "❌ No significant relationship"
        
        results.append({
            "Feature": feat,
            "Corr": round(corr, 3),
            "P-Value": f"{pval:.3e}",
            "Interpretation": interp
        })

    return pd.DataFrame(results).sort_values(by="Corr", ascending=False).reset_index(drop=True)


def categorical_analysis(df, target="Loan_Status_Y", cat_features=None):
    """
    Analyze categorical features vs binary target using Chi² test and Cramer's V.
    Displays results in a single row per feature.
    """
    if cat_features is None:
        cat_features = df.select_dtypes(include=["object"]).columns.tolist()
        if target in cat_features:
            cat_features.remove(target)

    y = df[target]
    results = []

    for feat in cat_features:
        # Contingency table
        table = pd.crosstab(df[feat], y)
        chi2, p, dof, exp = chi2_contingency(table)

        # Cramer's V
        n = table.sum().sum()
        phi2 = chi2 / n
        r, k = table.shape
        cramers_v = np.sqrt(phi2 / (min(r-1, k-1)))

        # Interpretation
        if p < 0.05:
            if cramers_v > 0.3:
                interp = "✔️ Strong"
            elif cramers_v > 0.1:
                interp = "⚠️ Moderate"
            else:
                interp = "ℹ️ Weak but significant"
        else:
            interp = "❌ None"

        # Append as a single-row string
        results.append(
            f"{feat:25s} | Chi²={chi2:8.3f} | P={p:.3e} | V={cramers_v:.3f} | {interp}"
        )

    return results

def approval_rate_by_feature(df, feature, target="Loan_Status_Y"):
    tbl = df.groupby(feature)[target].mean() * 100
    result = tbl.reset_index().rename(columns={target: "Approval %"})
    return result

def approval_summary(df, feature, target="Loan_Status_Y"):
    """
    Show approval counts and percentages by category for a given feature.
    Works for categorical variables (like Gender, Property_Area).
    """
    tbl = pd.crosstab(df[feature], df[target], margins=True)
    tbl["Approval %"] = (tbl[1] / (tbl[0] + tbl[1]) * 100).round(1)  # target=1 means approved
    return tbl

def draw_business_insights(df, target="Loan_Status_Y"):
    """
    Draws key business insight charts for loan prediction dataset.
    """

    # 1. Loan Approval by Credit History
    plt.figure(figsize=(6,4))
    sns.barplot(x="Credit_History", y=target, data=df,
                estimator=lambda x: 100*sum(x)/len(x))
    plt.ylabel("Approval %")
    plt.title("Loan Approval by Credit History")
    plt.show()

    # 2. Loan Approval by Property Area
    plt.figure(figsize=(6,4))
    sns.barplot(x="Property_Area", y=target, data=df,
                estimator=lambda x: 100*sum(x)/len(x))
    plt.ylabel("Approval %")
    plt.title("Loan Approval by Property Area")
    plt.show()

    # 3. Applicant Income Distribution by Loan Status
    plt.figure(figsize=(7,4))
    sns.violinplot(x=target, y="ApplicantIncome", data=df)
    plt.title("Applicant Income vs Loan Status")
    plt.ylabel("Applicant Income")
    plt.xlabel("Loan Approved (1=Yes, 0=No)")
    plt.show()

    # 4. Loan Approval by Gender
    plt.figure(figsize=(5,4))
    sns.barplot(x="Gender_Male", y=target, data=df,
                estimator=lambda x: 100*sum(x)/len(x))
    plt.xticks([0,1], ["Female","Male"])
    plt.ylabel("Approval %")
    plt.title("Loan Approval by Gender")
    plt.show()

    # 5. Loan Amount Distribution by Loan Status
    plt.figure(figsize=(7,4))
    sns.histplot(data=df, x="LoanAmount", hue=target, bins=30,
                 kde=True, multiple="stack")
    plt.title("Loan Amount Distribution: Approved vs Rejected")
    plt.xlabel("Loan Amount")
    plt.ylabel("Count")
    plt.show()
