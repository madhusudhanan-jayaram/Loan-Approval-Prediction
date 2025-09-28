import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def target_distribution(df, target="Loan_Status_Y"):
    counts = df[target].value_counts(dropna=False)
    pct = (counts / counts.sum() * 100).round(2)

    print(f"\n===== Target Variable: {target} =====")
    print(pd.DataFrame({"Count": counts, "Percent": pct}).to_string())
    print(f"👉 Approval rate: {pct.get('Y','NA')}% | Rejection rate: {pct.get('N','NA')}%")
    if max(pct.fillna(0)) >= 70:
        print("👉 Business Insight: Dataset imbalance risk (one class dominates).")

    # simple bar chart
    plt.bar(pct.index.astype(str), pct, color=["green","red"])
    plt.title("Loan Status Distribution")
    plt.ylabel("Percentage")
    for i, val in enumerate(pct):
        plt.text(i, val+1, f"{val}%", ha="center")
    plt.show()

def data_quality_report(df):
    print("\n===== DATA QUALITY REPORT =====")
    report = pd.DataFrame({
        "Missing Values": df.isnull().sum(),
        "Missing %": (df.isnull().sum() / len(df) * 100).round(2),
        "Unique Values": df.nunique()
    })

    print(report.to_string())

    # Business insights
    print("\n👉 Business Insights:")
    # Missing values
    high_missing = report[report["Missing %"] > 30]
    if not high_missing.empty:
        print(f"   - Columns with very high missing values: {list(high_missing.index)} → may need imputation or removal.")
    else:
        print("   - No column has extremely high missing values.")

    # Unique values
    for col in df.columns:
        unique_count = report.loc[col, "Unique Values"]
        if unique_count == 1:
            print(f"   - '{col}' has only 1 unique value → not useful for prediction.")
        elif unique_count == len(df):
            print(f"   - '{col}' is almost unique per row → might be an ID, not predictive.")
        elif unique_count < 5:
            print(f"   - '{col}' has very few categories → treat as categorical feature.")
