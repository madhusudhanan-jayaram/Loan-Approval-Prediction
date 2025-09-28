import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _normalize_target(df: pd.DataFrame, target_hint: str | None = None) -> pd.Series:
    """
    Return a Series with values 'Approved'/'Rejected', detecting either:
      - 'Loan_Status' as 'Y'/'N', or
      - 'Loan_Status_Y' as 1/0 (one-hot)
    If target_hint is provided, it is used first.
    """
    tgt = None

    # 1) Use the hint if valid
    if target_hint and target_hint in df.columns:
        col = df[target_hint]
        if col.dropna().isin(["Y", "N"]).any():
            tgt = col.map({"Y": "Approved", "N": "Rejected"})
        elif set(pd.unique(col.dropna())) <= {0, 1}:
            tgt = col.map({1: "Approved", 0: "Rejected"})
        else:
            # fall through to auto-detect below
            pass

    # 2) Auto-detect
    if tgt is None:
        if "Loan_Status" in df.columns:
            col = df["Loan_Status"]
            if col.dropna().isin(["Y", "N"]).any():
                tgt = col.map({"Y": "Approved", "N": "Rejected"})
        if tgt is None and "Loan_Status_Y" in df.columns:
            col = df["Loan_Status_Y"]
            if set(pd.unique(col.dropna())) <= {0, 1}:
                tgt = col.map({1: "Approved", 0: "Rejected"})

    if tgt is None:
        raise ValueError(
            "Could not detect target. Provide target_hint='Loan_Status' (Y/N) "
            "or 'Loan_Status_Y' (0/1)."
        )

    return tgt


def bivariate_business_insights(df: pd.DataFrame, target_hint: str | None = None):
    """
    Prints quick bivariate insights using your encoded/raw columns.
    No scaling is applied (not needed for bivariate analysis).
    """
    y_lbl = _normalize_target(df, target_hint)  # 'Approved'/'Rejected'
    data = df.copy()
    data["__Status__"] = y_lbl

    print("\n🔹 Bivariate Q&A Insights 🔹\n")

    # Helper: approval rate for a boolean mask
    def appr_rate(mask: pd.Series) -> float:
        sub = data.loc[mask, "__Status__"]
        return float((sub == "Approved").mean() * 100) if len(sub) else np.nan

    # ---- Credit History ----
    if "Credit_History" in data.columns:
        # normalize to 0/1
        ch = pd.to_numeric(data["Credit_History"], errors="coerce")
        r_good = appr_rate(ch == 1)
        r_bad  = appr_rate(ch == 0)
        print("Q: Does credit history impact loan approval?")
        print(f"👉 Insight: Credit_History=1 → {r_good:.1f}% | Credit_History=0 → {r_bad:.1f}%  (usually the strongest driver)\n")

    # ---- Gender ----
    if "Gender_Male" in data.columns:
        gm = pd.to_numeric(data["Gender_Male"], errors="coerce")
        r_m   = appr_rate(gm == 1)
        r_f   = appr_rate(gm == 0)
        print("Q: Do males or females have higher approval rates?")
        print(f"👉 Insight: Male={r_m:.1f}% | Female={r_f:.1f}%  (difference is typically small)\n")

    # ---- Married ----
    if "Married_Yes" in data.columns:
        my = pd.to_numeric(data["Married_Yes"], errors="coerce")
        r_y = appr_rate(my == 1)
        r_n = appr_rate(my == 0)
        print("Q: Do married applicants get more approvals?")
        print(f"👉 Insight: Married={r_y:.1f}% | Single={r_n:.1f}%  (joint income often helps)\n")

    # ---- Education ----
    if "Education_Not Graduate" in data.columns:
        ed = pd.to_numeric(data["Education_Not Graduate"], errors="coerce")
        r_grad = appr_rate(ed == 0)
        r_nogr = appr_rate(ed == 1)
        print("Q: Do graduates have better approval chances?")
        print(f"👉 Insight: Graduate={r_grad:.1f}% | Not Graduate={r_nogr:.1f}%\n")

    # ---- Applicant Income ----
    if "ApplicantIncome" in data.columns:
        ai = pd.to_numeric(data["ApplicantIncome"], errors="coerce")
        med_y = ai[data["__Status__"] == "Approved"].median()
        med_n = ai[data["__Status__"] == "Rejected"].median()
        print("Q: Do higher incomes lead to more approvals?")
        print(f"👉 Insight: Median income Approved={med_y:.0f} | Rejected={med_n:.0f}  (helpful but weaker than Credit History)\n")

    # ---- Loan Amount ----
    if "LoanAmount" in data.columns:
        la = pd.to_numeric(data["LoanAmount"], errors="coerce")
        med_y = la[data["__Status__"] == "Approved"].median()
        med_n = la[data["__Status__"] == "Rejected"].median()
        print("Q: Are larger loans harder to approve?")
        print(f"👉 Insight: Median loan Approved={med_y:.0f} | Rejected={med_n:.0f}\n")

    # ---- Loan Term (crosstab) ----
    if "Loan_Amount_Term" in data.columns:
        lt = pd.to_numeric(data["Loan_Amount_Term"], errors="coerce")
        ct = (pd.crosstab(lt, data["__Status__"], normalize="index") * 100).round(1)
        print("Q: Do shorter terms have better approval?")
        print(ct)
        print("👉 Insight: Shorter terms tend to show slightly higher approvals.\n")


def plot_bivariate(df: pd.DataFrame, target_hint: str | None = None):
    """
    Three simple, readable plots:
      1) Approval rate by Credit_History
      2) ApplicantIncome distribution by status (violin+box)
      3) Approval rate by Loan_Amount_Term
    """
    y_lbl = _normalize_target(df, target_hint)  # 'Approved'/'Rejected'
    data = df.copy()
    data["__Status__"] = y_lbl

    # 1) Approval rate by Credit_History
    if "Credit_History" in data.columns:
        ch = pd.to_numeric(data["Credit_History"], errors="coerce")
        plot_df = (pd.crosstab(ch, data["__Status__"], normalize="index") * 100).reset_index()
        plot_df = plot_df.rename(columns={"__Status__": "Status", 0: "Rejected", 1: "Approved"})
        plt.figure(figsize=(6,4))
        # Handle column names robustly
        cols = [c for c in plot_df.columns if c in ["Approved", "Rejected"]]
        plot_df.plot(x="Credit_History", y=cols, kind="bar", stacked=True, ax=plt.gca())
        plt.ylabel("Approval %")
        plt.title("Approval Rate by Credit History")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    # 2) ApplicantIncome violin
    if "ApplicantIncome" in data.columns:
        plt.figure(figsize=(6,5))
        sns.violinplot(
            data=data,
            x="__Status__", y="ApplicantIncome",
            inner="box"
        )
        plt.xlabel("Loan Status")
        plt.ylabel("Applicant Income")
        plt.title("Applicant Income by Loan Status")
        plt.tight_layout()
        plt.show()

    # 3) Approval rate by Loan_Amount_Term
    if "Loan_Amount_Term" in data.columns:
        lt = pd.to_numeric(data["Loan_Amount_Term"], errors="coerce")
        ct = (pd.crosstab(lt, data["__Status__"], normalize="index") * 100).reset_index()
        plt.figure(figsize=(6,4))
        sns.lineplot(data=ct, x="Loan_Amount_Term", y="Approved", marker="o")
        plt.ylabel("Approval %")
        plt.title("Approval % vs Loan Term")
        plt.tight_layout()
        plt.show()
