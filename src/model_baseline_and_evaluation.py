import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def run_baseline_and_evaluation(df: pd.DataFrame, target: str):

    X = df.drop(columns=[target])
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, solver="liblinear"),
        "SVM (Linear)": LinearSVC(),
        "SVM (RBF)": SVC(kernel="rbf"),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    # Train + evaluate
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append([name, acc, f1])

        print(f"\n{name}")
        print("Accuracy:", acc)
        print("F1 Score:", f1)
        print(classification_report(y_test, y_pred))

        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        plt.title(name)
        plt.show()

    # Summary table
    res_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1"])
    print("\n=== Model Comparison ===")
    print(res_df.sort_values("F1", ascending=False))

    return res_df
