"""
Breast Cancer Classification using Classical & Hybrid ML Models
Models: Logistic Regression, SVM, KNN, Random Forest, Naive Bayes,
        Gradient Boosting, XGBoost (if available), MLP Neural Network
Evaluation: Accuracy, F1-Score, ROC-AUC, Confusion Matrix, CV Score
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

# Optional XGBoost installation check
try:
    from xgboost import XGBClassifier
    xgb_available = True
except:
    xgb_available = False


def evaluate_model(name, model, X_train, y_train, X_tesI'll install Miniforge" — you will install it yourself; tell me when done and I will continue to create env, train, and start server.t, y_test):
    """Fit, evaluate model and return metrics"""
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["malignant(0)", "benign(1)"]
        )
    }
    return metrics, y_score


def main():
    # ---- Load Dataset ----
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    print(f"Dataset Loaded: {X.shape[0]} samples, {X.shape[1]} features\n")

    # ---- Train-Test Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ---- Model Dictionary ----
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced"))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=42
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "MLP Neural Network": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42))
        ])
    }

    if xgb_available:
        models["XGBoost"] = XGBClassifier(
            eval_metric="logloss",
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            use_label_encoder=False
        )

    # ---- Cross Validation ----
    print("Cross-validation ROC-AUC Scores:")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        score = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")
        print(f"{name:>22}: {score.mean():.4f} ± {score.std():.4f}")
    print()

    # ---- Evaluation ----
    results = []
    roc_curves = {}

    for name, model in models.items():
        metrics, y_score = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        roc_curves[name] = y_score
        results.append(metrics)

        print(f"====== {name} ======")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"ROC-AUC : {metrics['roc_auc']:.4f}")
        print(metrics["classification_report"])

        # Save confusion matrix
        sns.heatmap(metrics["confusion_matrix"], annot=True, cmap="Blues", fmt="d")
        plt.title(f"Confusion Matrix - {name}")
        os.makedirs("artifacts/confusion_matrices", exist_ok=True)
        plt.savefig(f"artifacts/confusion_matrices/{name}.png")
        plt.close()

    # ---- Ranking Best Model ----
    df_results = pd.DataFrame([
        {"model": m["model"], "accuracy": m["accuracy"], "f1": m["f1"], "roc_auc": m["roc_auc"]}
        for m in results
    ]).sort_values(by=["roc_auc", "f1"], ascending=False)

    print("\n===== Final Model Ranking =====")
    print(df_results)

    best_model_name = df_results.iloc[0]["model"]
    best_model = models[best_model_name]

    # ---- Train Best Model on Full Dataset ----
    final_model = best_model.fit(X, y)
    os.makedirs("artifacts", exist_ok=True)
    dump(final_model, "artifacts/best_model.joblib")

    print(f"\nBest Model Saved: {best_model_name}")

    # ---- ROC CURVES ----
    plt.figure(figsize=(8, 6))
    for name, y_score in roc_curves.items():
        RocCurveDisplay.from_predictions(y_test, y_score, name=name)
    plt.title("ROC Comparison Across Models")
    plt.tight_layout()
    plt.savefig("artifacts/roc_curves.png", dpi=150)


if __name__ == "__main__":
    main()
