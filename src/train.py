"""Huấn luyen va so sanh hai mo hinh chinh cho bai toan du doan hong may."""
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.evaluate import compute_metrics, plot_confusion_matrices, plot_roc_curves
from src.preprocess import FEATURES, ROOT, build_preprocessor, load_data, split_data

SEED = 42
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
}


def get_models() -> dict:
    """Tra ve dung hai mo hinh chinh cua do an."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
    }


def main():
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df, seed=SEED)
    print(f"Tong: {len(df)} mau | train: {len(X_train)} | test: {len(X_test)}")
    print(f"Ty le hong - train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")
    print("Hai mo hinh chinh: Logistic Regression va Random Forest\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1-score": "f1",
        "ROC-AUC": "roc_auc",
    }

    cv_rows, test_rows, preds, scores, fitted = [], [], {}, {}, {}
    for name, classifier in get_models().items():
        pipeline = Pipeline([
            ("prep", build_preprocessor()),
            ("clf", classifier),
        ])
        cv_result = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1
        )
        cv_rows.append({
            "Mô hình": name,
            **{metric: cv_result[f"test_{metric}"].mean() for metric in scoring},
        })

        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = pipeline.predict(X_test)
        test_rows.append({
            "Mô hình": name,
            **compute_metrics(y_test, predictions, probabilities),
        })
        preds[name] = predictions
        scores[name] = probabilities
        fitted[name] = pipeline

        # Luu rieng tung mo hinh de giao dien cho phep nguoi dung tu chon.
        joblib.dump(pipeline, MODELS / MODEL_FILES[name])
        print(f"Da huan luyen va luu: {name}")

    cv_df = pd.DataFrame(cv_rows).set_index("Mô hình").round(4)
    test_df = pd.DataFrame(test_rows).set_index("Mô hình").round(4)
    cv_df.to_csv(REPORTS / "metrics_cv_train.csv", encoding="utf-8-sig")
    test_df.to_csv(REPORTS / "metrics_test.csv", encoding="utf-8-sig")

    plot_confusion_matrices(preds, y_test, REPORTS / "confusion_matrices.png")
    plot_roc_curves(scores, y_test, REPORTS / "roc_curves.png")

    print("\n=== Kiem dinh cheo 5-fold tren tap train ===")
    print(cv_df.to_string())
    print("\n=== Ket qua tren tap test ===")
    print(test_df.to_string())

    # Chon mo hinh theo F1 cross-validation, khong chon theo tap test.
    best_name = cv_df["F1-score"].idxmax()
    joblib.dump(fitted[best_name], MODELS / "best_model.joblib")
    metadata = {
        "best_model": best_name,
        "compared_models": list(get_models().keys()),
        "selection_metric": "F1-score (5-fold CV)",
        "imbalance_strategy": "class_weight",
        "features": FEATURES,
        "seed": SEED,
    }
    (MODELS / "model_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nMo hinh co F1 CV cao nhat: {best_name}")
    print("Da luu ca hai mo hinh va best_model.joblib trong thu muc models/.")


if __name__ == "__main__":
    main()
