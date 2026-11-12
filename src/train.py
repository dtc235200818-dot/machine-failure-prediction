"""Huấn luyện và so sánh hai mô hình chính cho bài toán dự đoán hỏng máy.

Hai mô hình được chọn:
1. Logistic Regression: mô hình cơ sở, dễ giải thích.
2. Random Forest: mô hình phi tuyến, phù hợp quan hệ phức tạp giữa cảm biến.

Dữ liệu mất cân bằng được xử lý bằng class_weight; dùng StratifiedKFold
để giữ tỷ lệ lớp trong 5-fold cross-validation.

Chạy: python -m src.train
Kết quả: reports/*.csv, reports/*.png và models/best_model.joblib
"""
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


def get_models() -> dict:
    """Trả về đúng hai mô hình được sử dụng trong đồ án."""
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
    print(f"Tổng: {len(df)} mẫu | train: {len(X_train)} | test: {len(X_test)}")
    print(f"Tỷ lệ hỏng - train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")
    print("Hai mô hình: Logistic Regression và Random Forest\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1-score": "f1",
        "ROC-AUC": "roc_auc",
    }

    cv_rows, test_rows, preds, scores, fitted = [], [], {}, {}, {}
    for name, clf in get_models().items():
        pipe = Pipeline([
            ("prep", build_preprocessor()),
            ("clf", clf),
        ])

        # Đánh giá chéo trên tập train; không sử dụng tập test để chọn mô hình.
        cvres = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1
        )
        cv_rows.append({
            "Mô hình": name,
            **{metric: cvres[f"test_{metric}"].mean() for metric in scoring},
        })

        # Huấn luyện trên toàn bộ train và đánh giá đúng một lần trên test.
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        test_rows.append({"Mô hình": name, **compute_metrics(y_test, pred, proba)})
        preds[name] = pred
        scores[name] = proba
        fitted[name] = pipe
        print(f"Đã hoàn thành: {name}")

    cv_df = pd.DataFrame(cv_rows).set_index("Mô hình").round(4)
    test_df = pd.DataFrame(test_rows).set_index("Mô hình").round(4)
    cv_df.to_csv(REPORTS / "metrics_cv_train.csv", encoding="utf-8-sig")
    test_df.to_csv(REPORTS / "metrics_test.csv", encoding="utf-8-sig")

    plot_confusion_matrices(preds, y_test, REPORTS / "confusion_matrices.png")
    plot_roc_curves(scores, y_test, REPORTS / "roc_curves.png")

    print("\n=== Kiểm định chéo 5 phần trên tập train ===")
    print(cv_df.to_string())
    print("\n=== Kết quả trên tập test ===")
    print(test_df.to_string())

    # Chọn theo F1 trung bình của cross-validation, không chọn theo tập test.
    best = cv_df["F1-score"].idxmax()
    joblib.dump(fitted[best], MODELS / "best_model.joblib")
    (MODELS / "model_info.json").write_text(
        json.dumps(
            {
                "best_model": best,
                "compared_models": list(get_models().keys()),
                "selection_metric": "F1-score (5-fold CV)",
                "imbalance_strategy": "class_weight",
                "features": FEATURES,
                "seed": SEED,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nMô hình được chọn theo F1 (CV): {best}")
    print(f"Đã lưu mô hình tại: {MODELS / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
