"""Huấn luyện, kiểm định chéo và so sánh các mô hình (báo cáo mục 2.3, 3.3, 3.4).

Chạy:  python -m src.train
Kết quả: reports/*.csv, reports/*.png và models/best_model.joblib
"""
import json
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.evaluate import compute_metrics, plot_confusion_matrices, plot_roc_curves
from src.preprocess import FEATURES, ROOT, build_preprocessor, load_data, split_data

SEED = 42
# sklearn 1.9 cảnh báo tham số probability của SVC sẽ đổi trong bản tương lai; vẫn chạy bình thường
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"


def get_models(y_train) -> dict:
    """Các thuật toán so sánh. Dữ liệu mất cân bằng nên dùng trọng số lớp."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
    }
    try:  # XGBoost là tùy chọn
        from xgboost import XGBClassifier
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
            scale_pos_weight=neg / pos, eval_metric="logloss", random_state=SEED, n_jobs=-1)
    except ImportError:
        print("[!] Chưa cài xgboost -> bỏ qua XGBoost (pip install xgboost để thêm).")
    return models


def main():
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df, seed=SEED)
    print(f"Tổng: {len(df)} mẫu | train: {len(X_train)} | test: {len(X_test)}")
    print(f"Tỷ lệ hỏng - train: {y_train.mean():.2%} | test: {y_test.mean():.2%}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"Accuracy": "accuracy", "Precision": "precision", "Recall": "recall",
               "F1-score": "f1", "ROC-AUC": "roc_auc"}

    cv_rows, test_rows, preds, scores, fitted = [], [], {}, {}, {}
    for name, clf in get_models(y_train).items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])

        # Kiểm định chéo 5 phần trên tập train (đóng vai trò validation)
        cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        cv_rows.append({"Mô hình": name, **{m: cvres[f"test_{m}"].mean() for m in scoring}})

        # Huấn luyện trên toàn bộ train, đánh giá trên test
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        test_rows.append({"Mô hình": name, **compute_metrics(y_test, pred, proba)})
        preds[name], scores[name], fitted[name] = pred, proba, pipe
        print(f"  xong: {name}")

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

    # Chọn mô hình theo F1 của kiểm định chéo (không nhìn vào tập test)
    best = cv_df["F1-score"].idxmax()
    joblib.dump(fitted[best], MODELS / "best_model.joblib")
    (MODELS / "model_info.json").write_text(
        json.dumps({"best_model": best, "features": FEATURES, "seed": SEED}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\nMô hình tốt nhất theo F1 (CV): {best} -> models/best_model.joblib")


if __name__ == "__main__":
    main()
