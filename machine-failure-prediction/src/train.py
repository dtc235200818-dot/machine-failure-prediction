"""Huấn luyện và đánh giá hai mô hình chính (báo cáo mục 2.3, 3.3, 3.4).

Chạy:  python -m src.train
Kết quả: models/*.joblib, models/model_info.json, reports/*.csv, reports/*.png
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from src.config import MODEL_NAMES, MODEL_SLUGS, MODELS_DIR, REPORTS_DIR
from src.evaluate import (compute_metrics, plot_confusion_matrices, plot_feature_effects,
                          plot_pr_curves, plot_roc_curves)
from src.models import BUILDERS, SEED, best_threshold, permutation_effects
from src.preprocess import FEATURES, load_data, split_data


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df, seed=SEED)
    print(f"Tổng: {len(df)} mẫu | train: {len(X_train)} | test: {len(X_test)}")
    print(f"Tỷ lệ hỏng - train: {y_train.mean():.2%} | test: {y_test.mean():.2%}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    info, cv_rows, test_rows, preds, scores, effects = {}, [], [], {}, {}, {}

    for name in MODEL_NAMES:
        pipe, grid = BUILDERS[name]()
        print(f"[{name}] tìm tham số tốt nhất (GridSearchCV, tối ưu F1)...")
        gs = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        print(f"    tham số: {gs.best_params_}")

        # Ngưỡng quyết định chọn trên dự đoán out-of-fold của tập train
        oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        thr = best_threshold(y_train, oof)
        cv_m = compute_metrics(y_train, (oof >= thr).astype(int), oof)

        proba = best.predict_proba(X_test)[:, 1]
        pred = (proba >= thr).astype(int)
        test_m = compute_metrics(y_test, pred, proba)
        print(f"    ngưỡng = {thr:.3f} | F1 test = {test_m['F1-score']:.4f}\n")

        cv_rows.append({"Mô hình": name, **cv_m})
        test_rows.append({"Mô hình": name, **test_m})
        preds[name], scores[name] = pred, proba
        effects[name] = permutation_effects(best, X_test, y_test)
        info[name] = {"threshold": thr, "best_params": gs.best_params_,
                      "cv_metrics": cv_m, "test_metrics": test_m, "feature_effects": effects[name]}

        joblib.dump({"name": name, "pipeline": best, "threshold": thr, "features": FEATURES},
                    MODELS_DIR / f"{MODEL_SLUGS[name]}.joblib")

    cv_df = pd.DataFrame(cv_rows).set_index("Mô hình").round(4)
    test_df = pd.DataFrame(test_rows).set_index("Mô hình").round(4)
    cv_df.to_csv(REPORTS_DIR / "metrics_cv_train.csv", encoding="utf-8-sig")
    test_df.to_csv(REPORTS_DIR / "metrics_test.csv", encoding="utf-8-sig")
    plot_confusion_matrices(preds, y_test, REPORTS_DIR / "confusion_matrices.png")
    plot_roc_curves(scores, y_test, REPORTS_DIR / "roc_curves.png")
    plot_pr_curves(scores, y_test, REPORTS_DIR / "pr_curves.png")
    plot_feature_effects(effects, REPORTS_DIR / "feature_effects.png")

    meta = {"models": info, "best_model": cv_df["F1-score"].idxmax(), "features": FEATURES, "seed": SEED,
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "failure_rate": float(df["machine_failure"].mean())}
    (MODELS_DIR / "model_info.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8")

    print("=== Out-of-fold trên tập train (ngưỡng đã chọn) ===")
    print(cv_df.to_string())
    print("\n=== Kết quả trên tập test ===")
    print(test_df.to_string())


if __name__ == "__main__":
    main()
