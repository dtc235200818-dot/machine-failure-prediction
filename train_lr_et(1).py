#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dự đoán hỏng hóc máy (AI4I 2020) - Logistic Regression & Extra Trees
Nhóm 20 - Học phần Học máy (đặt tại src/train_lr_et.py)

Cách chạy:
    python -m src.train_lr_et                      # đọc data/ai4i2020.csv, ghi vào reports/ và models/experiments/
    python -m src.train_lr_et --synthetic --out /tmp/thu   # chỉ kiểm tra code, KHÔNG dùng số liệu này cho báo cáo

Cấu hình bám theo báo cáo (mục 2.1 - 2.4):
    - Đặc trưng: Type (One-Hot) + 5 thông số vận hành
    - Nhãn: Machine failure; bỏ UDI, Product ID, TWF, HDF, PWF, OSF, RNF (tránh rò rỉ nhãn)
    - Chia 80/20, random_state=42, stratify=y
    - Logistic Regression: class_weight='balanced', max_iter=2000
    - Extra Trees: n_estimators=300, class_weight='balanced', min_samples_leaf=2, random_state=42
    - Độ đo: Accuracy, Precision, Recall, F1, ROC-AUC; ngưỡng rủi ro 0.30 / 0.60
"""
import argparse
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
CAT_COL = "Type"
NUM_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"
DROP_COLS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
RISK_LOW, RISK_HIGH = 0.30, 0.60  # mục 2.4.3


# ----------------------------------------------------------------------------
# 1. Dữ liệu
# ----------------------------------------------------------------------------
def make_synthetic(n=10000, seed=RANDOM_STATE):
    """Dữ liệu mô phỏng theo quy tắc sinh của AI4I 2020, CHỈ để kiểm tra code chạy được."""
    rng = np.random.default_rng(seed)
    typ = rng.choice(["L", "M", "H"], size=n, p=[0.6, 0.3, 0.1])

    def smooth(mean, sd):
        x = np.zeros(n)
        e = rng.normal(size=n)
        for i in range(1, n):
            x[i] = 0.999 * x[i - 1] + e[i]
        return mean + sd * (x - x.mean()) / x.std()

    air = smooth(300.0, 2.0)
    proc = air + 10 + rng.normal(0, 1.0, n)
    torque = np.clip(rng.normal(40, 10, n), 3.8, None)
    rpm = np.clip(1538.8 - 14 * (torque - 40) + rng.normal(0, 110, n), 1168, None)
    wear = np.clip(rng.normal(108, 63, n), 0, 253)

    power = torque * rpm * 2 * np.pi / 60
    twf = (wear >= 200) & (wear <= 240) & (rng.random(n) < 0.03)
    hdf = ((proc - air) < 8.6) & (rpm < 1380)
    pwf = (power < 3500) | (power > 9000)
    lim = np.select([typ == "L", typ == "M"], [11000, 12000], 13000)
    osf = wear * torque > lim
    rnf = rng.random(n) < 0.001
    fail = (twf | hdf | pwf | osf | rnf).astype(int)

    return pd.DataFrame({
        "UDI": np.arange(1, n + 1),
        "Product ID": [f"{t}{i}" for i, t in enumerate(typ)],
        "Type": typ,
        "Air temperature [K]": air.round(1),
        "Process temperature [K]": proc.round(1),
        "Rotational speed [rpm]": rpm.round(0).astype(int),
        "Torque [Nm]": torque.round(1),
        "Tool wear [min]": wear.round(0).astype(int),
        "Machine failure": fail,
        "TWF": twf.astype(int), "HDF": hdf.astype(int), "PWF": pwf.astype(int),
        "OSF": osf.astype(int), "RNF": rnf.astype(int),
    })


def load_and_check(path):
    df = pd.read_csv(path)
    need = [CAT_COL, TARGET] + NUM_COLS
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"Thiếu cột trong file CSV: {miss}\nCác cột hiện có: {list(df.columns)}")
    return df


def describe_data(df):
    n = len(df)
    n_fail = int(df[TARGET].sum())
    print("=== Kiểm tra dữ liệu (mục 2.2.2) ===")
    print(f"Kích thước: {df.shape[0]} bản ghi x {df.shape[1]} cột")
    print(f"Giá trị thiếu: {int(df.isna().sum().sum())} | Dòng trùng lặp: {int(df.duplicated().sum())}")
    print(f"Bình thường: {n - n_fail} ({(n - n_fail) / n:.2%}) | Hỏng: {n_fail} ({n_fail / n:.2%})")
    print(f"Type: {df[CAT_COL].value_counts().to_dict()}\n")


# ----------------------------------------------------------------------------
# 2. Mô hình (Pipeline thống nhất giữa train / test / dữ liệu mới)
# ----------------------------------------------------------------------------
def build_logistic_regression():
    pre = ColumnTransformer([
        ("type", OneHotEncoder(handle_unknown="ignore"), [CAT_COL]),
        ("num", StandardScaler(), NUM_COLS),          # LR nhạy với thang đo -> chuẩn hóa
    ])
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    return Pipeline([("prep", pre), ("clf", clf)])


def build_extra_trees():
    pre = ColumnTransformer([
        ("type", OneHotEncoder(handle_unknown="ignore"), [CAT_COL]),
        ("num", "passthrough", NUM_COLS),             # cây quyết định không cần chuẩn hóa
    ])
    clf = ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                               min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline([("prep", pre), ("clf", clf)])


# ----------------------------------------------------------------------------
# 3. Đánh giá
# ----------------------------------------------------------------------------
def metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


def risk_level(p):
    return "Thấp" if p < RISK_LOW else ("Trung bình" if p < RISK_HIGH else "Cao")


def plot_confusion(cms, out):
    fig, axes = plt.subplots(1, len(cms), figsize=(5 * len(cms), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (name, cm) in zip(axes, cms.items()):
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Ma trận nhầm lẫn - {name}")
        ax.set_xticks([0, 1], ["Bình thường", "Hỏng"])
        ax.set_yticks([0, 1], ["Bình thường", "Hỏng"])
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_roc(curves, out):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Đường ROC trên tập kiểm thử")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_importance(models, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, pipe) in zip(axes, models.items()):
        names = pipe.named_steps["prep"].get_feature_names_out()
        names = [n.split("__", 1)[1] for n in names]
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            vals, title = clf.feature_importances_, "Độ quan trọng đặc trưng"
        else:
            vals, title = clf.coef_[0], "Hệ số hồi quy (đã chuẩn hóa)"
        order = np.argsort(np.abs(vals))
        ax.barh(np.array(names)[order], vals[order], color="#3b6ea5")
        ax.set_title(f"{name}\n{title}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 4. Dự đoán một bản ghi mới (mục 2.4.1 - 2.4.3)
# ----------------------------------------------------------------------------
def predict_one(pipe, record):
    """record: dict gồm Type và 5 thông số. Trả về nhãn, xác suất hỏng, mức rủi ro."""
    missing = [c for c in [CAT_COL] + NUM_COLS if c not in record]
    if missing:
        raise ValueError(f"Thiếu trường: {missing}")
    if record[CAT_COL] not in ("L", "M", "H"):
        raise ValueError("Type phải là L, M hoặc H")
    row = {CAT_COL: record[CAT_COL]}
    for c in NUM_COLS:
        try:
            row[c] = float(record[c])
        except (TypeError, ValueError):
            raise ValueError(f"{c} phải là số")
    X = pd.DataFrame([row])
    p = float(pipe.predict_proba(X)[0, 1])
    return {"label": int(pipe.predict(X)[0]), "FailureProbability": round(p, 4),
            "RiskLevel": risk_level(p)}


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ai4i2020.csv", help="Đường dẫn file ai4i2020.csv")
    ap.add_argument("--synthetic", action="store_true",
                    help="Dùng dữ liệu mô phỏng để kiểm tra code (không dùng cho báo cáo)")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--models-dir", default="models/experiments", help="Không ghi đè mô hình tốt nhất mà app.py đang dùng")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    if args.synthetic:
        print("!!! DỮ LIỆU MÔ PHỎNG - chỉ kiểm tra code, không phải kết quả thật !!!\n")
        df = make_synthetic()
    else:
        df = load_and_check(args.data)

    describe_data(df)

    X = df[[CAT_COL] + NUM_COLS]          # đã loại UDI, Product ID, TWF..RNF
    y = df[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print("=== Chia dữ liệu 80/20 (mục 2.2.6) ===")
    for name, yy in [("Train", y_tr), ("Test", y_te)]:
        print(f"{name}: {len(yy)} mẫu | hỏng {int(yy.sum())} ({yy.mean():.2%})")
    print()

    builders = {"Logistic Regression": build_logistic_regression,
                "Extra Trees": build_extra_trees}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"Accuracy": "accuracy", "Precision": "precision", "Recall": "recall",
               "F1-score": "f1", "ROC-AUC": "roc_auc"}

    fitted, rows, cms, curves, cv_rows, risk_rows = {}, [], {}, {}, [], []
    for name, build in builders.items():
        pipe = build().fit(X_tr, y_tr)
        fitted[name] = pipe

        # train / test
        for split, XX, yy in [("Train", X_tr, y_tr), ("Test", X_te, y_te)]:
            prob = pipe.predict_proba(XX)[:, 1]
            pred = pipe.predict(XX)
            rows.append({"Mô hình": name, "Tập": split, **metrics(yy, pred, prob)})
            if split == "Test":
                cms[name] = confusion_matrix(yy, pred)
                fpr, tpr, _ = roc_curve(yy, prob)
                curves[name] = (fpr, tpr, roc_auc_score(yy, prob))
                lv = pd.Series(prob).map(risk_level).value_counts()
                for lvl in ["Thấp", "Trung bình", "Cao"]:
                    mask = pd.Series(prob).map(risk_level).values == lvl
                    risk_rows.append({"Mô hình": name, "Mức rủi ro": lvl,
                                      "Số mẫu": int(lv.get(lvl, 0)),
                                      "Số mẫu hỏng thật": int(yy.values[mask].sum())})

        # cross-validation trên train (đánh giá quá khớp / ổn định)
        res = cross_validate(build(), X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
        cv_rows.append({"Mô hình": name, **{k: f"{res['test_' + k].mean():.4f} ± {res['test_' + k].std():.4f}"
                                            for k in scoring}})

    res_df = pd.DataFrame(rows)
    cv_df = pd.DataFrame(cv_rows)
    risk_df = pd.DataFrame(risk_rows)

    pd.set_option("display.width", 200)
    print("=== Kết quả Train / Test ===")
    print(res_df.round(4).to_string(index=False))
    print("\n=== Cross-validation 5 fold trên tập train (mean ± std) ===")
    print(cv_df.to_string(index=False))
    print("\n=== Ma trận nhầm lẫn trên tập test [[TN, FP], [FN, TP]] ===")
    for n, cm in cms.items():
        print(n, cm.tolist())
    print("\n=== Phân bố mức rủi ro trên tập test (ngưỡng 0.30 / 0.60) ===")
    print(risk_df.to_string(index=False))

    # lưu kết quả
    res_df.round(4).to_csv(f"{args.out}/metrics_train_test_lr_et.csv", index=False, encoding="utf-8-sig")
    cv_df.to_csv(f"{args.out}/metrics_cv_lr_et.csv", index=False, encoding="utf-8-sig")
    risk_df.to_csv(f"{args.out}/risk_levels_test_lr_et.csv", index=False, encoding="utf-8-sig")
    with open(f"{args.out}/confusion_matrices_lr_et.json", "w", encoding="utf-8") as f:
        json.dump({k: v.tolist() for k, v in cms.items()}, f, ensure_ascii=False, indent=2)
    plot_confusion(cms, f"{args.out}/confusion_matrices_lr_et.png")
    plot_roc(curves, f"{args.out}/roc_curves_lr_et.png")
    plot_importance(fitted, f"{args.out}/feature_importance_lr_et.png")
    for name, pipe in fitted.items():
        joblib.dump(pipe, f"{args.models_dir}/{name.lower().replace(' ', '_')}.joblib")

    # ví dụ dự đoán một bản ghi mới
    demo = {"Type": "L", "Air temperature [K]": 298.5, "Process temperature [K]": 308.7,
            "Rotational speed [rpm]": 1350, "Torque [Nm]": 62.0, "Tool wear [min]": 215}
    print("\n=== Ví dụ dự đoán bản ghi mới ===")
    for name, pipe in fitted.items():
        print(name, predict_one(pipe, demo))
    print(f"\nĐã lưu kết quả vào thư mục: {args.out}/")


if __name__ == "__main__":
    main()
