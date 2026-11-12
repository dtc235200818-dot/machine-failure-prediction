"""Độ đo và biểu đồ đánh giá (báo cáo mục 2.3.4, 3.3, 3.4)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, ConfusionMatrixDisplay, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_score),
    }


def plot_confusion_matrices(results: dict, y_true, path):
    """results: {tên mô hình: y_pred}"""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6))
    axes = np.atleast_1d(axes)
    for ax, (name, y_pred) in zip(axes, results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, ax=ax, colorbar=False, cmap="Blues",
            display_labels=["Bình thường", "Hỏng"])
        ax.set_title(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc_curves(scores: dict, y_true, path):
    """scores: {tên mô hình: xác suất lớp hỏng}"""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Tỷ lệ dương tính giả (FPR)")
    ax.set_ylabel("Tỷ lệ dương tính thật (TPR)")
    ax.set_title("Đường cong ROC trên tập test")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pr_curves(scores: dict, y_true, path):
    from sklearn.metrics import average_precision_score, precision_recall_curve
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y_true, s)
        ax.plot(r, p, label=f"{name} (AP={average_precision_score(y_true, s):.3f})")
    ax.axhline(float(np.mean(y_true)), color="k", ls="--", lw=1, label="Ngẫu nhiên")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Đường cong Precision–Recall trên tập test")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_feature_effects(effects: dict, path):
    """effects: {tên mô hình: {tên thông số: mức giảm Average Precision khi xáo trộn}}"""
    fig, axes = plt.subplots(1, len(effects), figsize=(6.2 * len(effects), 3.8), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, eff) in zip(axes, effects.items()):
        items = sorted(eff.items(), key=lambda kv: kv[1])
        ax.barh([k for k, _ in items], [v for _, v in items], color="#1f77b4")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Mức giảm Average Precision khi xáo trộn", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
