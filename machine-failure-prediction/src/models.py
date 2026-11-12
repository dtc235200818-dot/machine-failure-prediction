"""Thiết kế hai mô hình chính: Logistic Regression và Random Forest.

Mỗi mô hình là một Pipeline (tiền xử lý + bộ phân loại), được tinh chỉnh tham số bằng
GridSearchCV (kiểm định chéo 5 phần, tối ưu F1). Ngưỡng quyết định được chọn trên dự đoán
out-of-fold của tập train để cân bằng Precision và Recall, vì dữ liệu mất cân bằng.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline

from src.config import FEATURE_LABELS
from src.preprocess import FEATURES, RAW_NUMERIC, add_features, build_preprocessor

SEED = 42


def build_logistic_regression():
    """Logistic Regression: mô hình tuyến tính, thêm số hạng bậc 2 và tích chéo để bắt quan hệ phi tuyến."""
    pipe = Pipeline([
        ("prep", build_preprocessor(polynomial=True)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)),
    ])
    grid = {"clf__C": [0.01, 0.1, 1, 10, 100, 1000]}
    return pipe, grid


def build_random_forest():
    """Random Forest: tập hợp nhiều cây quyết định, bắt được quan hệ phi tuyến."""
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(class_weight="balanced_subsample", n_jobs=1, random_state=SEED)),
    ])
    grid = {
        "clf__n_estimators": [300],
        "clf__max_depth": [None, 12, 20],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", 0.5],
    }
    return pipe, grid


BUILDERS = {"Logistic Regression": build_logistic_regression, "Random Forest": build_random_forest}


BETA = 2.0  # ưu tiên Recall: bỏ sót máy sắp hỏng tốn kém hơn một cảnh báo sai


def best_threshold(y_true, proba, beta: float = BETA) -> float:
    """Ngưỡng xác suất làm F-beta lớn nhất (tính trên dự đoán out-of-fold của tập train)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    p, r = precision[:-1], recall[:-1]
    fbeta = (1 + beta**2) * p * r / np.clip(beta**2 * p + r, 1e-12, None)
    return float(thresholds[int(np.argmax(fbeta))])


RAW_INPUTS = ["type"] + RAW_NUMERIC


def permutation_effects(pipeline, X_test, y_test, n_repeats: int = 10, seed: int = SEED) -> dict:
    """Độ quan trọng hoán vị trên các thông số đầu vào gốc: xáo trộn từng cột trong tập test
    (tính lại các đặc trưng dẫn xuất) rồi đo mức giảm Average Precision.
    Giá trị càng lớn thì thông số càng quan trọng, so sánh được giữa hai mô hình."""
    rng = np.random.default_rng(seed)
    raw = X_test[RAW_INPUTS].reset_index(drop=True)
    y = np.asarray(y_test)
    base = average_precision_score(y, pipeline.predict_proba(add_features(raw)[FEATURES])[:, 1])
    effects = {}
    for col in RAW_INPUTS:
        drops = []
        for _ in range(n_repeats):
            shuffled = raw.copy()
            shuffled[col] = rng.permutation(shuffled[col].values)
            ap = average_precision_score(y, pipeline.predict_proba(add_features(shuffled)[FEATURES])[:, 1])
            drops.append(base - ap)
        effects[FEATURE_LABELS[col]] = float(np.mean(drops))
    return effects
