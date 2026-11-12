"""Tiền xử lý dữ liệu (báo cáo mục 2.2).

- Đọc dữ liệu AI4I 2020, đổi tên cột cho gọn.
- Loại các cột không dùng làm đầu vào (UDI, Product ID) và các cột dạng hỏng
  (TWF, HDF, PWF, OSF, RNF) để tránh rò rỉ nhãn: Machine failure = 1 khi có ít nhất
  một dạng hỏng xảy ra, nên giữ các cột này sẽ làm mô hình "biết trước đáp án".
- Tạo thêm đặc trưng dẫn xuất (trích chọn đặc trưng, mục 2.2.4).
- Xử lý giá trị thiếu, chuẩn hóa, mã hóa biến loại bằng ColumnTransformer.
- Chia tập train/test theo tỷ lệ lớp (stratified), vì dữ liệu mất cân bằng.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ai4i2020.csv"

TARGET = "machine_failure"
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]

RENAME = {
    "Type": "type",
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": TARGET,
}
RAW_NUMERIC = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
DERIVED_NUMERIC = ["temp_diff", "power_w", "wear_torque"]
NUMERIC = RAW_NUMERIC + DERIVED_NUMERIC
CATEGORICAL = ["type"]
FEATURES = CATEGORICAL + NUMERIC


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Đổi tên cột gốc của AI4I sang tên gọn (bỏ qua cột không có)."""
    return df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm đặc trưng dẫn xuất từ các cảm biến gốc."""
    out = df.copy()
    out["temp_diff"] = out["process_temp"] - out["air_temp"]          # chênh lệch nhiệt độ (K)
    out["power_w"] = out["torque"] * out["rpm"] * 2 * np.pi / 60      # công suất cơ (W)
    out["wear_torque"] = out["tool_wear"] * out["torque"]             # độ mòn x mô-men
    return out


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Đọc CSV và trả về bảng gồm FEATURES + TARGET (đã loại cột gây rò rỉ)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = standardize_columns(df)
    df = add_features(df)
    return df[FEATURES + [TARGET]]


def build_preprocessor(polynomial: bool = False) -> ColumnTransformer:
    """polynomial=True: thêm các số hạng bậc 2 và tích chéo giữa các đặc trưng số
    (dùng cho Logistic Regression để mô hình tuyến tính bắt được quan hệ phi tuyến)."""
    steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if polynomial:
        steps += [("poly", PolynomialFeatures(degree=2, include_bias=False)),
                  ("scaler2", StandardScaler())]
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", Pipeline(steps), NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Chia train/test giữ nguyên tỷ lệ hỏng/không hỏng."""
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
