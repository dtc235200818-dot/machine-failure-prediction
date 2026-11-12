"""Dự đoán tình trạng máy và mức cảnh báo (báo cáo mục 2.4)."""
import joblib
import pandas as pd

from src.config import FEATURE_LABELS, MODEL_NAMES, MODEL_SLUGS, MODELS_DIR
from src.preprocess import FEATURES, RAW_NUMERIC, add_features, standardize_columns

DEFAULT_MODEL = "Random Forest"

MESSAGES = {
    "Bình thường": "Máy hoạt động bình thường, tiếp tục theo dõi theo lịch.",
    "Sắp hỏng": "Có dấu hiệu bất thường, nên kiểm tra máy trong lần bảo trì gần nhất.",
    "Hỏng": "Nguy cơ hỏng cao, nên dừng máy kiểm tra và ưu tiên bảo trì ngay.",
}


def load_bundle(name: str = DEFAULT_MODEL) -> dict:
    """Tải gói mô hình: {'name', 'pipeline', 'threshold', 'features'}."""
    if name not in MODEL_SLUGS:
        raise ValueError(f"Mô hình không hợp lệ: {name}. Chọn một trong {MODEL_NAMES}")
    path = MODELS_DIR / f"{MODEL_SLUGS[name]}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Chưa có mô hình tại {path}. Hãy chạy trước: python -m src.train")
    return joblib.load(path)


def risk_level(p: float, threshold: float) -> str:
    """Hỏng: từ ngưỡng của mô hình; Sắp hỏng: từ một nửa ngưỡng; còn lại: Bình thường."""
    if p >= threshold:
        return "Hỏng"
    if p >= 0.5 * threshold:
        return "Sắp hỏng"
    return "Bình thường"


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    data = add_features(standardize_columns(df))
    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        raise ValueError(f"Thiếu cột: {missing}")
    return data[FEATURES]


def predict_df(df: pd.DataFrame, model_name: str = DEFAULT_MODEL, bundle: dict = None) -> pd.DataFrame:
    """Nhận bảng có cột gọn (type, air_temp, ...) hoặc tên cột gốc AI4I.
    Trả về bảng gốc kèm xác suất hỏng, mức cảnh báo và khuyến nghị."""
    bundle = bundle or load_bundle(model_name)
    proba = bundle["pipeline"].predict_proba(_prepare(df))[:, 1]
    out = df.copy()
    out["xac_suat_hong"] = proba.round(4)
    out["muc_canh_bao"] = [risk_level(p, bundle["threshold"]) for p in proba]
    out["khuyen_nghi"] = out["muc_canh_bao"].map(MESSAGES)
    return out


def predict_one(type_: str, air_temp: float, process_temp: float, rpm: float,
                torque: float, tool_wear: float, model_name: str = DEFAULT_MODEL) -> dict:
    row = pd.DataFrame([{"type": type_, "air_temp": air_temp, "process_temp": process_temp,
                         "rpm": rpm, "torque": torque, "tool_wear": tool_wear}])
    return predict_df(row, model_name).iloc[0].to_dict()


def explain_inputs(df_row: pd.DataFrame, bundle: dict, reference: dict) -> pd.Series:
    """Giải thích một dự đoán: đưa từng thông số về giá trị tham chiếu (trung vị của dữ liệu huấn luyện)
    rồi xem xác suất hỏng giảm bao nhiêu. Giá trị dương nghĩa là thông số đó đang làm tăng nguy cơ."""
    row = standardize_columns(df_row).iloc[[0]].reset_index(drop=True)
    pipe = bundle["pipeline"]
    base = pipe.predict_proba(_prepare(row))[:, 1][0]
    out = {}
    for col in RAW_NUMERIC:
        tmp = row.copy()
        tmp[col] = reference[col]
        out[FEATURE_LABELS[col]] = float(base - pipe.predict_proba(_prepare(tmp))[:, 1][0])
    s = pd.Series(out)
    return s.reindex(s.abs().sort_values(ascending=False).index)


if __name__ == "__main__":
    for n in MODEL_NAMES:
        print(n, predict_one("L", 298, 308.5, 1350, 68, 200, n))
