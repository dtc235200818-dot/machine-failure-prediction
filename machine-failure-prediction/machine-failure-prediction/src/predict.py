"""Dự đoán tình trạng máy và mức cảnh báo (báo cáo mục 2.4)."""
import joblib
import pandas as pd

from src.preprocess import FEATURES, ROOT, add_features, standardize_columns

MODEL_PATH = ROOT / "models" / "best_model.joblib"

# Ngưỡng xác suất hỏng -> mức cảnh báo (có thể chỉnh theo kết quả thực nghiệm)
THRESHOLD_WARN = 0.30   # từ ngưỡng này: sắp hỏng
THRESHOLD_FAIL = 0.60   # từ ngưỡng này: nguy cơ hỏng cao

MESSAGES = {
    "Bình thường": "Máy hoạt động bình thường, tiếp tục theo dõi theo lịch.",
    "Sắp hỏng": "Có dấu hiệu bất thường, nên kiểm tra máy trong lần bảo trì gần nhất.",
    "Hỏng": "Nguy cơ hỏng cao, nên dừng máy kiểm tra và ưu tiên bảo trì ngay.",
}


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có mô hình tại {path}. Hãy chạy trước: python -m src.train")
    return joblib.load(path)


def risk_level(p: float) -> str:
    if p >= THRESHOLD_FAIL:
        return "Hỏng"
    if p >= THRESHOLD_WARN:
        return "Sắp hỏng"
    return "Bình thường"


def predict_df(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Nhận bảng có các cột gọn (type, air_temp, ...) hoặc tên cột gốc của AI4I.
    Trả về bảng gốc kèm xác suất hỏng, mức cảnh báo và khuyến nghị."""
    model = model or load_model()
    data = add_features(standardize_columns(df))
    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        raise ValueError(f"Thiếu cột: {missing}")
    proba = model.predict_proba(data[FEATURES])[:, 1]
    out = df.copy()
    out["xac_suat_hong"] = proba.round(4)
    out["muc_canh_bao"] = [risk_level(p) for p in proba]
    out["khuyen_nghi"] = out["muc_canh_bao"].map(MESSAGES)
    return out


def predict_one(type_: str, air_temp: float, process_temp: float, rpm: float,
                torque: float, tool_wear: float, model=None) -> dict:
    row = pd.DataFrame([{"type": type_, "air_temp": air_temp, "process_temp": process_temp,
                         "rpm": rpm, "torque": torque, "tool_wear": tool_wear}])
    return predict_df(row, model).iloc[0].to_dict()


if __name__ == "__main__":
    print(predict_one("M", 300.0, 310.0, 1500, 40.0, 100))
