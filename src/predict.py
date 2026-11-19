"""Du doan tinh trang may bang Logistic Regression hoac Random Forest."""
import joblib
import pandas as pd

from src.preprocess import FEATURES, ROOT, add_features, standardize_columns

MODEL_FILES = {
    "Logistic Regression": ROOT / "models" / "logistic_regression.joblib",
    "Random Forest": ROOT / "models" / "random_forest.joblib",
}
MODEL_PATH = ROOT / "models" / "best_model.joblib"

THRESHOLD_WARN = 0.30
THRESHOLD_FAIL = 0.60

MESSAGES = {
    "Bình thường": "Máy hoạt động bình thường, tiếp tục theo dõi theo lịch.",
    "Sắp hỏng": "Có dấu hiệu bất thường, nên kiểm tra máy trong lần bảo trì gần nhất.",
    "Hỏng": "Nguy cơ hỏng cao, nên dừng máy kiểm tra và ưu tiên bảo trì ngay.",
}


def load_model(model_name: str = "best"):
    """Tai mo hinh duoc chon; 'best' van ho tro tuong thich phien ban cu."""
    path = MODEL_FILES.get(model_name, MODEL_PATH if model_name == "best" else None)
    if path is None:
        raise ValueError(
            f"Mo hinh khong hop le: {model_name}. Hay chon Logistic Regression hoac Random Forest."
        )
    if not path.exists():
        raise FileNotFoundError(
            f"Chua co tep mo hinh: {path}. Hay chay: python -m src.train"
        )
    return joblib.load(path)


def risk_level(probability: float) -> str:
    if probability >= THRESHOLD_FAIL:
        return "Hỏng"
    if probability >= THRESHOLD_WARN:
        return "Sắp hỏng"
    return "Bình thường"


def predict_df(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Du doan bang CSV co ten cot AI4I goc hoac ten cot rut gon cua du an."""
    estimator = model if model is not None else load_model()
    data = add_features(standardize_columns(df))
    missing = [column for column in FEATURES if column not in data.columns]
    if missing:
        raise ValueError(f"Thieu cot dau vao: {missing}")
    probabilities = estimator.predict_proba(data[FEATURES])[:, 1]
    output = df.copy()
    output["xac_suat_hong"] = probabilities.round(4)
    output["muc_canh_bao"] = [risk_level(value) for value in probabilities]
    output["khuyen_nghi"] = output["muc_canh_bao"].map(MESSAGES)
    return output


def predict_one(type_: str, air_temp: float, process_temp: float, rpm: float,
                torque: float, tool_wear: float, model=None) -> dict:
    row = pd.DataFrame([{
        "type": type_,
        "air_temp": air_temp,
        "process_temp": process_temp,
        "rpm": rpm,
        "torque": torque,
        "tool_wear": tool_wear,
    }])
    return predict_df(row, model).iloc[0].to_dict()


if __name__ == "__main__":
    print(predict_one("M", 300.0, 310.0, 1500, 40.0, 100))
