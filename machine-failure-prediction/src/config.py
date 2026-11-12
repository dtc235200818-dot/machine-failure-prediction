"""Cấu hình dùng chung: tên mô hình, nhãn đặc trưng, khoảng giá trị nhập, mẫu thử."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
INFO_PATH = MODELS_DIR / "model_info.json"

# Hai mô hình chính của dự án
MODEL_NAMES = ["Logistic Regression", "Random Forest"]
MODEL_SLUGS = {"Logistic Regression": "logistic_regression", "Random Forest": "random_forest"}

FEATURE_LABELS = {
    "type": "Loại sản phẩm",
    "air_temp": "Nhiệt độ không khí [K]",
    "process_temp": "Nhiệt độ quá trình [K]",
    "rpm": "Tốc độ quay [rpm]",
    "torque": "Mô-men xoắn [Nm]",
    "tool_wear": "Độ mòn dụng cụ [phút]",
    "temp_diff": "Chênh lệch nhiệt độ [K]",
    "power_w": "Công suất cơ [W]",
    "wear_torque": "Độ mòn × mô-men",
}

# Cấu hình ô nhập trên giao diện
INPUT_SPECS = {
    "type": {"key": "in_type", "options": ["L", "M", "H"], "default": "M"},
    "air_temp": {"key": "in_air", "min": 290.0, "max": 310.0, "step": 0.1, "default": 300.0},
    "process_temp": {"key": "in_proc", "min": 300.0, "max": 320.0, "step": 0.1, "default": 310.0},
    "rpm": {"key": "in_rpm", "min": 1000, "max": 3000, "step": 10, "default": 1500},
    "torque": {"key": "in_torque", "min": 0.0, "max": 100.0, "step": 0.1, "default": 40.0},
    "tool_wear": {"key": "in_wear", "min": 0, "max": 300, "step": 1, "default": 100},
}

# Mẫu thử nhanh (đã kiểm tra bằng mô hình sau khi huấn luyện)
PRESETS = {
    "Tự nhập": None,
    "Máy hoạt động bình thường": {"type": "M", "air_temp": 300.0, "process_temp": 310.0,
                                  "rpm": 1500, "torque": 40.0, "tool_wear": 100},
    "Mô-men xoắn cao, tốc độ thấp": {"type": "L", "air_temp": 298.0, "process_temp": 308.5,
                                     "rpm": 1350, "torque": 68.0, "tool_wear": 200},
    "Tản nhiệt kém (chênh lệch nhiệt độ nhỏ)": {"type": "M", "air_temp": 302.0, "process_temp": 310.5,
                                                 "rpm": 1300, "torque": 45.0, "tool_wear": 60},
    "Dụng cụ mòn nhiều": {"type": "L", "air_temp": 300.0, "process_temp": 310.0,
                          "rpm": 1500, "torque": 45.0, "tool_wear": 240},
}
