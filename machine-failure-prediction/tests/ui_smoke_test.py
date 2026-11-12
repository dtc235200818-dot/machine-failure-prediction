"""Kiểm tra nhanh các trang giao diện bằng Streamlit giả (không cần cài streamlit).

Chạy:  python tests/ui_smoke_test.py
Mục đích: bắt lỗi cú pháp, tên biến, đường dẫn trong từng trang. Không thay thế việc xem giao diện thật.
"""
import io
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(ST, name)


class State(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v


class Fake(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = State()
        self.sidebar = Ctx()
        self.button_value = False
        self.upload_value = None
        self.calls = []

    def columns(self, spec, **k):
        return [Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]

    def tabs(self, labels):
        return [Ctx() for _ in labels]

    def container(self, **k):
        return Ctx()

    def expander(self, *a, **k):
        return Ctx()

    def button(self, *a, **k):
        return self.button_value

    def download_button(self, *a, **k):
        return False

    def radio(self, label, options, key=None, **k):
        return self.session_state.get(key, options[0])

    def selectbox(self, label, options, key=None, **k):
        options = list(options)
        return self.session_state.get(key, options[0])

    def number_input(self, label, min_value=None, key=None, **k):
        return self.session_state.get(key, min_value)

    def file_uploader(self, *a, **k):
        return self.upload_value

    def cache_resource(self, f=None, **k):
        return f if f else (lambda g: g)

    cache_data = cache_resource

    def Page(self, fn, **k):
        return fn

    def navigation(self, pages):
        return types.SimpleNamespace(run=lambda: None)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None


ST = Fake()
sys.modules["streamlit"] = ST

from src.preprocess import DATA_PATH  # noqa: E402
from views import batch, data, evaluation, home, predict  # noqa: E402
from src.config import PRESETS  # noqa: E402
import pandas as pd  # noqa: E402

home.show_home()
data.show_data()
evaluation.show_evaluation()

# Trang dự đoán: chưa bấm nút, bấm nút với từng lựa chọn mô hình và từng mẫu thử
predict.show_predict()
for preset in PRESETS:
    ST.session_state["preset"] = preset
    predict._apply_preset()
    for choice in ["Random Forest", "Logistic Regression", "So sánh cả hai"]:
        ST.session_state["model_choice"] = choice
        ST.button_value = True
        predict.show_predict()
        ST.button_value = False
        predict.show_predict()

# Trang CSV: có file hợp lệ, file thiếu cột
sample = pd.read_csv(DATA_PATH, encoding="utf-8-sig").head(20)
for choice in ["Random Forest", "Logistic Regression", "So sánh cả hai"]:
    ST.session_state["batch_model"] = choice
    ST.upload_value = io.BytesIO(sample.to_csv(index=False).encode("utf-8"))
    batch.show_batch()
ST.upload_value = io.BytesIO(b"a,b\n1,2\n")
batch.show_batch()
print("OK: các trang chạy không lỗi")
