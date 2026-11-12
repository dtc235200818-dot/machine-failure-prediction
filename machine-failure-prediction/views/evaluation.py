"""Trang Đánh giá mô hình."""
import pandas as pd
import streamlit as st

from src.config import MODEL_NAMES, REPORTS_DIR
from ui.components import header, need_model_warning
from ui.state import get_info, models_ready

TABS = ["Chỉ số", "Ma trận nhầm lẫn", "ROC và PR", "Thông số quan trọng", "Tham số mô hình"]


def _image(name: str, caption: str):
    path = REPORTS_DIR / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.warning(f"Chưa có {name}. Hãy chạy python -m src.train")


def show_evaluation():
    header("📊 Đánh giá mô hình", "So sánh Logistic Regression và Random Forest trên tập test.")
    if not models_ready():
        need_model_warning()
        return
    info = get_info() or {}
    tabs = st.tabs(TABS)

    with tabs[0]:
        f = REPORTS_DIR / "metrics_test.csv"
        if f.exists():
            df = pd.read_csv(f, index_col=0, encoding="utf-8-sig")
            st.dataframe(df)
            st.bar_chart(df.T)
            st.caption("Accuracy cao dễ gây hiểu nhầm vì chỉ khoảng 3,4% mẫu là hỏng. "
                       "Hãy xem Recall (bắt được bao nhiêu ca hỏng) và Precision (cảnh báo đúng bao nhiêu).")
        else:
            st.warning("Chưa có metrics_test.csv. Hãy chạy python -m src.train")
    with tabs[1]:
        _image("confusion_matrices.png", "Ma trận nhầm lẫn trên tập test")
    with tabs[2]:
        a, b = st.columns(2)
        with a:
            _image("roc_curves.png", "Đường cong ROC")
        with b:
            _image("pr_curves.png", "Đường cong Precision–Recall")
    with tabs[3]:
        _image("feature_effects.png", "Mức giảm Average Precision khi xáo trộn từng thông số")
    with tabs[4]:
        for name in MODEL_NAMES:
            m = info.get("models", {}).get(name)
            if not m:
                continue
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.write(f"Ngưỡng quyết định: {m['threshold']:.3f}")
                st.json(m["best_params"])
        st.caption("Tham số được chọn bằng GridSearchCV (kiểm định chéo 5 phần, tối ưu F1). "
                   "Ngưỡng chọn bằng F2 trên dự đoán out-of-fold của tập train, ưu tiên Recall.")
