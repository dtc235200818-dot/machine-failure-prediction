"""Trang Tổng quan."""
import streamlit as st

from src.config import MODEL_NAMES
from ui.components import header, need_model_warning, step_card
from ui.state import get_dataset, get_info, models_ready

ROLES = {
    "Logistic Regression": "Mô hình tuyến tính, có thêm số hạng bậc 2 và tích chéo giữa các thông số. "
                           "Nhanh, nhẹ, ưu tiên bắt được nhiều ca hỏng.",
    "Random Forest": "Tập hợp nhiều cây quyết định, bắt tốt quan hệ phi tuyến giữa các thông số. "
                     "Cho độ chính xác cao nhất trong hai mô hình.",
}


def show_home():
    header("🛠️ Dự đoán hỏng hóc máy từ dữ liệu cảm biến",
           "Hệ thống hỗ trợ bảo trì dự đoán, dùng hai mô hình học máy: Logistic Regression và Random Forest.")

    df = get_dataset()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số mẫu dữ liệu", f"{len(df):,}")
    c2.metric("Tỷ lệ máy hỏng", f"{df['machine_failure'].mean():.2%}")
    c3.metric("Thông số cảm biến", "5")
    c4.metric("Mô hình chính", "2")

    st.subheader("Cách hoạt động")
    cols = st.columns(4)
    steps = [("Nhập dữ liệu", "Nhập thông số cảm biến của một máy, hoặc tải lên file CSV."),
             ("Xử lý", "Hệ thống tạo thêm đặc trưng (chênh lệch nhiệt độ, công suất) và chuẩn hóa."),
             ("Dự đoán", "Hai mô hình tính xác suất máy hỏng."),
             ("Cảnh báo", "Xác suất được đổi thành mức: Bình thường, Sắp hỏng, Hỏng.")]
    for i, (col, (t, d)) in enumerate(zip(cols, steps), start=1):
        with col:
            step_card(i, t, d)

    st.subheader("Hai mô hình chính")
    if not models_ready():
        need_model_warning()
        return
    info = get_info() or {}
    cols = st.columns(2)
    for col, name in zip(cols, MODEL_NAMES):
        with col:
            with st.container(border=True):
                st.markdown(f"#### {name}")
                st.caption(ROLES[name])
                m = info.get("models", {}).get(name, {}).get("test_metrics")
                if m:
                    a, b, c = st.columns(3)
                    a.metric("Recall", f"{m['Recall']:.2f}")
                    b.metric("F1-score", f"{m['F1-score']:.2f}")
                    c.metric("ROC-AUC", f"{m['ROC-AUC']:.3f}")
                    st.caption("Kết quả trên tập test.")
    st.info("Dùng menu bên trái để chuyển giữa các phần: Dự đoán, Dự đoán từ file CSV, Đánh giá mô hình, Dữ liệu.")
