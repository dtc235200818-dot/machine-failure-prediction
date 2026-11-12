"""Các thành phần giao diện dùng lại ở nhiều trang."""
import streamlit as st

LEVEL_COLORS = {"Bình thường": "#1e8e3e", "Sắp hỏng": "#e37400", "Hỏng": "#d93025"}
LEVEL_BG = {"Bình thường": "#d7f0de", "Sắp hỏng": "#ffe6c2", "Hỏng": "#f8d7da"}


def header(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def need_model_warning():
    st.warning("Chưa có mô hình đã huấn luyện. Hãy chạy `python -m src.train` rồi tải lại trang.")


def risk_card(model_name: str, level: str, prob: float, threshold: float, advice: str):
    color = LEVEL_COLORS.get(level, "#555")
    st.markdown(
        f"<div class='risk-card' style='background:{color}'>"
        f"<div class='risk-model'>{model_name}</div>"
        f"<div class='risk-title'>{level}</div>"
        f"<div class='risk-sub'>Xác suất hỏng: {prob:.1%}</div></div>",
        unsafe_allow_html=True)
    st.progress(min(max(float(prob), 0.0), 1.0))
    st.caption(f"Ngưỡng của mô hình: từ {threshold:.0%} là Hỏng, từ {threshold / 2:.0%} là Sắp hỏng.")
    st.write(advice)


def step_card(number: int, title: str, text: str):
    with st.container(border=True):
        st.markdown(f"**{number}. {title}**")
        st.caption(text)


def level_style(value):
    """Tô màu ô theo mức cảnh báo trong bảng kết quả."""
    return f"background-color: {LEVEL_BG[value]}" if value in LEVEL_BG else ""
