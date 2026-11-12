"""Ứng dụng web dự đoán hỏng hóc máy. Chạy: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Dự đoán hỏng hóc máy", page_icon="🛠️", layout="wide")

from ui.state import models_ready  # noqa: E402
from ui.style import inject_css  # noqa: E402
from views import batch, data, evaluation, home, predict  # noqa: E402

inject_css()

pages = [
    st.Page(home.show_home, title="Tổng quan", icon="🏠", url_path="tong-quan", default=True),
    st.Page(predict.show_predict, title="Dự đoán", icon="🔮", url_path="du-doan"),
    st.Page(batch.show_batch, title="Dự đoán từ file CSV", icon="📄", url_path="file-csv"),
    st.Page(evaluation.show_evaluation, title="Đánh giá mô hình", icon="📊", url_path="danh-gia"),
    st.Page(data.show_data, title="Dữ liệu", icon="🗂️", url_path="du-lieu"),
]
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("### Mô hình")
    st.caption("Logistic Regression và Random Forest")
    st.success("Đã sẵn sàng") if models_ready() else st.error("Chưa huấn luyện")
    st.caption("Dữ liệu: AI4I 2020 (UCI), CC BY 4.0")

nav.run()
