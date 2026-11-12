"""Trang Dữ liệu."""
import pandas as pd
import streamlit as st

from src.preprocess import DATA_PATH
from ui.components import header

FEATURE_TABLE = pd.DataFrame([
    ["Type", "Loại chất lượng sản phẩm: L, M hoặc H", "-"],
    ["Air temperature", "Nhiệt độ không khí", "K"],
    ["Process temperature", "Nhiệt độ quá trình", "K"],
    ["Rotational speed", "Tốc độ quay", "rpm"],
    ["Torque", "Mô-men xoắn", "Nm"],
    ["Tool wear", "Độ mòn dụng cụ", "phút"],
], columns=["Thuộc tính", "Ý nghĩa", "Đơn vị"])


def show_data():
    header("🗂️ Dữ liệu", "AI4I 2020 Predictive Maintenance Dataset (UCI), dữ liệu mô phỏng, giấy phép CC BY 4.0.")
    raw = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    failed = int(raw["Machine failure"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Số mẫu", f"{len(raw):,}")
    c2.metric("Số mẫu hỏng", f"{failed:,}")
    c3.metric("Tỷ lệ hỏng", f"{failed / len(raw):.2%}")

    tabs = st.tabs(["Thuộc tính", "Phân bố", "Xem dữ liệu", "Tiền xử lý"])
    with tabs[0]:
        st.dataframe(FEATURE_TABLE)
        st.caption("Nhãn là Machine failure (0 = bình thường, 1 = hỏng).")
    with tabs[1]:
        a, b = st.columns(2)
        with a:
            st.markdown("**Bình thường và hỏng**")
            st.bar_chart(pd.Series({"Bình thường": len(raw) - failed, "Hỏng": failed}))
        with b:
            st.markdown("**Số mẫu theo loại sản phẩm**")
            st.bar_chart(raw["Type"].value_counts())
        modes = raw[["TWF", "HDF", "PWF", "OSF", "RNF"]].sum()
        st.markdown("**Số mẫu theo dạng hỏng** (chỉ để tham khảo, không dùng làm đầu vào)")
        st.bar_chart(modes)
    with tabs[2]:
        st.dataframe(raw.head(100))
    with tabs[3]:
        st.markdown(
            "1. Bỏ cột UDI và Product ID.\n"
            "2. Bỏ các cột dạng hỏng TWF, HDF, PWF, OSF, RNF để tránh rò rỉ nhãn.\n"
            "3. Tạo thêm 3 đặc trưng: chênh lệch nhiệt độ, công suất cơ, độ mòn × mô-men.\n"
            "4. Xử lý giá trị thiếu, chuẩn hóa số, mã hóa one-hot cho Type.\n"
            "5. Với Logistic Regression: thêm số hạng bậc 2 và tích chéo.\n"
            "6. Chia 80% train, 20% test, giữ nguyên tỷ lệ lớp.\n"
            "7. Xử lý mất cân bằng bằng trọng số lớp.")
