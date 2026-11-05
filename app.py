"""Giao diện chương trình (báo cáo mục 3.5). Chạy: streamlit run app.py"""
import pandas as pd
import streamlit as st

from src.predict import MESSAGES, THRESHOLD_FAIL, THRESHOLD_WARN, load_model, predict_df
from src.preprocess import ROOT

st.set_page_config(page_title="Dự đoán hỏng hóc máy", page_icon="🛠️", layout="centered")
st.title("🛠️ Dự đoán hỏng hóc máy từ dữ liệu cảm biến")


@st.cache_resource
def get_model():
    return load_model()


try:
    model = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

COLORS = {"Bình thường": "green", "Sắp hỏng": "orange", "Hỏng": "red"}
tab1, tab2, tab3 = st.tabs(["Nhập dữ liệu cảm biến", "Dự đoán từ file CSV", "Kết quả mô hình"])

with tab1:
    c1, c2 = st.columns(2)
    type_ = c1.selectbox("Loại sản phẩm (Type)", ["L", "M", "H"], index=1)
    air = c1.number_input("Nhiệt độ không khí [K]", 290.0, 310.0, 300.0, 0.1)
    proc = c1.number_input("Nhiệt độ quá trình [K]", 300.0, 320.0, 310.0, 0.1)
    rpm = c2.number_input("Tốc độ quay [rpm]", 1000, 3000, 1500, 10)
    torque = c2.number_input("Mô-men xoắn [Nm]", 0.0, 100.0, 40.0, 0.1)
    wear = c2.number_input("Độ mòn dụng cụ [phút]", 0, 300, 100, 1)

    if st.button("Dự đoán", type="primary"):
        row = pd.DataFrame([{"type": type_, "air_temp": air, "process_temp": proc,
                             "rpm": rpm, "torque": torque, "tool_wear": wear}])
        res = predict_df(row, model).iloc[0]
        level = res["muc_canh_bao"]
        st.markdown(f"### Mức cảnh báo: :{COLORS[level]}[{level}]")
        st.metric("Xác suất hỏng", f"{res['xac_suat_hong']:.1%}")
        st.info(MESSAGES[level])
        st.caption(f"Ngưỡng: từ {THRESHOLD_WARN:.0%} là sắp hỏng, từ {THRESHOLD_FAIL:.0%} là nguy cơ hỏng cao.")

with tab2:
    st.write("Tải lên file CSV có các cột: Type, Air temperature [K], Process temperature [K], "
             "Rotational speed [rpm], Torque [Nm], Tool wear [min].")
    up = st.file_uploader("Chọn file CSV", type="csv")
    if up is not None:
        try:
            res = predict_df(pd.read_csv(up, encoding="utf-8-sig"), model)
            st.dataframe(res, use_container_width=True)
            st.write(res["muc_canh_bao"].value_counts())
            st.download_button("Tải kết quả (CSV)", res.to_csv(index=False).encode("utf-8-sig"),
                               "ket_qua_du_doan.csv", "text/csv")
        except Exception as e:  # hiển thị lỗi thay vì làm sập ứng dụng
            st.error(f"Không xử lý được file: {e}")

with tab3:
    reports = ROOT / "reports"
    f = reports / "metrics_test.csv"
    if f.exists():
        st.subheader("Kết quả trên tập test")
        st.dataframe(pd.read_csv(f, encoding="utf-8-sig"), use_container_width=True)
        for img, cap in [("confusion_matrices.png", "Ma trận nhầm lẫn"), ("roc_curves.png", "Đường cong ROC")]:
            if (reports / img).exists():
                st.image(str(reports / img), caption=cap)
    else:
        st.warning("Chưa có kết quả. Hãy chạy: python -m src.train")
