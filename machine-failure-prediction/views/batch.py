"""Trang Dự đoán hàng loạt từ file CSV."""
import pandas as pd
import streamlit as st

from src.config import MODEL_NAMES, MODEL_SLUGS
from src.predict import predict_df
from src.preprocess import DATA_PATH
from ui.components import header, level_style, need_model_warning
from ui.state import get_bundle, models_ready

LEVELS = ["Bình thường", "Sắp hỏng", "Hỏng"]
SAMPLE_COLS = ["Type", "Air temperature [K]", "Process temperature [K]",
               "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]


def _sample_csv() -> bytes:
    sample = pd.read_csv(DATA_PATH, encoding="utf-8-sig").head(30)[SAMPLE_COLS]
    return sample.to_csv(index=False).encode("utf-8-sig")


def _predict_all(data: pd.DataFrame, choice: str) -> pd.DataFrame:
    if choice != "So sánh cả hai":
        return predict_df(data, bundle=get_bundle(choice))
    out = data.copy()
    for name in MODEL_NAMES:
        r = predict_df(data, bundle=get_bundle(name))
        slug = MODEL_SLUGS[name]
        out[f"xac_suat_hong_{slug}"] = r["xac_suat_hong"]
        out[f"muc_canh_bao_{slug}"] = r["muc_canh_bao"]
    return out


def show_batch():
    header("📄 Dự đoán từ file CSV", "Dự đoán nhiều máy cùng lúc và tải kết quả về.")
    if not models_ready():
        need_model_warning()
        return

    with st.container(border=True):
        st.write("File cần có 6 cột: Type, Air temperature [K], Process temperature [K], "
                 "Rotational speed [rpm], Torque [Nm], Tool wear [min].")
        st.download_button("Tải file mẫu", _sample_csv(), "mau_du_lieu_cam_bien.csv", "text/csv")
        choice = st.radio("Mô hình", MODEL_NAMES[::-1] + ["So sánh cả hai"], horizontal=True, key="batch_model")
        uploaded = st.file_uploader("Chọn file CSV", type="csv")

    if uploaded is None:
        return
    try:
        data = pd.read_csv(uploaded, encoding="utf-8-sig")
        result = _predict_all(data, choice)
    except Exception as e:  # hiển thị lỗi thay vì làm sập ứng dụng
        st.error(f"Không xử lý được file: {e}")
        return

    st.subheader("Kết quả")
    level_cols = [c for c in result.columns if c.startswith("muc_canh_bao")]
    cols = st.columns(len(level_cols))
    for col, lc in zip(cols, level_cols):
        with col:
            st.markdown(f"**{lc.replace('muc_canh_bao', 'Mức cảnh báo').replace('_', ' ').strip()}**")
            st.bar_chart(result[lc].value_counts().reindex(LEVELS, fill_value=0))
    try:
        st.dataframe(result.style.map(level_style, subset=level_cols))
    except Exception:
        st.dataframe(result)
    st.download_button("Tải kết quả (CSV)", result.to_csv(index=False).encode("utf-8-sig"),
                       "ket_qua_du_doan.csv", "text/csv")
