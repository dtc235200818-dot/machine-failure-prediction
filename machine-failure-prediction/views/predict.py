"""Trang Dự đoán cho một máy."""
import pandas as pd
import streamlit as st

from src.config import INPUT_SPECS, MODEL_NAMES, PRESETS
from src.predict import MESSAGES, explain_inputs, predict_df
from src.preprocess import add_features
from ui.components import header, need_model_warning, risk_card
from ui.state import get_bundle, get_reference, models_ready

CHOICES = MODEL_NAMES[::-1] + ["So sánh cả hai"]  # Random Forest, Logistic Regression, So sánh


def _init_state():
    for spec in INPUT_SPECS.values():
        st.session_state.setdefault(spec["key"], spec["default"])


def _apply_preset():
    values = PRESETS.get(st.session_state.get("preset"))
    if values:
        for feature, value in values.items():
            st.session_state[INPUT_SPECS[feature]["key"]] = value


def _inputs() -> dict:
    s = INPUT_SPECS
    c1, c2, c3 = st.columns(3)
    row = {}
    row["type"] = c1.selectbox("Loại sản phẩm (Type)", s["type"]["options"], key=s["type"]["key"])
    row["air_temp"] = c1.number_input("Nhiệt độ không khí [K]", min_value=s["air_temp"]["min"],
                                      max_value=s["air_temp"]["max"], step=s["air_temp"]["step"],
                                      format="%.1f", key=s["air_temp"]["key"])
    row["process_temp"] = c2.number_input("Nhiệt độ quá trình [K]", min_value=s["process_temp"]["min"],
                                          max_value=s["process_temp"]["max"], step=s["process_temp"]["step"],
                                          format="%.1f", key=s["process_temp"]["key"])
    row["rpm"] = c2.number_input("Tốc độ quay [rpm]", min_value=s["rpm"]["min"], max_value=s["rpm"]["max"],
                                 step=s["rpm"]["step"], key=s["rpm"]["key"])
    row["torque"] = c3.number_input("Mô-men xoắn [Nm]", min_value=s["torque"]["min"],
                                    max_value=s["torque"]["max"], step=s["torque"]["step"],
                                    format="%.1f", key=s["torque"]["key"])
    row["tool_wear"] = c3.number_input("Độ mòn dụng cụ [phút]", min_value=s["tool_wear"]["min"],
                                       max_value=s["tool_wear"]["max"], step=s["tool_wear"]["step"],
                                       key=s["tool_wear"]["key"])
    return row


def show_predict():
    header("🔮 Dự đoán tình trạng máy", "Nhập thông số cảm biến, chọn mô hình rồi bấm Dự đoán.")
    if not models_ready():
        need_model_warning()
        return
    _init_state()

    with st.container(border=True):
        st.subheader("1. Thông số cảm biến")
        st.selectbox("Mẫu thử nhanh", list(PRESETS), key="preset", on_change=_apply_preset)
        row = _inputs()
        if row["process_temp"] <= row["air_temp"]:
            st.warning("Nhiệt độ quá trình thường cao hơn nhiệt độ không khí. Hãy kiểm tra lại giá trị nhập.")
        d = add_features(pd.DataFrame([row])).iloc[0]
        st.caption(f"Đặc trưng tính thêm: chênh lệch nhiệt độ {d['temp_diff']:.1f} K, "
                   f"công suất cơ {d['power_w']:.0f} W, độ mòn × mô-men {d['wear_torque']:.0f}.")

    with st.container(border=True):
        st.subheader("2. Chọn mô hình")
        choice = st.radio("Mô hình dùng để dự đoán", CHOICES, horizontal=True, key="model_choice",
                          label_visibility="collapsed")
        if st.button("Dự đoán", type="primary", key="btn_predict"):
            st.session_state["last_row"] = row

    last = st.session_state.get("last_row")
    if not last:
        st.info("Chưa có kết quả. Bấm Dự đoán để xem.")
        return

    st.subheader("3. Kết quả")
    names = MODEL_NAMES[::-1] if choice == "So sánh cả hai" else [choice]
    frame = pd.DataFrame([last])
    cols = st.columns(len(names))
    for col, name in zip(cols, names):
        bundle = get_bundle(name)
        res = predict_df(frame, bundle=bundle).iloc[0]
        with col:
            risk_card(name, res["muc_canh_bao"], res["xac_suat_hong"], bundle["threshold"],
                      MESSAGES[res["muc_canh_bao"]])

    with st.expander("Thông số nào đang làm tăng nguy cơ?"):
        st.caption("Với mỗi thông số, đưa về giá trị trung vị của dữ liệu rồi xem xác suất hỏng giảm bao nhiêu. "
                   "Cột dương là thông số đang làm tăng nguy cơ.")
        ref = get_reference()
        for name in names:
            st.markdown(f"**{name}**")
            st.bar_chart(explain_inputs(frame, get_bundle(name), ref))
