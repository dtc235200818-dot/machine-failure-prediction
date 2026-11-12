"""Dashboard giao dien cho he thong du doan hong hoc may."""
import json
import pandas as pd
import streamlit as st
from src.predict import MESSAGES, THRESHOLD_FAIL, THRESHOLD_WARN, load_model, predict_df
from src.preprocess import DATA_PATH, ROOT

st.set_page_config(page_title="MachineCare | Predictive Maintenance", page_icon="🏭", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background:#f5f8fc; }
[data-testid="stHeader"] { background:rgba(245,248,252,.92); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#102a56 0%,#173d70 100%); }
[data-testid="stSidebar"] * { color:#f8fbff !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:.48rem .65rem; border-radius:.55rem; }
.main-title { font-size:2rem; font-weight:800; color:#142b4d; margin-bottom:.15rem; }
.sub-title { color:#64748b; margin-bottom:1.35rem; }
.eyebrow { color:#2563eb; font-size:.75rem; font-weight:800; letter-spacing:.12em; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #e5edf7; padding:1rem 1.1rem; border-radius:14px; box-shadow:0 3px 14px rgba(15,39,75,.045); }
div[data-testid="stMetricLabel"] { color:#64748b; }
div[data-testid="stMetricValue"] { color:#142b4d; }
div.stButton > button[kind="primary"] { background:#2563eb; border:0; border-radius:9px; padding:.55rem 1.1rem; font-weight:700; }
.info-card { background:#fff; border:1px solid #e5edf7; border-radius:14px; padding:1rem 1.1rem; min-height:112px; }
.info-card small { color:#64748b; }
.info-card strong { display:block; color:#142b4d; font-size:1.12rem; margin:.35rem 0; }
.section-heading { color:#142b4d; font-size:1.2rem; font-weight:750; margin:.35rem 0 .8rem; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_model()

@st.cache_data
def get_dataset():
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig") if DATA_PATH.exists() else pd.DataFrame()

@st.cache_data
def get_metrics():
    path = ROOT / "reports" / "metrics_test.csv"
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()

@st.cache_data
def get_model_info():
    path = ROOT / "models" / "model_info.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {}

try:
    model = get_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.info("Hãy huấn luyện mô hình bằng lệnh: python -m src.train")
    st.stop()

with st.sidebar:
    st.markdown("## 🏭 MachineCare")
    st.caption("PREDICTIVE MAINTENANCE SYSTEM")
    st.markdown("---")
    page = st.radio("ĐIỀU HƯỚNG", ["Tổng quan", "Dự đoán máy", "Dự đoán hàng loạt", "Đánh giá mô hình"])
    st.markdown("---")
    st.markdown("**Trạng thái hệ thống**")
    st.success("Mô hình đã được tải")
    st.caption("Dữ liệu: AI4I 2020")
    st.caption("Đồ án học máy · Python / Streamlit")

dataset = get_dataset()
metrics = get_metrics()
model_info = get_model_info()
best_model = model_info.get("best_model", "Mô hình đã huấn luyện")
st.markdown('<div class="eyebrow">PREDICTIVE MAINTENANCE</div>', unsafe_allow_html=True)

if page == "Tổng quan":
    st.markdown('<div class="main-title">Tổng quan hệ thống</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Theo dõi dữ liệu cảm biến và hỗ trợ phát hiện sớm nguy cơ hỏng hóc máy.</div>', unsafe_allow_html=True)
    total = len(dataset)
    failures = int(dataset["Machine failure"].sum()) if total and "Machine failure" in dataset else 0
    rate = failures / total if total else 0
    best_f1 = pd.to_numeric(metrics["F1-score"], errors="coerce").max() if not metrics.empty and "F1-score" in metrics else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng số mẫu dữ liệu", f"{total:,}" if total else "Chưa có dữ liệu")
    c2.metric("Số mẫu máy hỏng", f"{failures:,}" if total else "—")
    c3.metric("Tỷ lệ hỏng trong dữ liệu", f"{rate:.2%}" if total else "—")
    c4.metric("F1-score cao nhất", f"{best_f1:.3f}" if best_f1 is not None else "—")
    st.markdown('<div class="section-heading">Thông tin hệ thống</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        st.markdown(f'<div class="info-card"><small>MÔ HÌNH ĐANG SỬ DỤNG</small><strong>{best_model}</strong><small>Mô hình được lưu trong thư mục models/</small></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="info-card"><small>BỘ DỮ LIỆU</small><strong>AI4I 2020</strong><small>Dữ liệu bảo trì dự đoán từ cảm biến công nghiệp</small></div>', unsafe_allow_html=True)
    with c:
        st.markdown(f'<div class="info-card"><small>NGƯỠNG CẢNH BÁO</small><strong>{THRESHOLD_WARN:.0%} / {THRESHOLD_FAIL:.0%}</strong><small>Bình thường · Sắp hỏng · Hỏng</small></div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="section-heading">Phân bố nhãn trong dữ liệu</div>', unsafe_allow_html=True)
        if total and "Machine failure" in dataset:
            counts = dataset["Machine failure"].map({0:"Bình thường", 1:"Hỏng"}).value_counts()
            st.bar_chart(counts, color="#3b82f6", use_container_width=True)
        else:
            st.info("Chưa tìm thấy data/ai4i2020.csv.")
    with right:
        st.markdown('<div class="section-heading">Quy trình sử dụng</div>', unsafe_allow_html=True)
        st.markdown("1. Chọn **Dự đoán máy** để nhập thông số cảm biến.\n2. Chọn **Dự đoán hàng loạt** để tải tệp CSV.\n3. Xem xác suất hỏng và mức cảnh báo.\n4. Mở **Đánh giá mô hình** để xem các chỉ số thực nghiệm.")

elif page == "Dự đoán máy":
    st.markdown('<div class="main-title">Dự đoán tình trạng máy</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Nhập sáu thông số đầu vào để ước lượng xác suất hỏng của máy.</div>', unsafe_allow_html=True)
    with st.form("single_prediction"):
        st.markdown("#### 1. Thông tin và nhiệt độ")
        c1, c2, c3 = st.columns(3)
        with c1: type_ = st.selectbox("Loại sản phẩm (Type)", ["L", "M", "H"], index=1)
        with c2: air = st.number_input("Nhiệt độ không khí (K)", 290.0, 310.0, 300.0, 0.1)
        with c3: process = st.number_input("Nhiệt độ quá trình (K)", 300.0, 320.0, 310.0, 0.1)
        st.markdown("#### 2. Thông số vận hành")
        c4, c5, c6 = st.columns(3)
        with c4: rpm = st.number_input("Tốc độ quay (rpm)", 1000, 3000, 1500, 10)
        with c5: torque = st.number_input("Mô-men xoắn (Nm)", 0.0, 100.0, 40.0, 0.1)
        with c6: wear = st.number_input("Độ mòn dụng cụ (phút)", 0, 300, 100, 1)
        submitted = st.form_submit_button("🔍 Phân tích nguy cơ hỏng", type="primary", use_container_width=True)
    if submitted:
        row = pd.DataFrame([{"type":type_, "air_temp":air, "process_temp":process, "rpm":rpm, "torque":torque, "tool_wear":wear}])
        try:
            result = predict_df(row, model).iloc[0]
            level = result["muc_canh_bao"]
            probability = float(result["xac_suat_hong"])
            item = {"Loại máy":type_, "Nhiệt độ không khí (K)":air, "Nhiệt độ quá trình (K)":process, "Tốc độ quay (rpm)":rpm, "Mô-men xoắn (Nm)":torque, "Độ mòn (phút)":wear, "Xác suất hỏng":probability, "Mức cảnh báo":level, "Khuyến nghị":MESSAGES[level]}
            st.session_state["last_prediction"] = item
            st.session_state.setdefault("prediction_history", []).append(item)
            st.markdown("### Kết quả phân tích")
            icon = {"Bình thường":"🟢", "Sắp hỏng":"🟠", "Hỏng":"🔴"}.get(level, "⚪")
            x, y = st.columns([1, 2])
            with x:
                st.metric("Xác suất hỏng", f"{probability:.1%}")
                st.progress(min(max(probability, 0.0), 1.0))
            with y:
                st.subheader(f"{icon} {level}")
                st.info(MESSAGES[level])
            st.caption(f"Ngưỡng cảnh báo: từ {THRESHOLD_WARN:.0%} là Sắp hỏng; từ {THRESHOLD_FAIL:.0%} là Hỏng (nguy cơ cao).")
        except Exception as error:
            st.error(f"Không thể thực hiện dự đoán: {error}")
    if st.session_state.get("prediction_history"):
        with st.expander("Lịch sử dự đoán trong phiên hiện tại"):
            st.dataframe(pd.DataFrame(st.session_state["prediction_history"]), use_container_width=True)

elif page == "Dự đoán hàng loạt":
    st.markdown('<div class="main-title">Dự đoán từ tệp CSV</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Tải dữ liệu cảm biến lên để dự đoán nhiều mẫu trong một lần.</div>', unsafe_allow_html=True)
    st.markdown("Tệp CSV cần có sáu cột đầu vào: **Type**, **Air temperature [K]**, **Process temperature [K]**, **Rotational speed [rpm]**, **Torque [Nm]**, **Tool wear [min]**. Cũng có thể dùng tên cột rút gọn theo dự án.")
    uploaded = st.file_uploader("Chọn tệp dữ liệu CSV", type=["csv"])
    if uploaded is not None:
        try:
            input_df = pd.read_csv(uploaded, encoding="utf-8-sig")
            result_df = predict_df(input_df, model)
            normal = int((result_df["muc_canh_bao"] == "Bình thường").sum())
            warning = int((result_df["muc_canh_bao"] == "Sắp hỏng").sum())
            failure = int((result_df["muc_canh_bao"] == "Hỏng").sum())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tổng mẫu", f"{len(result_df):,}")
            m2.metric("Bình thường", f"{normal:,}")
            m3.metric("Sắp hỏng", f"{warning:,}")
            m4.metric("Nguy cơ hỏng cao", f"{failure:,}")
            st.markdown("#### Kết quả dự đoán")
            st.dataframe(result_df, use_container_width=True, height=420)
            st.download_button("⬇️ Tải kết quả dự đoán (CSV)", result_df.to_csv(index=False).encode("utf-8-sig"), file_name="ket_qua_du_doan.csv", mime="text/csv", type="primary")
        except Exception as error:
            st.error(f"Không xử lý được tệp CSV: {error}")

elif page == "Đánh giá mô hình":
    st.markdown('<div class="main-title">Đánh giá mô hình học máy</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">So sánh hiệu quả các thuật toán trên tập kiểm tra theo các chỉ số phân loại.</div>', unsafe_allow_html=True)
    if metrics.empty:
        st.warning("Chưa có báo cáo. Hãy chạy lệnh: python -m src.train")
    else:
        st.markdown("#### Bảng kết quả thực nghiệm")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        available = [x for x in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"] if x in metrics.columns]
        if available and "Mô hình" in metrics.columns:
            st.markdown("#### So sánh chỉ số")
            selected = st.multiselect("Chọn chỉ số muốn hiển thị", available, default=[x for x in ["F1-score", "Recall", "ROC-AUC"] if x in available])
            if selected:
                chart = metrics.set_index("Mô hình")[selected].apply(pd.to_numeric, errors="coerce")
                st.bar_chart(chart, use_container_width=True)
        reports = ROOT / "reports"
        left, right = st.columns(2)
        with left:
            image = reports / "confusion_matrices.png"
            if image.exists(): st.image(str(image), caption="Ma trận nhầm lẫn", use_container_width=True)
        with right:
            image = reports / "roc_curves.png"
            if image.exists(): st.image(str(image), caption="Đường cong ROC", use_container_width=True)

st.markdown("---")
st.caption("MachineCare · Hệ thống hỗ trợ bảo trì dự đoán. Kết quả là ước lượng từ mô hình học máy, cần kết hợp kiểm tra thực tế.")
