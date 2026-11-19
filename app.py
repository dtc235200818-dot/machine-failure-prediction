"""MachineCare - giao dien Streamlit cho du doan hong hoc may."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import MESSAGES, THRESHOLD_FAIL, THRESHOLD_WARN, load_model, predict_df
from src.preprocess import DATA_PATH, ROOT

st.set_page_config(
    page_title="MachineCare | Predictive Maintenance",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = ROOT / "data" / "prediction_history.db"
MODEL_OPTIONS = ["Logistic Regression", "Random Forest"]
MODEL_LABELS = {
    "Logistic Regression": "Logistic Regression",
    "Random Forest": "Random Forest",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #f4f7fb; color: #172b4d; }
[data-testid="stHeader"] { background: rgba(244,247,251,.92); }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#10233f 0%,#173b63 100%); }
[data-testid="stSidebar"] * { color: #f4f8ff !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  padding: .62rem .72rem; border-radius: 10px; margin: .15rem 0;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(255,255,255,.10); }
.main-title { font-size: 2.05rem; line-height: 1.2; font-weight: 800; color: #142b49; letter-spacing: -.04em; }
.sub-title { color: #64748b; margin: .35rem 0 1.35rem; font-size: .98rem; }
.eyebrow { color: #2875d7; font-size: .72rem; font-weight: 800; letter-spacing: .16em; margin-bottom: .4rem; }
.section-title { color: #183454; font-size: 1.18rem; font-weight: 750; margin: .5rem 0 .8rem; }
div[data-testid="stMetric"] {
  background: #fff; border: 1px solid #e5edf6; padding: 1rem 1.1rem;
  border-radius: 15px; box-shadow: 0 5px 18px rgba(16,42,76,.045);
}
div[data-testid="stMetricLabel"] { color: #667991; }
div[data-testid="stMetricValue"] { color: #142b49; font-weight: 800; }
div.stButton > button[kind="primary"], button[kind="primaryFormSubmit"] {
  background: #1769d2; border: 0; border-radius: 10px; font-weight: 700;
  min-height: 2.7rem;
}
div.stButton > button { border-radius: 10px; }
div[data-testid="stDataFrame"] { border: 1px solid #e5edf6; border-radius: 12px; overflow: hidden; }
.panel { background: #fff; border: 1px solid #e5edf6; border-radius: 16px; padding: 1.1rem 1.25rem; }
.panel-label { color: #728198; font-size: .72rem; font-weight: 800; letter-spacing: .09em; }
.panel-value { color: #183454; font-size: 1.22rem; font-weight: 800; margin-top: .3rem; }
.small-muted { color: #718096; font-size: .84rem; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def init_history():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model_name TEXT NOT NULL,
                input_data TEXT NOT NULL,
                probability REAL NOT NULL,
                risk_level TEXT NOT NULL
            )
        """)
        conn.commit()


def save_history(input_df: pd.DataFrame, result_df: pd.DataFrame, model_name: str):
    init_history()
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    records = []
    for index, row in input_df.reset_index(drop=True).iterrows():
        result = result_df.iloc[index]
        records.append((
            now,
            model_name,
            json.dumps(row.to_dict(), ensure_ascii=False, default=str),
            float(result["xac_suat_hong"]),
            str(result["muc_canh_bao"]),
        ))
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            "INSERT INTO predictions (created_at, model_name, input_data, probability, risk_level) VALUES (?, ?, ?, ?, ?)",
            records,
        )
        conn.commit()


def read_history():
    init_history()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT id, created_at AS 'Thời gian', model_name AS 'Mô hình', "
            "probability AS 'Xác suất hỏng', risk_level AS 'Mức cảnh báo', "
            "input_data AS 'Thông số đầu vào' FROM predictions ORDER BY id DESC",
            conn,
        )


@st.cache_data
def get_dataset():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


@st.cache_data
def get_metrics():
    path = ROOT / "reports" / "metrics_test.csv"
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


@st.cache_data
def get_model_info():
    path = ROOT / "models" / "model_info.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def cached_model(model_name: str):
    return load_model(model_name)


def page_header(title: str, subtitle: str):
    st.markdown('<div class="eyebrow">MACHINECARE / PREDICTIVE ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">{subtitle}</div>', unsafe_allow_html=True)


def model_selector(key: str):
    return st.selectbox("Mô hình dự đoán", MODEL_OPTIONS, key=key,
                        help="Chọn một trong hai thuật toán chính của đồ án.")


init_history()
dataset = get_dataset()
metrics = get_metrics()
model_info = get_model_info()

with st.sidebar:
    st.markdown("# 🏭 MachineCare")
    st.caption("PREDICTIVE MAINTENANCE PLATFORM")
    st.markdown("---")
    page = st.radio(
        "ĐIỀU HƯỚNG",
        ["Tổng quan", "Dự đoán một máy", "Dự đoán hàng loạt", "So sánh mô hình", "Lịch sử dự đoán", "Thông tin dự án"],
        label_visibility="visible",
    )
    st.markdown("---")
    st.markdown("**MÔ HÌNH CHÍNH**")
    st.caption("Logistic Regression")
    st.caption("Random Forest")
    st.markdown("---")
    if (ROOT / "models" / "logistic_regression.joblib").exists() and (ROOT / "models" / "random_forest.joblib").exists():
        st.success("Đã có đủ 2 mô hình")
    else:
        st.warning("Cần huấn luyện lại mô hình")
    st.caption("Dataset · AI4I 2020")

if page == "Tổng quan":
    page_header("Trung tâm giám sát thiết bị", "Tổng quan dữ liệu cảm biến, tình trạng máy và hiệu quả mô hình học máy.")
    total = len(dataset)
    failures = int(dataset["Machine failure"].sum()) if total and "Machine failure" in dataset else 0
    failure_rate = failures / total if total else 0
    history = read_history()
    risky = int(history["Mức cảnh báo"].isin(["Sắp hỏng", "Hỏng"]).sum()) if not history.empty else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng mẫu dữ liệu", f"{total:,}" if total else "—")
    c2.metric("Mẫu hỏng trong dataset", f"{failures:,}" if total else "—")
    c3.metric("Tỷ lệ hỏng trong dataset", f"{failure_rate:.2%}" if total else "—")
    c4.metric("Dự đoán cần chú ý", f"{risky:,}")
    st.markdown('<div class="section-title">Tình trạng dữ liệu và hệ thống</div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="panel-label">PHÂN BỐ NHÃN TRONG DATASET</div>', unsafe_allow_html=True)
        if total and "Machine failure" in dataset:
            labels = dataset["Machine failure"].map({0: "Bình thường", 1: "Hỏng"}).value_counts()
            st.bar_chart(labels, color="#2875d7", use_container_width=True)
        else:
            st.info("Chưa tìm thấy data/ai4i2020.csv.")
    with right:
        st.markdown('<div class="panel-label">HAI THUẬT TOÁN CỐT LÕI</div>', unsafe_allow_html=True)
        st.markdown("**Logistic Regression**  
Mô hình tuyến tính, dễ giải thích và làm mốc so sánh.")
        st.markdown("**Random Forest**  
Mô hình tổ hợp cây quyết định, học được quan hệ phi tuyến.")
        st.markdown("---")
        st.markdown(f"**Ngưỡng cảnh báo:** {THRESHOLD_WARN:.0%} · **Nguy cơ cao:** {THRESHOLD_FAIL:.0%}")
    st.markdown('<div class="section-title">Dự đoán gần đây</div>', unsafe_allow_html=True)
    if history.empty:
        st.info("Chưa có lịch sử dự đoán. Hãy mở trang Dự đoán một máy để bắt đầu.")
    else:
        st.dataframe(history.drop(columns=["Thông số đầu vào"]).head(8), use_container_width=True, hide_index=True)

elif page == "Dự đoán một máy":
    page_header("Phân tích một thiết bị", "Nhập thông số cảm biến và chọn thuật toán để ước lượng xác suất hỏng.")
    chosen_model = model_selector("single_model")
    with st.form("single_prediction"):
        st.markdown("#### 01 · Loại máy và nhiệt độ")
        c1, c2, c3 = st.columns(3)
        with c1:
            type_ = st.selectbox("Loại sản phẩm", ["L", "M", "H"], index=1)
        with c2:
            air = st.number_input("Nhiệt độ không khí (K)", 290.0, 310.0, 300.0, 0.1)
        with c3:
            process = st.number_input("Nhiệt độ quá trình (K)", 300.0, 320.0, 310.0, 0.1)
        st.markdown("#### 02 · Thông số vận hành")
        c4, c5, c6 = st.columns(3)
        with c4:
            rpm = st.number_input("Tốc độ quay (rpm)", 1000, 3000, 1500, 10)
        with c5:
            torque = st.number_input("Mô-men xoắn (Nm)", 0.0, 100.0, 40.0, 0.1)
        with c6:
            wear = st.number_input("Độ mòn dụng cụ (phút)", 0, 300, 100, 1)
        submitted = st.form_submit_button("Phân tích nguy cơ hỏng", type="primary", use_container_width=True)
    if submitted:
        input_df = pd.DataFrame([{
            "type": type_, "air_temp": air, "process_temp": process,
            "rpm": rpm, "torque": torque, "tool_wear": wear,
        }])
        try:
            estimator = cached_model(chosen_model)
            result_df = predict_df(input_df, estimator)
            save_history(input_df, result_df, chosen_model)
            result = result_df.iloc[0]
            probability = float(result["xac_suat_hong"])
            level = str(result["muc_canh_bao"])
            st.markdown("### Kết quả phân tích")
            a, b = st.columns([1, 1.6])
            with a:
                st.metric("Xác suất hỏng", f"{probability:.1%}")
                st.progress(min(max(probability, 0.0), 1.0))
                st.caption(f"Thuật toán: {chosen_model}")
            with b:
                if level == "Bình thường":
                    st.success(f"🟢 {level}")
                elif level == "Sắp hỏng":
                    st.warning(f"🟠 {level}")
                else:
                    st.error(f"🔴 {level}")
                st.write(MESSAGES[level])
            st.caption(f"Quy ước trong ứng dụng: dưới {THRESHOLD_WARN:.0%} là Bình thường; từ {THRESHOLD_WARN:.0%} là Sắp hỏng; từ {THRESHOLD_FAIL:.0%} là Hỏng/nguy cơ cao.")
        except FileNotFoundError as error:
            st.error(str(error))
            st.info("Mở Terminal tại thư mục dự án và chạy: python -m src.train")
        except Exception as error:
            st.error(f"Không thể dự đoán: {error}")

elif page == "Dự đoán hàng loạt":
    page_header("Dự đoán hàng loạt", "Tải dữ liệu CSV, phân tích nhiều thiết bị và xuất báo cáo kết quả.")
    chosen_model = model_selector("batch_model")
    st.info("CSV cần có Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min] hoặc các tên cột rút gọn tương ứng.")
    uploaded = st.file_uploader("Chọn tệp dữ liệu CSV", type=["csv"])
    if uploaded is not None:
        try:
            input_df = pd.read_csv(uploaded, encoding="utf-8-sig")
            st.write(f"Đã tải lên **{len(input_df):,}** dòng dữ liệu.")
            if st.button("Chạy dự đoán cho tệp này", type="primary"):
                estimator = cached_model(chosen_model)
                result_df = predict_df(input_df, estimator)
                save_history(input_df, result_df, chosen_model)
                normal = int((result_df["muc_canh_bao"] == "Bình thường").sum())
                warning = int((result_df["muc_canh_bao"] == "Sắp hỏng").sum())
                high = int((result_df["muc_canh_bao"] == "Hỏng").sum())
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Tổng thiết bị", f"{len(result_df):,}")
                m2.metric("Bình thường", f"{normal:,}")
                m3.metric("Cần theo dõi", f"{warning:,}")
                m4.metric("Nguy cơ cao", f"{high:,}")
                st.dataframe(result_df, use_container_width=True, height=420, hide_index=True)
                st.download_button(
                    "Tải kết quả CSV",
                    result_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="machinecare_predictions.csv",
                    mime="text/csv",
                    type="primary",
                )
        except FileNotFoundError as error:
            st.error(str(error))
            st.info("Chạy python -m src.train để tạo các mô hình.")
        except Exception as error:
            st.error(f"Không xử lý được CSV: {error}")

elif page == "So sánh mô hình":
    page_header("Đánh giá và so sánh thuật toán", "Đối chiếu Logistic Regression và Random Forest trên cùng tập kiểm tra.")
    if metrics.empty:
        st.warning("Chưa có báo cáo đánh giá. Hãy chạy python -m src.train.")
    else:
        if "Mô hình" not in metrics.columns and len(metrics.columns):
            metrics = metrics.rename(columns={metrics.columns[0]: "Mô hình"})
        st.markdown("#### Bảng chỉ số trên tập kiểm tra")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        available = [m for m in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"] if m in metrics.columns]
        if available and "Mô hình" in metrics.columns:
            chosen_metrics = st.multiselect("Chọn chỉ số hiển thị", available, default=[m for m in ["F1-score", "Recall", "ROC-AUC"] if m in available])
            if chosen_metrics:
                chart = metrics.set_index("Mô hình")[chosen_metrics].apply(pd.to_numeric, errors="coerce")
                st.bar_chart(chart, use_container_width=True)
        reports = ROOT / "reports"
        left, right = st.columns(2)
        cm = reports / "confusion_matrices.png"
        roc = reports / "roc_curves.png"
        with left:
            if cm.exists():
                st.image(str(cm), caption="Ma trận nhầm lẫn trên tập kiểm tra", use_container_width=True)
        with right:
            if roc.exists():
                st.image(str(roc), caption="Đường cong ROC", use_container_width=True)
        best = model_info.get("best_model", "Chưa xác định")
        st.info(f"Mô hình có F1-score cross-validation cao nhất: **{best}**. Việc lựa chọn dựa trên tập train, không chọn theo kết quả tập test.")

elif page == "Lịch sử dự đoán":
    page_header("Lịch sử phân tích", "Theo dõi các lần dự đoán đã lưu trong cơ sở dữ liệu SQLite.")
    history = read_history()
    if history.empty:
        st.info("Chưa có dữ liệu lịch sử.")
    else:
        f1, f2 = st.columns(2)
        with f1:
            model_filter = st.selectbox("Lọc theo mô hình", ["Tất cả"] + MODEL_OPTIONS)
        with f2:
            risk_filter = st.selectbox("Lọc theo cảnh báo", ["Tất cả", "Bình thường", "Sắp hỏng", "Hỏng"])
        filtered = history.copy()
        if model_filter != "Tất cả":
            filtered = filtered[filtered["Mô hình"] == model_filter]
        if risk_filter != "Tất cả":
            filtered = filtered[filtered["Mức cảnh báo"] == risk_filter]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button("Xuất lịch sử CSV", filtered.to_csv(index=False).encode("utf-8-sig"), "prediction_history.csv", "text/csv")
        if st.button("Xóa toàn bộ lịch sử", type="secondary"):
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("DELETE FROM predictions")
                conn.commit()
            st.rerun()

else:
    page_header("Thông tin dự án", "MachineCare là ứng dụng học máy hỗ trợ nhận diện sớm nguy cơ hỏng hóc thiết bị.")
    st.markdown("### Mục tiêu")
    st.write("Phân tích các thông số cảm biến vận hành để ước lượng xác suất máy gặp sự cố, hỗ trợ ưu tiên kiểm tra và bảo trì.")
    st.markdown("### Hai mô hình chính")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Logistic Regression")
        st.write("Mô hình phân loại tuyến tính, ước lượng xác suất hỏng. Ưu điểm là tốc độ nhanh và dễ giải thích.")
    with right:
        st.markdown("#### Random Forest")
        st.write("Mô hình tổ hợp nhiều cây quyết định, có khả năng học các quan hệ phi tuyến giữa thông số cảm biến.")
    st.markdown("### Dữ liệu và giới hạn")
    st.write("Dự án sử dụng AI4I 2020 Predictive Maintenance Dataset. Đây là dữ liệu mô phỏng; kết quả chỉ là ước lượng từ các thuộc tính có trong bộ dữ liệu, không phải chẩn đoán công nghiệp hoặc dự báo chính xác thời điểm máy hỏng.")
    st.markdown("### Quy trình")
    st.write("1. Tiền xử lý và tạo đặc trưng.  2. Huấn luyện hai thuật toán.  3. Đánh giá bằng 5-fold cross-validation và tập test độc lập.  4. Dự đoán, phân loại mức cảnh báo và lưu lịch sử.")

st.markdown("---")
st.caption("MachineCare · Hệ thống hỗ trợ bảo trì dự đoán · Kết quả cần được đối chiếu với kiểm tra kỹ thuật thực tế.")
