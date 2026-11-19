# Machine Failure Prediction — MachineCare

Ứng dụng Streamlit hỗ trợ dự đoán nguy cơ hỏng hóc máy từ dữ liệu cảm biến AI4I 2020.

## Hai mô hình chính

Dự án tập trung vào đúng hai thuật toán:
- **Logistic Regression** — mô hình tuyến tính, dễ giải thích, dùng làm mô hình cơ sở.
- **Random Forest** — mô hình tổ hợp cây quyết định, có thể học các quan hệ phi tuyến.

Cả hai mô hình được huấn luyện và lưu riêng để người dùng chủ động lựa chọn khi dự đoán. Dự án vẫn lưu thêm `best_model.joblib`, được chọn theo F1-score trung bình của 5-fold cross-validation trên tập train.

## Tính năng

1. **Tổng quan:** số mẫu dữ liệu, số mẫu hỏng, tỷ lệ hỏng và các dự đoán gần đây.
2. **Dự đoán một máy:** nhập sáu thông số cảm biến, chọn một trong hai mô hình và xem xác suất/mức cảnh báo/khuyến nghị.
3. **Dự đoán hàng loạt:** tải CSV, phân tích nhiều dòng và tải kết quả xuống.
4. **So sánh mô hình:** Accuracy, Precision, Recall, F1-score, ROC-AUC, ma trận nhầm lẫn và ROC.
5. **Lịch sử dự đoán:** lưu bằng SQLite, lọc theo mô hình/mức cảnh báo và xuất CSV.
6. **Thông tin dự án:** giới thiệu thuật toán, dữ liệu và giới hạn sử dụng.

## Cài đặt và chạy trên GitHub Codespaces

Mở Terminal tại thư mục gốc của repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
streamlit run app.py --server.address 0.0.0.0
```

Khi Codespaces hỏi về cổng 8501, chọn **Open in Browser**.

Windows PowerShell dùng lệnh kích hoạt:

```powershell
.venv\Scripts\Activate.ps1
```

## Cấu trúc chính

```
app.py
src/
  preprocess.py
  train.py
  predict.py
  evaluate.py
data/
  ai4i2020.csv
models/
  logistic_regression.joblib
  random_forest.joblib
  best_model.joblib
  model_info.json
reports/
  metrics_cv_train.csv
  metrics_test.csv
  confusion_matrices.png
  roc_curves.png
  prediction_history.db
requirements.txt
```

## Quy trình học máy

- Chuẩn hóa tên cột, loại bỏ các trường gây rò rỉ nhãn.
- Tạo đặc trưng dẫn xuất từ nhiệt độ, tốc độ quay, mô-men xoắn và độ mòn.
- Impute dữ liệu thiếu, chuẩn hóa dữ liệu số và one-hot encode loại máy.
- Chia train/test có stratify và đánh giá chéo 5-fold trên tập train.
- Đánh giá trên tập test độc lập; không dùng test để lựa chọn mô hình.

## Ngưỡng cảnh báo

- Dưới 30%: Bình thường.
- Từ 30% đến dưới 60%: Sắp hỏng.
- Từ 60%: Hỏng/nguy cơ cao.

Đây là ngưỡng minh họa có thể hiệu chỉnh theo yêu cầu vận hành và đánh giá thực nghiệm, không phải ngưỡng an toàn được chứng nhận.

## Giới hạn dữ liệu

AI4I 2020 là bộ dữ liệu mô phỏng. Ứng dụng chỉ ước lượng nguy cơ dựa trên các thuộc tính có trong dữ liệu; không dự báo chính xác thời điểm hỏng, không tính RUL và không thay thế kiểm tra kỹ thuật thực tế.

Nguồn dữ liệu: AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository, DOI: https://doi.org/10.24432/C5HS5C. License: CC BY 4.0.
