# machine-failure-prediction

Dự đoán hỏng hóc máy phục vụ bảo trì dựa trên dữ liệu cảm biến (học máy, Python).

Dữ liệu: **AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository), 10.000 mẫu, 6 thuộc tính đầu vào,
nhãn `Machine failure` (tỷ lệ hỏng khoảng 3,39%, dữ liệu mất cân bằng).
Giấy phép dữ liệu: CC BY 4.0. Trích dẫn: *AI4I 2020 Predictive Maintenance Dataset (2020), UCI Machine Learning Repository,
https://doi.org/10.24432/C5HS5C*.

## Cấu trúc

```
├── data/ai4i2020.csv      # dữ liệu gốc
├── src/
│   ├── preprocess.py      # đọc, tạo đặc trưng, chuẩn hóa, chia train/test   (mục 2.2)
│   ├── train.py           # huấn luyện, kiểm định chéo, so sánh mô hình      (mục 2.3, 3.3, 3.4)
│   ├── evaluate.py        # độ đo, ma trận nhầm lẫn, ROC
│   └── predict.py         # dự đoán và mức cảnh báo                          (mục 2.4)
├── app.py                 # giao diện Streamlit                              (mục 3.5)
├── reports/               # kết quả: bảng độ đo (.csv) và biểu đồ (.png)
├── models/                # mô hình đã huấn luyện (best_model.joblib)
└── requirements.txt
```

## Cài đặt và chạy

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train              # huấn luyện, in kết quả, lưu models/ và reports/
python -m src.predict            # chạy thử một dự đoán mẫu
streamlit run app.py             # mở giao diện web
```

## Quy trình xử lý

1. Bỏ cột `UDI`, `Product ID`.
2. Bỏ các cột dạng hỏng `TWF, HDF, PWF, OSF, RNF`. Nhãn `Machine failure = 1` khi có ít nhất một dạng hỏng,
   nên nếu giữ lại các cột này mô hình sẽ biết trước đáp án (rò rỉ nhãn) và kết quả cao giả tạo.
3. Tạo thêm 3 đặc trưng dẫn xuất: chênh lệch nhiệt độ, công suất cơ (mô-men x tốc độ quay), độ mòn x mô-men.
4. Xử lý giá trị thiếu (trung vị / giá trị phổ biến), chuẩn hóa số, one-hot cho `Type`.
5. Chia 80% train / 20% test, giữ nguyên tỷ lệ lớp (`stratify`).
6. Xử lý mất cân bằng bằng trọng số lớp (`class_weight`).
7. Kiểm định chéo 5 phần trên tập train, chọn mô hình theo F1 của kiểm định chéo (không nhìn tập test).

## Kết quả (tập test, 2.000 mẫu, seed = 42)

| Mô hình | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8585 | 0.1791 | 0.8824 | 0.2978 | 0.9385 |
| SVM (RBF) | 0.9245 | 0.2976 | 0.8971 | 0.4469 | 0.9719 |
| Random Forest | 0.9910 | 0.9630 | 0.7647 | 0.8525 | 0.9720 |
| Gradient Boosting | 0.9920 | 0.9194 | 0.8382 | 0.8769 | 0.9787 |

Mô hình được chọn theo F1 kiểm định chéo là Random Forest. Kết quả có thể thay đổi nhẹ giữa các phiên bản thư viện.
Do dữ liệu mất cân bằng nên cần xem Recall và F1, không chỉ Accuracy.
Nếu cài thêm `xgboost`, `train.py` tự thêm XGBoost vào bảng so sánh.

## Mức cảnh báo

Xác suất hỏng từ mô hình được đổi thành 3 mức trong `src/predict.py`:
dưới 30% là Bình thường, từ 30% là Sắp hỏng, từ 60% là Hỏng (nguy cơ cao). Có thể chỉnh `THRESHOLD_WARN`, `THRESHOLD_FAIL`.

## Lưu ý

- AI4I 2020 là dữ liệu tổng hợp (mô phỏng), mỗi dòng là một mẫu độc lập, không phải chuỗi thời gian, nên không dùng LSTM.
- Dữ liệu không có độ rung, áp suất hay dòng điện, và không có thời gian sử dụng còn lại (RUL).
