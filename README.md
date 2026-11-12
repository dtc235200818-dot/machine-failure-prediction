# machine-failure-prediction

Dự đoán hỏng hóc máy phục vụ bảo trì dựa trên dữ liệu cảm biến bằng học máy (Python).

**Phạm vi mô hình:** đồ án tập trung so sánh đúng hai thuật toán dễ triển khai:
- **Logistic Regression:** mô hình cơ sở, dễ giải thích; dùng trọng số lớp để chú ý hơn đến mẫu máy hỏng.
- **Random Forest:** mô hình tổ hợp nhiều cây quyết định, có thể học quan hệ phi tuyến giữa các thông số cảm biến; dùng `balanced_subsample` để giảm ảnh hưởng của mất cân bằng lớp.

Dữ liệu: **AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository), 10.000 mẫu, nhãn `Machine failure` có tỷ lệ dương khoảng 3,39%.
Giấy phép dữ liệu: CC BY 4.0. Trích dẫn: *AI4I 2020 Predictive Maintenance Dataset (2020), UCI Machine Learning Repository, https://doi.org/10.24432/C5HS5C*.

## Cấu trúc

```
├── data/ai4i2020.csv
├── src/
│   ├── preprocess.py      # tiền xử lý, tạo đặc trưng, chia train/test
│   ├── train.py           # huấn luyện và so sánh Logistic Regression, Random Forest
│   ├── evaluate.py        # Precision, Recall, F1, ROC-AUC, ma trận nhầm lẫn, ROC
│   └── predict.py         # dự đoán và mức cảnh báo
├── app.py                 # giao diện Streamlit
├── reports/               # bảng chỉ số và biểu đồ đánh giá
├── models/                # mô hình tốt nhất đã huấn luyện
└── requirements.txt
```

## Cài đặt và chạy

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

python -m src.train              # huấn luyện hai mô hình, tạo báo cáo và lưu mô hình tốt nhất
python -m src.predict            # thử dự đoán một mẫu
streamlit run app.py             # mở giao diện web
```

## Quy trình xử lý dữ liệu

1. Đọc dữ liệu cảm biến và chuẩn hóa tên cột.
2. Loại `UDI`, `Product ID` và các cột dạng hỏng `TWF, HDF, PWF, OSF, RNF` để tránh rò rỉ nhãn. Nhãn `Machine failure` được tạo từ các dạng hỏng này, vì vậy không được dùng chúng làm đầu vào.
3. Tạo ba đặc trưng dẫn xuất: chênh lệch nhiệt độ, công suất cơ và tích độ mòn với mô-men xoắn.
4. Điền giá trị thiếu, chuẩn hóa dữ liệu số và mã hóa biến loại `Type` bằng pipeline tiền xử lý.
5. Chia dữ liệu 80% train / 20% test theo phương pháp phân tầng (`stratify`) để giữ tỷ lệ lớp.
6. Xử lý mất cân bằng bằng `class_weight="balanced"` cho Logistic Regression và `class_weight="balanced_subsample"` cho Random Forest. Không áp dụng SMOTE để quy trình đơn giản, tránh tạo mẫu tổng hợp và giữ nguyên phân phối dữ liệu kiểm tra.
7. Dùng Stratified 5-fold cross-validation trên tập train. Chọn mô hình theo F1-score trung bình; sau đó mới đánh giá trên tập test độc lập.

## Hai mô hình được sử dụng

| Nội dung | Logistic Regression | Random Forest |
|---|---|---|
| Vai trò | Mô hình cơ sở để đối chiếu | Mô hình phi tuyến để so sánh |
| Cách hoạt động | Ước lượng xác suất hỏng từ tổ hợp tuyến tính của đặc trưng | Tổng hợp dự đoán từ nhiều cây quyết định |
| Xử lý mất cân bằng | `class_weight="balanced"` | `class_weight="balanced_subsample"` |
| Ưu điểm | Nhanh, dễ triển khai và giải thích | Học được quan hệ phi tuyến, ít cần giả định về dữ liệu |
| Hạn chế | Có thể chưa mô tả tốt quan hệ phức tạp | Khó giải thích hơn và tốn tài nguyên hơn Logistic Regression |

## Đánh giá

Sử dụng các chỉ số:
- **Precision:** trong các mẫu được dự đoán hỏng, tỷ lệ dự đoán đúng.
- **Recall:** trong các máy thực sự hỏng, tỷ lệ được phát hiện.
- **F1-score:** trung bình điều hòa Precision và Recall; phù hợp khi dữ liệu mất cân bằng.
- **ROC-AUC:** khả năng phân biệt hai lớp ở nhiều ngưỡng.
- **Confusion Matrix:** cho biết số dự đoán đúng/sai ở từng lớp.

Chạy `python -m src.train` để tạo lại các kết quả mới trong `reports/metrics_cv_train.csv`, `reports/metrics_test.csv`, `reports/confusion_matrices.png` và `reports/roc_curves.png`. Không dùng số liệu cũ trong README thay cho kết quả của lần chạy mới.

## Mức cảnh báo trong ứng dụng

Xác suất hỏng được chuyển thành ba mức trong `src/predict.py`: dưới 30% là Bình thường, từ 30% là Sắp hỏng, từ 60% là Hỏng (nguy cơ cao). Ngưỡng nằm ở `THRESHOLD_WARN` và `THRESHOLD_FAIL`, có thể điều chỉnh sau khi đánh giá thực nghiệm.

## Lưu ý về dữ liệu

AI4I 2020 là dữ liệu tổng hợp mô phỏng; mỗi dòng là một mẫu độc lập, không phải chuỗi thời gian. Bộ dữ liệu không có độ rung, áp suất, dòng điện hoặc thời gian sử dụng còn lại (RUL). Vì vậy đồ án chỉ kết luận trong phạm vi các thuộc tính cảm biến có sẵn, không khẳng định dự đoán được thời điểm chính xác máy sẽ hỏng.
