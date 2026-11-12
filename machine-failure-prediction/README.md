# machine-failure-prediction

Dự đoán hỏng hóc máy phục vụ bảo trì dựa trên dữ liệu cảm biến (học máy, Python).
Dự án có hai mô hình chính: **Logistic Regression** và **Random Forest**, kèm giao diện web nhiều trang bằng Streamlit.

Dữ liệu: **AI4I 2020 Predictive Maintenance Dataset** (UCI), 10.000 mẫu, 6 thuộc tính đầu vào, nhãn `Machine failure`
(tỷ lệ hỏng khoảng 3,39%, dữ liệu mất cân bằng). Giấy phép CC BY 4.0. Trích dẫn: *AI4I 2020 Predictive Maintenance Dataset (2020),
UCI Machine Learning Repository, https://doi.org/10.24432/C5HS5C*.

## Cấu trúc

```
├── app.py                 # điểm vào của ứng dụng, khai báo các trang
├── views/                 # mỗi trang là một file nhỏ
│   ├── home.py            #   Tổng quan
│   ├── predict.py         #   Dự đoán cho một máy
│   ├── batch.py           #   Dự đoán hàng loạt từ file CSV
│   ├── evaluation.py      #   Đánh giá mô hình
│   └── data.py            #   Giới thiệu dữ liệu
├── ui/                    # thành phần giao diện dùng chung (CSS, thẻ cảnh báo, tải mô hình)
├── src/
│   ├── config.py          # tên mô hình, nhãn, khoảng nhập, mẫu thử
│   ├── preprocess.py      # đọc dữ liệu, tạo đặc trưng, chuẩn hóa, chia train/test  (mục 2.2)
│   ├── models.py          # thiết kế Logistic Regression và Random Forest            (mục 2.3)
│   ├── train.py           # tinh chỉnh tham số, chọn ngưỡng, đánh giá, lưu mô hình   (mục 2.3, 3.3, 3.4)
│   ├── evaluate.py        # độ đo và biểu đồ
│   └── predict.py         # dự đoán, mức cảnh báo, giải thích                        (mục 2.4)
├── data/ai4i2020.csv
├── reports/               # bảng độ đo (.csv) và biểu đồ (.png)
├── models/                # mô hình đã huấn luyện (.joblib), thông tin mô hình (.json)
└── tests/ui_smoke_test.py # kiểm tra nhanh các trang giao diện
```

## Cài đặt và chạy

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train              # huấn luyện hai mô hình, lưu models/ và reports/
streamlit run app.py             # mở giao diện web
python tests/ui_smoke_test.py    # (tùy chọn) kiểm tra nhanh các trang
```

## Thiết kế hai mô hình

Cả hai là Pipeline gồm tiền xử lý và bộ phân loại, tinh chỉnh bằng GridSearchCV (kiểm định chéo 5 phần, tối ưu F1).

| | Logistic Regression | Random Forest |
|---|---|---|
| Tiền xử lý | Chuẩn hóa, thêm số hạng bậc 2 và tích chéo giữa các đặc trưng số | Chuẩn hóa |
| Xử lý mất cân bằng | `class_weight="balanced"` | `class_weight="balanced_subsample"` |
| Tham số tìm kiếm | `C` | số cây, độ sâu, số mẫu tối thiểu ở lá, số đặc trưng mỗi lần chia |
| Vai trò | Nhẹ, nhanh, bắt được nhiều ca hỏng | Chính xác nhất |

Ngưỡng quyết định của mỗi mô hình được chọn bằng F2 (ưu tiên Recall) trên dự đoán out-of-fold của tập train,
vì bỏ sót một máy sắp hỏng tốn kém hơn một cảnh báo sai. Tập test không được dùng để chọn tham số hay ngưỡng.

## Quy trình xử lý

1. Bỏ cột `UDI`, `Product ID`.
2. Bỏ các cột dạng hỏng `TWF, HDF, PWF, OSF, RNF`. Nhãn `Machine failure = 1` khi có ít nhất một dạng hỏng nên nếu giữ lại
   mô hình sẽ biết trước đáp án (rò rỉ nhãn).
3. Tạo 3 đặc trưng dẫn xuất: chênh lệch nhiệt độ, công suất cơ, độ mòn x mô-men.
4. Xử lý giá trị thiếu, chuẩn hóa, mã hóa one-hot cho `Type`.
5. Chia 80% train, 20% test theo tỷ lệ lớp (`stratify`).

## Kết quả (tập test, 2.000 mẫu, seed = 42)

| Mô hình | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9530 | 0.4000 | 0.7647 | 0.5253 | 0.9700 |
| Random Forest | 0.9930 | 0.9655 | 0.8235 | 0.8889 | 0.9716 |

Random Forest cho F1 cao hơn rõ rệt. Logistic Regression bắt được gần ba phần tư ca hỏng nhưng có nhiều cảnh báo sai hơn.
Do chỉ khoảng 3,4% mẫu là hỏng nên cần xem Recall và F1, không chỉ Accuracy.
Kết quả có thể thay đổi nhẹ giữa các phiên bản thư viện.

## Mức cảnh báo

Với ngưỡng `t` của mô hình: xác suất từ `t` là **Hỏng**, từ `t/2` là **Sắp hỏng**, còn lại là **Bình thường**.
Trang Dự đoán còn cho biết từng thông số đang làm tăng nguy cơ bao nhiêu, bằng cách đưa thông số đó về giá trị trung vị rồi xem xác suất giảm bao nhiêu.

## Lưu ý

- AI4I 2020 là dữ liệu tổng hợp (mô phỏng), mỗi dòng là một mẫu độc lập, không phải chuỗi thời gian.
- Dữ liệu không có độ rung, áp suất, dòng điện và không có thời gian sử dụng còn lại (RUL).
