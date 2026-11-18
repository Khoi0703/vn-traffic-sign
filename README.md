# 📊 Báo cáo Thống kê & Đánh giá Mô hình: Nhận diện Biển báo Giao thông Việt Nam (YOLO11)

Tài liệu tổng hợp toàn bộ số liệu thực nghiệm, thống kê hiệu suất mô hình, phân phối độ chính xác trên 52 lớp biển báo giao thông Việt Nam (`maitam/vietnamese-traffic-signs`) và benchmark phần cứng.

---

## 1. Tổng quan Cấu hình & Dữ liệu Thực nghiệm

| Hạng mục | Thông số thống kê |
| :--- | :--- |
| **Kiến trúc mô hình** | `YOLO11n` (101 layers, `2,609,444` tham số, `6.5 GFLOPs` @ 640 / `14.8 GFLOPs` @ 960) |
| **Số lượng lớp (Classes)** | `52` loại biển báo giao thông Việt Nam |
| **Tập Validation** | `324` ảnh \| `852` đối tượng (trung bình `2.63` biển/ảnh) |
| **Tập Test (Độc lập)** | `320` ảnh \| `823` đối tượng (trung bình `2.57` biển/ảnh, `46/52` lớp xuất hiện) |
| **Chống rò rỉ dữ liệu** | Gom nhóm cụm khung hình trùng lặp bằng `pHash` trước khi chia tập |
| **Phần cứng huấn luyện** | NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM) — `40 epochs` trong `0.758 giờ` (~45.5 phút) |
| **Phần cứng đo CPU** | AMD Ryzen 5 7535HS with Radeon Graphics |

---

## 2. Thống kê Độ chính xác Tổng quát (Validation vs. Test)

| Tập dữ liệu | Images | Instances | Precision ($P$) | Recall ($R$) | F1-Score (Max) | mAP@50 | mAP@50-95 | GPU Inference | GPU FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation** | 324 | 852 | 0.9360 | 0.8990 | — | 0.9750 | 0.7480 | 4.1 ms | ~243.9 |
| **Test** | 320 | 823 | **0.9408** | **0.9538** | **0.94** (`@0.591`) | **0.9820** | **0.7569** | 7.2 ms | ~138.9 |
| **Chênh lệch (Test - Val)** | -4 | -29 | `+0.0048` | `+0.0548` | — | `+0.0070` | `+0.0089` | +3.1 ms | -105.0 |

### Các điểm ngưỡng thống kê trên tập Test (`reports/eval_test/`)
* **Đỉnh F1-Score:** `0.94` tại ngưỡng tin cậy **`conf = 0.591`** (khuyến nghị dùng `conf = 0.50 – 0.60` khi suy luận).
* **Đỉnh Precision:** `1.00` tại **`conf = 1.000`**.
* **Đỉnh Recall:** `0.99` tại **`conf = 0.000`**.
* **Diện tích dưới đường cong PR (mAP@50):** **`0.982`**.

| Precision-Recall Curve | F1-Confidence Curve |
| :---: | :---: |
| ![PR Curve](reports/eval_test/BoxPR_curve.png) | ![F1 Curve](reports/eval_test/BoxF1_curve.png) |

| Confusion Matrix (Số lượng tuyệt đối) | Confusion Matrix (Chuẩn hóa tỷ lệ) |
| :---: | :---: |
| ![Confusion Matrix](reports/eval_test/confusion_matrix.jpg) | ![Confusion Matrix Normalized](reports/eval_test/confusion_matrix_normalized.jpg) |

---

## 3. Phân phối Hiệu suất trên 46 Lớp Tập Test (`reports/per_class_test.csv`)

### 3.1. Thống kê mô tả theo chỉ số ( trên 46 lớp có mẫu ở tập Test)

| Chỉ số | Trung bình (Mean) | Trung vị (Median) | Thấp nhất (Min) | Cao nhất (Max) | Số lớp đạt $\ge 0.95$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Precision** | 0.938 | 0.954 | 0.628 (`Low Clearance`) | 1.000 (10 lớp) | **27 / 46** (58.7%) |
| **Recall** | 0.948 | 1.000 | 0.761 (`Speed limit 50km/h`) | 1.000 (28 lớp) | **31 / 46** (67.4%) |
| **AP@50** | 0.980 | 0.995 | 0.880 (`Speed limit 50km/h`) | 0.995 (32 lớp) | **41 / 46** (89.1%) |
| **AP@50-95** | 0.761 | 0.765 | 0.497 (`Equal-level Intersection`) | 0.904 (`No Left or Right Turn`) | — |

### 3.2. Phân nhóm lớp theo mức độ chính xác (AP@50)

| Phân nhóm hiệu suất | Ngưỡng AP@50 | Số lượng lớp | Tỷ lệ | Đại diện tiêu biểu |
| :--- | :---: | :---: | :---: | :--- |
| **Xuất sắc (Tier 1)** | $\ge 0.990$ | **35 lớp** | 76.1% | *No Left Turn (0.995), No Right Turn (0.995), No Stopping & No Parking (0.994), Height Limit (0.995)* |
| **Tốt (Tier 2)** | $0.950 - 0.989$ | **6 lớp** | 13.0% | *Pedestrian Crossing (0.989), Right Turn Only (0.989), No Parking (0.981), Slow Down (0.974)* |
| **Cần cải thiện (Tier 3)** | $< 0.950$ | **5 lớp** | 10.9% | *Lane Allocation (0.928), One-way street (0.913), No U-Turn & No Right Turn (0.913), No Trucks Allowed (0.880), Speed limit 50km/h (0.880)* |

### 3.3. Bảng chi tiết Top 10 lớp cao nhất & Top 10 lớp thấp nhất (theo AP@50-95 & AP@50)

**Top 10 lớp có độ chính xác định vị khung hình cao nhất (AP@50-95):**

| Hạng | ID | Tên biển báo | Instances | Precision | Recall | AP@50 | AP@50-95 |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `33` | No Left or Right Turn | 6 | 0.950 | 1.000 | 0.995 | **0.904** |
| 2 | `15` | Narrow Road on the Right | 2 | 0.872 | 1.000 | 0.995 | **0.895** |
| 3 | `32` | No Containers Allowed | 7 | 1.000 | 0.972 | 0.995 | **0.882** |
| 4 | `34` | No Straight and Right Turn | 6 | 0.940 | 1.000 | 0.995 | **0.879** |
| 5 | `17` | No U-Turn | 7 | 0.953 | 1.000 | 0.995 | **0.851** |
| 6 | `23` | No Two or Three-wheeled Vehicles | 43 | 1.000 | 0.944 | 0.995 | **0.833** |
| 7 | `30` | Road with Surveillance Camera | 9 | 0.976 | 1.000 | 0.995 | **0.830** |
| 8 | `47` | No U-Turn for Cars | 3 | 0.974 | 1.000 | 0.995 | **0.830** |
| 9 | `48` | Level Crossing with Barriers | 3 | 0.935 | 1.000 | 0.995 | **0.830** |
| 10 | `9` | Roundabout | 4 | 0.921 | 1.000 | 0.995 | **0.829** |

**Top 10 lớp có AP@50 thấp nhất (Nhóm ưu tiên bổ sung dữ liệu):**

| Hạng | ID | Tên biển báo | Instances | Precision | Recall | AP@50 | AP@50-95 | Nguyên nhân chính từ Confusion Matrix |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `38` | Speed limit (50km/h) | 7 | 1.000 | 0.761 | **0.880** | 0.600 | Bỏ sót biển nhỏ (`background`) & nhầm với *One-way street* |
| 2 | `14` | No Trucks Allowed | 11 | 0.907 | 0.909 | **0.880** | 0.650 | Nhầm với *No Containers Allowed* |
| 3 | `19` | No U-Turn and No Right Turn | 3 | 0.703 | 0.814 | **0.913** | 0.814 | Thiếu mẫu huấn luyện & kiểm thử (chỉ có 3 mẫu) |
| 4 | `45` | One-way street | 20 | 0.941 | 0.799 | **0.913** | 0.708 | Bỏ sót (`background`: 20% false negative) |
| 5 | `12` | Lane Allocation | 16 | 0.848 | 0.938 | **0.928** | 0.823 | Nhận diện nhầm từ nền (`background` false positive) |
| 6 | `41` | Speed limit (40km/h) | 45 | 0.961 | 0.911 | **0.958** | 0.765 | Sót biển ở khoảng cách xa |
| 7 | `39` | Speed limit (60km/h) | 40 | 0.994 | 0.950 | **0.972** | 0.791 | Sót một số mẫu kích thước nhỏ |
| 8 | `13` | Slow Down | 36 | 0.912 | 0.972 | **0.974** | 0.761 | Lẫn nhẹ với nền phức tạp |
| 9 | `46` | No Parking | 42 | 0.954 | 0.997 | **0.981** | 0.749 | Nhầm nhẹ với *No Stopping and No Parking* |
| 10 | `6` | Danger zone on the left | 14 | 0.930 | 0.857 | **0.986** | 0.638 | Sót biển ở góc khuất/xa |

---

## 4. Benchmark Tốc độ & Dung lượng Triển khai (`reports/benchmark.csv`)

Đo lường thực tế trên 50 ảnh tập Test với CPU **AMD Ryzen 5 7535HS** (Input shape `1x3x960x960`):

| Định dạng | Đường dẫn Artifact | Dung lượng | FPS (CPU) | Latency (ms/ảnh) | So với PyTorch | Khuyến nghị sử dụng |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **OpenVINO** | `weights/best_openvino_model/` | 10.5 MB | **17.46** | **57.27 ms** | **`+10.2%` FPS** | Triển khai thực tế trên CPU Intel/AMD |
| **PyTorch** | `weights/best.pt` | **5.5 MB** | 15.85 | 63.09 ms | Baseline (`1.0x`) | Huấn luyện & suy luận trên GPU CUDA |
| **ONNX** | `weights/best.onnx` | 10.4 MB | 12.61 | 79.30 ms | `-20.4%` FPS | Tích hợp đa ngôn ngữ / C++ / Web |

---

## 5. Cài đặt & Tái lập Kết quả (Reproducibility)

### Cài đặt nhanh
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"
pytest -q
```

### Các lệnh tái lập toàn bộ Pipeline & Báo cáo
```bash
vnsign-eda                                        # 1. Thống kê phân bố dữ liệu thô
vnsign-prepare                                    # 2. Lọc nhãn & chia tập chống rò rỉ (pHash)
vnsign-train --epochs 40 --name yolo11n_baseline  # 3. Huấn luyện mô hình
vnsign-eval --split test                          # 4. Xuất per_class_test.csv & biểu đồ
vnsign-export --formats onnx openvino             # 5. Export & xuất benchmark.csv
vnsign-predict --source 0 --conf 0.59 --show      # 6. Suy luận webcam + ByteTrack
streamlit run app/streamlit_app.py                # 7. Khởi chạy Web App
```