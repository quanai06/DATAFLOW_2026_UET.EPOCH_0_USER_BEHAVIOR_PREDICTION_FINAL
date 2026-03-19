<div align="center">

  <img src="dataflow.png" width="200" alt="logo" />

  # User Behavior Prediction — DataFlow 2026

  **Dự đoán 6 thuộc tính hành vi người dùng từ chuỗi `action_id` bằng GRU, LSTM, Transformer và chiến lược lai (Hybrid Ensemble).**

  ![Python](https://img.shields.io/badge/Python-3.12-red?logo=python&logoColor=white)
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
  ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-FF6F00?logo=tensorflow&logoColor=white)
  ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
  ![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
</div>

---

## Mục lục

1. [Tổng quan bài toán](#1-tổng-quan-bài-toán)
2. [Cấu trúc repo](#2-cấu-trúc-repo)
3. [Pipeline dữ liệu](#3-pipeline-dữ-liệu)
4. [Họ mô hình 1: GRU + Transformer (Hybrid)](#4-họ-mô-hình-1-gru--transformer-hybrid)
5. [Họ mô hình 2: Transformer only](#5-họ-mô-hình-2-transformer-only)
6. [Họ mô hình 3: LSTM + GRU (Hồi quy)](#6-họ-mô-hình-3-lstm--gru-hồi-quy)
7. [So sánh 3 họ mô hình](#7-so-sánh-3-họ-mô-hình)
8. [Feature Engineering](#8-feature-engineering)
9. [Loss Function & Metric](#9-loss-function--metric)
10. [Web App Demo](#10-web-app-demo)
11. [Hướng dẫn chạy](#11-hướng-dẫn-chạy)
12. [Kết quả & Hình ảnh phân tích](#12-kết-quả--hình-ảnh-phân-tích)
13. [Lưu ý khi tái lập](#13-lưu-ý-khi-tái-lập)

---

## 1. Tổng quan bài toán

**Đầu vào:** Chuỗi `action_id` (lên tới 66 bước, giá trị từ 1 đến ~30.000) biểu diễn hành vi thao tác của người dùng.

**Đầu ra:** Dự đoán 6 thuộc tính:

| Thuộc tính | Ý nghĩa | Miền giá trị |
|---|---|---|
| `attr_1`, `attr_4` | Tháng | 1 – 12 |
| `attr_2`, `attr_5` | Ngày | 1 – 31 |
| `attr_3`, `attr_6` | Factory / Capacity score | 0 – 99 |

**Dữ liệu:**

| Tập | Số mẫu |
|---|---|
| Train | ~9.300 |
| Validation | ~7.300 |
| Test | ~21.600 |

**Metric thi đấu:** Weighted MSE có chuẩn hóa:

```
Score = (1 / 6N) × Σ Wⱼ × (predⱼ/Mⱼ − targetⱼ/Mⱼ)²
```

Trong đó:
- `M = [12, 31, 99, 12, 31, 99]` — hằng số chuẩn hóa
- `W = [1, 1, 100, 1, 1, 100]` — trọng số (capacity bị phạt **100×**)

---

## 2. Cấu trúc repo

```text
.
├── README.md
├── requirements.txt
├── .env                              # Config: SEED, BATCH_SIZE, EPOCHS, ...
├── run.py                            # Khởi chạy backend + frontend
├── precompute_x_test.py              # Tiền xử lý offline cho app
├── predict_each.py                   # Inference nhanh 1 fold
│
├── data/
│   ├── data_raw/                     # Dữ liệu gốc cuộc thi
│   │   ├── X_train.csv, X_val.csv, X_test.csv
│   │   ├── Y_train.csv, Y_val.csv
│   │   └── sample_submission.csv
│   ├── data_processed/transformer/   # Tensor đã tiền xử lý (npy)
│   ├── precomputed_orders.parquet    # Cache prediction cho app
│   └── precomputed_dataset_summary.json
│
├── model/                            # Checkpoint đã train
│   ├── kfold/                        # Transformer K-Fold (3 fold)
│   │   └── transformer_f{0,1,2}.pth
│   └── lstm_gru/                     # LSTM/GRU models
│       ├── model_{0,1,2}_{lstm,gru}.keras
│       ├── scaler_full.pkl
│       └── hubs.pkl
│
├── scripts/
│   ├── pipeline_training.py          # Orchestrate toàn bộ training
│   ├── pipeline_ai.py                # Pipeline AI hỗ trợ
│   └── predict_test.py               # Ensemble inference Transformer
│
├── src/
│   ├── models/
│   │   ├── transformer_model.py      # Kiến trúc Transformer
│   │   ├── lstm_gru_model.py         # Kiến trúc LSTM/GRU (TensorFlow)
│   │   └── losses.py                 # CompetitionLoss, LogCosh
│   ├── training/
│   │   ├── train_transformer.py      # Vòng train Transformer (PyTorch)
│   │   └── train_lstm_gru.py         # Vòng train RNN (TensorFlow)
│   ├── data/
│   │   ├── build_transformer_features.py   # Trích 16 expert features
│   │   └── build_lstm_gru_feature.py       # Feature cho LSTM/GRU
│   ├── metrics/
│   │   └── metrics.py                # Weighted MSE, scoring
│   ├── utils/
│   │   └── loaders.py                # PyTorch DataLoader
│   ├── r2/                           # Pipeline Round 2
│   │   ├── rebuild_r2_pipeline.py    # Dựng pipeline dữ liệu R2
│   │   ├── run_r2_regression.py      # Train LSTM/GRU/CNN regression
│   │   └── run_r2_softcls_v3.py      # Train Transformer Soft-Classification
│   ├── ai/
│   │   ├── translator.py             # Module dịch văn bản
│   │   └── slm.py                    # Small Language Model
│   └── app/
│       ├── backend/                  # FastAPI backend
│       │   ├── main.py               # API endpoints
│       │   ├── predictor.py          # Ensemble predictor
│       │   ├── feature_extractor.py  # Trích feature từ chuỗi
│       │   ├── data_store.py         # Quản lý dữ liệu đơn hàng
│       │   ├── precomputed_store.py  # Cache management
│       │   ├── planning_engine.py    # Logic lập kế hoạch
│       │   ├── risk_detector.py      # Đánh giá rủi ro
│       │   ├── scheduler.py          # Gợi ý lịch trình
│       │   ├── schemas.py            # Pydantic models
│       │   ├── config.py             # Cấu hình
│       │   └── sample_data.py        # Dữ liệu mẫu
│       └── frontend/                 # Streamlit dashboard
│           ├── app.py                # Dashboard chính
│           ├── api_client.py         # Giao tiếp API
│           ├── charts.py             # Biểu đồ Plotly
│           ├── components.py         # UI components
│           └── utils.py              # Tiện ích
│
├── notebooks/
│   └── processing_data.ipynb         # Khám phá dữ liệu
├── figures/                          # Hình ảnh phân tích
├── outputs/
│   ├── submission.csv                # File nộp bài
│   └── plots/                        # Biểu đồ kết quả
└── report/
    └── ai_assistance_report.csv
```

---

## 3. Pipeline dữ liệu

Luồng xử lý chính trong [`rebuild_r2_pipeline.py`](src/r2/rebuild_r2_pipeline.py):

```
CSV gốc → Trích manual features → Remap action_id → Padding/Mask → Chuẩn hóa → Tensor .npy
```

**Chi tiết từng bước:**

1. Đọc dữ liệu gốc `X_train.csv`, `X_val.csv`, `X_test.csv` (66 cột feature)
2. Tạo **manual features**: độ dài chuỗi, số action khác nhau, entropy, transition ratio, rollback pattern, rare action ratio, top hub counts, bigram diversity, repeat rate
3. Remap `action_id` về vocab liên tiếp `1..N`
4. Padding / mask chuỗi về `MAX_LEN = 66`
5. Chuẩn hóa stats bằng `StandardScaler`
6. Lưu tensor tại `data/data_processed/transformer/`

**File sinh ra:**
- `X_{train,val,test}_seq.npy` — chuỗi action đã mã hóa
- `X_{train,val,test}_mask.npy` — attention mask
- `X_{train,val,test}_stats.npy` — expert features (16 chiều)
- `y_{train,val}_raw.npy` — nhãn mục tiêu
- `action_remapper.pkl`, `stat_scaler.pkl` — đối tượng tiền xử lý

---

## 4. Họ mô hình 1: GRU + Transformer (Hybrid)

Hướng hybrid mạnh nhất về ý tưởng: kết hợp điểm mạnh của hai nhánh rồi ghép ở mức ensemble.

### Ý tưởng

- **Nhánh R2 Regression** (GRU/LSTM): hồi quy trực tiếp theo competition metric
- **Nhánh Transformer SoftCls**: học phân phối lớp, biểu diễn toàn chuỗi

### Cách ghép

- Ensemble trung bình / median prediction
- Dùng một nhánh cho date attrs (`attr_1,2,4,5`), nhánh còn lại cho capacity attrs (`attr_3,6`)
- Dùng một nhánh làm kiểm chứng / fallback

### File chính

- [`run_r2_regression.py`](src/r2/run_r2_regression.py) — train nhánh hồi quy
- [`run_r2_softcls_v3.py`](src/r2/run_r2_softcls_v3.py) — train Transformer Soft-Classification

| Ưu điểm | Hạn chế |
|---|---|
| Ổn định, linh hoạt, dễ blend nhiều kiểu | Triển khai phức tạp hơn |
| Tận dụng cả sequence branch và stats branch | Tốn thời gian train / infer |
| Ít phụ thuộc vào một kiến trúc duy nhất | Dễ overfit nếu ghép thiếu kiểm soát |

---

## 5. Họ mô hình 2: Transformer only

Mô hình Transformer thuần trong [`transformer_model.py`](src/models/transformer_model.py).

### Kiến trúc

<div align="center">
  <img src="figures/gru_architecture.png" width="600" alt="Kiến trúc mô hình" />
</div>

```
Token Embedding (vocab=30k → d_model=256)
  → Positional Embedding (learnable)
  → Thêm CLS token
  → N × TransformerBlock (MultiHead Attention + FFN + LayerNorm)
  → CLS pooling + Mean pooling → concat (d_model × 2)
  → Expert Features Branch (16 → 64 dim)
  → Ghép vào trunk
  → head_dates  → [attr_1, attr_2, attr_4, attr_5]  (× M_dates)
  → head_factory → [attr_3, attr_6]                  (× M_factory)
```

### File liên quan

- [`transformer_model.py`](src/models/transformer_model.py) — kiến trúc mô hình
- [`train_transformer.py`](src/training/train_transformer.py) — vòng train PyTorch
- [`predict_test.py`](scripts/predict_test.py) — ensemble inference (K-Fold)
- [`predict_each.py`](predict_each.py) — inference 1 fold kiểm tra nhanh

### Huấn luyện

- K-Fold Cross-Validation (thường 3–5 fold)
- Train độc lập từng fold, ensemble bằng soft voting trên tập test

| Ưu điểm | Hạn chế |
|---|---|
| Mô hình hóa quan hệ xa trong chuỗi tốt | Compute nặng hơn RNN |
| Attention dễ trực quan hóa, vẽ heatmap | Cần tuning kỹ |
| Phù hợp phân tích vùng token quan trọng | Cần dữ liệu đủ lớn để ổn định |

---

## 6. Họ mô hình 3: LSTM + GRU (Hồi quy)

Nhánh hồi quy trực tiếp trong [`run_r2_regression.py`](src/r2/run_r2_regression.py), chạy trên TensorFlow.

### Kiến trúc

```
Token Embedding (vocab → 128 dim)
  → Bidirectional LSTM / GRU (128 hidden)
  → Attention Pooling → context vector
  → Stats Branch (n_stats → 64 dim)
  → Concat context + stats
  → date_head  → [attr_1, attr_2, attr_4, attr_5]
  → cap_head   → [attr_3, attr_6]
  → Sigmoid scaling
```

Hỗ trợ 3 backbone: `lstm`, `gru`, `cnn`

### Nhánh thử nghiệm nâng cao

- `len_cond` — conditional encoding theo độ dài
- `prefix_attn` — attention trên prefix
- `dual_gate` — gating kép
- `film_cond` — FiLM conditioning
- `pos_emb` / `seg_emb` — positional / segment embedding

| Ưu điểm | Hạn chế |
|---|---|
| Ổn định, thực dụng | Yếu hơn nếu cần quan hệ rất xa |
| Phù hợp chuỗi ngắn–vừa | — |
| Dễ kết hợp feature engineering | — |

---

## 7. So sánh 3 họ mô hình

| Họ mô hình | Ý tưởng cốt lõi | Điểm mạnh | Rủi ro |
|---|---|---|---|
| **GRU + Transformer** | Ensemble 2 nhánh bổ sung | Ổn định, linh hoạt | Khó triển khai, dễ phức tạp |
| **Transformer only** | Attention toàn chuỗi, CLS + mean pool | Quan hệ xa, dễ phân tích | Nặng, cần tuning kỹ |
| **LSTM + GRU** | Hồi quy trực tiếp RNN + stats | Thực dụng, hợp feature eng. | Yếu quan hệ xa |

---

## 8. Feature Engineering

### Expert Features (16 chiều)

Trích xuất trong [`build_transformer_features.py`](src/data/build_transformer_features.py):

**Nhóm 1 — Định lượng (6 features):**
- Tỷ lệ độ dài chuỗi, tỷ lệ action unique, tỷ lệ trùng lặp
- Mức chi phối action (% action phổ biến nhất), độ phức tạp chuyển trạng thái, tỷ lệ padding

**Nhóm 2 — Signature Density (8 features):**
- Mật độ các "action quan trọng" `[105, 102, 1071, 4004, 1076, 10795, 15342, 21606]`

**Nhóm 3 — Checkpoints (2 features):**
- Action đầu / cuối chuỗi có phải signature hay không

### Manual Features (Layer 2)

- Entropy phân phối action, transition ratio, rollback patterns
- Rare action ratio, top-10 hub counts, bigram diversity, repeat rate

---

## 9. Loss Function & Metric

Định nghĩa trong [`losses.py`](src/models/losses.py):

```python
CompetitionLoss = 0.9 × WeightedMSE + 0.1 × LogCosh
```

- **WeightedMSE**: bám sát metric thi đấu, trọng số `W = [1, 1, 100, 1, 1, 100]`
- **LogCosh**: tăng tính bền vững (robust) với outlier, giảm attention drift

---

## 10. Web App Demo

Ứng dụng web hoàn chỉnh gồm **FastAPI backend** + **Streamlit frontend** để demo trực quan.

### Backend (FastAPI — port 8000)

| Endpoint | Chức năng |
|---|---|
| `GET /health` | Kiểm tra trạng thái API |
| `GET /dataset/overview` | Thống kê tập dữ liệu |
| `GET /orders` | Duyệt đơn hàng (phân trang, lọc) |
| `GET /orders/{id}` | Chi tiết 1 đơn hàng |
| `POST /predict` | Dự đoán realtime cho chuỗi tùy chỉnh |
| `GET /planning/overview` | Tổng quan lập kế hoạch |

### Frontend (Streamlit — port 8501)

- Dashboard tổng quan: health, thống kê, phân phối dữ liệu
- Duyệt & tìm kiếm đơn hàng
- Chi tiết dự đoán cho từng đơn
- Phân tích rủi ro và gợi ý lịch trình
- Biểu đồ tương tác (Plotly)

### Khởi chạy

```bash
python run.py
# Backend:  http://localhost:8000/docs
# Frontend: http://localhost:8501
```

---

## 11. Hướng dẫn chạy

### Cài đặt

```bash
pip install -r requirements.txt
```

### Dựng pipeline dữ liệu

```bash
python src/r2/rebuild_r2_pipeline.py
```

### Train nhánh hồi quy R2 (LSTM/GRU)

```bash
python src/r2/run_r2_regression.py --n_per_type 3 --epochs_full 20 --batch 512 --stats_version v3
```

### Train Transformer Soft-Classification

```bash
python src/r2/run_r2_softcls_v3.py --seed 42 --folds 5 --d_model 512 --layers 6 --epochs 40 --batch 128
```

### Train fullfit Transformer SoftCls

```bash
python src/r2/run_r2_softcls_v3.py --seed 42 --d_model 512 --layers 6 --epochs 40 --batch 128 --fullfit
```

### Pipeline training Transformer (K-Fold)

```bash
python scripts/pipeline_training.py
```

### Ensemble inference

```bash
python scripts/predict_test.py
```

### Tiền xử lý cho app demo

```bash
python precompute_x_test.py
```

### Chạy app demo

```bash
python run.py
```

---

## 12. Kết quả & Hình ảnh phân tích

<details>
<summary><b>Phân bố dữ liệu & nhãn</b></summary>

| Hình | Mô tả |
|---|---|
| ![](figures/phanbododai.png) | Phân bố độ dài chuỗi |
| ![](figures/phanbonhan.png) | Phân bố nhãn |
| ![](figures/phanboclasstrongnhanphobien.png) | Phân bố class trong nhãn phổ biến |

</details>

<details>
<summary><b>Phân tích feature & ảnh hưởng</b></summary>

| Hình | Mô tả |
|---|---|
| ![](figures/anhhuongcuacacfeature.png) | Ảnh hưởng của các feature |
| ![](figures/dodaianhhuongdennhan.png) | Độ dài ảnh hưởng đến nhãn |
| ![](figures/anchor.png) | Anchor pattern |
| ![](figures/matranaha.png) | Ma trận AHA |

</details>

<details>
<summary><b>Rollback & hành vi đặc biệt</b></summary>

| Hình | Mô tả |
|---|---|
| ![](figures/rollback.png) | Rollback pattern |
| ![](figures/firstrollback.png) | First rollback |
| ![](figures/top3nguyennhangayrung.png) | Top 3 nguyên nhân gây rung |
| ![](figures/viporpower.png) | VIP or Power user |

</details>

<details>
<summary><b>Kết quả mô hình</b></summary>

| Hình | Mô tả |
|---|---|
| ![](outputs/plots/1_global_train.png) | Global metrics — Train |
| ![](outputs/plots/1_global_val.png) | Global metrics — Validation |
| ![](outputs/plots/1_global_test.png) | Global metrics — Test |
| ![](outputs/plots/2_case_best.png) | Case tốt nhất |
| ![](outputs/plots/2_case_worst.png) | Case tệ nhất |
| ![](outputs/plots/2_case_best_ensemble.png) | Case tốt nhất (ensemble) |
| ![](outputs/plots/2_case_worst_ensemble.png) | Case tệ nhất (ensemble) |
| ![](outputs/plots/heatmap_correlation.png) | Heatmap tương quan |
| ![](outputs/plots/heatmap_attr_specific.png) | Heatmap theo thuộc tính |

</details>

---

## 13. Lưu ý khi tái lập

- Repo mang tính **thực nghiệm cao** — nhiều nhánh ý tưởng cùng tồn tại
- Kiểm tra kỹ trước khi chạy:
  - Đường dẫn dữ liệu phù hợp
  - Version feature đang dùng
  - Checkpoint model tương ứng
  - Cách hậu xử lý và ensemble
- Config mặc định trong `.env`: `SEED=42`, `BATCH_SIZE=128`, `EPOCHS=20`, `N_FOLDS=5`
