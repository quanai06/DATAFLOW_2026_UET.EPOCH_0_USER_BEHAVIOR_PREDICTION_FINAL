<div align="center">
    
  <img src="dataflow.png" width="200" alt="logo" />

  # Chủ đề: Autoscaling Analysis - DataFlow 2026
  
  **Giải pháp Tối ưu hóa Tự động hóa việc cấp phát tài nguyên máy chủ (Autoscaling) dựa trên Học máy và Chiến lược lai (Hybrid Strategy).**
  
  ![Python](https://img.shields.io/badge/Python-3.12-red?logo=python&logoColor=white)
  ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
</div>

# DATAFLOW_2026_UET.EPOCH_0_USER_BEHAVIOR_PREDICTION_FINAL

Repo này lưu toàn bộ pipeline xử lý dữ liệu, huấn luyện mô hình và sinh submission cho bài toán dự đoán 6 thuộc tính hành vi người dùng từ chuỗi `action_id`.

Mục tiêu dự đoán:
- `attr_1, attr_2, attr_4, attr_5`: thông tin dạng lịch/ngày
- `attr_3, attr_6`: thông tin dạng capacity / factory score

Project này phát triển theo 3 họ mô hình chính:
- `GRU + Transformer` theo kiểu hybrid / ensemble
- `Transformer only`
- `LSTM + GRU` trong nhánh hồi quy R2

## 1. Cấu trúc repo

Các file quan trọng:
- [rebuild_r2_pipeline.py](/home/hduong/dev/DATAFLOW_2026_push_target/rebuild_r2_pipeline.py): dựng lại pipeline dữ liệu Round 2, tạo sequence tensor và stats feature
- [run_r2_regression.py](/home/hduong/dev/DATAFLOW_2026_push_target/run_r2_regression.py): train nhánh hồi quy R2 với `LSTM / GRU / CNN`
- [run_r2_softcls_v3.py](/home/hduong/dev/DATAFLOW_2026_push_target/run_r2_softcls_v3.py): train nhánh `Transformer Soft-Classification`
- [src/models/transformer_model.py](/home/hduong/dev/DATAFLOW_2026_push_target/src/models/transformer_model.py): model Transformer thuần
- [src/training/train_transformer.py](/home/hduong/dev/DATAFLOW_2026_push_target/src/training/train_transformer.py): vòng train cho Transformer thuần
- [scripts/predict_test.py](/home/hduong/dev/DATAFLOW_2026_push_target/scripts/predict_test.py): ensemble inference cho Transformer
- [predict_each.py](/home/hduong/dev/DATAFLOW_2026_push_target/predict_each.py): inference 1 fold để kiểm tra nhanh
- [precompute_x_test.py](/home/hduong/dev/DATAFLOW_2026_push_target/precompute_x_test.py): tiền xử lý offline cho dữ liệu test phục vụ app/demo
- [run.py](/home/hduong/dev/DATAFLOW_2026_push_target/run.py): chạy backend + frontend demo

## 2. Pipeline dữ liệu

Luồng xử lý chính trong [rebuild_r2_pipeline.py](/home/hduong/dev/DATAFLOW_2026_push_target/rebuild_r2_pipeline.py):

1. Đọc dữ liệu gốc từ `data/X_train.csv`, `data/X_val.csv`, `data/X_test.csv`
2. Tạo `manual features`:
- độ dài chuỗi
- số action khác nhau
- entropy
- transition ratio
- rollback pattern
- rare action ratio
- top hub counts
- bigram diversity
- repeat rate

3. Remap `action_id` về vocab liên tiếp `1..N`
4. Padding / mask chuỗi về `MAX_LEN = 66`
5. Chuẩn hóa stats feature bằng `StandardScaler`
6. Lưu tensor tại `data/layer3_features/transformer_r2/`

Các file sinh ra thường gồm:
- `X_train_seq.npy`, `X_val_seq.npy`, `X_test_seq.npy`
- `X_train_mask.npy`, `X_val_mask.npy`, `X_test_mask.npy`
- `X_train_stats.npy`, `X_val_stats.npy`, `X_test_stats.npy`
- `y_train_raw.npy`, `y_val_raw.npy`
- `action_remapper.pkl`, `stat_scaler.pkl`

## 3. Họ mô hình 1: GRU + Transformer

Đây là hướng hybrid mạnh nhất về mặt ý tưởng của repo: tận dụng điểm mạnh của hai nhánh khác nhau rồi ghép lại ở mức ensemble hoặc chia vai trò theo nhóm nhãn.

### Ý tưởng
- Nhánh `R2 regression` mạnh ở phần hồi quy trực tiếp theo competition metric
- Nhánh `Transformer SoftCls` mạnh ở việc học phân phối lớp và biểu diễn toàn chuỗi
- Có thể ghép hai nhánh theo nhiều cách:
- ensemble trung bình / median prediction
- dùng một nhánh cho date attrs, nhánh còn lại cho capacity attrs
- dùng một nhánh làm kiểm chứng hoặc fallback cho nhánh còn lại

### Thành phần chính
- [run_r2_regression.py](/home/hduong/dev/DATAFLOW_2026_push_target/run_r2_regression.py)
- [run_r2_softcls_v3.py](/home/hduong/dev/DATAFLOW_2026_push_target/run_r2_softcls_v3.py)

### Khi nào dùng
- Khi muốn khai thác tính bổ sung giữa mô hình tuần tự kiểu RNN và mô hình attention
- Khi cần ensemble để tăng độ ổn định leaderboard
- Khi muốn tách riêng chiến lược học cho `date attrs` và `capacity attrs`

### Điểm mạnh
- ít phụ thuộc vào một kiến trúc duy nhất
- dễ thử nghiệm nhiều kiểu blend
- tận dụng được cả sequence branch và stats branch

### Điểm yếu
- triển khai phức tạp hơn
- tốn thời gian train / infer hơn
- dễ overfit nếu ghép thiếu kiểm soát

## 4. Họ mô hình 2: Transformer only

Nhánh này là mô hình Transformer thuần trong [src/models/transformer_model.py](/home/hduong/dev/DATAFLOW_2026_push_target/src/models/transformer_model.py).

### Kiến trúc
- embedding cho token
- positional embedding học được
- thêm `CLS token`
- stack nhiều `TransformerBlock`
- lấy `CLS + mean pooling`
- tách 2 head:
- `head_dates` cho `attr_1, attr_2, attr_4, attr_5`
- `head_factory` cho `attr_3, attr_6`

Đầu ra được scale trực tiếp về miền giá trị:
- tháng / ngày
- factory score 0..99

### File liên quan
- [src/models/transformer_model.py](/home/hduong/dev/DATAFLOW_2026_push_target/src/models/transformer_model.py)
- [src/training/train_transformer.py](/home/hduong/dev/DATAFLOW_2026_push_target/src/training/train_transformer.py)
- [scripts/predict_test.py](/home/hduong/dev/DATAFLOW_2026_push_target/scripts/predict_test.py)
- [predict_each.py](/home/hduong/dev/DATAFLOW_2026_push_target/predict_each.py)

### Ưu điểm
- mô hình hóa quan hệ xa trong chuỗi tốt
- attention dễ soi trực quan, dễ vẽ heatmap
- phù hợp khi muốn phân tích vùng token quan trọng

### Hạn chế
- cần dữ liệu và tuning tốt để ổn định
- compute nặng hơn RNN
- nếu chỉ dùng Transformer thuần, đôi khi chưa tận dụng hết handcrafted stats

## 5. Họ mô hình 3: LSTM + GRU

Nhánh này nằm trong [run_r2_regression.py](/home/hduong/dev/DATAFLOW_2026_push_target/run_r2_regression.py). Đây là hướng hồi quy trực tiếp theo competition metric.

### Ý tưởng chính
- encode chuỗi action bằng `LSTM` hoặc `GRU` hai chiều
- attention pooling trên hidden states
- ghép thêm `stats branch`
- đi qua shared trunk
- tách 2 head:
- `date_head` cho `attr_1, attr_2, attr_4, attr_5`
- `cap_head` cho `attr_3, attr_6`

Script này thực tế hỗ trợ 3 backbone:
- `lstm`
- `gru`
- `cnn`

Trong đó `LSTM + GRU` là hai họ chính để học thứ tự tuần tự của hành vi.

### Các điểm đáng chú ý
- loss bám sát competition metric
- có stats branch để tận dụng feature thủ công
- có thể ensemble nhiều seed, nhiều backbone
- có các nhánh thử nghiệm như:
- `len_cond`
- `prefix_attn`
- `dual_gate`
- `film_cond`
- `pos_emb`
- `seg_emb`

### Vì sao hướng này mạnh
- phù hợp với dữ liệu chuỗi ngắn-vừa
- ổn định hơn khi dữ liệu không quá lớn
- dễ kết hợp với feature engineering

## 6. So sánh nhanh 3 họ mô hình

| Họ mô hình | Ý tưởng cốt lõi | Điểm mạnh | Rủi ro |
|---|---|---|---|
| GRU + Transformer | Kết hợp 2 nhánh khác nhau để ensemble / chia vai trò | mạnh về độ ổn định và linh hoạt | khó triển khai, dễ phức tạp |
| Transformer only | Attention toàn chuỗi, CLS + mean pooling | mô hình hóa quan hệ xa tốt, dễ phân tích attention | nặng, cần tuning kỹ |
| LSTM + GRU | Hồi quy trực tiếp bằng RNN + stats branch | ổn định, thực dụng, hợp feature engineering | có thể yếu hơn nếu cần quan hệ rất xa |

## 7. Cách chạy cơ bản

### Cài dependencies
```bash
pip install -r requirements.txt
```

### Dựng pipeline dữ liệu R2
```bash
python rebuild_r2_pipeline.py
```

### Train nhánh hồi quy R2
```bash
python run_r2_regression.py --n_per_type 3 --epochs_full 20 --batch 512 --stats_version v3
```

### Train Transformer SoftCls v3
```bash
python run_r2_softcls_v3.py --seed 42 --folds 5 --d_model 512 --layers 6 --epochs 40 --batch 128
```

### Train fullfit cho SoftCls
```bash
python run_r2_softcls_v3.py --seed 42 --d_model 512 --layers 6 --epochs 40 --batch 128 --fullfit
```

### Predict bằng Transformer
```bash
python scripts/predict_test.py
```

### Chạy app demo
```bash
python run.py
```

## 8. Hướng dùng repo theo mục tiêu

Nếu muốn nghiên cứu pipeline dữ liệu:
- bắt đầu từ [rebuild_r2_pipeline.py](/home/hduong/dev/DATAFLOW_2026_push_target/rebuild_r2_pipeline.py)

Nếu muốn train mô hình tuần tự ổn định:
- bắt đầu từ [run_r2_regression.py](/home/hduong/dev/DATAFLOW_2026_push_target/run_r2_regression.py)

Nếu muốn train mô hình attention / phân tích heatmap:
- bắt đầu từ [src/models/transformer_model.py](/home/hduong/dev/DATAFLOW_2026_push_target/src/models/transformer_model.py)
- rồi tới [src/training/train_transformer.py](/home/hduong/dev/DATAFLOW_2026_push_target/src/training/train_transformer.py)

Nếu muốn thử hybrid:
- train riêng `R2 regression`
- train riêng `SoftCls Transformer`
- blend / ensemble ở bước cuối

## 9. Ghi chú

- Repo này mang tính thực nghiệm cao, có nhiều nhánh ý tưởng khác nhau cùng tồn tại.
- Không phải mọi script đều dùng chung một format dữ liệu tuyệt đối giống nhau.
- Khi tái lập kết quả, nên kiểm tra kỹ:
- đường dẫn dữ liệu
- version của feature
- checkpoint đang dùng
- cách hậu xử lý và ensemble

Nếu muốn chuẩn hóa repo hơn nữa, bước tiếp theo hợp lý là:
- gom script train vào một thư mục thống nhất
- chuẩn hóa config
- thêm notebook / report mô tả kết quả từng family model
