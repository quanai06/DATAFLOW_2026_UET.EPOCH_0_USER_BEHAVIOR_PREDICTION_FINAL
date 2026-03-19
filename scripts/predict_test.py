import torch
import numpy as np
import pandas as pd
import os
from src.models.transformer_model import TransformerModel

def make_submission():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_PATH = "data/data_processed/transformer"
    
    # 1. LOAD TEST DATA (Bao gồm cả Expert Features S_test)
    print("📂 Loading Test Data and Expert Features...")
    X_te = np.load(f"{DATA_PATH}/X_test.npy")
    S_te = np.load(f"{DATA_PATH}/S_test.npy") # Expert Features trích xuất từ Heatmap
    M_te = np.load(f"{DATA_PATH}/mask_test.npy")
    
    # 2. LOAD K-FOLD MODELS (3 Folds như trong Pipeline)
    models = []
    print(f"🤖 Loading 3-Fold Models from 'model/kfold/'...")
    for i in range(3):
        # Khởi tạo model với expert_dim=16
        m = TransformerModel(expert_dim=16, vocab_size=30000, d_model=256, num_layers=4).to(DEVICE)
        model_path = f"model/kfold/transformer_f{i}.pth"
        if os.path.exists(model_path):
            m.load_state_dict(torch.load(model_path, map_location=DEVICE))
            m.eval()
            models.append(m)
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy {model_path}")

    # 3. INFERENCE (DỰ ĐOÁN)
    all_preds = []
    print("🚀 Running Inference with Ensemble...")
    with torch.no_grad():
        for i in range(0, len(X_te), 128):
            # Cắt batch
            xb = torch.LongTensor(X_te[i:i+128]).to(DEVICE)
            sb = torch.FloatTensor(S_te[i:i+128]).to(DEVICE)
            mb = torch.BoolTensor(M_te[i:i+128]).to(DEVICE)
            
            # Ensemble: Trung bình cộng kết quả từ các Fold (Soft Voting)
            fold_outs = []
            for m in models:
                # Model bây giờ nhận 3 tham số: chuỗi, expert stats, và mask
                out, _ = m(xb, sb, mb)
                fold_outs.append(out.cpu().numpy())
            
            avg_out = np.mean(fold_outs, axis=0)
            all_preds.append(avg_out)

    final_preds = np.concatenate(all_preds, axis=0)
    
    # --- 4. HẬU XỬ LÝ (BÁM SÁT ĐỀ BÀI TRANG 2) ---
    print("🛠 Post-processing outputs...")
    
    # Clip giá trị theo ý nghĩa thực tế của đầu ra (Mục 2.1 - Cấu trúc dữ liệu)
    # attr_1, 4: Tháng (1-12)
    # attr_2, 5: Ngày (1-31)
    # attr_3, 6: Chỉ số hoạt động nhà máy (0-99)
    final_preds[:, [0, 3]] = np.clip(final_preds[:, [0, 3]], 1, 12)  
    final_preds[:, [1, 4]] = np.clip(final_preds[:, [1, 4]], 1, 31)  
    final_preds[:, [2, 5]] = np.clip(final_preds[:, [2, 5]], 0, 99)  
    
    # Ép kiểu UINT16 theo yêu cầu bắt buộc của BTC (trang 2)
    final_preds = np.round(final_preds).astype(np.uint16)

    # 5. TẠO FILE SUBMISSION
    df_sub = pd.DataFrame(final_preds, columns=['attr_1', 'attr_2', 'attr_3', 'attr_4', 'attr_5', 'attr_6'])
    
    # Lấy cột ID từ file test gốc hoặc sample_submission
    try:
        sample = pd.read_csv("data/data_raw/X_test.csv")
        df_sub.insert(0, 'id', sample['id'].values)
    except:
        print("⚠️ Không tìm thấy X_test.csv để lấy ID, dùng index tạm thời.")
        df_sub.insert(0, 'id', range(len(df_sub)))
    
    os.makedirs("outputs", exist_ok=True)
    sub_path = "outputs/submission.csv"
    df_sub.to_csv(sub_path, index=False)
    
    print(f"✅ HOÀN TẤT! File nộp bài đã được lưu tại: {sub_path}")
    print(df_sub.head())

if __name__ == "__main__":
    make_submission()