import torch
import numpy as np
import pandas as pd
import os
from src.models.transformer_model import TransformerModel

def make_submission():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Load test data
    X_te = np.load("data/data_processed/transformer/X_test.npy")
    M_te = np.load("data/data_processed/transformer/mask_test.npy")
    
    # Load 5 models
    models = []
    for i in range(5):
        m = TransformerModel(vocab_size=30000, d_model=256, num_layers=6).to(DEVICE)
        m.load_state_dict(torch.load(f"model/kfold/transformer_f{i}.pth"))
        m.eval()
        models.append(m)

    all_preds = []
    with torch.no_grad():
        # Predict theo batch để tránh tràn RAM
        for i in range(0, len(X_te), 128):
            xb = torch.LongTensor(X_te[i:i+128]).to(DEVICE)
            mb = torch.BoolTensor(M_te[i:i+128]).to(DEVICE)
            
            # Ensemble: Trung bình cộng 5 Fold
            fold_outs = []
            for m in models:
                out, _ = m(xb, mb)
                fold_outs.append(out.cpu().numpy())
            
            avg_out = np.mean(fold_outs, axis=0)
            all_preds.append(avg_out)

    final_preds = np.concatenate(all_preds, axis=0)
    
    # --- HẬU XỬ LÝ (QUAN TRỌNG) ---
    # 1. Clip giá trị để không bị âm hoặc vượt quá ngưỡng thực tế
    final_preds[:, [0, 3]] = np.clip(final_preds[:, [0, 3]], 1, 12)  # Tháng
    final_preds[:, [1, 4]] = np.clip(final_preds[:, [1, 4]], 1, 31)  # Ngày
    final_preds[:, [2, 5]] = np.clip(final_preds[:, [2, 5]], 0, 99)  # Nhà máy
    
    # 2. Làm tròn và ép kiểu UINT16
    final_preds = np.round(final_preds).astype(np.uint16)

    # 3. Tạo DataFrame
    df_sub = pd.DataFrame(final_preds, columns=['attr_1', 'attr_2', 'attr_3', 'attr_4', 'attr_5', 'attr_6'])
    # Giả sử bạn có file sample_submission để lấy cột ID
    sample = pd.read_csv("data/data_raw/sample_submission.csv")
    df_sub.insert(0, 'id', sample['id'])
    
    df_sub.to_csv("outputs/submission.csv", index=False)
    print("✅ Đã xuất file ssubmission.csv")

if __name__ == "__main__":
    make_submission()