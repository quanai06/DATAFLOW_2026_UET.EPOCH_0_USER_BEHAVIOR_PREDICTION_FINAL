import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys

# Đảm bảo Python ưu tiên tìm trong thư mục hiện tại
sys.path.append(os.getcwd())

from src.models.transformer_model import TransformerModel

def make_single_fold_submission():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load test data
    X_te = np.load("data/data_processed/transformer/X_test.npy")
    M_te = np.load("data/data_processed/transformer/mask_test.npy")
    
    # 2. Khởi tạo Model 
    model = TransformerModel(vocab_size=30000, d_model=256, num_layers=6).to(DEVICE)

    # 3. Load trọng số
    model_path = "model/kfold/transformer_f0.pth"
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file {model_path}!")
        return

    try:
        # Load state_dict
        state_dict = torch.load(model_path, map_location=DEVICE)
        
        # Kiểm tra xem state_dict có chứa head_dates không trước khi load
        if "head_dates.0.weight" in state_dict:
            model.load_state_dict(state_dict)
            print(f"🚀 Đã load thành công Fold 1 với kiến trúc Multi-Head.")
        else:
            print("⚠️ Cảnh báo: State dict không khớp với kiến trúc head_dates. Kiểm tra lại file .pth")
            return
            
        model.eval()
    except Exception as e:
        print(f"❌ Lỗi load model chi tiết: {e}")
        return

    all_preds = []
    print("🚀 Đang thực hiện inference...")
    
    with torch.no_grad():
        for i in range(0, len(X_te), 128):
            xb = torch.LongTensor(X_te[i:i+128]).to(DEVICE)
            mb = torch.BoolTensor(M_te[i:i+128]).to(DEVICE)
            
            # Forward pass trả về (out, all_attentions)
            out, _ = model(xb, mb)
            all_preds.append(out.cpu().numpy())

    final_preds = np.concatenate(all_preds, axis=0)
    
    # --- SANITY CHECK ---
    std_devs = np.std(final_preds, axis=0)
    print("\n📊 Kiểm tra độ biến thiên (Standard Deviation):")
    cols = ['Tháng 1', 'Ngày 1', 'Nhà máy 1', 'Tháng 2', 'Ngày 2', 'Nhà máy 2']
    for col, std in zip(cols, std_devs):
        print(f"   🔹 {col}: {std:.4f}")

    # --- HẬU XỬ LÝ ---
    final_preds[:, [0, 3]] = np.clip(final_preds[:, [0, 3]], 1, 12)  # Tháng
    final_preds[:, [1, 4]] = np.clip(final_preds[:, [1, 4]], 1, 31)  # Ngày
    final_preds[:, [2, 5]] = np.clip(final_preds[:, [2, 5]], 0, 99)  # Nhà máy
    
    final_preds = np.round(final_preds).astype(np.uint16)

    # 4. Tạo DataFrame
    df_sub = pd.DataFrame(final_preds, columns=['attr_1', 'attr_2', 'attr_3', 'attr_4', 'attr_5', 'attr_6'])
    
    sample_path = "data/data_raw/sample_submission.csv"
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        df_sub.insert(0, 'id', sample['id'])
    else:
        df_sub.insert(0, 'id', range(len(df_sub)))

    output_name = "submission_fold1_check.csv"
    df_sub.to_csv(output_name, index=False)
    print(f"\n✅ Đã xuất file: {output_name}")

if __name__ == "__main__":
    make_single_fold_submission()