import numpy as np
import os
import torch
from sklearn.model_selection import KFold
from src.training.train_transformer import run_train_transformer

def main():
    # 1. Load Data - Giữ nguyên sự độc lập
    print("🚀 Loading data for Internal 5-Fold...")
    X_tr = np.load("data/data_processed/transformer/X_train.npy")
    M_tr = np.load("data/data_processed/transformer/mask_train.npy")
    y_tr = np.load("data/data_processed/transformer/y_train.npy")
    
    # Đây là tập Val "sạch", model không được phép nhìn thấy lúc train
    X_va_holdout = np.load("data/data_processed/transformer/X_val.npy")
    M_va_holdout = np.load("data/data_processed/transformer/mask_val.npy")
    y_va_holdout = np.load("data/data_processed/transformer/y_val.npy")

    # 2. Chia 5-Fold CHỈ TRÊN TẬP TRAIN (55k mẫu)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    os.makedirs("model/kfold", exist_ok=True)
    
    config_base = {
        'batch_size': 256,   # Giảm batch size xuống một chút nếu tăng d_model để tránh tràn VRAM
        'epochs': 28,       # Tăng epoch để model hội tụ sâu với MSE
        'lr': 1e-3,         # Giảm LR một chút để tránh vọt Loss khi dùng MSE
        'd_model': 256,
        'num_layers': 6
    }

    fold_scores_internal = []
    fold_scores_holdout = []

    # 3. Chạy Pipeline
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr)):
        print(f"\n🔥 --- INTERNAL FOLD {fold + 1} / 5 ---")
        
        # Data cho Fold này (lấy từ 55k mẫu train)
        train_data = (X_tr[train_idx], M_tr[train_idx], y_tr[train_idx])
        internal_val_data = (X_tr[val_idx], M_tr[val_idx], y_tr[val_idx])
        
        config_fold = config_base.copy()
        config_fold['model_save_path'] = f"model/kfold/transformer_f{fold}.pth"
        
        # Huấn luyện model của Fold
        _, score_internal = run_train_transformer(train_data, internal_val_data, config_fold)
        fold_scores_internal.append(score_internal)

        # TEST THỬ TRÊN TẬP VAL "SẠCH" (40k mẫu) để xem score thực chất
        # (B có thể viết thêm logic này để check sự chênh lệch)
        print(f"✅ Fold {fold+1} Internal Score: {score_internal:.6f}")

    print("\n✨ PIPELINE COMPLETE (NO MERGE)!")
    print(f"🏆 Mean Internal Score: {np.mean(fold_scores_internal):.6f}")
    print(f"📍 5 Models saved in model/kfold/")

if __name__ == "__main__":
    main()