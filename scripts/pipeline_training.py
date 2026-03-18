import numpy as np
import os
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tabulate import tabulate # Nhớ pip install tabulate nếu chưa có

from src.training.train_transformer import run_train_transformer
from src.utils.loaders import SequenceDataset
from src.models.transformer_model import TransformerModel

# --- HÀM 1: TÍNH ĐIỂM ENSEMBLE VÀ IN BẢNG ---
def evaluate_ensemble_holdout(n_folds, holdout_data, device, phase="BEFORE FE"):
    print(f"\n🔮 --- ĐANG TÍNH ĐIỂM ENSEMBLE TRÊN TẬP HOLDOUT (GIAI ĐOẠN: {phase}) ---")
    X_val, M_val, y_val = holdout_data
    val_loader = DataLoader(SequenceDataset(X_val, M_val, y_val), batch_size=256, shuffle=False)
    
    all_fold_preds = []
    
    # Load từng model của mỗi Fold lên để dự đoán
    for fold in range(n_folds):
        model_path = f"model/kfold/transformer_f{fold}.pth"
        if not os.path.exists(model_path): 
            print(f"⚠️ Không tìm thấy model fold {fold}, bỏ qua.")
            continue
        
        # Khởi tạo lại kiến trúc model (phải khớp với d_model, num_layers lúc train)
        model = TransformerModel(d_model=256, num_layers=4).to(device) # Chỉnh lại num_layers=4 nếu bạn đã giảm xuống
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for xb, mb, _ in val_loader:
                out, _ = model(xb.to(device), mb.to(device))
                fold_preds.append(out.cpu())
        
        all_fold_preds.append(torch.cat(fold_preds))
        print(f"✅ Đã dự đoán xong với Fold {fold}")

    # Ensemble: Tính trung bình cộng kết quả của tất cả các Fold
    ensemble_preds = torch.stack(all_fold_preds).mean(dim=0)
    y_true = torch.FloatTensor(y_val)

    # Tính Weighted MSE chuẩn cho 6 cột
    M = torch.tensor([12, 31, 99, 12, 31, 99], dtype=torch.float32)
    W = torch.tensor([1, 1, 100, 1, 1, 100], dtype=torch.float32)
    
    p_norm = ensemble_preds / M
    t_norm = y_true / M
    sq_error = torch.pow(p_norm - t_norm, 2) * W
    
    # Tính điểm từng cột và điểm tổng
    col_scores = sq_error.mean(dim=0).numpy()
    overall_score = col_scores.mean()

    # IN BẢNG BÁO CÁO CUỐI CÙNG (Dùng cho slide/báo cáo)
    headers = ["Model/Phase", "Attr_1", "Attr_2", "Attr_3", "Attr_4", "Attr_5", "Attr_6", "OVERALL"]
    row = [f"Ensemble {n_folds} Folds"] + [f"{s:.6f}" for s in col_scores] + [f"{overall_score:.6f}"]
    
    print("\n" + "="*115)
    print(f"{'BẢNG ĐIỂM ENSEMBLE CHI TIẾT (' + phase + ')':^115}")
    print("="*115)
    print(tabulate([row], headers=headers, tablefmt="grid"))
    print("="*115)
    
    return overall_score

# --- HÀM 2: PIPELINE CHÍNH ---
def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    N_FOLDS = 3 # Bạn đã giảm xuống 3 Fold cho nhanh
    
    # 1. Load Data
    X_tr = np.load("data/data_processed/transformer/X_train.npy")
    M_tr = np.load("data/data_processed/transformer/mask_train.npy")
    y_tr = np.load("data/data_processed/transformer/y_train.npy")
    
    X_va_holdout = np.load("data/data_processed/transformer/X_val.npy")
    M_va_holdout = np.load("data/data_processed/transformer/mask_val.npy")
    y_va_holdout = np.load("data/data_processed/transformer/y_val.npy")

    # 2. Cấu hình
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    os.makedirs("model/kfold", exist_ok=True)
    
    config_base = {
        'batch_size': 256,
        'epochs': 25,       # Bạn đã giảm xuống 25 Epoch
        'lr': 1e-3,
        'd_model': 256,
        'num_layers': 4,    # Nhớ chỉnh khớp với model kiến trúc mới (4 layer)
        'model_save_path': "" 
    }

    fold_scores_internal = []

    # 3. Train K-Fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr)):
        print(f"\n🔥 --- TRAINING FOLD {fold + 1} / {N_FOLDS} ---")
        train_data = (X_tr[train_idx], M_tr[train_idx], y_tr[train_idx])
        internal_val_data = (X_tr[val_idx], M_tr[val_idx], y_tr[val_idx])
        
        config_fold = config_base.copy()
        config_fold['model_save_path'] = f"model/kfold/transformer_f{fold}.pth"
        
        _, score_internal = run_train_transformer(train_data, internal_val_data, config_fold)
        fold_scores_internal.append(score_internal)
        print(f"✅ Fold {fold+1} Internal Score: {score_internal:.6f}")

    print("\n" + "*"*50)
    print(f"🏆 Average Internal Score (Across Folds): {np.mean(fold_scores_internal):.6f}")
    print("*"*50)

    # 4. CHẠY ENSEMBLE TRÊN TẬP HOLDOUT VÀ IN BẢNG BÁO CÁO
    holdout_data = (X_va_holdout, M_va_holdout, y_va_holdout)
    evaluate_ensemble_holdout(
        n_folds=N_FOLDS, 
        holdout_data=holdout_data, 
        device=DEVICE,
        phase="BEFORE FEATURE ENGINEERING" # Giai đoạn trước khi làm FE
    )

if __name__ == "__main__":
    main()