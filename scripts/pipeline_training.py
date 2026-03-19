import numpy as np
import os
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tabulate import tabulate
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.train_transformer import run_train_transformer
from src.utils.loaders import SequenceDataset
from src.models.transformer_model import TransformerModel

import tensorflow as tf
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.training.train_lstm_gru import build_model
from src.metrics.metrics import set_seed, evaluate_report

import dotenv
dotenv.load_dotenv()

SEED = int(os.getenv("SEED", 42))
FINAL_MAX_LEN = int(os.getenv("FINAL_MAX_LEN", 37))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 128))
EPOCHS = int(os.getenv("EPOCHS", 20))
N_FOLDS = int(os.getenv("N_FOLDS", 5))
TARGET_COLS = os.getenv("TARGET_COLS", ['attr_1','attr_2','attr_3','attr_4','attr_5','attr_6'])
M_CONST_NP = np.array([float(x) for x in os.getenv("M_CONST_NP", "12.0, 31.0, 99.0, 12.0, 31.0, 99.0").split(",")], dtype=np.float32)
W_CONST_NP = np.array([float(x) for x in os.getenv("W_CONST_NP", "1.0, 1.0, 100.0, 1.0, 1.0, 100.0").split(",")], dtype=np.float32)

# --- HÀM 1: TÍNH ĐIỂM ENSEMBLE VÀ IN BẢNG ---
def evaluate_ensemble_holdout(n_folds, holdout_data, device, phase="AFTER FEATURE ENGINEERING"):
    """
    holdout_data: (X_val, S_val, M_val, y_val)
    """
    print(f"\n🔮 --- ĐANG TÍNH ĐIỂM ENSEMBLE TRÊN TẬP HOLDOUT (GIAI ĐOẠN: {phase}) ---")
    X_val, S_val, M_val, y_val = holdout_data
    
    # Dataset bây giờ nhận 4 tham số: X, S, Mask, y
    val_loader = DataLoader(SequenceDataset(X_val, S_val, M_val, y_val), batch_size=256, shuffle=False)
    
    all_fold_preds = []
    
    for fold in range(n_folds):
        model_path = f"model/kfold/transformer_f{fold}.pth"
        if not os.path.exists(model_path): 
            print(f"⚠️ Không tìm thấy model fold {fold}, bỏ qua.")
            continue
        
        # Khởi tạo model với expert_dim=22 (khớp với file build_features)
        model = TransformerModel(expert_dim=16, d_model=256, num_layers=4).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for xb, sb, mb, _ in val_loader:
                # Model forward nhận cả xb (chuỗi) và sb (expert features)
                out, _ = model(xb.to(device), sb.to(device), mb.to(device))
                fold_preds.append(out.cpu())
        
        all_fold_preds.append(torch.cat(fold_preds))
        print(f"✅ Đã dự đoán xong với Fold {fold}")

    # Ensemble: Trung bình cộng Soft Voting từ các Fold
    ensemble_preds = torch.stack(all_fold_preds).mean(dim=0)
    y_true = torch.FloatTensor(y_val)

    # Tính Weighted MSE chuẩn (Alpha = 100 cho Attr 3, 6)
    M = torch.tensor([12, 31, 99, 12, 31, 99], dtype=torch.float32)
    W = torch.tensor([1, 1, 100, 1, 1, 100], dtype=torch.float32)
    
    p_norm = ensemble_preds / M
    t_norm = y_true / M
    sq_error = torch.pow(p_norm - t_norm, 2) * W
    
    col_scores = sq_error.mean(dim=0).numpy()
    overall_score = col_scores.mean()

    # IN BẢNG BÁO CÁO (Bám sát báo cáo để so sánh hiệu quả FE)
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
    N_FOLDS = 3 
    DATA_PATH = "data/data_processed/transformer"
    
    # 1. Load Data (Bao gồm cả Expert Features S đã trích xuất từ Heatmap)
    print("📂 Loading data and Expert Features...")
    X_tr = np.load(f"{DATA_PATH}/X_train.npy")
    S_tr = np.load(f"{DATA_PATH}/S_train.npy") # Expert Features bám sát báo cáo 2.2
    M_tr = np.load(f"{DATA_PATH}/mask_train.npy")
    y_tr = np.load(f"{DATA_PATH}/y_train.npy")
    
    X_va_holdout = np.load(f"{DATA_PATH}/X_val.npy")
    S_va_holdout = np.load(f"{DATA_PATH}/S_val.npy")
    M_va_holdout = np.load(f"{DATA_PATH}/mask_val.npy")
    y_va_holdout = np.load(f"{DATA_PATH}/y_val.npy")

    # 2. Cấu hình
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    os.makedirs("model/kfold", exist_ok=True)
    
    config_base = {
        'batch_size': 256,
        'epochs': 25,       
        'lr': 1e-3,
        'd_model': 256,
        'num_layers': 4,    
        'model_save_path': "" 
    }

    fold_scores_internal = []

    # 3. Train K-Fold (Chia cả X, S, M, y theo Fold)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr)):
        print(f"\n🔥 --- TRAINING FOLD {fold + 1} / {N_FOLDS} ---")
        
        # Tạo tuple dữ liệu 4 thành phần
        train_data = (X_tr[train_idx], S_tr[train_idx], M_tr[train_idx], y_tr[train_idx])
        internal_val_data = (X_tr[val_idx], S_tr[val_idx], M_tr[val_idx], y_tr[val_idx])
        
        config_fold = config_base.copy()
        config_fold['model_save_path'] = f"model/kfold/transformer_f{fold}.pth"
        
        # Chạy train (Hàm này trong train_transformer.py đã được sửa để nhận S)
        _, score_internal = run_train_transformer(train_data, internal_val_data, config_fold)
        fold_scores_internal.append(score_internal)
        print(f"✅ Fold {fold+1} Internal Score: {score_internal:.6f}")

    print("\n" + "*"*50)
    print(f"🏆 Average Internal Score (Across Folds): {np.mean(fold_scores_internal):.6f}")
    print("*"*50)

    # 4. CHẠY ENSEMBLE TRÊN TẬP HOLDOUT VÀ IN BẢNG BÁO CÁO
    holdout_data = (X_va_holdout, S_va_holdout, M_va_holdout, y_va_holdout)
    evaluate_ensemble_holdout(
        n_folds=N_FOLDS, 
        holdout_data=holdout_data, 
        device=DEVICE,
        phase="AFTER EXPERT FEATURE ENGINEERING" 
    )

if __name__ == "__main__":
    # main()
    build_model()
