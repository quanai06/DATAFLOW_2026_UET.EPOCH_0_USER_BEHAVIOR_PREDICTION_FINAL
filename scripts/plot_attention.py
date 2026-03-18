import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from src.models.transformer_model import TransformerModel

# --- CẤU HÌNH ĐỒNG BỘ ---
VOCAB_SIZE = 30000
D_MODEL = 256
NUM_LAYERS = 6
MODEL_FOLDER = "model/kfold/"
OUTPUT_DIR = "outputs/plots/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_all_models():
    models = []
    for i in range(5):
        path = os.path.join(MODEL_FOLDER, f"transformer_f{i}.pth")
        if os.path.exists(path):
            m = TransformerModel(vocab_size=VOCAB_SIZE, d_model=D_MODEL, num_layers=NUM_LAYERS).to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            m.eval()
            models.append(m)
    return models

def get_ensemble_result(models, xb, mb):
    """Lấy trung bình Output và trung bình Attention từ list models"""
    all_outs = []
    all_attns = []
    with torch.no_grad():
        for m in models:
            out, att_list = m(xb, mb)
            all_outs.append(out.cpu().numpy())
            
            # Lấy layer cuối
            attn = att_list[-1]
            if len(attn.shape) == 4: # [B, H, S, S]
                attn = attn.mean(dim=1) # -> [B, S, S]
            all_attns.append(attn.cpu().numpy())
            
    # Trung bình cộng
    avg_out = np.mean(all_outs, axis=0)
    avg_attn = np.mean(all_attns, axis=0) # [B, S, S]
    return avg_out, avg_attn

def plot_heatmap(matrix, title, filename):
    # Đảm bảo matrix là 2D
    if len(matrix.shape) == 3: matrix = matrix.mean(0)
    
    plt.figure(figsize=(10, 8))
    # Slicing [1:, 1:] để bỏ CLS token ở index 0
    sns.heatmap(matrix[1:, 1:], cmap='magma', cbar=True)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Step Index (1-66)")
    plt.ylabel("Step Index (1-66)")
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()
    print(f"✅ Đã xuất: {filename}")

def main():
    print("Khởi động quy trình Ensemble Heatmap (Lai 5 Fold)...")
    models = load_all_models()
    if len(models) == 0:
        print("❌ Lỗi: Không tìm thấy model nào trong model/kfold/")
        return

    # 1. LOAD DATA
    X_tr = np.load("data/data_processed/transformer/X_train.npy")
    M_tr = np.load("data/data_processed/transformer/mask_train.npy")
    X_va = np.load("data/data_processed/transformer/X_val.npy")
    M_va = np.load("data/data_processed/transformer/mask_val.npy")
    y_va = np.load("data/data_processed/transformer/y_val.npy")
    X_te = np.load("data/data_processed/transformer/X_test.npy")
    M_te = np.load("data/data_processed/transformer/mask_test.npy")

    # --- PHẦN 1: GLOBAL ENSEMBLE (So sánh 3 tập dữ liệu) ---
    print("📊 1. Vẽ Global Ensemble (Train - Val - Test)...")
    for name, X, M in [("train", X_tr, M_tr), ("val", X_va, M_va), ("test", X_te, M_te)]:
        # Lấy 2000 mẫu đại diện cho mỗi tập để tính Attention trung bình
        xb = torch.LongTensor(X[:2000]).to(DEVICE)
        mb = torch.BoolTensor(M[:2000]).to(DEVICE)
        _, avg_attn = get_ensemble_result(models, xb, mb)
        plot_heatmap(avg_attn, f"ENSEMBLE GLOBAL ATTENTION: {name.upper()} SET", f"1_global_{name}.png")

    # --- PHẦN 2: CASE STUDIES (Tìm mẫu Giỏi/Dốt dựa trên trung bình 5 model) ---
    print("📊 2. Truy tìm Case Studies bằng Ensemble Error...")
    all_errors = []
    # Quét batch để tìm nhanh trong 5000 mẫu tập Val
    batch_size = 100
    search_range = min(5000, len(X_va))
    for i in range(0, search_range, batch_size):
        end_idx = min(i + batch_size, search_range)
        xb = torch.LongTensor(X_va[i:end_idx]).to(DEVICE)
        mb = torch.BoolTensor(M_va[i:end_idx]).to(DEVICE)
        avg_out, _ = get_ensemble_result(models, xb, mb)
        
        # Tính MSE cho từng mẫu
        err = np.mean((avg_out - y_va[i:end_idx])**2, axis=1)
        all_errors.extend(err.tolist())
    
    best_idx = np.argmin(all_errors)
    worst_idx = np.argmax(all_errors)

    # Vẽ Attention "Lai" cho 2 mẫu cá biệt (Best/Worst)
    for idx, name in [(best_idx, "BEST"), (worst_idx, "WORST")]:
        xb_single = torch.LongTensor(X_va[idx:idx+1]).to(DEVICE)
        mb_single = torch.BoolTensor(M_va[idx:idx+1]).to(DEVICE)
        _, single_avg_attn = get_ensemble_result(models, xb_single, mb_single)
        plot_heatmap(single_avg_attn[0], f"ENSEMBLE {name} SAMPLE (Error: {all_errors[idx]:.4f})", f"2_case_{name.lower()}_ensemble.png")

    # --- PHẦN 3: BEHAVIORAL SHIFT (SỚM vs MUỘN) ---
    print("📊 3. Vẽ Behavioral Shift (Early Birds vs Late Comers)...")
    # attr_4 là tháng, attr_5 là ngày
    days = y_va[:, 3] * 30 + y_va[:, 4]
    early_idx = np.argsort(days)[:1000]
    late_idx = np.argsort(days)[-1000:]

    for name, idxs in [("early", early_idx), ("late", late_idx)]:
        xb = torch.LongTensor(X_va[idxs]).to(DEVICE)
        mb = torch.BoolTensor(M_va[idxs]).to(DEVICE)
        _, avg_attn = get_ensemble_result(models, xb, mb)
        plot_heatmap(avg_attn, f"ENSEMBLE BEHAVIOR: {name.upper()} COMPLETION", f"3_behavior_{name}.png")

    print(f"\n✨ SIÊU PHẨM ENSEMBLE ĐÃ HOÀN TẤT TẠI: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()