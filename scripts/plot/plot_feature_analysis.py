import os
import sys
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Thêm root directory vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.transformer_model import TransformerModel

# --- CONFIG ---
DATA_DIR = "data/data_processed/transformer"
MODEL_DIR = "model/kfold"
OUTPUT_DIR = "outputs/plots"
ATTR_NAMES = [f"attr_{i+1}" for i in range(6)]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128 # Giới hạn batch size để tránh tràn RAM

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_ensemble_models():
    models = []
    folds = ["transformer_f0.pth", "transformer_f1.pth", "transformer_f2.pth"]
    for f in folds:
        path = os.path.join(MODEL_DIR, f)
        if os.path.exists(path):
            model = TransformerModel(vocab_size=30000, d_model=256, nhead=8, num_layers=4)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.to(DEVICE).eval()
            models.append(model)
            print(f"-> Loaded model: {f}")
    return models

def get_attr_saliency_single(models, x_input):
    """Tính Saliency cho 1 mẫu duy nhất (Đã fix lỗi .zero_grad)"""
    x_tensor = torch.from_numpy(x_input).long().to(DEVICE).unsqueeze(0)
    ensemble_sal = np.zeros((6, 66))
    
    for model in models:
        with torch.no_grad():
            emb = model.embedding(x_tensor)
        emb = emb.detach().requires_grad_(True)
        
        # Forward pass thủ công
        B = emb.shape[0]
        x = torch.cat((model.cls_token.expand(B, -1, -1), emb), dim=1)
        x = x + model.pos_embedding
        for layer in model.layers: x, _ = layer(x)
        x = model.norm_final(x)
        combined = torch.cat([x[:, 0, :], x.mean(dim=1)], dim=-1)
        
        out_dates = model.head_dates(combined) * model.M_dates
        out_factory = model.head_factory(combined) * model.M_factory
        final_out = torch.cat([out_dates[:, :2], out_factory[:, :1], out_dates[:, 2:], out_factory[:, 1:]], dim=1)

        attr_grads = []
        for i in range(6):
            if emb.grad is not None: emb.grad.zero_()
            final_out[0, i].backward(retain_graph=True)
            attr_grads.append(emb.grad.detach().abs().mean(dim=-1).squeeze().cpu().numpy())
            
        ensemble_sal += np.array(attr_grads)
    return ensemble_sal / len(models)

def extract_manual_features(X):
    feats = []
    for seq in X:
        v = seq[seq != 0]
        if len(v) == 0: feats.append([0]*5); continue
        feats.append([len(v), len(np.unique(v)), len(np.unique(v))/len(v), v[0], v[-1]])
    return pd.DataFrame(feats, columns=['length', 'nunique', 'velocity', 'first_item', 'last_item'])

def extract_attn_map(attn_tensor):
    attn = attn_tensor[0]  # lấy sample đầu

    if attn.dim() == 3:
        # [heads, seq, seq]
        attn = attn.mean(dim=0)
    elif attn.dim() == 2:
        # [seq, seq]
        pass
    else:
        raise ValueError(f"Unexpected attention shape: {attn.shape}")

    return attn[1:, 1:].cpu().numpy()
            
def main():
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    models = load_ensemble_models()
    if not models: return

    # --- 1. CORRELATION ---
    print("\n[1/3] Correlation Heatmap...")
    df_f = extract_manual_features(X_val)
    df_t = pd.DataFrame(y_val, columns=ATTR_NAMES)
    corr = pd.concat([df_f, df_t], axis=1).corr().loc[df_f.columns, ATTR_NAMES]
    plt.figure(figsize=(10, 6)); sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_correlation.png")); plt.close()

    # --- 2. SALIENCY ---
    print("[2/3] Attribute Saliency Map (50 samples)...")
    total_sal = np.zeros((6, 66))
    indices = np.random.choice(len(X_val), 50, replace=False)
    for idx in indices:
        total_sal += get_attr_saliency_single(models, X_val[idx])
    avg_sal = total_sal / 50
    avg_sal = (avg_sal - avg_sal.min()) / (avg_sal.max() - avg_sal.min() + 1e-9)
    plt.figure(figsize=(15, 7)); sns.heatmap(avg_sal, xticklabels=2, yticklabels=ATTR_NAMES, cmap="magma")
    plt.title("Attribute-Specific Positional Importance", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_attr_specific.png"), dpi=150); plt.close()

    # --- 3. DELTA (FIX MEMORY ERROR) ---
    print("[3/3] Delta Heatmap (Optimized Memory)...")
    all_preds = []
    # Dự báo theo Batch để tránh tràn RAM
    with torch.no_grad():
        for i in range(0, len(X_val), BATCH_SIZE):
            batch_x = torch.from_numpy(X_val[i:i+BATCH_SIZE]).to(DEVICE)
            preds, _ = models[0](batch_x)
            all_preds.append(preds.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    errors = np.mean((all_preds - y_val)**2, axis=1)
    b_idx, w_idx = np.argmin(errors), np.argmax(errors)
    
    # Chỉ lấy Attention của 2 mẫu cụ thể
    with torch.no_grad():
        _, b_attn_list = models[0](torch.from_numpy(X_val[b_idx:b_idx+1]).to(DEVICE))
        _, w_attn_list = models[0](torch.from_numpy(X_val[w_idx:w_idx+1]).to(DEVICE))
        
        # Layer cuối, mean over heads, bỏ CLS
        b_map = extract_attn_map(b_attn_list[-1])
        w_map = extract_attn_map(w_attn_list[-1])
        
    delta = b_map - w_map
    plt.figure(figsize=(10, 9)); sns.heatmap(delta, cmap="RdBu_r", center=0)
    plt.title("DELTA ATTENTION: BEST - WORST", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_delta_best_worst.png")); plt.close()

    print(f"\nSuccessfully generated 3 heatmaps in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()