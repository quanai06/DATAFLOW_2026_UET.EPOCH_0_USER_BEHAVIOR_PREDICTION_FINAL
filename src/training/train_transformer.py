import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

# Giả định cấu trúc thư mục của bạn
from src.utils.loaders import SequenceDataset
from src.models.transformer_model import TransformerModel
from src.models.losses import get_loss
from src.metrics.metrics import CompetitionMetric 

def run_train_transformer(train_data, val_data, config):
    """
    train_data: (X_train, S_train, M_train, Y_train)
    val_data: (X_val, S_val, M_val, Y_val)
    """
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- 1. LOAD DATA ---
    # SequenceDataset bây giờ nhận 4 tham số: X (chuỗi), S (expert), M (mask), Y (target)
    train_loader = DataLoader(SequenceDataset(*train_data), batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(SequenceDataset(*val_data), batch_size=config['batch_size'])

    # Khởi tạo Model với expert_dim=22 (khớp với file build_features)
    model = TransformerModel(expert_dim=16).to(DEVICE)
    
    # Optimizer & Scheduler (Bám sát báo cáo để xử lý Attention Drift)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.05)
    total_steps = len(train_loader) * config['epochs']
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], total_steps=total_steps)
    
    criterion = get_loss().to(DEVICE)
    metrics_calc = CompetitionMetric(DEVICE)
    best_score = float('inf')

    print(f"🔥 Training started on {DEVICE}. Focus: Attr 3 & 6 (Alpha=100)")

    for epoch in range(config['epochs']):
        model.train()
        train_loss_epoch = 0
        
        pbar = tqdm(train_loader, desc=f"🚀 Fold Epoch {epoch+1}")
        for xb, sb, mb, yb in pbar:
            # Đưa toàn bộ dữ liệu (bao gồm cả nhánh S) lên GPU
            xb, sb, mb, yb = xb.to(DEVICE), sb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward: Model nhận cả X (chuỗi) và S (Expert features)
            outputs, _ = model(xb, sb, mb)
            
            loss = criterion(outputs, yb)
            loss.backward()
            
            # Cắt gradient để tránh "nổ" loss khi gặp các ca "Worst Sample" (Mục 2.2.2)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            current_loss = loss.item()
            train_loss_epoch += current_loss
            
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}', 
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        # --- VALIDATION ---
        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for xb, sb, mb, yb in val_loader:
                # Validation cũng phải truyền cả sb vào
                out, _ = model(xb.to(DEVICE), sb.to(DEVICE), mb.to(DEVICE))
                all_preds.append(out)
                all_trues.append(yb.to(DEVICE))
        
        y_pred_cat = torch.cat(all_preds)
        y_true_cat = torch.cat(all_trues)
        results = metrics_calc.compute_all(y_pred_cat, y_true_cat)
        
        # --- LOG CHI TIẾT THEO BÁO CÁO ---
        print(f"\n --- [EPOCH {epoch+1} SUMMARY] ---")
        print(f"   🔹 Avg Train Loss: {train_loss_epoch/len(train_loader):.6f}")
        print(f"   🏆 [KAGGLE SCORE]: {results['comp_score']:.6f}")
        
        # Soi kỹ MSE của Attr 3 (index 2) và Attr 6 (index 5)
        mse_cols = results['mse_per_col']
        print(f"   📍 MSE Dates (1,2,4,5): [{mse_cols[0]:.2f}, {mse_cols[1]:.2f}, {mse_cols[3]:.2f}, {mse_cols[4]:.2f}]")
        print(f"   🔥 MSE Factory (3,6):   [{mse_cols[2]:.2f}, {mse_cols[5]:.2f}] <-- Cần giảm mạnh cái này!")

        # Lưu model nếu đạt điểm Kaggle tốt nhất
        if results['comp_score'] < best_score:
            best_score = results['comp_score']
            os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"   🌟 New Best Score! Saving to {config['model_save_path']}")
        print("-" * 50)

    return model, best_score