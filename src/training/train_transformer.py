import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from src.utils.loaders import SequenceDataset
from src.models.transformer_model import TransformerModel
from src.models.losses import get_loss
from src.metrics.metrics import CompetitionMetric 

def run_train_transformer(train_data, val_data, config):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_loader = DataLoader(SequenceDataset(*train_data), batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(SequenceDataset(*val_data), batch_size=config['batch_size'])

    model = TransformerModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.05)
    
    # OneCycleLR giúp Loss giảm cực đẹp ở giai đoạn cuối
    total_steps = len(train_loader) * config['epochs']
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], total_steps=total_steps)
    
    criterion = get_loss().to(DEVICE)
    metrics_calc = CompetitionMetric(DEVICE)
    best_score = float('inf')

    for epoch in range(config['epochs']):
        model.train()
        train_loss_epoch = 0
        
        # --- LIVE OBSERVER TRONG TQDM ---
        pbar = tqdm(train_loader, desc=f"🚀 Fold Epoch {epoch+1}")
        for xb, mb, yb in pbar:
            xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()
            outputs, _ = model(xb, mb)
            loss = criterion(outputs, yb)
            loss.backward()
            
            # Cắt gradient để tránh "nổ" loss làm đường đi bị giật
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            current_loss = loss.item()
            train_loss_epoch += current_loss
            
            # Hiển thị loss hiện tại ngay trên thanh tiến trình
            pbar.set_postfix({'batch_loss': f'{current_loss:.6f}', 'lr': f'{optimizer.param_groups[0]['lr']:.8f}'})

        # --- VALIDATION ---
        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for xb, mb, yb in val_loader:
                out, _ = model(xb.to(DEVICE), mb.to(DEVICE))
                all_preds.append(out)
                all_trues.append(yb.to(DEVICE))
        
        results = metrics_calc.compute_all(torch.cat(all_preds), torch.cat(all_trues))
        
        # --- CHI TIẾT LOSS GIẢM DẦN ---
        print(f"\n --- [EPOCH {epoch+1} SUMMARY] ---")
        print(f"   🔹 Train Loss Avg: {train_loss_epoch/len(train_loader):.6f}")
        print(f"   🏆 [KAGGLE SCORE]: {results['comp_score']:.6f}")
        print(f"   📍 MSE Per Col: {np.round(results['mse_per_col'], 2)}") 
        # Soi kỹ cái index 2 và 5 (cột 99), nếu nó giảm dần từ 500 -> 300 -> 100 là b đang thắng.

        if results['comp_score'] < best_score:
            best_score = results['comp_score']
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"   🌟 New Best Score! Model saved.")
        print("-" * 40)

    return model, best_score