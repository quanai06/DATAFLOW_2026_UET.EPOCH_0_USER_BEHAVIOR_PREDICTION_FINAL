"""
Round 2 Soft-Classification Model v3.
- Stable architecture from v1.
- Upgraded stats MLP: 267 -> 256 -> 128 -> 64.
- Uses X_*_stats_v3.npy (267 dims).
- NO auxiliary losses (only expected-value MSE).
- Added --fullfit flag.
"""
import sys, pickle, os, torch, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, '.')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_PATH = "data/layer3_features/transformer_r2"
MODEL_DIR = Path("models/r2_softcls_v3")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

with open(f"{FEATURE_PATH}/action_remapper.pkl", "rb") as f:
    remap = pickle.load(f)
    VOCAB_SIZE = remap["vocab_size"]

MASK_TOKEN = VOCAB_SIZE
MAX_LEN = 66

# attr sizes (number of output classes)
NC = [12, 31, 100, 12, 31, 100]          # classes per attr
OFFSETS = [1, 1, 0, 1, 1, 0]            # raw value = class_index + offset
M_NORM = np.array([12, 31, 99, 12, 31, 99], dtype=np.float32)
W_PENALTY = np.array([1, 1, 100, 1, 1, 100], dtype=np.float32)

print(f"Device: {DEVICE}, VocabSize: {VOCAB_SIZE}")


# ─────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────
class SoftClsTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6,
                 dim_ff=2048, dropout=0.1, max_len=66, num_stat_features=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=vocab_size)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        pool_dim = d_model * 3  # mean + max + last
        
        # Upgraded stats projection: 267 -> 256 -> 128 -> 64 (3-layer MLP)
        stat_dim = 64 if num_stat_features > 0 else 0
        if num_stat_features > 0:
            self.stat_proj = nn.Sequential(
                nn.Linear(num_stat_features, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.GELU(), # Final GELU as per original F.gelu
            )
        else:
            self.stat_proj = None
            
        feat_dim = pool_dim + stat_dim

        self.shared = nn.Sequential(
            nn.Linear(feat_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Separate head per attribute
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(d_model, nc)
            ) for nc in NC
        ])

    def forward(self, x, mask, stats=None):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.embedding(x) + self.pos_embedding(positions)

        pad_mask = mask | (x == VOCAB_SIZE)
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        lengths = (~pad_mask).sum(dim=1).clamp(min=1)
        h_masked = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        mean_pool = h_masked.sum(dim=1) / lengths.unsqueeze(-1).float()
        max_pool = h_masked.masked_fill(pad_mask.unsqueeze(-1), -1e9).max(dim=1).values
        last_idx = (lengths - 1).clamp(min=0)
        last_pool = h[torch.arange(B, device=x.device), last_idx]

        feat = torch.cat([mean_pool, max_pool, last_pool], dim=1)
        if stats is not None and self.stat_proj is not None:
            feat = torch.cat([feat, self.stat_proj(stats)], dim=1)

        shared = self.shared(feat)
        return [head(shared) for head in self.heads]


# ─────────────────────────────────────────────────────────
# Loss: expected-value MSE (NO CE auxiliary loss as requested)
# ─────────────────────────────────────────────────────────
class SoftClsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Pre-register class index tensors
        self.register_buffer('cls_100', torch.arange(100, dtype=torch.float32))
        self.register_buffer('cls_12',  torch.arange(12,  dtype=torch.float32))
        self.register_buffer('cls_31',  torch.arange(31,  dtype=torch.float32))
        self._cls = [self.cls_12, self.cls_31, self.cls_100,
                     self.cls_12, self.cls_31, self.cls_100]

    def forward(self, logits_list, y_raw):
        total = 0.0
        for i in range(6):
            probs = F.softmax(logits_list[i], dim=1)
            cls_idx = self._cls[i].to(logits_list[i].device)
            expected = (probs * cls_idx).sum(dim=1)
            expected_raw = expected + OFFSETS[i]

            # Competition MSE (normalized)
            mse = F.mse_loss(expected_raw / M_NORM[i],
                             y_raw[:, i].float() / M_NORM[i])

            total = total + W_PENALTY[i] * mse

        return total


# ─────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────
class SoftClsDataset(Dataset):
    def __init__(self, seqs, masks, stats, y_raw, augment=True):
        self.seqs = seqs
        self.masks = masks
        self.stats = stats
        self.y_raw = y_raw
        self.augment = augment

    def __len__(self): return len(self.seqs)

    def __getitem__(self, idx):
        seq  = self.seqs[idx].copy()
        mask = self.masks[idx].copy()
        stats = self.stats[idx].copy()

        if self.augment:
            L = int((~mask).sum())
            if L > 4 and np.random.random() < 0.40:
                ratio = np.clip(np.random.normal(0.85, 0.10), 0.5, 1.0)
                crop_len = max(3, int(L * ratio))
                start = np.random.randint(0, max(1, L - crop_len + 1))
                new_seq = np.full_like(seq, VOCAB_SIZE)
                new_mask = np.ones_like(mask)
                new_seq[:crop_len] = seq[start:start + crop_len]
                new_mask[:crop_len] = False
                seq, mask = new_seq, new_mask

            for pos in np.where(~mask)[0]:
                if np.random.random() < 0.15:
                    seq[pos] = VOCAB_SIZE

        return (torch.LongTensor(seq), torch.BoolTensor(mask),
                torch.FloatTensor(stats), torch.LongTensor(self.y_raw[idx]))


# ─────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────
CLS_IDX = [np.arange(nc, dtype=np.float32) for nc in NC]

def decode_expected(logits_list):
    preds = np.zeros((len(logits_list[0]), 6), dtype=int)
    for i in range(6):
        probs = logits_list[i]
        expected = (probs * CLS_IDX[i]).sum(axis=1)
        raw = np.round(expected + OFFSETS[i]).astype(int)
        lo = OFFSETS[i]
        hi = NC[i] - 1 + OFFSETS[i]
        preds[:, i] = raw.clip(lo, hi)
    return preds


def competition_score(y_true, y_pred):
    s = 0.0
    for j in range(6):
        d = y_true[:, j] / M_NORM[j] - y_pred[:, j] / M_NORM[j]
        s += W_PENALTY[j] * np.mean(d ** 2)
    return s / 6.0


def evaluate(model, seqs, masks, stats, y_raw, batch_size=512):
    model.eval()
    acc = [[] for _ in range(6)]
    with torch.no_grad():
        for s in range(0, len(seqs), batch_size):
            e = min(s + batch_size, len(seqs))
            x = torch.LongTensor(seqs[s:e]).to(DEVICE)
            m = torch.BoolTensor(masks[s:e]).to(DEVICE)
            st = torch.FloatTensor(stats[s:e]).to(DEVICE)
            out = model(x, m, st)
            for i in range(6):
                acc[i].append(F.softmax(out[i], 1).cpu().numpy())
    probs = [np.concatenate(acc[i]) for i in range(6)]
    preds = decode_expected(probs)
    return competition_score(y_raw, preds), np.mean(np.all(y_raw == preds, axis=1))


# ─────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────
def train(seed=42, n_folds=5, d_model=512, num_layers=6, num_epochs=40,
          batch_size=128, lr=2e-4, resume=False, fullfit=False):
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"SoftCls V3 Training: seed={seed}, d={d_model}, L={num_layers}, bs={batch_size}, fullfit={fullfit}")
    print(f"{'='*60}")

    # Load V3 Stats Features (267 dims)
    X_tr_seq   = np.load(f"{FEATURE_PATH}/X_train_seq.npy")
    X_tr_mask  = np.load(f"{FEATURE_PATH}/X_train_mask.npy")
    X_tr_stats = np.load(f"{FEATURE_PATH}/X_train_stats_v3.npy")
    y_tr_raw   = np.load(f"{FEATURE_PATH}/y_train_raw.npy")

    X_va_seq   = np.load(f"{FEATURE_PATH}/X_val_seq.npy")
    X_va_mask  = np.load(f"{FEATURE_PATH}/X_val_mask.npy")
    X_va_stats = np.load(f"{FEATURE_PATH}/X_val_stats_v3.npy")
    y_va_raw   = np.load(f"{FEATURE_PATH}/y_val_raw.npy")

    n_stat = X_tr_stats.shape[1]
    print(f"Train={X_tr_seq.shape[0]}, Val={X_va_seq.shape[0]}, Stats={n_stat}")

    criterion = SoftClsLoss().to(DEVICE)

    if fullfit:
        # Train on all data
        print("\n--- Full Fit Mode (Training on all data) ---")
        X_all_seq = np.concatenate([X_tr_seq, X_va_seq])
        X_all_mask = np.concatenate([X_tr_mask, X_va_mask])
        X_all_stats = np.concatenate([X_tr_stats, X_va_stats])
        y_all_raw = np.concatenate([y_tr_raw, y_va_raw])

        ds = SoftClsDataset(X_all_seq, X_all_mask, X_all_stats, y_all_raw, augment=True)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)

        model = SoftClsTransformer(
            vocab_size=VOCAB_SIZE, d_model=d_model, nhead=8,
            num_layers=num_layers, dim_ff=d_model * 4,
            dropout=0.1, max_len=MAX_LEN, num_stat_features=n_stat
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(loader),
            epochs=num_epochs, pct_start=0.1
        )

        for epoch in range(num_epochs):
            model.train()
            ep_loss, nb = 0.0, 0
            for seqs, msks, sts, y_raw_b in loader:
                seqs = seqs.to(DEVICE); msks = msks.to(DEVICE)
                sts = sts.to(DEVICE);   y_raw_b = y_raw_b.to(DEVICE)
                optimizer.zero_grad()
                out = model(seqs, msks, sts)
                loss = criterion(out, y_raw_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
                ep_loss += loss.item(); nb += 1
            print(f"  Ep {epoch+1:2d}/{num_epochs}: loss={ep_loss/nb:.4f}")

        model_path = MODEL_DIR / f"softcls_v3_fullfit_seed{seed}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": VOCAB_SIZE, "d_model": d_model, "nhead": 8,
                "num_layers": num_layers, "dim_ff": d_model * 4,
                "dropout": 0.1, "max_len": MAX_LEN, "num_stat_features": n_stat
            },
            "seed": seed,
        }, model_path)
        print(f"  Saved FullFit: {model_path}")
        return [0]

    # CV Mode
    strat_key = (y_tr_raw[:, 2] // 10) * 10 + (y_tr_raw[:, 5] // 10)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best_scores = []

    for fold, (tr_idx, va_fold_idx) in enumerate(skf.split(X_tr_seq, strat_key)):
        model_path = MODEL_DIR / f"softcls_v3_seed{seed}_fold{fold}.pt"
        if resume and model_path.exists():
            print(f"Fold {fold}: skip (exists)")
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            best_scores.append(ckpt.get('competition_score', 0))
            continue

        print(f"\n--- Fold {fold} ---")

        ds = SoftClsDataset(X_tr_seq[tr_idx], X_tr_mask[tr_idx],
                            X_tr_stats[tr_idx], y_tr_raw[tr_idx], augment=True)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)

        model = SoftClsTransformer(
            vocab_size=VOCAB_SIZE, d_model=d_model, nhead=8,
            num_layers=num_layers, dim_ff=d_model * 4,
            dropout=0.1, max_len=MAX_LEN, num_stat_features=n_stat
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(loader),
            epochs=num_epochs, pct_start=0.1
        )

        best_score = float('inf')
        best_state = None
        patience = 0
        MAX_PATIENCE = 10

        for epoch in range(num_epochs):
            model.train()
            ep_loss, nb = 0.0, 0
            for seqs, msks, sts, y_raw_b in loader:
                seqs = seqs.to(DEVICE); msks = msks.to(DEVICE)
                sts = sts.to(DEVICE);   y_raw_b = y_raw_b.to(DEVICE)
                optimizer.zero_grad()
                out = model(seqs, msks, sts)
                loss = criterion(out, y_raw_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
                ep_loss += loss.item(); nb += 1

            # Evaluate on fold holdout
            fold_score, fold_em = evaluate(
                model, X_tr_seq[va_fold_idx], X_tr_mask[va_fold_idx],
                X_tr_stats[va_fold_idx], y_tr_raw[va_fold_idx])
            print(f"  Ep {epoch+1:2d}/{num_epochs}: loss={ep_loss/nb:.4f}  fold_val={fold_score:.4f}  em={fold_em:.4f}")

            if fold_score < best_score:
                best_score = fold_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= MAX_PATIENCE:
                    print(f"  Early stop at epoch {epoch+1}")
                    break

        model.load_state_dict(best_state)
        val_score, val_em = evaluate(model, X_va_seq, X_va_mask, X_va_stats, y_va_raw)
        print(f"  -> Full val: {val_score:.4f}, EM: {val_em:.4f}")

        torch.save({
            "model_state_dict": best_state,
            "config": {
                "vocab_size": VOCAB_SIZE, "d_model": d_model, "nhead": 8,
                "num_layers": num_layers, "dim_ff": d_model * 4,
                "dropout": 0.1, "max_len": MAX_LEN, "num_stat_features": n_stat
            },
            "competition_score": val_score,
            "fold": fold, "seed": seed,
        }, model_path)
        print(f"  Saved: {model_path}")
        best_scores.append(val_score)

    print(f"\nSeed {seed} avg: {np.mean(best_scores):.4f}")
    return best_scores


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--folds",  type=int, default=5)
    p.add_argument("--d_model",type=int, default=512)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch",  type=int, default=128)
    p.add_argument("--lr",     type=float, default=2e-4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fullfit",action="store_true")
    args = p.parse_args()
    train(seed=args.seed, n_folds=args.folds, d_model=args.d_model,
          num_layers=args.layers, num_epochs=args.epochs,
          batch_size=args.batch, lr=args.lr, resume=args.resume,
          fullfit=args.fullfit)
