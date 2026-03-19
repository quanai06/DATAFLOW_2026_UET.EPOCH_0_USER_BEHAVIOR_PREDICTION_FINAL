"""
Simple regression approach (inspired by god_tier 0.25329).
Pure regression with sigmoid, NOT classification.
Split heads: date head (attr_1,2,4,5) + capacity head (attr_3,6).
Loss: exact competition metric (weighted MSE).
Ensemble: median of LSTM + GRU + CNN models.
"""
import argparse, sys, os, torch, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_PATH = "data/layer3_features/transformer_r2"
MAX_LEN = 66
VOCAB_SIZE = 951  # tokens 1-950, PAD=0

M_CONST = np.array([12.0, 31.0, 99.0, 12.0, 31.0, 99.0], dtype=np.float32)
W_CONST = np.array([1.0, 1.0, 100.0, 1.0, 1.0, 100.0], dtype=np.float32)
M_CONST_T = torch.tensor(M_CONST, device=DEVICE)
W_CONST_T = torch.tensor(W_CONST, device=DEVICE)

print(f"Device: {DEVICE}")


# ─────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────
_KEY_TOKENS_LIST = [783, 764, 801, 723]

class RegressionModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=192, model_type='lstm',
                 dropout=0.3, max_len=MAX_LEN, num_stat_features=30,
                 prefix_attn=0, len_cond=False, dual_gate=False, prefix_len=15,
                 scalable_softmax=False, film_cond=False,
                 pos_emb=False, seg_emb=False):
        super().__init__()
        self.model_type = model_type
        self.prefix_attn = prefix_attn
        self.len_cond = len_cond
        self.dual_gate = dual_gate
        self.prefix_len = prefix_len
        self.max_len = max_len
        self.scalable_softmax = scalable_softmax
        self.film_cond = film_cond
        self.pos_emb = pos_emb
        self.seg_emb = seg_emb
        self._last_hn = None
        self.embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        # Positional embedding: one learned vector per absolute position (0..MAX_LEN-1)
        if pos_emb:
            self.pos_embedding = nn.Embedding(max_len, d_model)
        # Segment embedding: 3 learned vectors for early/middle/late thirds of sequence
        if seg_emb:
            self.seg_embedding = nn.Embedding(3, d_model)

        if model_type in ('lstm', 'gru'):
            rnn_cls = nn.LSTM if model_type == 'lstm' else nn.GRU
            self.rnn = rnn_cls(d_model, d_model, batch_first=True,
                               bidirectional=True, num_layers=2, dropout=0.15)
            self.embed_drop = nn.Dropout(0.1)
            feat_dim = d_model * 2  # bidirectional
            # Attention
            self.attention = nn.Sequential(
                nn.Linear(feat_dim, 1),
                nn.Tanh()
            )
            # FiLM: modulate GRU hidden states with [len_norm, key_frac] before pooling
            # gamma/beta applied per-feature: rnn_out = (1+gamma) * rnn_out + beta
            if film_cond:
                self.film_net = nn.Sequential(
                    nn.Linear(2, 64),
                    nn.ReLU(),
                    nn.Linear(64, feat_dim * 2)  # outputs gamma + beta, each (feat_dim,)
                )
            # Dual-gate: separate attention head for prefix pool + gate projection
            if dual_gate:
                self.attention_prefix = nn.Sequential(
                    nn.Linear(feat_dim, 1),
                    nn.Tanh()
                )
                self.gate_proj = nn.Linear(2, 1)  # [len_norm, key_frac] → scalar gate
        else:  # cnn — multi-scale for more compute
            self.convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, k, padding=k//2),
                    nn.BatchNorm1d(d_model),
                    nn.ReLU()
                ) for k in (3, 5, 7)
            ])
            self.embed_drop = nn.Dropout(0.1)
            feat_dim = d_model * 3  # concat 3 kernel sizes

        # Stats branch (larger for v3 features with 155 dims)
        if num_stat_features > 0:
            stat_hidden = 128 if num_stat_features > 60 else 64
            self.stat_proj = nn.Sequential(
                nn.Linear(num_stat_features, stat_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(stat_hidden, 64),
                nn.ReLU()
            )
        else:
            self.stat_proj = None

        merged_dim = feat_dim + (64 if num_stat_features > 0 else 0)

        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(merged_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Length-conditioned correction for cap_head (attr_3/attr_6 only)
        # Projects [len_norm, key_frac] into a residual added to shared before cap_head
        self.len_proj = nn.Linear(2, 256) if len_cond else None

        # Split head 1: dates (attr_1, attr_2, attr_4, attr_5) — 6% of score
        self.date_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

        # Split head 2: capacity (attr_3, attr_6) — 94% of score, deeper
        self.cap_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, x, mask, stats=None, h0=None):
        self._last_hn = None
        B, L = x.shape
        h = self.embedding(x)  # (B, L, d)

        if self.pos_emb:
            pos_ids = torch.arange(L, device=x.device).unsqueeze(0)  # (1, L)
            h = h + self.pos_embedding(pos_ids)

        if self.seg_emb:
            # Segment index per position based on actual sequence length (not MAX_LEN)
            lengths_f_seg = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)  # (B, 1)
            pos_f = torch.arange(L, device=x.device).unsqueeze(0).float()          # (1, L)
            seg_ids = (pos_f / lengths_f_seg * 3).long().clamp(max=2)              # (B, L)
            h = h + self.seg_embedding(seg_ids)

        h = self.embed_drop(h)

        if self.model_type in ('lstm', 'gru'):
            # mask: True=padding, False=real
            lengths_f = (~mask).sum(dim=1).float()
            lengths = lengths_f.clamp(min=1).cpu()
            packed = pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
            if self.model_type == 'lstm' and h0 is not None and not isinstance(h0, tuple):
                h0 = (h0, torch.zeros_like(h0))
            if self.model_type == 'lstm':
                rnn_out, (hn, cn) = self.rnn(packed, h0)
                self._last_hn = (hn.detach(), cn.detach())
            else:
                rnn_out, hn = self.rnn(packed, h0)
                self._last_hn = hn.detach()
            rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True, total_length=L)

            # FiLM: scale+shift hidden states conditioned on [len_norm, key_frac]
            # Applied before pooling so what gets aggregated is already length-adapted
            if self.film_cond:
                key_t = torch.tensor(_KEY_TOKENS_LIST, device=x.device)
                is_key_f = (x.unsqueeze(-1) == key_t).any(-1) & ~mask       # (B, L)
                key_frac_f = is_key_f.float().sum(1) / lengths_f.clamp(min=1)
                film_in = torch.stack([lengths_f / self.max_len, key_frac_f], dim=1)  # (B, 2)
                film_params = self.film_net(film_in)                          # (B, feat_dim*2)
                feat_dim = rnn_out.shape[-1]
                gamma = film_params[:, :feat_dim].unsqueeze(1)                # (B, 1, feat_dim)
                beta  = film_params[:, feat_dim:].unsqueeze(1)                # (B, 1, feat_dim)
                rnn_out = (1.0 + gamma) * rnn_out + beta                      # identity init

            # Attention pooling (masked)
            pos_ids = torch.arange(L, device=mask.device).unsqueeze(0)
            if self.dual_gate:
                # Full attention pool
                att_full = self.attention(rnn_out).squeeze(-1)
                att_full = att_full.masked_fill(mask, -1e4)
                context_full = (rnn_out * F.softmax(att_full, dim=1).unsqueeze(-1)).sum(dim=1)

                # Prefix attention pool (first prefix_len tokens)
                prefix_mask = mask | (pos_ids >= self.prefix_len)
                att_pre = self.attention_prefix(rnn_out).squeeze(-1)
                att_pre = att_pre.masked_fill(prefix_mask, -1e4)
                context_prefix = (rnn_out * F.softmax(att_pre, dim=1).unsqueeze(-1)).sum(dim=1)

                # Gate conditioned on [len_norm, key_frac]
                lengths_f = (~mask).sum(dim=1).float()
                len_norm_g = lengths_f / self.max_len
                key_t = torch.tensor(_KEY_TOKENS_LIST, device=x.device)
                is_key = (x.unsqueeze(-1) == key_t).any(-1) & ~mask
                key_frac_g = is_key.float().sum(dim=1) / lengths_f.clamp(min=1)
                len_feat_g = torch.stack([len_norm_g, key_frac_g], dim=1)
                g = torch.sigmoid(self.gate_proj(len_feat_g))  # (B, 1)
                context = g * context_prefix + (1 - g) * context_full
            else:
                att_scores = self.attention(rnn_out).squeeze(-1)
                att_mask = mask
                if self.prefix_attn > 0:
                    att_mask = mask | (pos_ids >= self.prefix_attn)
                if self.scalable_softmax:
                    len_scale = torch.sqrt(torch.tensor(13.9, device=att_scores.device) /
                                           lengths_f.clamp(min=1.0))
                    att_scores = att_scores * len_scale.unsqueeze(-1)
                att_scores = att_scores.masked_fill(att_mask, -1e4)
                att_weights = F.softmax(att_scores, dim=1)
                context = (rnn_out * att_weights.unsqueeze(-1)).sum(dim=1)  # (B, feat_dim)
        else:  # CNN — multi-scale
            h = h.transpose(1, 2)  # (B, d, L)
            pooled = []
            for conv in self.convs:
                c = conv(h)  # (B, d, L)
                c = c.masked_fill(mask.unsqueeze(1), -1e4)
                pooled.append(c.max(dim=2).values)  # (B, d)
            context = torch.cat(pooled, dim=1)  # (B, d*3)

        if stats is not None and self.stat_proj is not None:
            context = torch.cat([context, self.stat_proj(stats)], dim=1)

        shared = self.shared(context)

        if self.len_proj is not None:
            lengths_f = (~mask).sum(dim=1).float()
            len_norm = lengths_f / self.max_len                          # (B,)
            key_t = torch.tensor(_KEY_TOKENS_LIST, device=x.device)
            is_key = (x.unsqueeze(-1) == key_t).any(-1) & ~mask         # (B, L)
            key_frac = is_key.float().sum(dim=1) / lengths_f.clamp(min=1)  # (B,)
            len_feat = torch.stack([len_norm, key_frac], dim=1)          # (B, 2)
            cap_shared = shared + self.len_proj(len_feat)
        else:
            cap_shared = shared

        date_out = self.date_head(shared)      # (B, 4) in [0,1]
        cap_out = self.cap_head(cap_shared)    # (B, 2) in [0,1]

        # Order: attr_1, attr_2, attr_3, attr_4, attr_5, attr_6
        output = torch.cat([
            date_out[:, 0:2],  # attr_1, attr_2
            cap_out[:, 0:1],   # attr_3
            date_out[:, 2:4],  # attr_4, attr_5
            cap_out[:, 1:2],   # attr_6
        ], dim=1)  # (B, 6) all sigmoid [0,1]

        return output


# ─────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────
class RegDataset(Dataset):
    def __init__(self, seqs, masks, stats, y_scaled=None, augment=False):
        self.seqs = seqs; self.masks = masks
        self.stats = stats; self.y_scaled = y_scaled
        self.augment = augment

    def __len__(self): return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx].copy()
        mask = self.masks[idx].copy()
        stats = self.stats[idx].copy()

        if self.augment:
            L = int((~mask).sum())
            # Light token masking only (10%)
            for pos in np.where(~mask)[0]:
                if np.random.random() < 0.10:
                    seq[pos] = 0  # PAD token

        out = (torch.LongTensor(seq), torch.BoolTensor(mask), torch.FloatTensor(stats))
        if self.y_scaled is not None:
            out = out + (torch.FloatTensor(self.y_scaled[idx]),)
        return out


# ─────────────────────────────────────────────────────
# Loss: exact competition metric
# ─────────────────────────────────────────────────────
_W_CAP_ONLY_T = torch.tensor([0., 0., 100., 0., 0., 100.], device=DEVICE)

def competition_loss(pred_scaled, true_scaled, cap_only=False, reduction='mean'):
    """pred_scaled and true_scaled are both in [0,1] range (y/M)."""
    sq_err = (pred_scaled - true_scaled) ** 2
    w = _W_CAP_ONLY_T if cap_only else W_CONST_T
    per_sample = (sq_err * w).sum(dim=1) / 6.0   # (B,)
    return per_sample.mean() if reduction == 'mean' else per_sample


def competition_score_np(y_true_raw, y_pred_raw):
    """Compute competition score from raw integer predictions."""
    score = 0.0
    for j in range(6):
        d = y_true_raw[:, j] / M_CONST[j] - y_pred_raw[:, j] / M_CONST[j]
        score += W_CONST[j] * np.mean(d ** 2)
    return score / 6.0


def decode_predictions(pred_scaled):
    """Convert sigmoid [0,1] predictions to raw integer predictions."""
    raw = pred_scaled * M_CONST
    raw = np.round(raw).astype(int)
    # Clip to valid ranges
    raw[:, 0] = np.clip(raw[:, 0], 1, 12)   # attr_1: month
    raw[:, 1] = np.clip(raw[:, 1], 1, 31)   # attr_2: day
    raw[:, 2] = np.clip(raw[:, 2], 0, 99)   # attr_3: capacity
    raw[:, 3] = np.clip(raw[:, 3], 1, 12)   # attr_4: month
    raw[:, 4] = np.clip(raw[:, 4], 1, 31)   # attr_5: day
    raw[:, 5] = np.clip(raw[:, 5], 0, 99)   # attr_6: capacity
    return raw


def calendar_postprocess(preds):
    """Swap start/end dates if start > end, clip days by month."""
    preds = preds.copy()
    for i in range(len(preds)):
        m1, d1, m2, d2 = preds[i, 0], preds[i, 1], preds[i, 3], preds[i, 4]
        # Swap if start > end
        if m1 > m2:
            preds[i, 0], preds[i, 3] = m2, m1
            preds[i, 1], preds[i, 4] = d2, d1
        elif m1 == m2 and d1 > d2:
            preds[i, 1], preds[i, 4] = d2, d1

        # Clip days by month
        for m_idx, d_idx in [(0, 1), (3, 4)]:
            m = preds[i, m_idx]
            if m == 2:
                preds[i, d_idx] = np.clip(preds[i, d_idx], 1, 29)
            elif m in (4, 6, 9, 11):
                preds[i, d_idx] = np.clip(preds[i, d_idx], 1, 30)
            else:
                preds[i, d_idx] = np.clip(preds[i, d_idx], 1, 31)
    return preds


# ─────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────
def train_model(model_type, seqs, masks, stats, y_scaled,
                val_seqs=None, val_masks=None, val_stats=None, val_y_raw=None,
                epochs=15, batch_size=128, lr=2e-3, seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    n_stat = stats.shape[1] if stats is not None else 0

    model = RegressionModel(
        vocab_size=VOCAB_SIZE, d_model=192, model_type=model_type,
        dropout=0.3, max_len=MAX_LEN, num_stat_features=n_stat
    ).to(DEVICE)

    ds = RegDataset(seqs, masks, stats, y_scaled, augment=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Exponential decay like god_tier: lr * 0.8^epoch
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    best_val = float('inf')
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0; n_batches = 0
        for batch in loader:
            seq_b, mask_b, stat_b, y_b = [b.to(DEVICE) for b in batch]
            pred = model(seq_b, mask_b, stat_b)
            loss = competition_loss(pred, y_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1

        scheduler.step()

        msg = f"  Ep {ep:2d}/{epochs}: loss={total_loss/n_batches:.4f}"

        if val_seqs is not None:
            val_preds = predict(model, val_seqs, val_masks, val_stats)
            val_raw = decode_predictions(val_preds)
            val_score = competition_score_np(val_y_raw, val_raw)
            msg += f"  val={val_score:.4f}"
            if val_score < best_val:
                best_val = val_score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(msg, flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val


def predict(model, seqs, masks, stats, batch_size=4096):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for s in range(0, len(seqs), batch_size):
            e = min(s + batch_size, len(seqs))
            x = torch.LongTensor(seqs[s:e]).to(DEVICE)
            m = torch.BoolTensor(masks[s:e]).to(DEVICE)
            st = torch.FloatTensor(stats[s:e]).to(DEVICE)
            pred = model(x, m, st)
            all_preds.append(pred.cpu().numpy())
    return np.concatenate(all_preds)


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_per_type', type=int, default=3, help='Models per type (default 3)')
    parser.add_argument('--epochs_check', type=int, default=20, help='Epochs for val check')
    parser.add_argument('--epochs_full', type=int, default=20, help='Epochs for fullfit')
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--skip_check', action='store_true', help='Skip phase 1')
    parser.add_argument('--stats_version', type=str, default='v3', choices=['v1', 'v2', 'v3'],
                        help='Feature version: v1=30, v2=60, v3=155 (with token features)')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    X_tr_seq   = np.load(f"{FEATURE_PATH}/X_train_seq.npy")
    X_tr_mask  = np.load(f"{FEATURE_PATH}/X_train_mask.npy")
    y_tr_raw   = np.load(f"{FEATURE_PATH}/y_train_raw.npy")

    X_va_seq   = np.load(f"{FEATURE_PATH}/X_val_seq.npy")
    X_va_mask  = np.load(f"{FEATURE_PATH}/X_val_mask.npy")
    y_va_raw   = np.load(f"{FEATURE_PATH}/y_val_raw.npy")

    X_te_seq   = np.load(f"{FEATURE_PATH}/X_test_seq.npy")
    X_te_mask  = np.load(f"{FEATURE_PATH}/X_test_mask.npy")
    ids_test   = np.load(f"{FEATURE_PATH}/ids_test.npy", allow_pickle=True)

    # Load stats based on version
    stat_suffix = {'v1': '', 'v2': '_v2', 'v3': '_v3'}[args.stats_version]
    X_tr_stats = np.load(f"{FEATURE_PATH}/X_train_stats{stat_suffix}.npy")
    X_va_stats = np.load(f"{FEATURE_PATH}/X_val_stats{stat_suffix}.npy")
    X_te_stats = np.load(f"{FEATURE_PATH}/X_test_stats{stat_suffix}.npy")
    print(f"  Stats version: {args.stats_version} ({X_tr_stats.shape[1]} features)")

    # Scale targets to [0,1]
    y_tr_scaled = y_tr_raw.astype(np.float32) / M_CONST
    y_va_scaled = y_va_raw.astype(np.float32) / M_CONST

    # Fullfit data
    X_full_seq   = np.concatenate([X_tr_seq, X_va_seq])
    X_full_mask  = np.concatenate([X_tr_mask, X_va_mask])
    X_full_stats = np.concatenate([X_tr_stats, X_va_stats])
    y_full_raw   = np.concatenate([y_tr_raw, y_va_raw])
    y_full_scaled = y_full_raw.astype(np.float32) / M_CONST

    print(f"Train: {len(X_tr_seq)}, Val: {len(X_va_seq)}, Test: {len(X_te_seq)}")
    print(f"Fullfit: {len(X_full_seq)}, Stats: {X_tr_stats.shape[1]}")

    model_types = ['lstm', 'gru', 'cnn']
    seeds = list(range(args.n_per_type))

    # ───── PHASE 1: Check on val ─────
    if not args.skip_check:
        print("\n" + "=" * 50)
        print("PHASE 1: CHECK ON VALIDATION")
        print("=" * 50)

        for mt in model_types:
            print(f"\n--- {mt.upper()} (seed=0) ---")
            model, best_val = train_model(
                mt, X_tr_seq, X_tr_mask, X_tr_stats, y_tr_scaled,
                val_seqs=X_va_seq, val_masks=X_va_mask,
                val_stats=X_va_stats, val_y_raw=y_va_raw,
                epochs=args.epochs_check, batch_size=args.batch, seed=42
            )
            print(f"  Best val: {best_val:.4f}")

    # ───── PHASE 2: Fullfit + Ensemble ─────
    print("\n" + "=" * 50)
    print(f"PHASE 2: FULLFIT ENSEMBLE ({args.n_per_type} x {len(model_types)} = {args.n_per_type * len(model_types)} models)")
    print("=" * 50)

    all_test_preds = []
    all_val_preds = []
    model_seeds = [42, 123, 456, 789, 2026][:args.n_per_type]

    for mt in model_types:
        for si, seed in enumerate(model_seeds):
            print(f"\n--- {mt.upper()} seed={seed} ({si+1}/{len(model_seeds)}) ---")
            model, _ = train_model(
                mt, X_full_seq, X_full_mask, X_full_stats, y_full_scaled,
                epochs=args.epochs_full, batch_size=args.batch, lr=2e-3, seed=seed
            )

            te_pred = predict(model, X_te_seq, X_te_mask, X_te_stats)
            va_pred = predict(model, X_va_seq, X_va_mask, X_va_stats)
            all_test_preds.append(te_pred)
            all_val_preds.append(va_pred)

            # Save model
            os.makedirs("models/r2_regression", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {'vocab_size': VOCAB_SIZE, 'd_model': 192,
                           'model_type': mt, 'dropout': 0.3,
                           'max_len': MAX_LEN, 'num_stat_features': X_tr_stats.shape[1],
                           'scalable_softmax': getattr(model, 'scalable_softmax', False)},
            }, f"models/r2_regression/{mt}_seed{seed}.pt")

    # ───── Ensemble ─────
    print("\n" + "=" * 50)
    print("ENSEMBLE")
    print("=" * 50)

    # Median ensemble (like god_tier)
    te_median = np.median(all_test_preds, axis=0)
    va_median = np.median(all_val_preds, axis=0)

    # Also try mean
    te_mean = np.mean(all_test_preds, axis=0)
    va_mean = np.mean(all_val_preds, axis=0)

    for name, va_pred, te_pred in [("MEDIAN", va_median, te_median),
                                     ("MEAN", va_mean, te_mean)]:
        va_raw = decode_predictions(va_pred)
        va_raw_cal = calendar_postprocess(va_raw)
        score_raw = competition_score_np(y_va_raw, va_raw)
        score_cal = competition_score_np(y_va_raw, va_raw_cal)
        print(f"  {name}: val={score_raw:.4f}  cal={score_cal:.4f}")

    # Use whichever is better
    va_raw_med = decode_predictions(va_median)
    va_raw_mean = decode_predictions(va_mean)
    score_med = competition_score_np(y_va_raw, calendar_postprocess(va_raw_med))
    score_mean = competition_score_np(y_va_raw, calendar_postprocess(va_raw_mean))

    if score_med <= score_mean:
        final_pred = te_median
        print(f"\n  -> Using MEDIAN ensemble")
    else:
        final_pred = te_mean
        print(f"\n  -> Using MEAN ensemble")

    te_raw = decode_predictions(final_pred)
    te_raw = calendar_postprocess(te_raw)
    te_raw = te_raw.astype(np.uint16)

    sub = pd.DataFrame({
        'id': ids_test,
        'attr_1': te_raw[:, 0], 'attr_2': te_raw[:, 1], 'attr_3': te_raw[:, 2],
        'attr_4': te_raw[:, 3], 'attr_5': te_raw[:, 4], 'attr_6': te_raw[:, 5],
    })
    sub.to_csv('submission_regression.csv', index=False)
    print(f"\nSaved: submission_regression.csv")


if __name__ == "__main__":
    main()
