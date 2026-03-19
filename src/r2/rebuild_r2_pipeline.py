"""
Round 2 Data Pipeline Rebuild
1. Create manual features (layer2) from new data/
2. Build transformer tensors (layer3) with MAX_LEN=66
"""
import pandas as pd
import numpy as np
import pickle
import os
import math
from collections import Counter
from sklearn.preprocessing import StandardScaler

# ======================================================
# CONFIG
# ======================================================
RAW_PATH = "data"
LAYER2_PATH = "data/layer2_r2"
LAYER3_PATH = "data/layer3_features/transformer_r2"
MAX_LEN = 66
PAD = 0

os.makedirs(LAYER2_PATH, exist_ok=True)
os.makedirs(LAYER3_PATH, exist_ok=True)

# ======================================================
# 1. LOAD DATA
# ======================================================
print("Loading data...")
df_train = pd.read_csv(f'{RAW_PATH}/X_train.csv')
df_val   = pd.read_csv(f'{RAW_PATH}/X_val.csv')
df_test  = pd.read_csv(f'{RAW_PATH}/X_test.csv')

df_y_train = pd.read_csv(f'{RAW_PATH}/Y_train.csv')
df_y_val   = pd.read_csv(f'{RAW_PATH}/Y_val.csv')

X_full = pd.concat([df_train, df_val], ignore_index=True)

print(f"  Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

feature_cols = [c for c in df_train.columns if c.startswith('feature_')]
print(f"  Feature columns: {len(feature_cols)}, MAX_LEN: {MAX_LEN}")

# ======================================================
# 2. TOP-10 HUBS + RARE ACTIONS
# ======================================================
print("Finding hubs and rare actions...")
all_actions = X_full[feature_cols].values.ravel()
all_actions = all_actions[~pd.isna(all_actions)].astype(int)
all_actions = all_actions[all_actions != 0]

cnt = Counter(all_actions)
TOP_10_HUBS = [k for k, v in cnt.most_common(10)]
print(f"  Top hubs: {TOP_10_HUBS}")

total_seqs = len(X_full)
rare_threshold = max(2, int(total_seqs * 0.001))
rare_actions = set(a for a, c in cnt.items() if c < rare_threshold)
print(f"  Rare actions (count < {rare_threshold}): {len(rare_actions)}")

# ======================================================
# 3. FEATURE ENGINEERING
# ======================================================
def _entropy(seq):
    if len(seq) == 0:
        return 0.0
    n = len(seq)
    counts = Counter(seq)
    return -sum((c/n) * math.log2(c/n + 1e-12) for c in counts.values())

def _rb3_count(seq):
    n = len(seq)
    if n < 3: return 0
    return sum(1 for i in range(n - 2) if seq[i] == seq[i + 2] and seq[i] != seq[i + 1])

def _rb4_count(seq):
    n = len(seq)
    if n < 4: return 0
    return sum(1 for i in range(n - 3) if seq[i] == seq[i + 3] and seq[i] != seq[i + 1])

def create_manual_features(df, top_hubs, rare_actions_set):
    feature_df = df.filter(like='feature_')
    stats_list = []

    for row in feature_df.values:
        seq = row[~pd.isna(row)].astype(int)
        seq = seq[seq != 0].tolist()

        if len(seq) == 0:
            stats_list.append([0] * (17 + len(top_hubs)))
            continue

        length = len(seq)
        nunique = len(set(seq))
        first_item = seq[0]
        last_item = seq[-1]

        cnt_seq = Counter(seq)
        mode_val = cnt_seq.most_common(1)[0][0]
        action_dominance = cnt_seq.most_common(1)[0][1] / length
        entropy = _entropy(seq)

        n_transitions = sum(1 for i in range(length - 1) if seq[i] != seq[i + 1])
        transition_ratio = n_transitions / max(length - 1, 1)

        rb_3_steps = _rb3_count(seq)
        rb_4_steps = _rb4_count(seq)
        first_action_rb = 1 if (length >= 3 and seq[0] == seq[2]) else 0

        n_rare = sum(1 for a in seq if a in rare_actions_set)
        rare_ratio = n_rare / length

        hub_counts = [seq.count(hub) for hub in top_hubs]

        # NEW R2 features
        # Bigram diversity
        bigrams = set()
        for i in range(length - 1):
            bigrams.add((seq[i], seq[i+1]))
        bigram_diversity = len(bigrams) / max(length - 1, 1)

        # Position of most common action (normalized)
        mode_positions = [i/length for i, a in enumerate(seq) if a == mode_val]
        mode_avg_pos = np.mean(mode_positions) if mode_positions else 0.5

        # Repetition rate (consecutive same actions)
        n_repeats = sum(1 for i in range(length - 1) if seq[i] == seq[i+1])
        repeat_rate = n_repeats / max(length - 1, 1)

        row_stats = [
            length, nunique, first_item, last_item, mode_val,
            action_dominance, entropy, n_transitions, transition_ratio,
            rb_3_steps, rb_4_steps, first_action_rb,
            n_rare, rare_ratio,
            nunique / length,  # diversity ratio
            last_item == first_item,  # loop flag
            length / float(MAX_LEN),  # normalized length (updated for R2)
            bigram_diversity, mode_avg_pos, repeat_rate,
        ] + hub_counts

        stats_list.append(row_stats)

    columns = [
        "length", "nunique", "first_item", "last_item", "mode",
        "action_dominance", "entropy", "n_transitions", "transition_ratio",
        "rb_3_steps", "rb_4_steps", "first_action_rb",
        "n_rare_actions", "rare_action_ratio",
        "diversity_ratio", "loop_flag", "norm_length",
        "bigram_diversity", "mode_avg_pos", "repeat_rate",
    ] + [f"hub_{hub}" for hub in top_hubs]

    return pd.DataFrame(stats_list, columns=columns)


print("Creating manual features...")
train_stats = create_manual_features(df_train, TOP_10_HUBS, rare_actions)
val_stats   = create_manual_features(df_val,   TOP_10_HUBS, rare_actions)
test_stats  = create_manual_features(df_test,  TOP_10_HUBS, rare_actions)

# Merge and export layer2
df_train_final = pd.concat([df_train.reset_index(drop=True), train_stats.reset_index(drop=True)], axis=1)
df_val_final   = pd.concat([df_val.reset_index(drop=True),   val_stats.reset_index(drop=True)],   axis=1)
df_test_final  = pd.concat([df_test.reset_index(drop=True),  test_stats.reset_index(drop=True)],  axis=1)

df_train_final.to_csv(f"{LAYER2_PATH}/X_train.csv", index=False)
df_val_final.to_csv(f"{LAYER2_PATH}/X_val.csv",     index=False)
df_test_final.to_csv(f"{LAYER2_PATH}/X_test.csv",   index=False)

# Copy Y files
df_y_train.to_csv(f"{LAYER2_PATH}/Y_train.csv", index=False)
df_y_val.to_csv(f"{LAYER2_PATH}/Y_val.csv", index=False)

print(f"  Layer2 saved. Features per row: {len(train_stats.columns)}")

# ======================================================
# 4. BUILD TRANSFORMER TENSORS (Layer 3)
# ======================================================
print("\nBuilding transformer tensors...")

# Action remapper (contiguous 1..N, 0=PAD)
all_actions_list = []
for df in [df_train, df_val, df_test]:
    vals = df[feature_cols].values.ravel()
    vals = vals[~pd.isna(vals)].astype(int)
    all_actions_list.extend(vals[vals != 0].tolist())

unique_actions = sorted(set(all_actions_list))
action_remapper = {a: idx + 1 for idx, a in enumerate(unique_actions)}
vocab_size = len(unique_actions) + 1  # +1 for PAD

print(f"  Unique actions: {len(unique_actions)}, vocab_size: {vocab_size}")

with open(f"{LAYER3_PATH}/action_remapper.pkl", "wb") as f:
    pickle.dump({"remapper": action_remapper, "vocab_size": vocab_size}, f)

# Build sequences
def build_features(df, feat_cols, action_remapper):
    sequences = []
    masks = []
    for _, row in df.iterrows():
        seq = row[feat_cols].dropna().values.astype(int)
        remapped = np.array([action_remapper.get(int(v), 0) for v in seq], dtype=np.int64)

        padded = np.full(MAX_LEN, PAD, dtype=np.int64)
        mask = np.ones(MAX_LEN, dtype=bool)
        length = min(len(remapped), MAX_LEN)
        padded[:length] = remapped[:length]
        mask[:length] = False

        sequences.append(padded)
        masks.append(mask)
    return np.array(sequences), np.array(masks)

print("  Building train sequences...")
train_seq, train_mask = build_features(df_train_final, feature_cols, action_remapper)
print("  Building val sequences...")
val_seq, val_mask = build_features(df_val_final, feature_cols, action_remapper)
print("  Building test sequences...")
test_seq, test_mask = build_features(df_test_final, feature_cols, action_remapper)

# Stats features
def get_stat_cols(df):
    feat_cols_set = set(c for c in df.columns if c.startswith('feature_'))
    return [c for c in df.columns if c not in feat_cols_set and c != 'id']

stat_cols = get_stat_cols(df_train_final)
train_stats_raw = df_train_final[stat_cols].fillna(0).values.astype(np.float32)
val_stats_raw   = df_val_final[stat_cols].fillna(0).values.astype(np.float32)
test_stats_raw  = df_test_final[stat_cols].fillna(0).values.astype(np.float32)

scaler = StandardScaler()
train_stats_scaled = scaler.fit_transform(train_stats_raw).astype(np.float32)
val_stats_scaled   = scaler.transform(val_stats_raw).astype(np.float32)
test_stats_scaled  = scaler.transform(test_stats_raw).astype(np.float32)

with open(f"{LAYER3_PATH}/stat_scaler.pkl", "wb") as f:
    pickle.dump({"scaler": scaler, "stat_cols": stat_cols}, f)

# Save sequences and stats
np.save(f"{LAYER3_PATH}/X_train_seq.npy", train_seq)
np.save(f"{LAYER3_PATH}/X_train_mask.npy", train_mask)
np.save(f"{LAYER3_PATH}/X_train_stats.npy", train_stats_scaled)

np.save(f"{LAYER3_PATH}/X_val_seq.npy", val_seq)
np.save(f"{LAYER3_PATH}/X_val_mask.npy", val_mask)
np.save(f"{LAYER3_PATH}/X_val_stats.npy", val_stats_scaled)

np.save(f"{LAYER3_PATH}/X_test_seq.npy", test_seq)
np.save(f"{LAYER3_PATH}/X_test_mask.npy", test_mask)
np.save(f"{LAYER3_PATH}/X_test_stats.npy", test_stats_scaled)

# ======================================================
# 5. ENCODE LABELS (both classification and raw for regression)
# ======================================================
print("\nEncoding labels...")

# Raw labels (for regression / competition score)
y_train_raw = df_y_train[['attr_1','attr_2','attr_3','attr_4','attr_5','attr_6']].values.astype(np.int64)
y_val_raw   = df_y_val[['attr_1','attr_2','attr_3','attr_4','attr_5','attr_6']].values.astype(np.int64)

np.save(f"{LAYER3_PATH}/y_train_raw.npy", y_train_raw)
np.save(f"{LAYER3_PATH}/y_val_raw.npy", y_val_raw)

# Classification encoding (for classification heads)
def encode_multilabel(y_df):
    y = y_df.iloc[:, 1:].values
    encoders = []
    y_encoded_cols = []
    for i in range(y.shape[1]):
        vals = np.unique(y[:, i])
        mapping = {v: idx for idx, v in enumerate(vals)}
        encoders.append(mapping)
        col = np.vectorize(mapping.get)(y[:, i])
        y_encoded_cols.append(col)
    return np.stack(y_encoded_cols, axis=1).astype(np.int64), encoders

def apply_encoders(y_df, encoders):
    y = y_df.iloc[:, 1:].values
    y_encoded_cols = []
    for i in range(y.shape[1]):
        mapping = encoders[i]
        col = np.array([mapping.get(v, 0) for v in y[:, i]])
        y_encoded_cols.append(col)
    return np.stack(y_encoded_cols, axis=1).astype(np.int64)

# Use combined train+val for building encoders to cover all possible values
y_combined = pd.concat([df_y_train, df_y_val], ignore_index=True)
_, encoders = encode_multilabel(y_combined)

y_train_enc = apply_encoders(df_y_train, encoders)
y_val_enc   = apply_encoders(df_y_val, encoders)

np.save(f"{LAYER3_PATH}/y_train.npy", y_train_enc)
np.save(f"{LAYER3_PATH}/y_val.npy", y_val_enc)

with open(f"{LAYER3_PATH}/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Save IDs for submission
np.save(f"{LAYER3_PATH}/ids_train.npy", df_train['id'].values)
np.save(f"{LAYER3_PATH}/ids_val.npy", df_val['id'].values)
np.save(f"{LAYER3_PATH}/ids_test.npy", df_test['id'].values)

print(f"\n Pipeline complete!")
print(f"  Layer2: {LAYER2_PATH}/")
print(f"  Layer3: {LAYER3_PATH}/")
print(f"  vocab_size={vocab_size}, max_len={MAX_LEN}, stat_features={len(stat_cols)}")
print(f"  Train: {train_seq.shape}, Val: {val_seq.shape}, Test: {test_seq.shape}")
print(f"  y_train_raw: {y_train_raw.shape}, y_val_raw: {y_val_raw.shape}")
print(f"  Encoders: {[len(e) for e in encoders]} classes per attr")
