import pandas as pd, numpy as np, joblib, tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.build_lstm_gru_feature import *
from models.lstm_gru_model import build_model
from metrics.metrics import evaluate_report, set_seed

import os
import dotenv
dotenv.load_dotenv()
SEED = int(os.getenv("SEED", 42))
N_FOLDS = int(os.getenv("N_FOLDS", 5))
EPOCHS = int(os.getenv("EPOCHS", 20))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 128))
TARGET_COLS = os.getenv("TARGET_COLS").split(",")
M_CONST = np.array([float(x) for x in os.getenv("M_CONST", "12.0, 31.0, 99.0, 12.0, 31.0, 99.0").split(",")], dtype=np.float32)
W_CONST = np.array([float(x) for x in os.getenv("W_CONST", "1.0, 1.0, 100.0, 1.0, 1.0, 100.0").split(",")], dtype=np.float32)

set_seed()

# 1. Load Data (Giữ tách biệt hoàn toàn theo yêu cầu)
df_tr = pd.read_csv('data/data_raw/X_train.csv')
df_va = pd.read_csv('data/data_raw/X_val.csv')
y_tr_raw = pd.read_csv('data/data_raw/Y_train.csv')
y_va_raw = pd.read_csv('data/data_raw/Y_val.csv')

y_tr = pd.merge(df_tr[['id']], y_tr_raw, on='id').fillna(0)[TARGET_COLS].values / M_CONST
y_va = pd.merge(df_va[['id']], y_va_raw, on='id').fillna(0)[TARGET_COLS].values / M_CONST

# 2. Features
X_tr_seq = process_seqs(df_tr)
X_va_seq = process_seqs(df_va)
hubs = get_hubs(df_tr)
joblib.dump(hubs, 'model/lstm_gru/hubs.pkl')

sc = StandardScaler()
X_tr_st = sc.fit_transform(create_stats(df_tr, hubs))
X_va_st = sc.transform(create_stats(df_va, hubs))
joblib.dump(sc, 'model/lstm_gru/scaler_full.pkl')

v_size = max(np.max(X_tr_seq), np.max(X_va_seq)) + 1

# 3. K-Fold Training on df_tr
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for fold, (t_idx, v_idx) in enumerate(kf.split(X_tr_seq)):
    for mt in ['lstm', 'gru']:
        tf.keras.backend.clear_session()
        m = build_model(mt, v_size + 10, X_tr_st.shape[1])
        m.fit([X_tr_seq[t_idx], X_tr_st[t_idx]], y_tr[t_idx],
              validation_data=([X_tr_seq[v_idx], X_tr_st[v_idx]], y_tr[v_idx]),
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
        m.save(f'model/lstm_gru/model_{fold}_{mt}.keras')
        print(f" Fold {fold} {mt.upper()} Done.")