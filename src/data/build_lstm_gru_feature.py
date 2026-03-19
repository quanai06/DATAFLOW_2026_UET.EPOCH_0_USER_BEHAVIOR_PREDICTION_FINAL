import numpy as np, pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
from scipy import stats
import os
import dotenv

from src.metrics.metrics import FINAL_MAX_LEN
dotenv.load_dotenv()
FINAL_MAX_LEN = int(os.getenv("FINAL_MAX_LEN", 37))

def process_seqs(df):
    seq_cols = [c for c in df.columns if c.startswith('feature_')]
    seqs = [row[~np.isnan(row)].astype(int).tolist() for row in df[seq_cols].values]
    return pad_sequences(seqs, maxlen=FINAL_MAX_LEN, padding='post', truncating='post', value=0)

def get_hubs(df):
    seq_cols = [c for c in df.columns if c.startswith('feature_')]
    vals = df[seq_cols].values.ravel()
    cnt = Counter(vals[~np.isnan(vals)].astype(int))
    return [k for k, v in cnt.most_common(10)]

def create_stats(df, hubs):
    vals = df.filter(like='feature_').values
    res = []
    for row in vals:
        s = row[~np.isnan(row)].astype(int)
        if len(s) == 0: res.append([0]*17); continue
        
        mode_val = stats.mode(s, keepdims=True).mode[0]
        s_list = s.tolist()
        has_783 = 1 if (783 in s_list and s_list.index(783) <= 5) else 0
        
        row_feat = [len(s), len(np.unique(s)), s[0], s[-1], mode_val, len(np.unique(s))/len(s), has_783]
        row_feat += [s_list.count(h) for h in hubs]
        res.append(row_feat)
    return np.array(res)