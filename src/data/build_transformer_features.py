import os
import numpy as np
import pandas as pd

RAW_DIR = "data/data_raw"
OUT_DIR = "data/data_processed/transformer"
os.makedirs(OUT_DIR, exist_ok=True)

# Các mã ID "Signature" có sức ảnh hưởng lớn nhất từ báo cáo
SIGNATURE_CODES = [105, 102, 1071, 4004, 1076, 10795, 15342, 21606]

def extract_expert_features(data_values):
    all_stats = []
    for seq in data_values:
        # 1. Lấy chuỗi thực tế (bỏ padding 0)
        real_seq = seq[seq != 0]
        eff_len = len(real_seq)
        
        if eff_len > 0:
            # --- NHÓM 1: ĐỊNH LƯỢNG (6 đặc trưng - Mục 2.1.1) ---
            f_len = eff_len / 66.0
            f_uniq = len(np.unique(real_seq)) / eff_len
            f_dup = 1.0 - f_uniq
            # Action Dominance: Thằng xuất hiện nhiều nhất chiếm bao nhiêu %
            counts = np.bincount(real_seq.astype(np.int32))
            f_dom = np.max(counts) / eff_len
            # Độ phức tạp: số lần thay đổi hành động (Mục 2.1.1)
            f_trans = np.sum(real_seq[1:] != real_seq[:-1]) / eff_len if eff_len > 1 else 0
            f_pad = (66.0 - eff_len) / 66.0
            
            quantitative = [f_len, f_uniq, f_dup, f_dom, f_trans, f_pad]

            # --- NHÓM 2: MẬT ĐỘ SIGNATURE (8 đặc trưng - Giải quyết Attr 6 "rải rác") ---
            # Thay vì lấy vị trí cố định, ta đếm tần suất xuất hiện của từng mã
            sig_densities = []
            for code in SIGNATURE_CODES:
                density = np.sum(real_seq == code) / eff_len
                sig_densities.append(density)
                
            # --- NHÓM 3: ĐIỂM CHỐT (2 đặc trưng) ---
            # Kiểm tra xem hành động đầu và cuối có phải là "Signature" không
            f_start_sig = 1.0 if real_seq[0] in SIGNATURE_CODES else 0.0
            f_end_sig = 1.0 if real_seq[-1] in SIGNATURE_CODES else 0.0
            checkpoints = [f_start_sig, f_end_sig]
        else:
            quantitative = [0.0] * 6
            sig_densities = [0.0] * 8
            checkpoints = [0.0] * 2

        # TỔNG CỘNG: 6 + 8 + 2 = 16 ĐẶC TRƯNG
        combined = quantitative + sig_densities + checkpoints
        all_stats.append(combined)
    
    return np.array(all_stats, dtype=np.float32)

def process_data(path, is_y=False):
    df = pd.read_csv(path).fillna(0)
    data = df.iloc[:, 1:].copy()
    if is_y:
        return data.values.astype(np.float32)
    return data.values.astype(np.int64)

def main():
    xtr_raw = process_data(f"{RAW_DIR}/X_train.csv")
    xva_raw = process_data(f"{RAW_DIR}/X_val.csv")
    xte_raw = process_data(f"{RAW_DIR}/X_test.csv")

    print("🚀 Đang tạo 16 Expert Features (Mật độ tín hiệu) bám sát báo cáo...")
    str_feat = extract_expert_features(xtr_raw)
    sva_feat = extract_expert_features(xva_raw)
    ste_feat = extract_expert_features(xte_raw)

    # Lưu file
    np.save(f"{OUT_DIR}/X_train.npy", xtr_raw)
    np.save(f"{OUT_DIR}/X_val.npy", xva_raw)
    np.save(f"{OUT_DIR}/X_test.npy", xte_raw)
    
    np.save(f"{OUT_DIR}/S_train.npy", str_feat)
    np.save(f"{OUT_DIR}/S_val.npy", sva_feat)
    np.save(f"{OUT_DIR}/S_test.npy", ste_feat)
    
    ytr = process_data(f"{RAW_DIR}/Y_train.csv", is_y=True)
    yva = process_data(f"{RAW_DIR}/Y_val.csv", is_y=True)
    np.save(f"{OUT_DIR}/y_train.npy", ytr)
    np.save(f"{OUT_DIR}/y_val.npy", yva)
    
    np.save(f"{OUT_DIR}/mask_train.npy", (xtr_raw == 0))
    np.save(f"{OUT_DIR}/mask_val.npy", (xva_raw == 0))
    np.save(f"{OUT_DIR}/mask_test.npy", (xte_raw == 0))

    print(f"✨ XONG! Đã sẵn sàng 16 đặc trưng (Dim={str_feat.shape[1]}).")

if __name__ == "__main__":
    main()