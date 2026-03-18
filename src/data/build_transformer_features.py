import os
import numpy as np
import pandas as pd

RAW_DIR = "data/data_raw"
OUT_DIR = "data/data_processed/transformer"
os.makedirs(OUT_DIR, exist_ok=True)

def process_data(path, is_y=False):
    print(f"Reading {path}...")
    df = pd.read_csv(path)
    # Bỏ cột ID, lấy data
    data = df.iloc[:, 1:].copy()
    
    # Fill missing bằng 0 (0 sẽ là Padding index)
    data = data.fillna(0)
    
    if is_y:
        # Tập Y giữ nguyên giá trị (12, 31, 99...) dạng Float để tính Loss
        return data.values.astype(np.float32)
    else:
        # TẬP X PHẢI LÀ SỐ NGUYÊN (INT64) - KHÔNG ĐƯỢC NORMALIZE
        # Ép về kiểu nguyên để nạp vào Embedding layer
        return data.values.astype(np.int64)

def main():
    # 1. Xử lý X (Dạng Nguyên)
    xtr = process_data(f"{RAW_DIR}/X_train.csv")
    xva = process_data(f"{RAW_DIR}/X_val.csv")
    xte = process_data(f"{RAW_DIR}/X_test.csv")

    # 2. Xử lý Y (Dạng thực)
    ytr = process_data(f"{RAW_DIR}/Y_train.csv", is_y=True)
    yva = process_data(f"{RAW_DIR}/Y_val.csv", is_y=True)

    # 3. Lưu file
    np.save(f"{OUT_DIR}/X_train.npy", xtr)
    np.save(f"{OUT_DIR}/X_val.npy", xva)
    np.save(f"{OUT_DIR}/X_test.npy", xte)
    np.save(f"{OUT_DIR}/y_train.npy", ytr)
    np.save(f"{OUT_DIR}/y_val.npy", yva)
    
    # Mask: Những chỗ bằng 0 là Padding
    np.save(f"{OUT_DIR}/mask_train.npy", (xtr == 0))
    np.save(f"{OUT_DIR}/mask_val.npy", (xva == 0))
    np.save(f"{OUT_DIR}/mask_test.npy", (xte == 0))

    print("\n✨ XONG! Dữ liệu đã sẵn sàng cho K-Fold Cross Validation.")
    print(f"X_train sample: {xtr[0, :5]} (Phải là số nguyên như 102, 50, 0...)")

if __name__ == "__main__":
    main()