import torch
import numpy as np
from tabulate import tabulate

def print_detailed_report(preds, targets, phase="AFTER EXPERT FE"):
    """
    In ra bảng điểm chi tiết bám sát báo cáo DATAFLOW 2026.
    Giúp soi kỹ Attr 3 & 6 (nhà máy) - nơi có trọng số x100.
    """
    # 1. Cấu hình hằng số chuẩn hóa M_j và trọng số W_j theo đề bài
    M = torch.tensor([12, 31, 99, 12, 31, 99], dtype=torch.float32).to(preds.device)
    W = torch.tensor([1, 1, 100, 1, 1, 100], dtype=torch.float32).to(preds.device)
    
    # 2. Tính toán theo công thức Score của BTC
    # (pred/M - target/M)^2 * W
    p_norm = preds / M
    t_norm = targets / M
    weighted_sq_error = torch.pow(p_norm - t_norm, 2) * W
    
    # 3. Tính điểm chi tiết từng cột (Weighted MSE per column)
    # Đây là giá trị đóng góp trực tiếp vào Overall Score
    col_scores = weighted_sq_error.mean(dim=0).cpu().numpy()
    overall_score = col_scores.mean() 
    
    # 4. Tính sai số thực tế (Raw MSE) - để chuyên gia dễ hình dung mức độ sai lệch
    raw_mse_per_col = torch.mean((preds - targets)**2, dim=0).cpu().numpy()
    
    # Tạo bảng dữ liệu bám sát báo cáo
    headers = ["Metric", "Attr_1", "Attr_2", "Attr_3", "Attr_4", "Attr_5", "Attr_6", "OVERALL"]
    
    # Dòng 1: Điểm trọng số (Dùng để nộp Kaggle)
    row_weighted = [f"Weighted Score ({phase})"] + [f"{s:.6f}" for s in col_scores] + [f"{overall_score:.6f}"]
    
    # Dòng 2: Sai số thực (Dùng để soi logic kinh doanh - x54, x55...)
    row_raw = ["Raw MSE (Real Error)"] + [f"{s:.2f}" for s in raw_mse_per_col] + ["-"]
    
    print(f"\n" + "═"*120)
    print(f"{'REPORT: USER BEHAVIOR PREDICTION PERFORMANCE':^120}")
    print("═"*120)
    print(tabulate([row_weighted, row_raw], headers=headers, tablefmt="grid"))
    
    # Cảnh báo dựa trên phân tích Heatmap
    if col_scores[2] > 0.5 or col_scores[5] > 0.5:
        print(f"⚠️ CẢNH BÁO: Attr 3 hoặc 6 đang có sai số cao. Hãy kiểm tra lại index 54, 55 và 44!")
    else:
        print(f"✅ TÍN HIỆU TỐT: Các Late-sequence Triggers đang hoạt động ổn định.")
    print("═"*120 + "\n")
    
    return col_scores, overall_score


class CompetitionMetric:
    def __init__(self, device):
        # M_j: [12, 31, 99, 12, 31, 99]
        self.M = torch.tensor([12, 31, 99, 12, 31, 99]).float().to(device)
        # W_j: Alpha=100 cho Attr 3 và 6
        self.W = torch.tensor([1, 1, 100, 1, 1, 100]).float().to(device)

    def compute_all(self, y_pred, y_true):
        with torch.no_grad():
            # Tính toán chuẩn hóa
            p_norm = y_pred / self.M
            t_norm = y_true / self.M
            
            # Weighted MSE theo từng mẫu và từng cột
            diff_sq = torch.pow(p_norm - t_norm, 2) * self.W
            
            # Score tổng theo công thức 1/6N
            score = diff_sq.mean() # mean() của matrix [N, 6] tương đương sum / (6*N)
            
            # MSE từng cột để soi lỗi Drift (Mục 2.2.2 báo cáo)
            weighted_mse_per_col = torch.mean(diff_sq, dim=0).cpu().numpy()
            raw_mse_per_col = torch.mean((y_pred - y_true)**2, dim=0).cpu().numpy()
            
        return {
            "comp_score": score.item(),
            "weighted_mse_per_col": weighted_mse_per_col,
            "mse_per_col": raw_mse_per_col # Giữ tên cũ để không gãy code cũ
        }