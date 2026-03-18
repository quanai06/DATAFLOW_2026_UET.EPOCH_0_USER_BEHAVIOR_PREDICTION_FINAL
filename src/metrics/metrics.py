import torch
import numpy as np
from tabulate import tabulate # Cài nếu chưa có: pip install tabulate

def print_detailed_report(preds, targets, phase="BEFORE FEATURE ENGINEERING"):
    """
    In ra bảng điểm chi tiết cho 6 thuộc tính theo chuẩn Weighted MSE
    """
    # Cấu hình chuẩn của đề bài
    M = torch.tensor([12, 31, 99, 12, 31, 99], dtype=torch.float32).to(preds.device)
    W = torch.tensor([1, 1, 100, 1, 1, 100], dtype=torch.float32).to(preds.device)
    
    # Tính toán sai số chuẩn hóa theo công thức Score đề bài
    p_norm = preds / M
    t_norm = targets / M
    
    # Tính (pred/M - target/M)^2 * W cho từng cột
    sq_error = torch.pow(p_norm - t_norm, 2) * W
    
    # Lấy trung bình theo từng cột (cho toàn bộ mẫu N)
    col_scores = sq_error.mean(dim=0).cpu().numpy()
    overall_score = col_scores.mean() # Score tổng là trung bình cộng của 6 cột này
    
    # Tạo bảng dữ liệu
    headers = ["Phase", "Attr_1", "Attr_2", "Attr_3", "Attr_4", "Attr_5", "Attr_6", "OVERALL"]
    row = [phase] + [f"{s:.6f}" for s in col_scores] + [f"{overall_score:.6f}"]
    
    print(f"\n" + "="*100)
    print(f"{'DETAILED PERFORMANCE REPORT':^100}")
    print("="*100)
    print(tabulate([row], headers=headers, tablefmt="grid"))
    print("="*100 + "\n")
    
    return col_scores, overall_score


class CompetitionMetric:
    def __init__(self, device):
        self.M = torch.tensor([12, 31, 99, 12, 31, 99]).float().to(device)
        self.W = torch.tensor([1, 1, 100, 1, 1, 100]).float().to(device)

    def compute_all(self, y_pred, y_true):
        with torch.no_grad():
            # Công thức chuẩn: 1/6N * sum(W * (pred/M - true/M)^2)
            p_norm = y_pred / self.M
            t_norm = y_true / self.M
            
            diff_sq = torch.pow(p_norm - t_norm, 2) * self.W
            # Score trung bình trên 6 cột và N mẫu
            score = diff_sq.sum() / (6 * y_pred.shape[0])
            
            mse_per_col = torch.mean((y_pred - y_true)**2, dim=0)
            
        return {
            "comp_score": score.item(),
            "mse_per_col": mse_per_col.cpu().numpy()
        }