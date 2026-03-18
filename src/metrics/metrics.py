import torch
import numpy as np

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