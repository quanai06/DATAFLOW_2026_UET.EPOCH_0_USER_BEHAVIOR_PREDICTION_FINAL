import torch
import torch.nn as nn

class CompetitionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # M: [tháng, ngày, nhà máy, tháng, ngày, nhà máy]
        self.register_buffer('M', torch.tensor([12, 31, 99, 12, 31, 99], dtype=torch.float32))
        # W: Cột nhà máy (index 2 và 5) nhân hệ số 100
        self.register_buffer('W', torch.tensor([1, 1, 100, 1, 1, 100], dtype=torch.float32))

    def forward(self, pred, target):
        # Công thức: w * ((pred/M) - (target/M))^2
        p_norm = pred / self.M
        t_norm = target / self.M
        
        # MSE có trọng số
        loss = torch.pow(p_norm - t_norm, 2) * self.W
        return loss.mean() # Trả về trung bình để lan truyền ngược

def get_loss():
    return CompetitionLoss()