import torch
import torch.nn as nn

class CompetitionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # M: Hằng số chuẩn hóa [tháng, ngày, nhà máy, tháng, ngày, nhà máy]
        # Theo báo cáo mục 2.2.4: M_j = [12, 31, 99, 12, 31, 99]
        self.register_buffer('M', torch.tensor([12, 31, 99, 12, 31, 99], dtype=torch.float32))
        
        # W: Hệ số phạt alpha = 100 cho Attr 3 (index 2) và Attr 6 (index 5)
        # Theo báo cáo mục 3: Các thuộc tính nhà máy có trọng số cực cao
        self.register_buffer('W', torch.tensor([1.0, 1.0, 100.0, 1.0, 1.0, 100.0], dtype=torch.float32))

    def forward(self, pred, target):
        """
        pred, target: [Batch_Size, 6]
        """
        # 1. Chuẩn hóa về cùng thang đo (Normalization) như công thức phần 3 trong đề
        p_norm = pred / self.M
        t_norm = target / self.M
        
        # 2. Tính MSE có trọng số (Weighted MSE)
        # Trọng số W giúp mô hình tập trung tối đa vào Attr 3 và Attr 6
        loss_mse = torch.pow(p_norm - t_norm, 2) * self.W
        
        # 3. Bổ sung Log-Cosh component (Mẹo bám sát báo cáo mục 2.2.2)
        # Báo cáo của bạn chỉ ra lỗi "Attention Drift" gây ra Error > 2000 (sai số rất lớn)
        # MSE thuần túy có thể bị "nhiễu" bởi các outlier này. 
        # Log-Cosh giúp mô hình bền bỉ (robust) hơn với các sai số quá lớn ở giai đoạn đầu.
        diff = p_norm - t_norm
        loss_logcosh = torch.log(torch.cosh(diff + 1e-12)) * self.W
        
        # Kết hợp 90% MSE (đúng luật chơi) và 10% Log-Cosh (để ổn định Attention Drift)
        final_loss = 0.9 * loss_mse + 0.1 * loss_logcosh
        
        # Trả về trung bình trên toàn bộ Batch và 6 thuộc tính (đúng 1/6N trong công thức)
        return final_loss.mean()

def get_loss():
    """
    Hàm này sẽ được gọi trong train.py
    """
    return CompetitionLoss()