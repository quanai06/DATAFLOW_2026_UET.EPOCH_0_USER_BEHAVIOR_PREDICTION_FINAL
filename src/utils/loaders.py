import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, X, S, mask, y=None):
        """
        X: [N, 66] - Chuỗi hành động (Long cho Embedding)
        S: [N, 22] - Expert features trích xuất từ Heatmap (Float)
        mask: [N, 66] - Padding mask (Bool)
        y: [N, 6] - Target attributes (Float)
        """
        self.X = torch.LongTensor(X) 
        self.S = torch.FloatTensor(S) # Expert features bổ trợ
        self.mask = torch.BoolTensor(mask)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Lấy các thành phần tại vị trí idx
        x_idx = self.X[idx]
        s_idx = self.S[idx]
        m_idx = self.mask[idx]
        
        if self.y is None:
            # Dùng cho giai đoạn Inference (Predict tập Test)
            return x_idx, s_idx, m_idx
        
        # Dùng cho giai đoạn Training/Validation
        y_idx = self.y[idx]
        return x_idx, s_idx, m_idx, y_idx