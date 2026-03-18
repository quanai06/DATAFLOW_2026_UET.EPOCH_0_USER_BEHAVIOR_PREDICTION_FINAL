import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, X, mask, y=None):
        self.X = torch.LongTensor(X) # Phải là Long cho Embedding
        self.mask = torch.BoolTensor(mask)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx], self.mask[idx]
        return self.X[idx], self.mask[idx], self.y[idx]