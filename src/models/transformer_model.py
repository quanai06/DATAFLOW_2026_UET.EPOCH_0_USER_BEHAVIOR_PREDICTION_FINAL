import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout=0.3): # Tăng Dropout
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, attn_weights

class TransformerModel(nn.Module):
    def __init__(self, vocab_size=30000, d_model=256, nhead=8, num_layers=6, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, 67, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_model*4, dropout) # FF tăng lên x4
            for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(d_model)

        # Nhánh 1: Ngày tháng (4 cột: 1, 2, 4, 5)
        self.head_dates = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )
        
        # Nhánh 2: Nhà máy (2 cột: 3, 6) - Tăng độ sâu để học kỹ hơn vì W=100
        self.head_factory = nn.Sequential(
            nn.Linear(d_model, 128), # Tăng width nhánh này
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

        self.register_buffer('M_dates', torch.tensor([12, 31, 12, 31], dtype=torch.float32))
        self.register_buffer('M_factory', torch.tensor([99, 99], dtype=torch.float32))

    def forward(self, x, mask=None):
        B = x.shape[0]
        x = self.embedding(x.long())
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        
        if mask is not None:
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)
        else: full_mask = None

        all_attentions = []
        for layer in self.layers:
            x, attn = layer(x, full_mask)
            all_attentions.append(attn)
            
        cls_out = self.norm_final(x[:, 0, :])
        
        out_dates = self.head_dates(cls_out) * self.M_dates
        out_factory = self.head_factory(cls_out) * self.M_factory
        
        # Ghép đúng thứ tự: 1, 2, 3, 4, 5, 6
        final_out = torch.cat([
            out_dates[:, :2],   # attr_1, 2
            out_factory[:, :1], # attr_3
            out_dates[:, 2:],   # attr_4, 5
            out_factory[:, 1:]  # attr_6
        ], dim=1)
        return final_out, all_attentions