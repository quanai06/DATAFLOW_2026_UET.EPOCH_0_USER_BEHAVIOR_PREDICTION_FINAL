import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout=0.2):
        super().__init__()
        # Để ý cái batch_first=True
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Quan trọng: Lấy attn_weights ở đây
        attn_out, attn_weights = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, attn_weights

class TransformerModel(nn.Module):
    def __init__(self, vocab_size=30000, d_model=256, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, 67, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_model*4, dropout)
            for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(d_model)

        # Head dates (attr 1, 2, 4, 5)
        self.head_dates = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.GELU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )
        
        # Head factory (attr 3, 6)
        self.head_factory = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Linear(256, 2),
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
            # Lấy cả x và attention weights từ từng layer
            x, attn = layer(x, full_mask)
            all_attentions.append(attn)
            
        x = self.norm_final(x)
        
        # CLS + Mean Pooling cho xịn
        cls_feat = x[:, 0, :]
        if full_mask is not None:
            mask_weights = (~full_mask).float().unsqueeze(-1)
            mean_feat = (x * mask_weights).sum(dim=1) / mask_weights.sum(dim=1)
        else:
            mean_feat = x.mean(dim=1)
            
        combined = torch.cat([cls_feat, mean_feat], dim=-1)
        
        # Sigmoid * M để Score đẹp ngay từ Epoch 1
        out_dates = self.head_dates(combined) * self.M_dates
        out_factory = self.head_factory(combined) * self.M_factory
        
        final_out = torch.cat([
            out_dates[:, :2],   # attr_1, 2
            out_factory[:, :1], # attr_3
            out_dates[:, 2:],   # attr_4, 5
            out_factory[:, 1:]  # attr_6
        ], dim=1)
        
        # Trả về kết quả và danh sách attention cho bạn vẽ Heatmap
        return final_out, all_attentions