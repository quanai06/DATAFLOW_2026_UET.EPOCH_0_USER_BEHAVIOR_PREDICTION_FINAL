import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout=0.2):
        super().__init__()
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
        attn_out, attn_weights = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, attn_weights

class TransformerModel(nn.Module):
    def __init__(self, vocab_size=30000, d_model=256, nhead=8, num_layers=4, dropout=0.2, expert_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, 67, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_model*4, dropout)
            for _ in range(num_layers)
        ])
        self.norm_final = nn.LayerNorm(d_model)

        # Nhánh xử lý Expert Features (S) từ file build_features trước đó
        self.s_mlp = nn.Sequential(
            nn.Linear(expert_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Head dates (attr 1, 2, 4, 5): Nhận Transformer (d_model*2) + Expert Stats (64)
        self.head_dates = nn.Sequential(
            nn.Linear(d_model * 2 + 64, 128),
            nn.GELU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )
        
        # Head factory (attr 3, 6): Nhận Transformer + Expert Stats (64)
        # Lưu ý: Sẽ thực hiện "Bác bỏ nhiễu" (mục 2.2.4) trong Forward
        self.head_factory = nn.Sequential(
            nn.Linear(d_model * 2 + 64, 256),
            nn.GELU(),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

        self.register_buffer('M_dates', torch.tensor([12, 31, 12, 31], dtype=torch.float32))
        self.register_buffer('M_factory', torch.tensor([99, 99], dtype=torch.float32))

    def forward(self, x, s, mask=None):
        """
        x: [B, 66] (Chuỗi hành động)
        s: [B, expert_dim] (Expert features trích xuất từ Heatmap)
        """
        B = x.shape[0]
        
        # --- NHÁNH TRANSFORMER (TRÍCH XUẤT NGỮ CẢNH) ---
        x_emb = self.embedding(x.long())
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_emb = torch.cat((cls_tokens, x_emb), dim=1)
        x_emb = x_emb + self.pos_embedding
        
        if mask is not None:
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)
        else: full_mask = None

        all_attentions = []
        for layer in self.layers:
            x_emb, attn = layer(x_emb, full_mask)
            all_attentions.append(attn)
            
        x_emb = self.norm_final(x_emb)
        
        cls_feat = x_emb[:, 0, :]
        if full_mask is not None:
            mask_weights = (~full_mask).float().unsqueeze(-1)
            mean_feat = (x_emb * mask_weights).sum(dim=1) / mask_weights.sum(dim=1)
        else:
            mean_feat = x_emb.mean(dim=1)
        
        combined_transformer = torch.cat([cls_feat, mean_feat], dim=-1)

        # --- NHÁNH EXPERT FEATURES (BÁM SÁT BÁO CÁO) ---
        # 2.2.4: Phân tách Expert Features
        # Giả định s đã được trích xuất: [length, nunique, first_item, x0, x2..., x54, x55...]
        
        # s_general: chứa length, nunique, first_item (3 cột đầu)
        # s_specialized: chứa các Late Triggers (x54, x55, x44, tail_5...)
        
        s_feat = self.s_mlp(s) # Đưa qua MLP để nén thông tin

        # Kết hợp thông tin Transformer và Expert Features
        final_combined = torch.cat([combined_transformer, s_feat], dim=-1)
        
        # Dự đoán
        # Mục 2.2.4: head_factory học từ final_combined 
        # (Lớp MLP bên trong head_factory sẽ tự động tập trung vào các triggers trong s_feat 
        # đã được trích xuất riêng biệt ở bước build_features)
        
        out_dates = self.head_dates(final_combined) * self.M_dates
        out_factory = self.head_factory(final_combined) * self.M_factory
        
        final_out = torch.cat([
            out_dates[:, :2],   # attr_1, 2
            out_factory[:, :1], # attr_3
            out_dates[:, 2:],   # attr_4, 5
            out_factory[:, 1:]  # attr_6
        ], dim=1)
        
        return final_out, all_attentions