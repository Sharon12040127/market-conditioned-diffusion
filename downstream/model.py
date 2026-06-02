# downstream/model.py
import torch
import torch.nn as nn
import math


# downstream/model.py 在LSTMPredictor里加两个方法
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def encode(self, x):
        """只跑encoder，返回最后时间步的隐状态"""
        out, _ = self.lstm(x)
        return out[:, -1, :]   # (B, hidden)

    def forward(self, x):
        return self.head(self.encode(x)).squeeze(-1)


class TransformerPredictor(nn.Module):
    def __init__(self, input_size, d_model=64, n_heads=4,
                 n_layers=2, dropout=0.1, max_len=100):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_emb    = nn.Embedding(max_len, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def encode(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x   = self.input_proj(x) + self.pos_emb(pos)
        x   = self.encoder(x)
        return x[:, -1, :]   # (B, d_model)

    def forward(self, x):
        return self.head(self.encode(x)).squeeze(-1)

# get_model不变

def get_model(model_name: str, input_size: int, **kwargs) -> nn.Module:
    """工厂函数，按名字返回模型。"""
    if model_name == "lstm":
        return LSTMPredictor(input_size=input_size, **kwargs)
    elif model_name == "transformer":
        return TransformerPredictor(input_size=input_size, **kwargs)
    else:
        raise ValueError(f"未知模型: {model_name}，可选: lstm / transformer")