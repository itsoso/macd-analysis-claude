"""
模型定义: Temporal Fusion Transformer + LightGBM
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════
#  Focal Loss (处理类别不平衡)
# ══════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, logits, targets):
        """logits: (B, C), targets: (B,) int64"""
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


# ══════════════════════════════════════════════════════════
#  位置编码
# ══════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ══════════════════════════════════════════════════════════
#  Gated Residual Network (GRN) — TFT 核心组件
# ══════════════════════════════════════════════════════════

class GatedResidualNetwork(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, d_context=None, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_input, d_hidden)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_hidden, d_output)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d_hidden, d_output)
        self.layer_norm = nn.LayerNorm(d_output)
        self.skip = nn.Linear(d_input, d_output) if d_input != d_output else nn.Identity()
        self.context_proj = nn.Linear(d_context, d_hidden, bias=False) if d_context else None

    def forward(self, x, context=None):
        skip = self.skip(x)
        h = self.fc1(x)
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)
        h = self.elu(h)
        h = self.dropout(h)
        gate = torch.sigmoid(self.gate(h))
        out = self.fc2(h)
        out = gate * out + (1 - gate) * skip
        return self.layer_norm(out)


# ══════════════════════════════════════════════════════════
#  Variable Selection Network (VSN) — 特征选择 + 可解释性
# ══════════════════════════════════════════════════════════

class VariableSelectionNetwork(nn.Module):
    def __init__(self, n_vars, d_model, d_hidden, dropout=0.1, d_context=None):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        # 每个变量的 GRN
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(d_model, d_hidden, d_model, d_context=d_context, dropout=dropout)
            for _ in range(n_vars)
        ])
        # 权重生成 GRN
        self.weight_grn = GatedResidualNetwork(
            n_vars * d_model, d_hidden, n_vars, d_context=d_context, dropout=dropout
        )

    def forward(self, inputs, context=None):
        """
        inputs: (B, T, n_vars, d_model)
        returns: (B, T, d_model), weights: (B, T, n_vars)
        """
        B, T, V, D = inputs.shape
        # 展平变量维度计算权重
        flat = inputs.reshape(B, T, V * D)
        ctx = context.unsqueeze(1).expand(-1, T, -1) if context is not None else None
        weights = self.weight_grn(flat, ctx)  # (B, T, n_vars)
        weights = F.softmax(weights, dim=-1)

        # 每个变量通过对应 GRN
        var_outputs = []
        for i, grn in enumerate(self.var_grns):
            var_outputs.append(grn(inputs[:, :, i, :], ctx))  # (B, T, D)
        var_outputs = torch.stack(var_outputs, dim=2)  # (B, T, V, D)

        # 加权求和
        out = (var_outputs * weights.unsqueeze(-1)).sum(dim=2)  # (B, T, D)
        return out, weights


# ══════════════════════════════════════════════════════════
#  Interpretable Multi-Head Attention
# ══════════════════════════════════════════════════════════

class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, self.d_k)  # 共享 V 投影 (可解释性)
        self.W_o = nn.Linear(self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape
        Q = self.W_q(q).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # (B, H, T, d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 所有 head 的 attention 平均 (可解释性)
        attn_avg = attn.mean(dim=1)  # (B, T, T)
        out = torch.matmul(attn_avg, V[:, 0])  # (B, T, d_k)
        return self.W_o(out), attn_avg


# ══════════════════════════════════════════════════════════
#  Temporal Fusion Transformer
# ══════════════════════════════════════════════════════════

class TemporalFusionTransformer(nn.Module):
    """
    简化版 TFT:
      - 静态特征: regime one-hot
      - 时变特征: OHLCV + 技术指标 + 六书评分
      - 输出: 分类 (long/hold/short) + 回归 (收益率)
    """

    def __init__(self, n_time_features, n_static=3, d_model=128,
                 n_heads=4, d_ff=256, num_encoder_layers=3,
                 num_decoder_layers=1, dropout=0.1, forecast_horizon=6):
        super().__init__()
        self.d_model = d_model
        self.n_time_features = n_time_features
        self.forecast_horizon = forecast_horizon

        # 输入投影: 每个特征投射到 d_model
        self.input_proj = nn.Linear(1, d_model)  # 逐特征投影
        self.static_proj = nn.Linear(n_static, d_model)

        # 静态变量 GRN (生成 context 向量)
        self.static_grn = GatedResidualNetwork(d_model, d_ff, d_model, dropout=dropout)

        # Variable Selection
        self.vsn = VariableSelectionNetwork(
            n_vars=n_time_features, d_model=d_model, d_hidden=d_ff,
            dropout=dropout, d_context=d_model
        )

        # 位置编码
        self.pos_enc = PositionalEncoding(d_model)

        # LSTM encoder (局部时间模式)
        self.lstm_enc = nn.LSTM(d_model, d_model, batch_first=True,
                                num_layers=num_encoder_layers, dropout=dropout)
        self.lstm_grn = GatedResidualNetwork(d_model, d_ff, d_model, dropout=dropout)
        self.lstm_norm = nn.LayerNorm(d_model)

        # Self-attention (长距依赖)
        self.self_attn = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.attn_grn = GatedResidualNetwork(d_model, d_ff, d_model, dropout=dropout)
        self.attn_norm = nn.LayerNorm(d_model)

        # Position-wise FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        # 输出头
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 3),  # short / hold / long
        )
        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, forecast_horizon),  # 多步收益预测
        )

    def forward(self, x_time, x_static):
        """
        x_time:   (B, T, n_features) — 时变特征序列
        x_static: (B, n_static) — 静态 regime 特征

        returns:
            cls_logits: (B, 3)
            reg_pred:   (B, forecast_horizon)
            attn_weights: (B, T, T)
            var_weights:  (B, T, n_features)
        """
        B, T, F = x_time.shape

        # 1. 静态 context
        static_emb = self.static_proj(x_static)  # (B, d_model)
        static_ctx = self.static_grn(static_emb)  # (B, d_model)

        # 2. 逐特征投影 + Variable Selection
        # x_time: (B, T, F) → (B, T, F, 1) → (B, T, F, d_model)
        x_proj = self.input_proj(x_time.unsqueeze(-1))  # (B, T, F, d_model)
        x_selected, var_weights = self.vsn(x_proj, static_ctx)  # (B, T, d_model)

        # 3. 位置编码
        x_selected = self.pos_enc(x_selected)

        # 4. LSTM encoder
        lstm_out, _ = self.lstm_enc(x_selected)
        lstm_out = self.lstm_grn(lstm_out)
        lstm_out = self.lstm_norm(lstm_out + x_selected)

        # 5. Self-attention + residual
        attn_out, attn_weights = self.self_attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_grn(attn_out)
        attn_out = self.attn_norm(attn_out + lstm_out)

        # 6. FFN + residual
        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_norm(ffn_out + attn_out)

        # 7. 取最后一个时间步输出
        last = ffn_out[:, -1, :]  # (B, d_model)

        # 8. 输出
        cls_logits = self.cls_head(last)    # (B, 3)
        reg_pred = self.reg_head(last)      # (B, forecast_horizon)

        return cls_logits, reg_pred, attn_weights, var_weights


# ══════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════

class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    滑动窗口时间序列数据集。
    """

    def __init__(self, df: "pd.DataFrame", feat_cols: list, label_cols: list,
                 lookback: int = 96):
        """
        df: 已标准化的 DataFrame, 包含特征列和标签列
        """
        self.feat_cols = feat_cols
        self.label_cols = label_cols
        self.lookback = lookback

        self.features = np.nan_to_num(df[feat_cols].values.astype(np.float32), nan=0.0)
        self.cls_labels = df["cls_label"].values.astype(np.int64)
        self.reg_labels = df["reg_label"].values.astype(np.float32)

        # regime 列 (用作静态特征)
        if "regime" in df.columns:
            self.regime = df["regime"].values.astype(np.int64)
        else:
            self.regime = np.ones(len(df), dtype=np.int64)

        self.n_samples = len(df) - lookback

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        start = idx
        end = idx + self.lookback

        x_time = torch.tensor(self.features[start:end])
        # 静态 regime: 取窗口末尾的 regime, one-hot
        regime_val = int(self.regime[end - 1])
        x_static = torch.zeros(3)
        x_static[min(regime_val, 2)] = 1.0

        cls_label = torch.tensor(self.cls_labels[end - 1])
        reg_label = torch.tensor(self.reg_labels[end - 1])

        return x_time, x_static, cls_label, reg_label
