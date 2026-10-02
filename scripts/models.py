"""
7 Model Deep Learning Suite for Solar Flare Forecasting
Supports: 1D CNN, CNN+BiLSTM, CNN+Attention+BiLSTM, TCN, Transformer Encoder, InceptionTime, ResNet1D.
Input Shape: (Batch, Sequence Length = 3600, Channels = 4) or (Batch, Channels = 4, Sequence Length = 3600).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. 1D CNN Architecture
# =====================================================================
class Conv1DModel(nn.Module):
    def __init__(self, in_channels: int = 4, sequence_length: int = 3600, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x input shape: (B, C, L) or (B, L, C)
        if x.dim() == 3 and x.shape[1] != 4 and x.shape[2] == 4:
            x = x.transpose(1, 2)
        feat = self.features(x).squeeze(-1)
        return self.classifier(feat)


# =====================================================================
# 2. CNN + BiLSTM Hybrid Architecture
# =====================================================================
class CNNBiLSTMModel(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] != 4 and x.shape[2] == 4:
            x = x.transpose(1, 2)
        feat = self.conv(x)  # (B, 64, L')
        feat = feat.transpose(1, 2)  # (B, L', 64)
        lstm_out, _ = self.lstm(feat)  # (B, L', 2*hidden_dim)
        pooled = torch.mean(lstm_out, dim=1)
        return self.fc(pooled)


# =====================================================================
# 3. CNN + Attention + BiLSTM Architecture
# =====================================================================
class CNNAttentionBiLSTMModel(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.bilstm = nn.LSTM(64, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] != 4 and x.shape[2] == 4:
            x = x.transpose(1, 2)
        feat = self.conv(x).transpose(1, 2)
        lstm_out, _ = self.bilstm(feat)
        weights = F.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(lstm_out * weights, dim=1)
        return self.fc(context)


# =====================================================================
# 4. Temporal Convolutional Network (TCN)
# =====================================================================
class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.res(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 2):
        super().__init__()
        self.layer1 = TCNBlock(in_channels, 32, dilation=1)
        self.pool1 = nn.MaxPool1d(2)
        self.layer2 = TCNBlock(32, 64, dilation=2)
        self.pool2 = nn.MaxPool1d(2)
        self.layer3 = TCNBlock(64, 128, dilation=4)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] != 4 and x.shape[2] == 4:
            x = x.transpose(1, 2)
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x)).squeeze(-1)
        return self.classifier(x)


# =====================================================================
# 5. Transformer Encoder Architecture
# =====================================================================
class TransformerModel(nn.Module):
    def __init__(self, in_channels: int = 4, d_model: int = 64, nhead: int = 4, num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] != 4 and x.shape[2] == 4:
            x = x.transpose(1, 2)
        emb = self.embedding(x).transpose(1, 2)  # (B, L', d_model)
        trans = self.transformer(emb)
        pooled = torch.mean(trans, dim=1)
        return self.classifier(pooled)


# =====================================================================
# 6. InceptionTime Architecture
# =====================================================================
class InceptionModule1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, 16, 1) if in_ch > 4 else nn.Identity()
        ch = 16 if in_ch > 4 else in_ch
        
        self.conv10 = nn.Conv1d(ch, out_ch // 3, kernel_size=10, padding=4)
        self.conv20 = nn.Conv1d(ch, out_ch // 3, kernel_size=20, padding=9)
        self.conv40 = nn.Conv1d(ch, out_ch // 3, kernel_size=40, padding=19)

        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_ch, out_ch // 3, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_ch // 3 * 4)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bottleneck(x)
        c1 = self.conv10(b)
        c2 = self.conv20(b)
        c3 = self.conv40(b)
        cp = self.conv_pool(self.maxpool(x))
        
        # Align sequence lengths across branches
        min_len = min(c1.shape[2], c2.shape[2], c3.shape[2], cp.shape[2])
        out = torch.cat([c1[:, :, :min_len], c2[:, :, :min_len], c3[:, :, :min_len], cp[:, :, :min_len]], dim=1)
        return self.relu(self.bn(out))

class InceptionTimeModel(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 2):
        super().__init__()
        self.inc1 = InceptionModule1D(in_channels, 32)
        self.pool1 = nn.MaxPool1d(2)
        self.inc2 = InceptionModule1D(32 // 3 * 4, 64)
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 // 3 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] != 4 and x.shape[2] == 4:
            x = x.transpose(1, 2)
        x = self.pool1(self.inc1(x))
        x = self.pool2(self.inc2(x)).squeeze(-1)
        return self.fc(x)


# =====================================================================
# 7. ResNet1D Architecture
# =====================================================================
class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=stride, padding=2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride),
            nn.BatchNorm1d(out_ch)
        ) if stride != 1 or in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)

class ResNet1DModel(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 2):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.layer1 = ResBlock1D(32, 64, stride=2)
        self.layer2 = ResBlock1D(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] != 4 and x.shape[2] == 4:
            x = x.transpose(1, 2)
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


# Model Registry Dictionary
MODEL_REGISTRY = {
    "1d_cnn": Conv1DModel,
    "cnn_bilstm": CNNBiLSTMModel,
    "cnn_attention_bilstm": CNNAttentionBiLSTMModel,
    "tcn": TCNModel,
    "transformer": TransformerModel,
    "inceptiontime": InceptionTimeModel,
    "resnet1d": ResNet1DModel
}
