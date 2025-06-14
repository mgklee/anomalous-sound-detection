import torch
from torch import nn
import torch.nn.functional as F


class SpecNet(nn.Module):
    """1D CNN over FFT magnitudes (Spectral Domain)"""
    def __init__(self, in_channels=1, out_channels=128, kernel_size=8):
        super(SpecNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=kernel_size, stride=kernel_size)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=kernel_size, stride=kernel_size)
        self.conv3 = nn.Conv1d(128, out_channels, kernel_size=kernel_size, stride=kernel_size, padding=2)

    def forward(self, x_wav):
        # x_wav: [B, 1, 160000] raw waveform
        # 1) compute FFT along time dimension
        Xf = torch.fft.fft(x_wav.squeeze(1), dim=-1)    # [B, 160000] complex tensor

        # 2) magnitude spectrum
        mag = torch.abs(Xf).unsqueeze(1)    # [B, 1, 160000]

        # 3) pass through Conv1d layers
        y = F.relu(self.conv1(mag))         # [B, 128, 20000]
        y = F.relu(self.conv2(y))           # [B, 128, 2500]
        out = F.relu(self.conv3(y))         # [B, 128, 313]
        return out


class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super(TgramNet, self).__init__()
        # if "center=True" of stft, padding = win_len / 2
        self.conv_extractor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len//2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                # 313(10), 63(2), 126(4)
                nn.LayerNorm(313),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for _ in range(num_layer)])

    def forward(self, x):
        out = self.conv_extractor(x)
        out = self.conv_encoder(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ResidualBlock, self).__init__()
        self.dilated_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=3,
            dilation=dilation, padding=dilation
        )
        self.residual_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)

    def forward(self, x):
        out = torch.tanh(self.dilated_conv(x)) * torch.sigmoid(self.dilated_conv(x))
        residual = self.residual_conv(out)
        skip = self.skip_conv(out)
        return x + residual, skip


class WaveNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super(WaveNet, self).__init__()
        self.conv_extractor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len//2, bias=False)

        # Causal Conv1D
        self.causal_conv = nn.Conv1d(mel_bins, 512, kernel_size=2)

        # Dilated residual blocks (4 layers, dilation = [1, 2, 4, 8])
        self.blocks = nn.ModuleList()
        for _ in range(num_layer):
            for dilation in [1, 2, 4, 8]:
                self.blocks.append(ResidualBlock(512, 512, dilation))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(512, 128, kernel_size=1)
        )

    def forward(self, x):
        x = self.conv_extractor(x)     # [B, mel_bins, T]
        x = F.pad(x, (1, 0))           # Causal padding
        x = self.causal_conv(x)        # [B, 512, T]

        skip_total = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_total = skip if isinstance(skip_total, int) else skip_total + skip

        x = self.output_proj(skip_total)          # [B, 128, 313]
        return x


class Temporal_Attention(nn.Module):
    def __init__(self, feature_dim=128):
        super(Temporal_Attention, self).__init__()
        self.feature_dim = feature_dim
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, 1, 128, 313)
        x = x.squeeze(1)
        x = x.transpose(1,2)    # (B, 313, 128)

        x1 = self.max_pool(x)   # (B, 313, 1)
        x2 = self.avg_pool(x)   # (B, 313, 1)

        feats = x1 + x2
        feats = feats.repeat(1, 1, self.feature_dim)

        refined_feats = self.sigmoid(feats).transpose(1,2) * x.transpose(1,2)
        return refined_feats