import torch
from torch import nn
import torch.nn.functional as F


class SpecNet(nn.Module):
    """1D CNN over FFT magnitudes (Spectral Domain)"""
    def __init__(self, in_channels=1, out_channels=128, channels=128, kernel_size=8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, channels, kernel_size=kernel_size, stride=kernel_size)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=kernel_size)
        self.conv3 = nn.Conv1d(channels, out_channels, kernel_size=kernel_size, stride=kernel_size, padding=2)

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