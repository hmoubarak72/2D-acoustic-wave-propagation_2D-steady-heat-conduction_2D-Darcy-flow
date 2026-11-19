"""2D Fourier Neural Operator (Li et al., 2021) in plain PyTorch."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, cin: int, cout: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(scale * torch.randn(cin, cout, modes, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(cin, cout, modes, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        m = self.modes
        xf = torch.fft.rfft2(x)
        out = torch.zeros(b, self.w1.shape[1], h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], self.w1)
        out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], self.w2)
        return torch.fft.irfft2(out, s=(h, w))


class FNO2d(nn.Module):
    """Input [B, Cin, H, W] -> output [B, Cout, H, W]. Grid coordinates are appended internally."""

    def __init__(self, cin: int = 2, cout: int = 24, modes: int = 16, width: int = 40, layers: int = 4):
        super().__init__()
        self.lift = nn.Conv2d(cin + 2, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, modes) for _ in range(layers)])
        self.skip = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.proj1 = nn.Conv2d(width, 128, 1)
        self.proj2 = nn.Conv2d(128, cout, 1)

    @staticmethod
    def grid(x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        gy = torch.linspace(0, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        gx = torch.linspace(0, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([gy, gx], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(torch.cat([x, self.grid(x)], dim=1))
        for i, (s, k) in enumerate(zip(self.spec, self.skip, strict=True)):
            x = s(x) + k(x)
            if i < len(self.spec) - 1:
                x = F.gelu(x)
        return self.proj2(F.gelu(self.proj1(x)))
