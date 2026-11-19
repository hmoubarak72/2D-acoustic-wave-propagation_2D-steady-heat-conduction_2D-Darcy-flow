"""Small U-Net CNN baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.GELU(),
    )


class UNet2d(nn.Module):
    def __init__(self, cin: int = 2, cout: int = 24, base: int = 32):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.e1 = block(cin + 2, c1)
        self.e2 = block(c1, c2)
        self.e3 = block(c2, c3)
        self.pool = nn.MaxPool2d(2)
        self.u2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.d2 = block(c2 * 2, c2)
        self.u1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.d1 = block(c1 * 2, c1)
        self.out = nn.Conv2d(c1, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        gy = torch.linspace(0, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        gx = torch.linspace(0, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        x = torch.cat([x, gy, gx], dim=1)
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.u2(e3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)
