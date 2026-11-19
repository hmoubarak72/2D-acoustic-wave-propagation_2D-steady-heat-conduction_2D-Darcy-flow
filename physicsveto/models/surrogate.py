"""Input/output normalisation and a CPU inference wrapper shared by training, benchmark and the app."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from physicsveto.data.fields import source_map
from physicsveto.models.fno import FNO2d
from physicsveto.models.unet import UNet2d

V_CENTER, V_SCALE = 2750.0, 750.0

ARCHS = {"fno": FNO2d, "unet": UNet2d}
ARCH_KWARGS = {
    "fno": {"cin": 2, "cout": 24, "modes": 16, "width": 40, "layers": 4},
    "unet": {"cin": 2, "cout": 24, "base": 32},
}


def encode_inputs(v: np.ndarray, srcs: np.ndarray) -> np.ndarray:
    """v: [B, n, n], srcs: [B, 2] -> [B, 2, n, n] float32."""
    n = v.shape[-1]
    vn = (v - V_CENTER) / V_SCALE
    sm = np.stack([source_map(n, tuple(s)) for s in srcs])
    return np.stack([vn, sm], axis=1).astype(np.float32)


def channel_scales(y: np.ndarray) -> np.ndarray:
    """Per-channel RMS over the training set (p0, d1, d2 differ by orders of magnitude)."""
    return np.sqrt(np.mean(y.astype(np.float64) ** 2, axis=(0, 2, 3))).astype(np.float32) + 1e-12


class Surrogate:
    def __init__(self, arch: str, model: torch.nn.Module, scales: np.ndarray, meta: dict):
        self.arch = arch
        self.model = model.eval()
        self.scales = scales
        self.meta = meta

    @classmethod
    def load(cls, weights_dir: Path, arch: str) -> Surrogate:
        meta = json.loads((weights_dir / f"case_a_{arch}.json").read_text())
        model = ARCHS[arch](**meta["kwargs"])
        state = torch.load(weights_dir / f"case_a_{arch}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return cls(arch, model, np.array(meta["scales"], dtype=np.float32), meta)

    @torch.no_grad()
    def predict(self, v: np.ndarray, srcs: np.ndarray, batch: int = 64) -> np.ndarray:
        single = v.ndim == 2
        if single:
            v, srcs = v[None], np.asarray(srcs)[None]
        x = encode_inputs(v, srcs)
        outs = []
        for i in range(0, len(x), batch):
            outs.append(self.model(torch.from_numpy(x[i : i + batch])).numpy())
        y = np.concatenate(outs) * self.scales[None, :, None, None]
        return y[0] if single else y

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
