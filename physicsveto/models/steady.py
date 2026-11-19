"""Surrogates for the steady cases B and C: input encoding, physics loss, and CPU inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from physicsveto.models.fno import FNO2d
from physicsveto.models.unet import UNet2d

ARCHS = {"fno": FNO2d, "unet": UNet2d}
CIN = {"case_b": 2, "case_c": 1}


def arch_kwargs(case_key: str, arch: str) -> dict:
    if arch == "fno":
        return {"cin": CIN[case_key], "cout": 1, "modes": 12, "width": 32, "layers": 4}
    return {"cin": CIN[case_key], "cout": 1, "base": 32}


def encode_inputs(case_key: str, k: np.ndarray, f: np.ndarray) -> np.ndarray:
    """[B, n, n] fields -> [B, C, n, n]. Case C does not see f: training always uses f = 1."""
    if case_key == "case_b":
        return np.stack([np.log(k), f / 100.0], axis=1).astype(np.float32)
    return ((k - 7.5) / 4.5)[:, None].astype(np.float32)


def torch_operator(u: torch.Tensor, k: torch.Tensor, h: float) -> torch.Tensor:
    """-div(k grad u) on interior nodes, same stencil as physicsveto.solvers.elliptic. Shapes [B, n, n]."""
    kz = 2 * k[:, 1:, :] * k[:, :-1, :] / (k[:, 1:, :] + k[:, :-1, :])
    kx = 2 * k[:, :, 1:] * k[:, :, :-1] / (k[:, :, 1:] + k[:, :, :-1])
    fz = kz * (u[:, 1:, :] - u[:, :-1, :]) / h
    fx = kx * (u[:, :, 1:] - u[:, :, :-1]) / h
    return -((fz[:, 1:, 1:-1] - fz[:, :-1, 1:-1]) + (fx[:, 1:-1, 1:] - fx[:, 1:-1, :-1])) / h


class SteadySurrogate:
    def __init__(self, case_key: str, name: str, model: torch.nn.Module, scale: float, meta: dict):
        self.case_key, self.name, self.arch = case_key, name, meta["arch"]
        self.model = model.eval()
        self.scale = scale
        self.meta = meta

    @classmethod
    def load(cls, weights_dir: Path, case_key: str, name: str) -> SteadySurrogate:
        meta = json.loads((weights_dir / f"{case_key}_{name}.json").read_text())
        model = ARCHS[meta["arch"]](**meta["kwargs"])
        model.load_state_dict(torch.load(weights_dir / f"{case_key}_{name}.pt", map_location="cpu", weights_only=True))
        return cls(case_key, name, model, float(meta["scale"]), meta)

    @torch.no_grad()
    def predict(self, k: np.ndarray, f: np.ndarray, batch: int = 64) -> np.ndarray:
        single = k.ndim == 2
        if single:
            k, f = k[None], f[None]
        x = encode_inputs(self.case_key, k, f)
        out = np.concatenate([self.model(torch.from_numpy(x[i: i + batch]))[:, 0].numpy()
                              for i in range(0, len(x), batch)]) * self.scale
        return out[0] if single else out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
