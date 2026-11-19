"""Classic PINN baseline for Case B (steady heat conduction).

A PINN (Raissi et al., 2019) trains one small network per problem instance by minimising the PDE
residual at random collocation points, with no solver data. That makes it a solver, not a surrogate:
it has to be retrained for every new conductivity / source layout. This script measures what that
costs on CPU for a few test instances and writes results/case_b/pinn.json.

    python scripts/pinn_case_b.py --instances 3 --iters 3000
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from physicsveto.config import DATA_DIR, RESULTS_DIR


class MLP(torch.nn.Module):
    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        layers, d = [], 2
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
            d = width
        layers.append(torch.nn.Linear(d, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        z, x = xy[:, :1], xy[:, 1:]
        return (z * (1 - z) * x * (1 - x) * 16.0) * self.net(xy)  # hard Dirichlet u = 0 on the boundary


def sampler(field: np.ndarray):
    """Bilinear interpolation of a grid field at points in [0,1]^2 (differentiable in the coordinates)."""
    g = torch.from_numpy(field.astype(np.float32))[None, None]

    def f(xy: torch.Tensor) -> torch.Tensor:
        grid = (xy[:, [1, 0]] * 2 - 1)[None, None]  # grid_sample wants (x, y) in [-1, 1]
        return F.grid_sample(g, grid, mode="bilinear", align_corners=True)[0, 0, 0][:, None]

    return f


def train_pinn(k: np.ndarray, q: np.ndarray, iters: int, n_pts: int, seed: int) -> tuple[MLP, float]:
    torch.manual_seed(seed)
    model = MLP()
    h = 1.0 / (k.shape[-1] - 1)
    kz_grid, kx_grid = np.gradient(k.astype(np.float64), h)
    kf, kzf, kxf, qf = sampler(k), sampler(kz_grid), sampler(kx_grid), sampler(q)
    q_rms = float(np.sqrt(np.mean(q**2))) + 1e-9
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    t0 = time.time()
    for _ in range(iters):
        xy = torch.rand(n_pts, 2, requires_grad=True)
        u = model(xy)
        gu = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
        lap = sum(torch.autograd.grad(gu[:, i].sum(), xy, create_graph=True)[0][:, i] for i in range(2))
        p = xy.detach()  # coefficients are sampled, not differentiated: div(k grad u) = k lap u + grad k . grad u
        div = kf(p)[:, 0] * lap + kzf(p)[:, 0] * gu[:, 0] + kxf(p)[:, 0] * gu[:, 1]
        res = (-div[:, None] - qf(p)) / q_rms
        loss = (res**2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--points", type=int, default=2048)
    a = ap.parse_args()
    torch.set_num_threads(1)
    d = np.load(DATA_DIR / "case_b_test.npz")
    n = d["k"].shape[-1]
    zz, xx = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    grid = torch.from_numpy(np.stack([zz.ravel(), xx.ravel()], 1).astype(np.float32))
    rows = []
    for i in range(a.instances):
        k, q, u_ref = d["k"][i], d["f"][i], d["u"][i]
        model, secs = train_pinn(k, q, a.iters, a.points, seed=i)
        with torch.no_grad():
            u = model(grid).numpy().reshape(n, n)
        err = float(np.linalg.norm(u - u_ref) / np.linalg.norm(u_ref))
        rows.append({"instance": i, "train_seconds": round(secs, 1), "rel_l2": round(err, 4)})
        print(rows[-1], flush=True)
        if i == 0:
            np.save(RESULTS_DIR / "case_b" / "pinn_example.npy", np.stack([u, u_ref, k, q]).astype(np.float32))
    out = {
        "method": "PINN (MLP 4x64 tanh, hard Dirichlet BC, residual loss only, Adam)",
        "iters": a.iters, "collocation_points": a.points, "hardware": "CPU, 1 thread",
        "instances": rows,
        "median_rel_l2": float(np.median([r["rel_l2"] for r in rows])),
        "median_train_seconds": float(np.median([r["train_seconds"] for r in rows])),
        "note": "Each instance is trained from scratch. No solver data is used.",
    }
    (RESULTS_DIR / "case_b" / "pinn.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
