"""Train a surrogate for Case B or C on CPU.

    python scripts/train_steady.py --case case_b --arch fno
    python scripts/train_steady.py --case case_b --arch fno --physics 0.2 --name fno_pino   # physics-informed
    python scripts/train_steady.py --case case_b --arch fno --n-train 100 --name fno_n100  # low-data
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from physicsveto.config import DATA_DIR, RESULTS_DIR, WEIGHTS_DIR
from physicsveto.models.steady import ARCHS, arch_kwargs, encode_inputs, torch_operator


def load(key, split):
    d = np.load(DATA_DIR / f"{key}_{split}.npz")
    return d["k"], d["f"], d["u"]


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["case_b", "case_c"], required=True)
    ap.add_argument("--arch", choices=list(ARCHS), default="fno")
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--physics", type=float, default=0.0, help="weight of the PDE-residual loss (0 = data only)")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=40.0)
    a = ap.parse_args()
    name = a.name or a.arch
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)

    kt, ft, ut = load(a.case, "train")
    if a.n_train:
        kt, ft, ut = kt[: a.n_train], ft[: a.n_train], ut[: a.n_train]
    kv, fv, uv = load(a.case, "val")
    scale = float(np.sqrt(np.mean(ut.astype(np.float64) ** 2)))
    xt = torch.from_numpy(encode_inputs(a.case, kt, ft))
    yt = torch.from_numpy(ut / scale)
    kt_t, ft_t = torch.from_numpy(kt), torch.from_numpy(ft)
    xv = torch.from_numpy(encode_inputs(a.case, kv, fv))
    h = 1.0 / (kt.shape[-1] - 1)

    model = ARCHS[a.arch](**arch_kwargs(a.case, a.arch))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    steps_per_epoch = int(np.ceil(len(xt) / a.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.1)
    t0, hist = time.time(), []
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), a.batch):
            idx = perm[i: i + a.batch]
            pred = model(xt[idx])[:, 0]
            loss = (torch.linalg.vector_norm(pred - yt[idx], dim=(1, 2))
                    / torch.linalg.vector_norm(yt[idx], dim=(1, 2))).mean()
            if a.physics > 0:
                res = torch_operator(pred * scale, kt_t[idx], h) - ft_t[idx][:, 1:-1, 1:-1]
                rnorm = torch.linalg.vector_norm(res, dim=(1, 2)) / torch.linalg.vector_norm(ft_t[idx][:, 1:-1, 1:-1], dim=(1, 2))
                loss = loss + a.physics * rnorm.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(xv[i: i + 64])[:, 0] for i in range(0, len(xv), 64)]).numpy() * scale
        val = float(np.median([rel_l2(p, u) for p, u in zip(pv, uv, strict=True)]))
        el = time.time() - t0
        hist.append({"epoch": ep + 1, "train_loss": tot / len(xt), "val_rel_l2_median": val, "seconds": el})
        print(f"[{a.case}/{name}] ep {ep + 1:3d} loss {tot / len(xt):.4f} val {val:.4f} {el / 60:.1f} min", flush=True)
        if el > a.max_minutes * 60:
            break
    WEIGHTS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_DIR / f"{a.case}_{name}.pt")
    meta = {"arch": a.arch, "kwargs": arch_kwargs(a.case, a.arch), "scale": scale, "physics_weight": a.physics,
            "n_train": len(xt), "epochs_run": len(hist), "train_minutes": round((time.time() - t0) / 60, 2),
            "train_hardware": f"CPU, {a.threads} thread(s)", "seed": a.seed}
    (WEIGHTS_DIR / f"{a.case}_{name}.json").write_text(json.dumps(meta, indent=2))
    out = RESULTS_DIR / a.case
    out.mkdir(parents=True, exist_ok=True)
    (out / f"train_history_{name}.json").write_text(json.dumps(hist, indent=2))


if __name__ == "__main__":
    main()
