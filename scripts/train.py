"""Train a Case A surrogate on CPU.   python scripts/train.py --arch fno --epochs 80"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from physicsveto.config import DATA_DIR, RESULTS_DIR, WEIGHTS_DIR
from physicsveto.models.surrogate import ARCH_KWARGS, ARCHS, channel_scales, encode_inputs
from physicsveto.verify.checks import relative_l2

KWARGS = ARCH_KWARGS


def load(split: str):
    d = np.load(DATA_DIR / f"case_a_{split}.npz")
    return d["v"], d["src"], d["y"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCHS), default="fno")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=45.0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)

    vt, st, yt = load("train")
    vv, sv, yv = load("val")
    scales = channel_scales(yt)
    xt = torch.from_numpy(encode_inputs(vt, st))
    yt_n = torch.from_numpy(yt / scales[None, :, None, None])
    xv = torch.from_numpy(encode_inputs(vv, sv))

    model = ARCHS[a.arch](**KWARGS[a.arch])
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    steps = a.epochs * int(np.ceil(len(xt) / a.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)

    t0 = time.time()
    history = []
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), a.batch):
            idx = perm[i : i + a.batch]
            pred = model(xt[idx])
            tgt = yt_n[idx]
            # relative L2 per sample, averaged
            loss = (torch.linalg.vector_norm(pred - tgt, dim=(1, 2, 3))
                    / torch.linalg.vector_norm(tgt, dim=(1, 2, 3))).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(xv[i : i + 64]) for i in range(0, len(xv), 64)]).numpy()
        pv = pv * scales[None, :, None, None]
        val = float(np.median([relative_l2(p, y) for p, y in zip(pv, yv, strict=True)]))
        elapsed = time.time() - t0
        history.append({"epoch": ep + 1, "train_loss": tot / len(xt), "val_rel_l2_median": val, "seconds": elapsed})
        print(f"[{a.arch}] ep {ep + 1:3d}  loss {tot / len(xt):.4f}  val relL2 {val:.4f}  {elapsed / 60:.1f} min", flush=True)
        if elapsed > a.max_minutes * 60:
            print(f"[{a.arch}] time budget reached, stopping")
            break

    WEIGHTS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_DIR / f"case_a_{a.arch}.pt")
    meta = {
        "arch": a.arch,
        "kwargs": KWARGS[a.arch],
        "scales": scales.tolist(),
        "epochs_run": len(history),
        "train_minutes": round((time.time() - t0) / 60, 2),
        "train_hardware": f"CPU, {a.threads} threads",
        "n_params": sum(p.numel() for p in model.parameters()),
        "seed": a.seed,
    }
    (WEIGHTS_DIR / f"case_a_{a.arch}.json").write_text(json.dumps(meta, indent=2))
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"train_history_{a.arch}.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
