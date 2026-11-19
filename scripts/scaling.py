"""Where does the surrogate start paying off? Measure FD solver vs FNO inference cost vs grid size.

Inference cost does not depend on trained weights, so this uses the production FNO
architecture with random weights at each resolution. The FD solver runs the same physical
duration at each resolution (time steps scale with N through the CFL condition).
All numbers are CPU, 1 thread, this machine. Writes results/scaling.json.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from physicsveto.config import RESULTS_DIR, load_case
from physicsveto.data.fields import velocity_model
from physicsveto.models.fno import FNO2d
from physicsveto.models.surrogate import ARCH_KWARGS
from physicsveto.solvers.wave import AcousticWaveSolver, WaveConfig, anchor_steps


def med(fn, r=3):
    fn()
    ts = []
    for _ in range(r):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main() -> None:
    torch.set_num_threads(1)
    case = load_case("case_a")
    rng = np.random.default_rng(0)
    rows = []
    for n in (64, 128, 256, 384, 512):
        f = n // 64
        cfg = WaveConfig(n=n, dx=case["dx"] / f, dt=case["dt"] / f, nt=case["nt"] * f,
                         sponge=case["sponge"] * f, sponge_strength=case["sponge_strength"],
                         f0=case["f0"], t0=case["t0"])
        solver = AcousticWaveSolver(cfg)
        v = velocity_model(n, rng, case["v_min"], case["v_max"], 7.0 * f)
        src = (case["source_depth"] * f, 32 * f)
        anchors = [a * f for a in case["anchors"]]
        t_fd = med(lambda s=solver, v=v, src=src, a=anchors: s.solve(v, src, save_steps=anchor_steps(a)),
                   r=2 if n > 200 else 3)
        model = FNO2d(**ARCH_KWARGS["fno"]).eval()
        x = torch.randn(8, 2, n, n)
        with torch.no_grad():
            t_nn = med(lambda m=model, x=x: m(x)) / 8
        rows.append({"n": n, "cells": n * n, "time_steps": cfg.nt, "fd_ms": t_fd * 1e3, "fno_ms_per_sample": t_nn * 1e3,
                     "speedup": t_fd / t_nn})
        print(rows[-1], flush=True)
    # fit power laws  t ~ a * N^b  to extrapolate (clearly labelled as extrapolation)
    ns = np.array([r["n"] for r in rows], dtype=float)
    b_fd = float(np.polyfit(np.log(ns), np.log([r["fd_ms"] for r in rows]), 1)[0])
    b_nn = float(np.polyfit(np.log(ns), np.log([r["fno_ms_per_sample"] for r in rows]), 1)[0])
    out = {
        "hardware": "CPU, 1 thread (this machine)",
        "measured": rows,
        "fd_exponent": round(b_fd, 2),
        "fno_exponent": round(b_nn, 2),
        "note": "2D only. FD cost grows ~N^3 (N^2 cells x N steps); FNO ~N^2 log N. "
                "In 3D the gap widens (FD ~N^4). Exponents are fitted to the measured points.",
    }
    (RESULTS_DIR / "scaling.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
