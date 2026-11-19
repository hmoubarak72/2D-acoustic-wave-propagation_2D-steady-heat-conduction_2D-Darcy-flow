"""Calibrate and validate the physics veto for Case B or C, time solver vs surrogates.

    python scripts/benchmark_steady.py --case case_b
Writes results/<case>/{calibration.json, veto_eval.csv, benchmark_summary.json, leaderboard.json}.
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
import torch

from physicsveto.config import DATA_DIR, RESULTS_DIR, WEIGHTS_DIR, load_case, load_veto
from physicsveto.data.steady import PLANTED
from physicsveto.models.steady import SteadySurrogate
from physicsveto.solvers.elliptic import EllipticSolver
from physicsveto.verify.ood import MahalanobisOOD
from physicsveto.verify.steady import SteadyVeto, steady_features
from physicsveto.verify.trust import ACCEPT, VETO

CORRUPTIONS = ("checkerboard_artifact", "amplitude_drift")
LABELS = {"fno": "FNO", "unet": "U-Net", "fno_pino": "FNO + physics loss",
          "fno_n100": "FNO, 100 samples", "fno_pino_n100": "FNO + physics loss, 100 samples"}


def corrupt(u, kind, rng):
    out = u.copy()
    if kind == "checkerboard_artifact":
        n = u.shape[-1]
        cb = np.indices((n, n)).sum(axis=0) % 2 * 2.0 - 1.0
        out += rng.uniform(0.02, 0.06) * np.abs(u).max() * cb
    else:
        out *= rng.uniform(1.25, 1.5)
    return out


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def time_it(fn, repeat=5):
    fn()
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def load(key, split):
    d = np.load(DATA_DIR / f"{key}_{split}.npz")
    return d["k"], d["f"], d["u"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["case_b", "case_c"], required=True)
    key = ap.parse_args().case
    torch.set_num_threads(1)
    case, veto_cfg = load_case(key), load_veto(key)
    out_dir = RESULTS_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [p.stem[len(key) + 1:] for p in sorted(WEIGHTS_DIR.glob(f"{key}_*.pt"))]
    sur = {n: SteadySurrogate.load(WEIGHTS_DIR, key, n) for n in names}
    main_m, alt_m = sur["fno"], sur["unet"]
    rng = np.random.default_rng(321)

    # calibration
    kt, ft, _ = load(key, "train")
    ood = MahalanobisOOD.fit(np.stack([steady_features(key, k, f) for k, f in zip(kt, ft, strict=True)]))
    kv, fv, uv = load(key, "val")
    pf, pu = main_m.predict(kv, fv), alt_m.predict(kv, fv)
    prov = SteadyVeto(key, veto_cfg, {"levels": {c: 1.0 for c in veto_cfg["checks"]}, "ood": ood.to_dict()})
    raw = [prov.raw_metrics(k, f, a, b)[0] for k, f, a, b in zip(kv, fv, pf, pu, strict=True)]
    levels = {c: max(float(np.percentile([r[c] for r in raw], veto_cfg["calibration_percentile"])),
                     float(spec.get("floor", 1e-9))) for c, spec in veto_cfg["checks"].items()}
    calibration = {"levels": levels, "ood": ood.to_dict(), "percentile": veto_cfg["calibration_percentile"]}
    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))
    veto = SteadyVeto(key, veto_cfg, calibration)
    tol = veto_cfg["error_tolerance"]

    rows = []

    def add(name, group, i, k, f, u, u_alt, ref):
        rep = veto.evaluate(k, f, u, u_alt)
        e = rel(u, ref)
        rows.append({"set": name, "group": group, "idx": i, "verdict": rep.verdict,
                     "trust_score": round(rep.trust_score, 2),
                     **{f"m_{c}": v for c, v in rep.metrics.items()},
                     **{f"r_{c}": v for c, v in rep.ratios.items()},
                     "rel_l2_fno": e, "rel_l2_unet": rel(u_alt, ref), "true_bad": bool(e > tol)})

    kte, fte, ute = load(key, "test")
    p_f, p_u = main_m.predict(kte, fte), alt_m.predict(kte, fte)
    for i in range(len(kte)):
        add("in_distribution", "in-distribution", i, kte[i], fte[i], p_f[i], p_u[i], ute[i])
    for kind in CORRUPTIONS:
        for i in range(40):
            add(kind, "model corruption", i, kte[i], fte[i], corrupt(p_f[i], kind, rng), p_u[i], ute[i])
    for i in range(20):
        add("reference_solver_output", "sanity", i, kte[i], fte[i], ute[i], ute[i], ute[i])
    planted = np.load(DATA_DIR / f"{key}_planted.npz")
    for m in PLANTED[key]:
        ks, fs, us = planted[f"{m}__k"], planted[f"{m}__f"], planted[f"{m}__u"]
        a, b = main_m.predict(ks, fs), alt_m.predict(ks, fs)
        for i in range(len(ks)):
            add(m, "input failure", i, ks[i], fs[i], a[i], b[i], us[i])
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "veto_eval.csv", index=False)

    ev = df[df.group != "sanity"]
    bad = ev.true_bad.values

    def pr(pred):
        tp, fp, fn = int((pred & bad).sum()), int((pred & ~bad).sum()), int((~pred & bad).sum())
        return tp / max(tp + fp, 1), tp / max(tp + fn, 1)

    v_p, v_r = pr((ev.verdict == VETO).values)
    f_p, f_r = pr((ev.verdict != ACCEPT).values)
    per_mode = {}
    for m, g in ev.groupby("set"):
        nb = int(g.true_bad.sum())
        per_mode[m] = {"group": g.group.iloc[0], "n": int(len(g)),
                       "share_truly_bad": round(float(g.true_bad.mean()), 3),
                       "veto_rate": round(float((g.verdict == VETO).mean()), 3),
                       "flag_rate": round(float((g.verdict != ACCEPT).mean()), 3),
                       "recall_of_bad": round(float(((g.verdict != ACCEPT) & g.true_bad).sum() / nb), 3) if nb else None,
                       "median_rel_l2": round(float(g.rel_l2_fno.median()), 4)}
    ind = df[df.set == "in_distribution"]
    acc = ev[ev.verdict == ACCEPT]
    sanity = df[df.set == "reference_solver_output"]

    solver = EllipticSolver(case["n"])
    t_fd = time_it(lambda: solver.solve(kte[0], fte[0]))
    leaderboard = [{"model": "FD solver (reference)", "rel_l2_median": 0.0, "latency_ms": round(t_fd * 1e3, 2),
                    "batched_ms_per_sample": round(t_fd * 1e3, 2), "params": 0, "train_minutes": 0.0,
                    "n_train": 0, "physics_weight": 0.0, "residual_median": 0.0}]
    timing = {}
    for n, s in sur.items():
        t1 = time_it(lambda s=s: s.predict(kte[0], fte[0]))
        tb = time_it(lambda s=s: s.predict(kte[:64], fte[:64])) / 64
        timing[n] = (t1, tb)
        pred = p_f if n == "fno" else (p_u if n == "unet" else s.predict(kte, fte))
        res_med = float(np.median([veto.raw_metrics(k, f, u)[0]["pde_residual"]
                                   for k, f, u in zip(kte, fte, pred, strict=True)]))
        leaderboard.append({"model": LABELS.get(n, n),
                            "rel_l2_median": round(float(np.median([rel(a, b) for a, b in zip(pred, ute, strict=True)])), 4),
                            "latency_ms": round(t1 * 1e3, 2), "batched_ms_per_sample": round(tb * 1e3, 3),
                            "params": s.n_params(), "train_minutes": s.meta["train_minutes"],
                            "n_train": s.meta["n_train"], "physics_weight": s.meta["physics_weight"],
                            "residual_median": round(res_med, 4)})
    (out_dir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))

    summary = {
        "case": f"{case['name']}, {case['n']}x{case['n']}",
        "hardware": f"{platform.machine()}, 1 CPU thread",
        "fd_solver_ms": round(t_fd * 1e3, 2),
        "fno_single_ms": round(timing["fno"][0] * 1e3, 2),
        "fno_batched_ms_per_sample": round(timing["fno"][1] * 1e3, 3),
        "speedup_batched": round(t_fd / timing["fno"][1], 2),
        "fno_in_dist_rel_l2_median": round(float(ind.rel_l2_fno.median()), 4),
        "unet_in_dist_rel_l2_median": round(float(ind.rel_l2_unet.median()), 4),
        "in_dist_veto_rate": round(float((ind.verdict == VETO).mean()), 3),
        "in_dist_flag_rate": round(float((ind.verdict != ACCEPT).mean()), 3),
        "error_tolerance": tol,
        "veto_precision": round(v_p, 3), "veto_recall": round(v_r, 3),
        "flag_precision": round(f_p, 3), "flag_recall": round(f_r, 3),
        "n_eval": int(len(ev)), "n_truly_bad": int(bad.sum()),
        "accepted_share_truly_bad": round(float(acc.true_bad.mean()), 4) if len(acc) else None,
        "unfiltered_share_truly_bad": round(float(ev.true_bad.mean()), 4),
        "reference_output_accept_rate": round(float((sanity.verdict == ACCEPT).mean()), 3),
        "per_mode": per_mode,
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_mode"}, indent=2))
    print(pd.DataFrame(per_mode).T[["share_truly_bad", "veto_rate", "flag_rate", "recall_of_bad", "median_rel_l2"]])
    print(pd.DataFrame(leaderboard))


if __name__ == "__main__":
    main()
