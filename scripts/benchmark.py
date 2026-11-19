"""Calibrate the physics veto, validate it on planted failures, and time solver vs surrogates.

Writes:
  results/calibration.json      reference levels for each check + OOD model
  results/veto_eval.csv         one row per evaluated prediction
  results/benchmark_summary.json headline numbers used by the README and the app
  results/leaderboard.json      per-model accuracy / latency / size / veto rate
"""

from __future__ import annotations

import json
import platform
import time

import numpy as np
import pandas as pd
import torch

from physicsveto.config import DATA_DIR, RESULTS_DIR, WEIGHTS_DIR, load_case, load_veto
from physicsveto.models.surrogate import Surrogate
from physicsveto.solvers.wave import AcousticWaveSolver, WaveConfig, anchor_steps
from physicsveto.verify.checks import relative_l2
from physicsveto.verify.ood import MahalanobisOOD, input_features
from physicsveto.verify.trust import ACCEPT, REVIEW, VETO, PhysicsVeto

CORRUPTIONS = ("checkerboard_artifact", "amplitude_drift")


def corrupt(pred: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """Simulated model bugs applied to otherwise good FNO predictions."""
    out = pred.copy()
    if kind == "checkerboard_artifact":
        n = pred.shape[-1]
        cb = np.indices((n, n)).sum(axis=0) % 2 * 2.0 - 1.0
        amp = rng.uniform(0.04, 0.10) * np.abs(pred[1::3][:, None]).max()
        out[0::3] += amp * cb[None]  # p0 channels only: breaks the PDE, keeps the look
    elif kind == "amplitude_drift":
        out *= rng.uniform(1.35, 1.6)
    return out


def load(name: str):
    d = np.load(DATA_DIR / f"case_a_{name}.npz")
    return d["v"], d["src"], d["y"]


def time_it(fn, repeat: int = 5) -> float:
    fn()
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main() -> None:
    torch.set_num_threads(1)  # single-core timing for a fair CPU-vs-CPU comparison
    case, veto_cfg = load_case("case_a"), load_veto()
    cfg = WaveConfig.from_dict(case)
    anchors = case["anchors"]
    fno = Surrogate.load(WEIGHTS_DIR, "fno")
    unet = Surrogate.load(WEIGHTS_DIR, "unet")
    rng = np.random.default_rng(123)

    # ---------- calibration on held-out in-distribution validation data ----------
    vt, st, _ = load("train")
    ood = MahalanobisOOD.fit(np.stack([input_features(v, s) for v, s in zip(vt, st, strict=True)]))
    vv, sv, yv = load("val")
    pf, pu = fno.predict(vv, sv), unet.predict(vv, sv)
    provisional = {"levels": {k: 1.0 for k in veto_cfg["checks"]}, "ood": ood.to_dict()}
    pv = PhysicsVeto(case, veto_cfg, provisional)
    raw = [pv.raw_metrics(v, tuple(s), p, q)[0] for v, s, p, q in zip(vv, sv, pf, pu, strict=True)]
    pct = veto_cfg["calibration_percentile"]
    levels = {}
    for k, spec in veto_cfg["checks"].items():
        lvl = float(np.percentile([r[k] for r in raw], pct))
        levels[k] = max(lvl, float(spec.get("floor", 1e-9)))
    calibration = {"levels": levels, "ood": ood.to_dict(), "percentile": pct, "n_val": len(vv)}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "calibration.json").write_text(json.dumps(calibration, indent=2))
    veto = PhysicsVeto(case, veto_cfg, calibration)

    # ---------- evaluation sets ----------
    sets: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    vte, ste, yte = load("test")
    sets.append(("in_distribution", "in-distribution", vte, ste, yte))
    planted = np.load(DATA_DIR / "case_a_planted.npz")
    modes = sorted({k.split("__")[0] for k in planted.files})
    for m in modes:
        sets.append((m, "input failure", planted[f"{m}__v"], planted[f"{m}__src"], planted[f"{m}__y"]))

    rows = []
    tol = veto_cfg["error_tolerance"]
    for name, group, vs, ss, ys in sets:
        p_f, p_u = fno.predict(vs, ss), unet.predict(vs, ss)
        for i, (v, s, y) in enumerate(zip(vs, ss, ys, strict=True)):
            src = tuple(int(x) for x in s)
            rep = veto.evaluate(v, src, p_f[i], p_u[i])
            rows.append(_row(name, group, i, rep, relative_l2(p_f[i], y), relative_l2(p_u[i], y), tol))
        if name == "in_distribution":
            # simulated model bugs on in-distribution inputs
            for kind in CORRUPTIONS:
                for i, (v, s, y) in enumerate(zip(vs, ss, ys, strict=True)):
                    if i >= 40:
                        break
                    src = tuple(int(x) for x in s)
                    bad = corrupt(p_f[i], kind, rng)
                    rep = veto.evaluate(v, src, bad, p_u[i])
                    rows.append(_row(kind, "model corruption", i, rep, relative_l2(bad, y), relative_l2(p_u[i], y), tol))
        # the exact solver output must always be accepted (sanity check of the veto)
        if name == "in_distribution":
            for i, (v, s, y) in enumerate(zip(vs[:20], ss[:20], ys[:20], strict=True)):
                rep = veto.evaluate(v, tuple(int(x) for x in s), y, y)
                rows.append(_row("reference_solver_output", "sanity", i, rep, 0.0, 0.0, tol))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "veto_eval.csv", index=False)

    # ---------- veto quality ----------
    ev = df[df.group != "sanity"]
    bad = ev.true_bad.values
    vetoed = (ev.verdict == VETO).values
    flagged = (ev.verdict != ACCEPT).values

    def pr(pred, truth):
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        return (tp / max(tp + fp, 1), tp / max(tp + fn, 1), tp, fp, fn)

    v_p, v_r, *_ = pr(vetoed, bad)
    f_p, f_r, *_ = pr(flagged, bad)
    per_mode = {}
    for m, g in ev.groupby("set"):
        per_mode[m] = {
            "group": g.group.iloc[0],
            "n": int(len(g)),
            "share_truly_bad": round(float(g.true_bad.mean()), 3),
            "veto_rate": round(float((g.verdict == VETO).mean()), 3),
            "flag_rate": round(float((g.verdict != ACCEPT).mean()), 3),
            "recall_of_bad": round(float(((g.verdict != ACCEPT) & g.true_bad).sum() / max(g.true_bad.sum(), 1)), 3)
            if g.true_bad.sum() else None,
            "median_rel_l2": round(float(g.rel_l2_fno.median()), 4),
        }
    ind = df[df.set == "in_distribution"]
    accepted_all = ev[ev.verdict == ACCEPT]
    sanity = df[df.set == "reference_solver_output"]

    # ---------- timing ----------
    solver = AcousticWaveSolver(cfg)
    v0, s0 = vte[0], tuple(int(x) for x in ste[0])
    t_fd = time_it(lambda: solver.solve(v0, s0, save_steps=anchor_steps(anchors)))
    vb, sb = vte[:64], ste[:64]
    timing = {}
    for sur in (fno, unet):
        t1 = time_it(lambda sur=sur: sur.predict(v0, np.array(s0)))
        tb = time_it(lambda sur=sur: sur.predict(vb, sb, batch=64)) / len(vb)
        timing[sur.arch] = {"single_ms": t1 * 1e3, "batched_ms_per_sample": tb * 1e3}
    t_veto = time_it(lambda: veto.evaluate(v0, s0, pf[0], pu[0]))

    hw = f"{platform.processor() or platform.machine()}, 1 CPU thread"
    summary = {
        "case": "A: 2D acoustic wave, 64x64, 252 steps",
        "hardware": hw,
        "fd_solver_ms": round(t_fd * 1e3, 2),
        "fno_single_ms": round(timing["fno"]["single_ms"], 2),
        "fno_batched_ms_per_sample": round(timing["fno"]["batched_ms_per_sample"], 3),
        "veto_check_ms": round(t_veto * 1e3, 2),
        "speedup_single": round(t_fd / (timing["fno"]["single_ms"] / 1e3), 2),
        "speedup_batched": round(t_fd / (timing["fno"]["batched_ms_per_sample"] / 1e3), 2),
        "speedup_note": "FNO on CPU vs FD solver on CPU at 64x64, same machine, 1 thread. "
                        "Not a production-scale or GPU number.",
        "fno_in_dist_rel_l2_median": round(float(ind.rel_l2_fno.median()), 4),
        "fno_in_dist_rel_l2_p90": round(float(ind.rel_l2_fno.quantile(0.9)), 4),
        "unet_in_dist_rel_l2_median": round(float(ind.rel_l2_unet.median()), 4),
        "in_dist_veto_rate": round(float((ind.verdict == VETO).mean()), 3),
        "in_dist_flag_rate": round(float((ind.verdict != ACCEPT).mean()), 3),
        "error_tolerance": tol,
        "veto_precision": round(v_p, 3),
        "veto_recall": round(v_r, 3),
        "flag_precision": round(f_p, 3),
        "flag_recall": round(f_r, 3),
        "n_eval": int(len(ev)),
        "n_truly_bad": int(bad.sum()),
        "accepted_rel_l2_p95": round(float(accepted_all.rel_l2_fno.quantile(0.95)), 4) if len(accepted_all) else None,
        "accepted_share_truly_bad": round(float(accepted_all.true_bad.mean()), 4) if len(accepted_all) else None,
        "unfiltered_share_truly_bad": round(float(ev.true_bad.mean()), 4),
        "reference_output_accept_rate": round(float((sanity.verdict == ACCEPT).mean()), 3),
        "per_mode": per_mode,
        "train_minutes": {"fno": fno.meta["train_minutes"], "unet": unet.meta["train_minutes"]},
    }
    (RESULTS_DIR / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))

    leaderboard = [
        {"model": "FD solver (reference)", "rel_l2_median": 0.0, "latency_ms": round(t_fd * 1e3, 2),
         "batched_ms_per_sample": round(t_fd * 1e3, 2), "params": 0, "train_minutes": 0.0,
         "in_dist_veto_rate": 0.0},
    ]
    for sur in (fno, unet):
        col = f"rel_l2_{sur.arch}"
        lb_vr = summary["in_dist_veto_rate"] if sur.arch == "fno" else None
        leaderboard.append({
            "model": {"fno": "FNO", "unet": "U-Net"}[sur.arch],
            "rel_l2_median": round(float(ind[col].median()), 4),
            "latency_ms": round(timing[sur.arch]["single_ms"], 2),
            "batched_ms_per_sample": round(timing[sur.arch]["batched_ms_per_sample"], 3),
            "params": sur.n_params(),
            "train_minutes": sur.meta["train_minutes"],
            "in_dist_veto_rate": lb_vr,
        })
    (RESULTS_DIR / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_mode"}, indent=2))
    print(pd.DataFrame(per_mode).T)


def _row(name, group, i, rep, err_f, err_u, tol):
    return {
        "set": name, "group": group, "idx": i,
        "verdict": rep.verdict, "trust_score": round(rep.trust_score, 2),
        **{f"m_{k}": v for k, v in rep.metrics.items()},
        **{f"r_{k}": v for k, v in rep.ratios.items()},
        "rel_l2_fno": err_f, "rel_l2_unet": err_u, "true_bad": bool(err_f > tol),
    }


if __name__ == "__main__":
    _ = (REVIEW,)
    main()
