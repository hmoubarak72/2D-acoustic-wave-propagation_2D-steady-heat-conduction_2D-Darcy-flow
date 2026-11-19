"""Physics veto for the steady cases B (heat) and C (Darcy). Same scoring as Case A (trust.score_metrics)."""

from __future__ import annotations

import numpy as np

from physicsveto.solvers.elliptic import apply_operator, boundary_outflux
from physicsveto.verify.ood import MahalanobisOOD
from physicsveto.verify.trust import VetoReport, score_metrics

STEADY_FEATURES = {
    "case_b": ["logk_mean", "logk_std", "logk_min", "logk_max", "logk_roughness", "logk_max_grad",
               "q_total", "q_peak", "q_edge_share", "q_spread"],
    "case_c": ["k_mean", "k_std", "k_min", "k_max", "k_roughness", "k_anisotropy",
               "f_mean", "f_std", "f_max"],
}


def _rough(a: np.ndarray) -> tuple[float, float, float]:
    gz, gx = np.gradient(a.astype(np.float64))
    g = np.hypot(gz, gx)
    return g.mean() / (a.std() + 1e-9), g.max(), (np.mean(gz**2) + 1e-12) / (np.mean(gx**2) + 1e-12)


def steady_features(case_key: str, k: np.ndarray, f: np.ndarray) -> np.ndarray:
    if case_key == "case_b":
        lk = np.log(k.astype(np.float64))
        rough, gmax, _ = _rough(lk)
        n = k.shape[-1]
        edge = np.zeros_like(f, dtype=bool)
        edge[:10, :] = edge[-10:, :] = edge[:, :10] = edge[:, -10:] = True
        tot = float(f.sum()) + 1e-9
        zz, xx = np.mgrid[0:n, 0:n]
        cz, cx = (f * zz).sum() / tot, (f * xx).sum() / tot
        spread = np.sqrt((f * ((zz - cz) ** 2 + (xx - cx) ** 2)).sum() / tot)
        return np.array([lk.mean(), lk.std(), lk.min(), lk.max(), rough, gmax,
                         tot, f.max(), f[edge].sum() / tot, spread])
    rough, _, aniso = _rough(k)
    return np.array([k.mean(), k.std(), k.min(), k.max(), rough, np.log(aniso), f.mean(), f.std(), f.max()])


def steady_metrics(k, f, u) -> tuple[dict, np.ndarray]:
    n = k.shape[-1]
    h = 1.0 / (n - 1)
    u64, k64, f64 = u.astype(np.float64), k.astype(np.float64), f.astype(np.float64)
    res = apply_operator(u64, k64, h) - f64
    res[0, :] = res[-1, :] = res[:, 0] = res[:, -1] = 0.0
    f_rms = np.sqrt(np.mean(f64[1:-1, 1:-1] ** 2)) + 1e-12
    pde = float(np.sqrt(np.mean(res[1:-1, 1:-1] ** 2)) / f_rms)
    res_map = np.abs(res) / f_rms
    res_map[0, :] = res_map[-1, :] = res_map[:, 0] = res_map[:, -1] = np.nan
    ring = np.concatenate([u64[0], u64[-1], u64[:, 0], u64[:, -1]])
    bnd = float(np.abs(ring).max() / (np.abs(u64).max() + 1e-12))
    injected = h**2 * f64[1:-1, 1:-1].sum()
    cons = float(abs(boundary_outflux(u64, k64, h) - injected) / (abs(injected) + 1e-12))
    return {"pde_residual": pde, "boundary_error": bnd, "conservation": cons}, res_map


class SteadyVeto:
    def __init__(self, case_key: str, veto_cfg: dict, calibration: dict):
        self.case_key = case_key
        self.veto_cfg = veto_cfg
        self.calib = calibration["levels"]
        self.ood = MahalanobisOOD.from_dict(calibration["ood"])

    def raw_metrics(self, k, f, u, u_alt=None):
        m, res_map = steady_metrics(k, f, u)
        m["ood_distance"] = self.ood.distance(steady_features(self.case_key, k, f))
        if u_alt is not None:
            m["ensemble_disagreement"] = float(np.linalg.norm(u - u_alt) / (np.linalg.norm(u) + 1e-12))
        return m, res_map

    def evaluate(self, k, f, u, u_alt=None) -> VetoReport:
        m, res_map = self.raw_metrics(k, f, u, u_alt)
        score, verdict, ratios, subs, reasons = score_metrics(m, self.calib, self.veto_cfg)
        return VetoReport(score, verdict, m, ratios, subs, reasons, res_map, None)
