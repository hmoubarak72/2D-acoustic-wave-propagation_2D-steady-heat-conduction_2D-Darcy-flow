"""Input generators for the steady cases B (heat) and C (Darcy), including planted failure modes."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from physicsveto.data.fields import grf
from physicsveto.solvers.elliptic import EllipticSolver


def gaussian_blob(n: int, cz: float, cx: float, w: float) -> np.ndarray:
    zz, xx = np.mgrid[0:n, 0:n]
    return np.exp(-((zz - cz) ** 2 + (xx - cx) ** 2) / (2 * w**2))


# ------------------------------- Case B: heat ----------------------------------------

def heat_conductivity(n, rng, logk_range, sigma):
    z = grf(n, sigma, rng)
    z = 2 * (z - z.min()) / (z.max() - z.min() + 1e-12) - 1
    return np.exp(logk_range * z).astype(np.float32)


def heat_sources(n, rng, n_src, width_range, pos_range, power_range):
    q = np.zeros((n, n))
    for _ in range(n_src):
        q += rng.uniform(*power_range) * gaussian_blob(
            n, rng.uniform(*pos_range), rng.uniform(*pos_range), rng.uniform(*width_range)
        )
    q[0, :] = q[-1, :] = q[:, 0] = q[:, -1] = 0.0
    return q.astype(np.float32)


def heat_in_distribution(case, rng):
    n = case["n"]
    k = heat_conductivity(n, rng, case["logk_range"], rng.uniform(*case["corr_len_range"]))
    q = heat_sources(n, rng, int(rng.integers(case["n_sources_range"][0], case["n_sources_range"][1] + 1)),
                     case["source_width_range"], case["source_pos_range"], case["source_power_range"])
    return k, q


def heat_edge_hotspot(case, rng):
    """A heat source right next to the cold boundary (training keeps sources in the middle)."""
    k, _ = heat_in_distribution(case, rng)
    n = case["n"]
    q = heat_sources(n, rng, 1, case["source_width_range"], (3, 8), case["source_power_range"])
    if rng.random() < 0.5:
        q = q[::-1].copy()
    return k, q


def heat_extreme_contrast(case, rng):
    """Conductivity spanning e^-3 .. e^3 (e.g. copper next to an air gap)."""
    n = case["n"]
    k = heat_conductivity(n, rng, 3.0, rng.uniform(*case["corr_len_range"]))
    _, q = heat_in_distribution(case, rng)
    return k, q


def heat_many_sources(case, rng):
    """Six to nine small hot chips, far more than the 1-3 seen in training."""
    n = case["n"]
    k, _ = heat_in_distribution(case, rng)
    q = heat_sources(n, rng, int(rng.integers(6, 10)), (1.0, 2.0), case["source_pos_range"], case["source_power_range"])
    return k, q


def heat_overpowered(case, rng):
    """Same layout but 3-5x the training power: a pure extrapolation in amplitude."""
    k, q = heat_in_distribution(case, rng)
    return k, (q * rng.uniform(3.0, 5.0)).astype(np.float32)


PLANTED_B = {
    "edge_hotspot": heat_edge_hotspot,
    "extreme_contrast": heat_extreme_contrast,
    "many_sources": heat_many_sources,
    "overpowered": heat_overpowered,
}


# ------------------------------- Case C: Darcy ---------------------------------------

def two_phase_k(n, rng, k_low, k_high, sigma, threshold, aniso=1.0):
    z = gaussian_filter(rng.standard_normal((n, n)), sigma=(sigma, sigma * aniso), mode="reflect")
    z = (z - z.mean()) / (z.std() + 1e-12)
    return np.where(z > threshold, k_high, k_low).astype(np.float32)


def darcy_in_distribution(case, rng):
    n = case["n"]
    k = two_phase_k(n, rng, case["k_low"], case["k_high"], rng.uniform(*case["corr_len_range"]),
                    rng.uniform(*case["threshold_range"]))
    f = np.full((n, n), case["forcing"], dtype=np.float32)
    return k, f


def darcy_high_contrast(case, rng):
    """Permeability contrast 1:100 instead of 3:12 (e.g. shale streaks in a sand)."""
    n = case["n"]
    k = two_phase_k(n, rng, 0.5, 50.0, rng.uniform(*case["corr_len_range"]), rng.uniform(*case["threshold_range"]))
    return k, np.full((n, n), case["forcing"], dtype=np.float32)


def darcy_channels(case, rng):
    """Long, thin, anisotropic channels (fluvial geology), unseen in training."""
    n = case["n"]
    k = two_phase_k(n, rng, case["k_low"], case["k_high"], rng.uniform(1.0, 1.5), rng.uniform(0.3, 0.7),
                    aniso=rng.uniform(6.0, 10.0))
    if rng.random() < 0.5:
        k = k.T.copy()
    return k, np.full((n, n), case["forcing"], dtype=np.float32)


def darcy_well_injection(case, rng):
    """A concentrated injection well instead of uniform forcing. The surrogate cannot see f:
    only the physics checks (and the OOD features of f) can catch this."""
    k, _ = darcy_in_distribution(case, rng)
    n = case["n"]
    f = 0.3 * case["forcing"] + 60.0 * gaussian_blob(n, rng.uniform(16, 48), rng.uniform(16, 48), 2.0)
    f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = 0.0
    return k, f.astype(np.float32)


def darcy_fine_scale(case, rng):
    """Very fine-scale heterogeneity (correlation length ~1 cell)."""
    n = case["n"]
    k = two_phase_k(n, rng, case["k_low"], case["k_high"], rng.uniform(0.6, 1.2), rng.uniform(*case["threshold_range"]))
    return k, np.full((n, n), case["forcing"], dtype=np.float32)


PLANTED_C = {
    "high_contrast": darcy_high_contrast,
    "channels": darcy_channels,
    "well_injection": darcy_well_injection,
    "fine_scale": darcy_fine_scale,
}

IN_DIST = {"case_b": heat_in_distribution, "case_c": darcy_in_distribution}
PLANTED = {"case_b": PLANTED_B, "case_c": PLANTED_C}


def build_steady_set(case_key, case, n_samples, rng, maker=None):
    solver = EllipticSolver(case["n"])
    ks, fs, us = [], [], []
    for _ in range(n_samples):
        k, f = (maker or IN_DIST[case_key])(case, rng)
        ks.append(k)
        fs.append(f)
        us.append(solver.solve(k, f))
    return np.stack(ks), np.stack(fs), np.stack(us)
