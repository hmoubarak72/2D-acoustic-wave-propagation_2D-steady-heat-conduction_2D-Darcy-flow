"""Physics checks run on every surrogate prediction (Case A, acoustic wave).

A prediction is an array [3K, n, n] holding, for each of K anchor steps, the channels
(p0, d1, d2) where p0 = p[n], d1 = (p[n+1] - p[n-1]) / 2 and d2 = p[n+1] - 2 p[n] + p[n-1].

None of these checks needs the reference solution. They only use the input
(velocity model, source) and the prediction itself, which is what makes them usable
in production, where the reference does not exist.
"""

from __future__ import annotations

import numpy as np

from physicsveto.solvers.wave import WaveConfig, laplacian, mur_coefficient, ricker


def split_channels(pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = pred.shape[0] // 3
    f = pred.reshape(k, 3, *pred.shape[1:])
    return f[:, 0], f[:, 1], f[:, 2]


def interior_mask(cfg: WaveConfig) -> np.ndarray:
    m = np.zeros((cfg.n, cfg.n), dtype=bool)
    w = max(cfg.sponge, 1)
    m[w:-w, w:-w] = True
    return m


def pde_residual(
    pred: np.ndarray,
    v: np.ndarray,
    src: tuple[int, int],
    anchors: list[int],
    cfg: WaveConfig,
) -> tuple[float, np.ndarray]:
    """Discrete wave-equation residual  d2 - (v dt)^2 lap(p0) - (v dt)^2 w[n] delta_s.

    Returns the relative L2 norm over the interior (all cells but the edge ring) and the
    per-cell residual map averaged over anchors, normalised by the RMS of (v dt)^2 lap(p0).
    """
    p0, _, d2 = split_channels(pred.astype(np.float64))
    c2 = (v.astype(np.float64) * cfg.dt) ** 2
    wav = ricker(cfg.nt, cfg.dt, cfg.f0, cfg.t0)
    rhs = c2[None] * laplacian(p0, cfg.dx)
    for i, a in enumerate(anchors):
        rhs[i, src[0], src[1]] += c2[src] * wav[a]
    res = d2 - rhs
    m = interior_mask(cfg)
    scale = np.sqrt(np.mean(rhs[:, m] ** 2)) + 1e-12
    rel = float(np.sqrt(np.mean(res[:, m] ** 2)) / scale)
    res_map = np.sqrt(np.mean(res**2, axis=0)) / scale
    res_map[~m] = np.nan
    return rel, res_map


def boundary_error(pred: np.ndarray, v: np.ndarray, cfg: WaveConfig) -> float:
    """Absorbing-boundary check. The edges must satisfy Mur's one-way wave condition
        p[n+1]_edge - p[n]_inner - k (p[n+1]_inner - p[n]_edge) = 0
    Returns the RMS mismatch along all four edges, relative to the RMS of the field."""
    p0, d1, d2 = split_channels(pred.astype(np.float64))
    pp = p0 + d1 + 0.5 * d2  # p[n+1]
    k = mur_coefficient(v, cfg)
    r = [
        pp[:, 0, 1:-1] - p0[:, 1, 1:-1] - k[0, 1:-1] * (pp[:, 1, 1:-1] - p0[:, 0, 1:-1]),
        pp[:, -1, 1:-1] - p0[:, -2, 1:-1] - k[-1, 1:-1] * (pp[:, -2, 1:-1] - p0[:, -1, 1:-1]),
        pp[:, 1:-1, 0] - p0[:, 1:-1, 1] - k[1:-1, 0] * (pp[:, 1:-1, 1] - p0[:, 1:-1, 0]),
        pp[:, 1:-1, -1] - p0[:, 1:-1, -2] - k[1:-1, -1] * (pp[:, 1:-1, -2] - p0[:, 1:-1, -1]),
    ]
    rms = np.sqrt(np.mean(np.concatenate([x.ravel() for x in r]) ** 2))
    return float(rms / (np.sqrt(np.mean(p0**2)) + 1e-12))


def wave_energy(pred: np.ndarray, v: np.ndarray, cfg: WaveConfig) -> np.ndarray:
    """Discrete acoustic energy in the interior at each anchor:
    E = sum( (dp/dt)^2 / v^2 + |grad p|^2 ) * dx^2   (up to a constant factor 1/2)."""
    p0, d1, _ = split_channels(pred.astype(np.float64))
    m = interior_mask(cfg)
    pt = d1 / cfg.dt
    gz = np.gradient(p0, cfg.dx, axis=1)
    gx = np.gradient(p0, cfg.dx, axis=2)
    dens = pt**2 / (v.astype(np.float64) ** 2)[None] + gz**2 + gx**2
    return (dens[:, m].sum(axis=1) * cfg.dx**2).astype(np.float64)


def energy_violation(pred: np.ndarray, v: np.ndarray, anchors: list[int], cfg: WaveConfig) -> tuple[float, np.ndarray]:
    """Conservation check. After the source has stopped, energy inside the domain can only
    leave (through the absorbing boundary). Any growth between anchors is unphysical.
    Returns the largest relative energy increase among post-source anchors, and the curve."""
    e = wave_energy(pred, v, cfg)
    t_end = cfg.t0 + 0.75 / cfg.f0  # Ricker amplitude < 0.5% after this
    post = [i for i, a in enumerate(anchors) if a * cfg.dt > t_end]
    viol = 0.0
    for i, j in zip(post[:-1], post[1:], strict=False):
        viol = max(viol, (e[j] - e[i]) / (e[i] + 1e-12))
    return float(max(viol, 0.0)), e


def ensemble_disagreement(pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """Relative L2 difference between two independent surrogates (FNO vs U-Net) on p0."""
    a, _, _ = split_channels(pred_a)
    b, _, _ = split_channels(pred_b)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    """True error on the wavefield snapshots p0 (needs the reference; used for evaluation only)."""
    a, _, _ = split_channels(pred)
    b, _, _ = split_channels(ref)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
