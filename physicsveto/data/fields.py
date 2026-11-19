"""Random input generators for Case A: velocity models (Gaussian random fields) and sources.

In-distribution samples follow `config/cases.yaml`. The `planted_*` generators build the
failure cases used to validate the physics veto (they are never used for training).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from physicsveto.solvers.wave import AcousticWaveSolver, WaveConfig, anchor_steps


def grf(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    z = gaussian_filter(rng.standard_normal((n, n)), sigma=sigma, mode="reflect")
    return (z - z.mean()) / (z.std() + 1e-12)


def velocity_model(
    n: int,
    rng: np.random.Generator,
    v_lo: float,
    v_hi: float,
    sigma: float,
) -> np.ndarray:
    z = grf(n, sigma, rng)
    z = (z - z.min()) / (z.max() - z.min() + 1e-12)
    return (v_lo + (v_hi - v_lo) * z).astype(np.float32)


def sample_in_distribution(case: dict, rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, int]]:
    n = case["n"]
    span = case["v_max"] - case["v_min"]
    v_lo = case["v_min"] + rng.uniform(0.0, 0.25) * span
    v_hi = case["v_max"] - rng.uniform(0.0, 0.25) * span
    sigma = rng.uniform(*case["corr_len_range"])
    v = velocity_model(n, rng, v_lo, v_hi, sigma)
    sx = int(rng.integers(case["source_x_range"][0], case["source_x_range"][1] + 1))
    return v, (int(case["source_depth"]), sx)


# ---- planted failure modes (inputs) -------------------------------------------------

def planted_salt_body(case: dict, rng: np.random.Generator):
    """A high-velocity salt body (4,500-5,200 m/s) inside an in-distribution background."""
    v, src = sample_in_distribution(case, rng)
    n = case["n"]
    zz, xx = np.mgrid[0:n, 0:n]
    cz, cx = rng.integers(28, 44), rng.integers(18, 46)
    a, b = rng.uniform(5, 10), rng.uniform(8, 16)
    mask = ((zz - cz) / a) ** 2 + ((xx - cx) / b) ** 2 <= 1.0
    v = v.copy()
    v[mask] = rng.uniform(4500.0, 5200.0)
    return v.astype(np.float32), src


def planted_high_contrast(case: dict, rng: np.random.Generator):
    """Velocity range far wider than training (1,500-5,200 m/s)."""
    n = case["n"]
    v = velocity_model(n, rng, 1500.0, 5200.0, rng.uniform(*case["corr_len_range"]))
    sx = int(rng.integers(case["source_x_range"][0], case["source_x_range"][1] + 1))
    return v, (int(case["source_depth"]), sx)


def planted_rough(case: dict, rng: np.random.Generator):
    """Much shorter correlation length than training (rough, small-scale geology)."""
    n = case["n"]
    v = velocity_model(n, rng, case["v_min"], case["v_max"], rng.uniform(1.0, 2.0))
    sx = int(rng.integers(case["source_x_range"][0], case["source_x_range"][1] + 1))
    return v, (int(case["source_depth"]), sx)


def planted_edge_source(case: dict, rng: np.random.Generator):
    """Source moved toward the edges / deeper than any training shot."""
    v, _ = sample_in_distribution(case, rng)
    n = case["n"]
    if rng.random() < 0.5:
        sx = int(rng.choice([13, 14, n - 15, n - 14]))
        sz = int(case["source_depth"])
    else:
        sx = int(rng.integers(20, 44))
        sz = int(rng.integers(32, 46))
    return v, (sz, sx)


def planted_mild_shift(case: dict, rng: np.random.Generator):
    """A subtle shift: velocities up to ~15% faster and slightly shorter correlation length than training."""
    n = case["n"]
    v_hi = rng.uniform(case["v_max"], case["v_max"] * 1.15)
    v = velocity_model(n, rng, case["v_min"], v_hi, rng.uniform(3.0, 5.0))
    sx = int(rng.integers(case["source_x_range"][0] - 3, case["source_x_range"][1] + 4))
    return v, (int(case["source_depth"]), sx)


PLANTED_INPUTS = {
    "salt_body": planted_salt_body,
    "high_contrast": planted_high_contrast,
    "rough_geology": planted_rough,
    "moved_source": planted_edge_source,
    "mild_shift": planted_mild_shift,
}


# ---- target encoding ------------------------------------------------------------------

def encode_triplets(frames: np.ndarray) -> np.ndarray:
    """frames: [3K, n, n] laid out as (n-1, n, n+1) per anchor -> [3K] as (p0, d1, d2)."""
    k = frames.shape[0] // 3
    f = frames.reshape(k, 3, *frames.shape[1:])
    pm, p0, pp = f[:, 0], f[:, 1], f[:, 2]
    d1 = 0.5 * (pp - pm)
    d2 = pp - 2.0 * p0 + pm
    return np.stack([p0, d1, d2], axis=1).reshape(3 * k, *frames.shape[1:]).astype(np.float32)


def source_map(n: int, src: tuple[int, int], width: float = 1.5) -> np.ndarray:
    zz, xx = np.mgrid[0:n, 0:n]
    return np.exp(-((zz - src[0]) ** 2 + (xx - src[1]) ** 2) / (2 * width**2)).astype(np.float32)


def solve_targets(solver: AcousticWaveSolver, v: np.ndarray, src, anchors: list[int]) -> np.ndarray:
    frames = solver.solve(v, src, save_steps=anchor_steps(anchors))
    return encode_triplets(frames)


def build_set(case: dict, n_samples: int, rng: np.random.Generator, maker=None):
    cfg = WaveConfig.from_dict(case)
    solver = AcousticWaveSolver(cfg)
    vs, srcs, ys = [], [], []
    for _ in range(n_samples):
        v, src = (maker or sample_in_distribution)(case, rng)
        vs.append(v)
        srcs.append(src)
        ys.append(solve_targets(solver, v, src, case["anchors"]))
    return np.stack(vs), np.array(srcs, dtype=np.int16), np.stack(ys)
