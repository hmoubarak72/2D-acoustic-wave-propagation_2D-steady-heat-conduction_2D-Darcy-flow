"""2D acoustic wave solver (second order in time and space) with Mur absorbing boundaries.

Governing equation:  d2p/dt2 = v(x)^2 * lap(p) + s(t) * delta(x - x_s)

Discretization (interior cells):
    p[n+1] = 2 p[n] - p[n-1] + (v dt)^2 * lap_h(p[n]) + (v dt)^2 * w[n] * delta_s
then the outermost ring is updated with Mur's first-order absorbing condition
    p[n+1]_edge = p[n]_inner + k (p[n+1]_inner - p[n]_edge),   k = (v dt - dx) / (v dt + dx)
which lets outgoing waves leave the domain. (An optional sponge layer is supported but
off by default: at 64x64 a sponge thinner than one wavelength reflects more than it absorbs.)

The same stencil is used by the physics-veto residual, so the reference solver output
has a residual at machine precision in the interior. That is the property the
veto tests rely on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CFL_LIMIT_2D = 1.0 / np.sqrt(2.0)  # von Neumann limit for the 5-point Laplacian


def ricker(nt: int, dt: float, f0: float, t0: float) -> np.ndarray:
    t = np.arange(nt) * dt - t0
    a = (np.pi * f0 * t) ** 2
    return ((1.0 - 2.0 * a) * np.exp(-a)).astype(np.float64)


def laplacian(p: np.ndarray, dx: float) -> np.ndarray:
    """5-point Laplacian; zero on the outer ring."""
    lap = np.zeros_like(p)
    lap[..., 1:-1, 1:-1] = (
        p[..., 2:, 1:-1] + p[..., :-2, 1:-1] + p[..., 1:-1, 2:] + p[..., 1:-1, :-2]
        - 4.0 * p[..., 1:-1, 1:-1]
    ) / dx**2
    return lap


def sponge_profile(n: int, width: int, strength: float) -> np.ndarray:
    idx = np.arange(n)
    d = np.minimum(idx, n - 1 - idx)
    prof = np.where(d < width, ((width - d) / width) ** 2, 0.0)
    return strength * np.maximum(prof[:, None], prof[None, :])


def mur_coefficient(v: np.ndarray, cfg) -> np.ndarray:
    c = v.astype(np.float64) * cfg.dt
    return (c - cfg.dx) / (c + cfg.dx)


def apply_mur(p_next: np.ndarray, p_curr: np.ndarray, k: np.ndarray) -> None:
    """First-order Mur absorbing boundary on all four edges (in place)."""
    p_next[0, :] = p_curr[1, :] + k[0, :] * (p_next[1, :] - p_curr[0, :])
    p_next[-1, :] = p_curr[-2, :] + k[-1, :] * (p_next[-2, :] - p_curr[-1, :])
    p_next[:, 0] = p_curr[:, 1] + k[:, 0] * (p_next[:, 1] - p_curr[:, 0])
    p_next[:, -1] = p_curr[:, -2] + k[:, -1] * (p_next[:, -2] - p_curr[:, -1])


def cfl_number(v_max: float, dt: float, dx: float) -> float:
    return float(v_max * dt / dx)


@dataclass
class WaveConfig:
    n: int = 64
    dx: float = 10.0
    dt: float = 0.0008
    nt: int = 212
    sponge: int = 0
    sponge_strength: float = 0.0
    f0: float = 15.0
    t0: float = 0.07

    @classmethod
    def from_dict(cls, d: dict) -> WaveConfig:
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: d[k] for k in keys if k in d})


class AcousticWaveSolver:
    def __init__(self, cfg: WaveConfig):
        self.cfg = cfg
        self.damping = (sponge_profile(cfg.n, cfg.sponge, cfg.sponge_strength)
                        if cfg.sponge > 0 else np.zeros((cfg.n, cfg.n)))
        self.wavelet = ricker(cfg.nt, cfg.dt, cfg.f0, cfg.t0)

    def check_cfl(self, v: np.ndarray) -> float:
        c = cfl_number(float(v.max()), self.cfg.dt, self.cfg.dx)
        if c >= CFL_LIMIT_2D:
            raise ValueError(f"CFL violation: v_max*dt/dx = {c:.3f} >= {CFL_LIMIT_2D:.3f}")
        return c

    def solve(
        self,
        v: np.ndarray,
        source: tuple[int, int],
        save_steps: list[int] | None = None,
    ) -> np.ndarray:
        """Run the solver. Returns p at `save_steps` (shape [len, n, n]) or the full history."""
        cfg = self.cfg
        self.check_cfl(v)
        v = v.astype(np.float64)
        c2 = (v * cfg.dt) ** 2
        k_mur = mur_coefficient(v, cfg)
        keep = 1.0 - self.damping
        sz, sx = source
        p_prev = np.zeros((cfg.n, cfg.n))
        p_curr = np.zeros((cfg.n, cfg.n))
        steps = range(cfg.nt) if save_steps is None else save_steps
        want = {s: i for i, s in enumerate(steps)}
        out = np.zeros((len(want), cfg.n, cfg.n), dtype=np.float32)
        last = max(want)
        # p at step 0 is zero; step k holds the field after k updates.
        if 0 in want:
            out[want[0]] = p_curr
        for k in range(1, last + 1):
            p_next = 2.0 * p_curr - p_prev + c2 * laplacian(p_curr, cfg.dx)
            p_next[sz, sx] += c2[sz, sx] * self.wavelet[k - 1]
            p_next *= keep
            apply_mur(p_next, p_curr, k_mur)
            p_prev, p_curr = p_curr, p_next
            if k in want:
                out[want[k]] = p_curr
        return out


def anchor_steps(anchors: list[int]) -> list[int]:
    """Expand anchors into consecutive triplets (n-1, n, n+1)."""
    return [s for a in anchors for s in (a - 1, a, a + 1)]
