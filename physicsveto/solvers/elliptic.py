"""Steady variable-coefficient elliptic solver for Cases B and C.

    -div( k(x) grad u ) = f   on the unit square,   u = 0 on the boundary (Dirichlet)

Case B (heat conduction): u = temperature rise, k = conductivity, f = heat sources.
Case C (Darcy flow):      u = pressure, k = permeability, f = injection / source term.

Discretization: cell-centred-style 5-point finite volumes on an n x n node grid (boundary nodes
included), face conductivities by harmonic averaging, sparse direct solve (SciPy).
The physics veto evaluates the same discrete operator, so the solver output has a residual at
round-off level, and the boundary-flux balance closes exactly.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def face_k(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic-mean conductivities on vertical faces (between rows) and horizontal faces (between cols)."""
    kz = 2.0 * k[..., 1:, :] * k[..., :-1, :] / (k[..., 1:, :] + k[..., :-1, :])
    kx = 2.0 * k[..., :, 1:] * k[..., :, :-1] / (k[..., :, 1:] + k[..., :, :-1])
    return kz, kx


def apply_operator(u: np.ndarray, k: np.ndarray, h: float) -> np.ndarray:
    """A(u) = -div(k grad u) at interior nodes (0 on the boundary ring). Works on [..., n, n]."""
    kz, kx = face_k(k)
    fz = kz * (u[..., 1:, :] - u[..., :-1, :]) / h  # flux between rows i and i+1 (sign: +grad)
    fx = kx * (u[..., :, 1:] - u[..., :, :-1]) / h
    out = np.zeros_like(u)
    out[..., 1:-1, 1:-1] = -(
        (fz[..., 1:, 1:-1] - fz[..., :-1, 1:-1]) + (fx[..., 1:-1, 1:] - fx[..., 1:-1, :-1])
    ) / h
    return out


def boundary_outflux(u: np.ndarray, k: np.ndarray, h: float) -> float:
    """Total flux leaving through the boundary faces: sum over faces of k_face * (u_inner - u_boundary).

    (flux density k * du/h times face length h). For the discrete solution this equals
    h^2 * sum(f) over interior nodes exactly: the discrete energy / mass balance."""
    kz, kx = face_k(k)
    top = kz[0, 1:-1] * (u[1, 1:-1] - u[0, 1:-1])
    bot = kz[-1, 1:-1] * (u[-2, 1:-1] - u[-1, 1:-1])
    left = kx[1:-1, 0] * (u[1:-1, 1] - u[1:-1, 0])
    right = kx[1:-1, -1] * (u[1:-1, -2] - u[1:-1, -1])
    return float(top.sum() + bot.sum() + left.sum() + right.sum())


class EllipticSolver:
    def __init__(self, n: int = 64):
        self.n = n
        self.h = 1.0 / (n - 1)
        m = n - 2
        self._idx = np.arange(m * m).reshape(m, m)

    def assemble(self, k: np.ndarray) -> sp.csr_matrix:
        n, h = self.n, self.h
        kz, kx = face_k(k.astype(np.float64))
        full = -np.ones((n, n), dtype=np.int64)
        full[1:-1, 1:-1] = self._idx
        ii, jj = np.mgrid[1 : n - 1, 1 : n - 1]
        me = full[ii, jj].ravel()
        faces = [
            (kz[ii - 1, jj], ii - 1, jj),  # up
            (kz[ii, jj], ii + 1, jj),      # down
            (kx[ii, jj - 1], ii, jj - 1),  # left
            (kx[ii, jj], ii, jj + 1),      # right
        ]
        diag = sum(c for c, _, _ in faces).ravel() / h**2
        rows, cols, vals = [me], [me], [diag]
        for c, ni, nj in faces:
            nb = full[ni, nj].ravel()
            keep = nb >= 0  # boundary neighbours are u = 0 and drop out
            rows.append(me[keep])
            cols.append(nb[keep])
            vals.append(-c.ravel()[keep] / h**2)
        m = (n - 2) ** 2
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(m, m))

    def solve(self, k: np.ndarray, f: np.ndarray) -> np.ndarray:
        if np.any(k <= 0):
            raise ValueError("conductivity / permeability must be positive")
        a = self.assemble(k)
        u_in = spla.spsolve(a.tocsc(), f[1:-1, 1:-1].astype(np.float64).ravel())
        u = np.zeros((self.n, self.n))
        u[1:-1, 1:-1] = u_in.reshape(self.n - 2, self.n - 2)
        return u.astype(np.float32)
