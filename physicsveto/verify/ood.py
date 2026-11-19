"""Out-of-distribution detector on interpretable input features (Mahalanobis distance)."""

from __future__ import annotations

import numpy as np

FEATURES = [
    "v_mean",
    "v_std",
    "v_min",
    "v_max",
    "roughness",       # mean |grad v| / std(v): high for short correlation length
    "max_gradient",    # largest velocity jump between neighbouring cells
    "src_depth",
    "src_x",
]


def input_features(v: np.ndarray, src: tuple[int, int]) -> np.ndarray:
    v = v.astype(np.float64)
    gz, gx = np.gradient(v)
    g = np.hypot(gz, gx)
    return np.array(
        [
            v.mean(),
            v.std(),
            v.min(),
            v.max(),
            g.mean() / (v.std() + 1e-9),
            g.max(),
            float(src[0]),
            float(src[1]),
        ]
    )


class MahalanobisOOD:
    def __init__(self, mean: np.ndarray, cov_inv: np.ndarray):
        self.mean = np.asarray(mean)
        self.cov_inv = np.asarray(cov_inv)

    @classmethod
    def fit(cls, feats: np.ndarray, ridge: float = 1e-3) -> MahalanobisOOD:
        mu = feats.mean(axis=0)
        x = feats - mu
        cov = x.T @ x / max(len(x) - 1, 1)
        # ridge relative to each feature's variance keeps constant features (e.g. a fixed
        # source depth) from making the covariance singular
        cov += np.diag(ridge * np.diag(cov) + 1e-6)
        return cls(mu, np.linalg.inv(cov))

    def distance(self, f: np.ndarray) -> float:
        d = f - self.mean
        return float(np.sqrt(max(d @ self.cov_inv @ d, 0.0)))

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "cov_inv": self.cov_inv.tolist(), "features": FEATURES}

    @classmethod
    def from_dict(cls, d: dict) -> MahalanobisOOD:
        return cls(np.array(d["mean"]), np.array(d["cov_inv"]))
