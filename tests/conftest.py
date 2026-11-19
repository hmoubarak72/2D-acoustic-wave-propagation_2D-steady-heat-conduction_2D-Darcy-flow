import numpy as np
import pytest

from physicsveto.config import load_case, load_veto
from physicsveto.data.fields import sample_in_distribution, solve_targets
from physicsveto.solvers.wave import AcousticWaveSolver, WaveConfig
from physicsveto.verify.ood import MahalanobisOOD, input_features


@pytest.fixture(scope="session")
def case():
    return load_case("case_a")


@pytest.fixture(scope="session")
def cfg(case):
    return WaveConfig.from_dict(case)


@pytest.fixture(scope="session")
def solver(cfg):
    return AcousticWaveSolver(cfg)


@pytest.fixture(scope="session")
def sample(case, solver):
    rng = np.random.default_rng(42)
    v, src = sample_in_distribution(case, rng)
    y = solve_targets(solver, v, src, case["anchors"])
    return v, src, y


@pytest.fixture(scope="session")
def calibration(case):
    """Small synthetic calibration so veto tests do not depend on trained weights."""
    rng = np.random.default_rng(0)
    feats = np.stack([input_features(*sample_in_distribution(case, rng)) for _ in range(200)])
    return {
        "levels": {
            "pde_residual": 0.05,
            "boundary_error": 0.01,
            "energy_violation": 0.02,
            "ood_distance": float(np.percentile(
                [MahalanobisOOD.fit(feats).distance(f) for f in feats], 99)),
            "ensemble_disagreement": 0.1,
        },
        "ood": MahalanobisOOD.fit(feats).to_dict(),
    }


@pytest.fixture(scope="session")
def veto_cfg():
    return load_veto()
