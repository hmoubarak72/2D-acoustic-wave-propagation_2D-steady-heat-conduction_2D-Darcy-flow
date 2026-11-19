from dataclasses import replace

import numpy as np
import pytest
import torch

from physicsveto.models.fno import FNO2d
from physicsveto.models.surrogate import ARCH_KWARGS
from physicsveto.models.unet import UNet2d
from physicsveto.roi.cost import RoiInputs, compute_roi, tornado


def test_roi_baseline_cost():
    p = RoiInputs(sims_per_cycle=100, cycles_per_year=10, core_hours_per_run=10, usd_per_core_hour=1.0)
    r = compute_roi(p)
    assert r["baseline_compute_usd"] == pytest.approx(100 * 10 * 10)


def test_vetoed_runs_cost_a_full_solve():
    p = RoiInputs(veto_rate=0.2, surrogate_speedup=1e12)
    r = compute_roi(p)
    assert r["hybrid_compute_usd"] == pytest.approx(0.2 * r["baseline_compute_usd"], rel=1e-6)


def test_full_veto_rate_means_no_compute_savings():
    r = compute_roi(RoiInputs(veto_rate=1.0))
    assert r["compute_savings_usd"] < 0


def test_effective_speedup_capped_by_veto_rate():
    r = compute_roi(RoiInputs(veto_rate=0.1, surrogate_speedup=1000))
    assert r["effective_speedup"] < 10.0
    assert r["effective_speedup"] == pytest.approx(1 / (0.1 + 0.001))


def test_break_even_positive_and_monotonic():
    a = compute_roi(RoiInputs(veto_rate=0.05))["break_even_months"]
    b = compute_roi(RoiInputs(veto_rate=0.4))["break_even_months"]
    assert 0 < a < b


def test_tornado_sorted_by_swing():
    rows = tornado(RoiInputs())
    swings = [r["swing"] for r in rows]
    assert swings == sorted(swings, reverse=True)
    assert replace(RoiInputs(), veto_rate=0.3).veto_rate == 0.3


@pytest.mark.parametrize("arch", [FNO2d, UNet2d])
def test_model_shapes(arch):
    kw = ARCH_KWARGS["fno" if arch is FNO2d else "unet"]
    m = arch(**kw)
    x = torch.randn(2, 2, 64, 64)
    assert m(x).shape == (2, 24, 64, 64)
    assert np.isfinite(m(x).detach().numpy()).all()
