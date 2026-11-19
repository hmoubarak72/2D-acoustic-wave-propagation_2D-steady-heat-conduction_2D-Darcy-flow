import numpy as np
import pytest

from physicsveto.solvers.wave import (
    CFL_LIMIT_2D,
    AcousticWaveSolver,
    WaveConfig,
    anchor_steps,
    cfl_number,
    laplacian,
    ricker,
)


def test_ricker_peaks_at_delay():
    dt, t0 = 0.001, 0.07
    w = ricker(200, dt, 15.0, t0)
    assert np.argmax(w) == round(t0 / dt)
    assert w.max() == pytest.approx(1.0, abs=1e-6)


def test_laplacian_exact_for_quadratic():
    n, dx = 32, 0.5
    z, x = np.mgrid[0:n, 0:n] * dx
    p = 3 * x**2 + 2 * z**2
    lap = laplacian(p, dx)
    assert np.allclose(lap[1:-1, 1:-1], 10.0)
    assert np.all(lap[0] == 0) and np.all(lap[:, -1] == 0)


def test_laplacian_second_order_convergence():
    """Manufactured solution sin(kx)sin(kz): error must fall ~4x per grid halving."""
    errs = []
    for n in (32, 64, 128):
        h = np.pi / (n - 1)
        z, x = np.mgrid[0:n, 0:n] * h
        p = np.sin(2 * x) * np.sin(3 * z)
        exact = -(4 + 9) * p
        errs.append(np.abs(laplacian(p, h) - exact)[1:-1, 1:-1].max())
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(rates > 1.8)


def test_cfl_violation_raises(cfg):
    s = AcousticWaveSolver(cfg)
    v_bad = np.full((cfg.n, cfg.n), 1.01 * CFL_LIMIT_2D * cfg.dx / cfg.dt)
    with pytest.raises(ValueError, match="CFL"):
        s.solve(v_bad, (16, 32))


def test_cfl_margin_for_training_and_planted_velocities(cfg):
    assert cfl_number(5200.0, cfg.dt, cfg.dx) < CFL_LIMIT_2D


def test_stable_and_bounded(solver, cfg):
    v = np.full((cfg.n, cfg.n), 3500.0, dtype=np.float32)
    h = solver.solve(v, (32, 32))
    assert np.all(np.isfinite(h))
    assert np.abs(h).max() < 1e3


def test_absorbing_edges_satisfy_mur_condition(solver, sample, cfg):
    from physicsveto.verify.checks import boundary_error

    _, _, y = sample
    v = sample[0]
    assert boundary_error(y, v, cfg) < 1e-5


def test_save_steps_match_full_history(solver, sample, case):
    v, src, _ = sample
    full = solver.solve(v, src)
    steps = anchor_steps(case["anchors"])
    part = solver.solve(v, src, save_steps=steps)
    assert np.allclose(full[steps], part)


def test_symmetry_homogeneous_medium(solver, cfg):
    v = np.full((cfg.n, cfg.n), 2500.0, dtype=np.float32)
    c = cfg.n // 2
    h = solver.solve(v, (c, c), save_steps=[100])[0]
    # source at (c, c) on an even grid: mirror about column c
    assert np.allclose(h[:, c - 5], h[:, c + 5], atol=1e-6 * np.abs(h).max())


def test_wavefront_travels_at_velocity():
    cfg = WaveConfig(n=128, sponge=12)
    s = AcousticWaveSolver(cfg)
    vel = 2500.0
    v = np.full((cfg.n, cfg.n), vel, dtype=np.float32)
    c = cfg.n // 2
    step = 170
    h = s.solve(v, (c, c), save_steps=[step])[0]
    radius_cells = np.argmax(np.abs(h[c, c:])) * cfg.dx
    expected = vel * (step * cfg.dt - cfg.t0)
    assert abs(radius_cells - expected) < 4 * cfg.dx


def test_waves_leave_through_absorbing_edges(cfg):
    """With Mur edges the domain must empty out: almost no energy is reflected back."""
    long = AcousticWaveSolver(WaveConfig(**{**cfg.__dict__, "nt": 800}))
    v = np.full((cfg.n, cfg.n), 3000.0, dtype=np.float32)
    h = long.solve(v, (32, 32), save_steps=[120, 799])
    assert np.sum(h[1] ** 2) < 0.02 * np.sum(h[0] ** 2)
