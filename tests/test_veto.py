import numpy as np

from physicsveto.data.fields import PLANTED_INPUTS
from physicsveto.verify import checks
from physicsveto.verify.ood import MahalanobisOOD, input_features
from physicsveto.verify.trust import ACCEPT, REVIEW, VETO, PhysicsVeto, score_metrics, sub_score


def test_reference_output_has_near_zero_residual(sample, case, cfg):
    v, src, y = sample
    rel, res_map = checks.pde_residual(y, v, src, case["anchors"], cfg)
    assert rel < 1e-4
    assert np.isnan(res_map[0, 0])  # sponge masked out


def test_noise_raises_residual(sample, case, cfg):
    v, src, y = sample
    noisy = y.copy()
    noisy[0::3] += 0.02 * np.abs(y[0::3]).max() * np.random.default_rng(0).standard_normal(y[0::3].shape)
    rel, _ = checks.pde_residual(noisy, v, src, case["anchors"], cfg)
    assert rel > 0.1


def test_boundary_error(sample, cfg):
    v, _, y = sample
    assert checks.boundary_error(y, v, cfg) < 1e-5
    bad = y.copy()
    bad[0::3, 0, :] += 0.2 * np.abs(y[0::3]).max()  # a reflecting / pinned top edge
    assert checks.boundary_error(bad, v, cfg) > 0.05


def test_energy_does_not_grow_for_reference(sample, case, cfg):
    v, _, y = sample
    viol, curve = checks.energy_violation(y, v, case["anchors"], cfg)
    assert viol < 0.02
    assert len(curve) == len(case["anchors"])


def test_energy_growth_detected(sample, case, cfg):
    v, _, y = sample
    bad = y.copy()
    k = len(case["anchors"])
    for i in range(k):
        bad[3 * i : 3 * i + 3] *= 1.5**i  # later frames get louder
    viol, _ = checks.energy_violation(bad, v, case["anchors"], cfg)
    assert viol > 0.2


def test_ensemble_disagreement_and_rel_l2(sample):
    _, _, y = sample
    assert checks.ensemble_disagreement(y, y) == 0.0
    assert abs(checks.relative_l2(1.1 * y, y) - 0.1) < 1e-6


def test_ood_distance_orders_inputs(case, calibration):
    ood = MahalanobisOOD.from_dict(calibration["ood"])
    rng = np.random.default_rng(5)
    v, s = PLANTED_INPUTS["salt_body"](case, rng)
    d_salt = ood.distance(input_features(v, s))
    assert d_salt > calibration["levels"]["ood_distance"]


def test_sub_score_shape():
    assert sub_score(0.5, 3.0) == 100.0
    assert sub_score(2.0, 3.0) == 50.0
    assert sub_score(5.0, 3.0) == 0.0


def test_verdict_logic(veto_cfg):
    calib = {k: 1.0 for k in veto_cfg["checks"]}
    ok = {k: 0.5 for k in calib}
    assert score_metrics(ok, calib, veto_cfg)[1] == ACCEPT
    hard = dict(ok, pde_residual=10.0)
    assert score_metrics(hard, calib, veto_cfg)[1] == VETO
    mid = {k: 1.4 for k in calib}  # everything a bit worse than usual, nothing hard-failed
    assert score_metrics(mid, calib, veto_cfg)[1] == REVIEW


def test_exact_solver_output_is_accepted(sample, case, veto_cfg, calibration):
    v, src, y = sample
    rep = PhysicsVeto(case, veto_cfg, calibration).evaluate(v, src, y, y)
    assert rep.verdict == ACCEPT and rep.trust_score > 95


def test_planted_bad_prediction_is_vetoed(sample, case, veto_cfg, calibration):
    v, src, y = sample
    n = y.shape[-1]
    bad = y.copy()
    cb = np.indices((n, n)).sum(axis=0) % 2 * 2.0 - 1.0
    bad[0::3] += 0.05 * np.abs(y[0::3]).max() * cb
    rep = PhysicsVeto(case, veto_cfg, calibration).evaluate(v, src, bad, y)
    assert rep.verdict == VETO
    assert any("PDE residual" in r for r in rep.reasons)


def test_ood_input_is_vetoed_even_if_prediction_is_exact(case, solver, veto_cfg, calibration):
    from physicsveto.data.fields import solve_targets

    rng = np.random.default_rng(9)
    v, src = PLANTED_INPUTS["moved_source"](case, rng)
    y = solve_targets(solver, v, src, case["anchors"])
    rep = PhysicsVeto(case, veto_cfg, calibration).evaluate(v, src, y, y)
    assert rep.verdict in (REVIEW, VETO)
