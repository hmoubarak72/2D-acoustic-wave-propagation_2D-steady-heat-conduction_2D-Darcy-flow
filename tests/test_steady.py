from types import SimpleNamespace

import numpy as np
import pytest
import torch

from physicsveto.config import load_case, load_veto
from physicsveto.data.steady import IN_DIST, PLANTED, build_steady_set
from physicsveto.gtm import llm
from physicsveto.models.steady import torch_operator
from physicsveto.solvers.elliptic import EllipticSolver, apply_operator, boundary_outflux
from physicsveto.verify.ood import MahalanobisOOD
from physicsveto.verify.steady import STEADY_FEATURES, SteadyVeto, steady_features, steady_metrics
from physicsveto.verify.trust import ACCEPT, VETO


def _poisson_error(n):
    z, x = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    exact = np.sin(np.pi * z) * np.sin(np.pi * x)
    f = 2 * np.pi**2 * exact
    u = EllipticSolver(n).solve(np.ones((n, n)), f)
    return np.abs(u - exact).max()


def test_elliptic_second_order():
    e1, e2 = _poisson_error(17), _poisson_error(33)
    assert 3.3 < e1 / e2 < 4.7


@pytest.fixture(scope="module", params=["case_b", "case_c"])
def steady(request):
    key = request.param
    case = load_case(key)
    k, f, u = build_steady_set(key, case, 4, np.random.default_rng(0), IN_DIST[key])
    return key, case, k, f, u


def test_reference_residual_and_balance(steady):
    _, case, k, f, u = steady
    n = case["n"]
    h = 1.0 / (n - 1)
    m, res_map = steady_metrics(k[0], f[0], u[0])
    assert m["pde_residual"] < 1e-4 and m["boundary_error"] == 0.0 and m["conservation"] < 1e-4
    assert np.isnan(res_map[0, 0])
    inj = h**2 * f[0][1:-1, 1:-1].astype(np.float64).sum()
    assert abs(boundary_outflux(u[0].astype(np.float64), k[0], h) - inj) / inj < 1e-4


def test_torch_operator_matches_numpy(steady):
    _, case, k, _, u = steady
    h = 1.0 / (case["n"] - 1)
    a = apply_operator(u[:2].astype(np.float64), k[:2].astype(np.float64), h)[:, 1:-1, 1:-1]
    b = torch_operator(torch.from_numpy(u[:2].astype(np.float64)), torch.from_numpy(k[:2].astype(np.float64)), h)
    assert np.allclose(a, b.numpy(), rtol=1e-8, atol=1e-8)


def test_features_and_planted_modes(steady):
    key, case, k, f, _ = steady
    assert steady_features(key, k[0], f[0]).shape == (len(STEADY_FEATURES[key]),)
    assert len(PLANTED[key]) == 4


def test_veto_accepts_reference_and_vetoes_corruption(steady):
    key, case, k, f, u = steady
    cfg = load_veto(key)
    feats = np.stack([steady_features(key, a, b) for a, b in zip(k, f, strict=True)])
    feats = np.concatenate([feats, feats + 1e-3 * np.random.default_rng(1).standard_normal(feats.shape)] * 4)
    calib = {"levels": {"pde_residual": 0.05, "boundary_error": 0.01, "conservation": 0.02,
                        "ood_distance": 50.0, "ensemble_disagreement": 0.05},
             "ood": MahalanobisOOD.fit(feats).to_dict()}
    veto = SteadyVeto(key, cfg, calib)
    assert veto.evaluate(k[0], f[0], u[0], u[0]).verdict == ACCEPT
    n = u.shape[-1]
    cb = (np.indices((n, n)).sum(axis=0) % 2 * 2.0 - 1.0) * 0.1 * np.abs(u[0]).max()
    assert veto.evaluate(k[0], f[0], u[0] + cb, u[0]).verdict == VETO


def test_llm_off_in_mock_mode(monkeypatch):
    monkeypatch.setenv("PHYSICSVETO_MOCK", "1")
    ok, why = llm.available()
    assert not ok and "Mock" in why


def test_llm_generate_with_fake_client():
    pytest.importorskip("anthropic")

    class FakeMessages:
        def create(self, **kw):
            self.kw = kw
            return SimpleNamespace(stop_reason="end_turn", model="fake-model",
                                   content=[SimpleNamespace(type="text", text="# Draft")])

    fake = SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages()))
    r = llm.generate("solution_brief", "Electronics & thermal", "Thermal lead", "template", {"x": 1}, client=fake)
    assert r.text == "# Draft" and r.model == "fake-model"
    kw = fake.beta.messages.kw
    assert kw["fallbacks"] == "default" and "MEASURED_RESULTS" in kw["messages"][0]["content"]


def test_llm_refusal_raises():
    pytest.importorskip("anthropic")
    msgs = SimpleNamespace(create=lambda **kw: SimpleNamespace(stop_reason="refusal", content=[], model="m"))
    with pytest.raises(RuntimeError):
        llm.generate("talk_track", "v", "a", "t", {}, client=SimpleNamespace(beta=SimpleNamespace(messages=msgs)))
