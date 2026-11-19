"""Combine physics checks into a Trust Score (0-100) and an ACCEPT / REVIEW / VETO verdict."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from physicsveto.solvers.wave import WaveConfig
from physicsveto.verify import checks
from physicsveto.verify.ood import MahalanobisOOD, input_features

ACCEPT, REVIEW, VETO = "ACCEPT", "REVIEW", "VETO"


@dataclass
class VetoReport:
    trust_score: float
    verdict: str
    metrics: dict[str, float]
    ratios: dict[str, float]
    sub_scores: dict[str, float]
    reasons: list[str]
    residual_map: np.ndarray | None = None
    energy_curve: np.ndarray | None = None
    extras: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "trust_score": round(self.trust_score, 1),
            "verdict": self.verdict,
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "ratios": {k: float(v) for k, v in self.ratios.items()},
            "reasons": self.reasons,
        }


def sub_score(ratio: float, hard_ratio: float) -> float:
    if ratio <= 1.0:
        return 100.0
    return float(100.0 * np.clip(1.0 - (ratio - 1.0) / (hard_ratio - 1.0), 0.0, 1.0))


def score_metrics(metrics: dict[str, float], calib: dict[str, float], veto_cfg: dict) -> tuple:
    ratios, subs, reasons = {}, {}, []
    hard_fail = False
    wsum = 0.0
    total = 0.0
    for name, spec in veto_cfg["checks"].items():
        if name not in metrics:
            continue
        ref = max(float(calib[name]), 1e-12)
        r = float(metrics[name]) / ref
        s = sub_score(r, spec["hard_ratio"])
        ratios[name], subs[name] = r, s
        total += spec["weight"] * s
        wsum += spec["weight"]
        if r >= spec["hard_ratio"]:
            hard_fail = True
            reasons.append(f"{spec['label']}: {r:.1f}x the in-distribution level (hard limit {spec['hard_ratio']}x)")
        elif r > 1.0:
            reasons.append(f"{spec['label']}: {r:.1f}x the in-distribution level")
    score = total / max(wsum, 1e-12)
    v = veto_cfg["verdicts"]
    if hard_fail or score < v["review_min_score"]:
        verdict = VETO
    elif score < v["accept_min_score"]:
        verdict = REVIEW
    else:
        verdict = ACCEPT
    if not reasons:
        reasons = ["All physics checks are within the in-distribution range."]
    return score, verdict, ratios, subs, reasons


class PhysicsVeto:
    """Runs every check on one prediction and returns a VetoReport."""

    def __init__(self, case: dict, veto_cfg: dict, calibration: dict):
        self.case = case
        self.cfg = WaveConfig.from_dict(case)
        self.veto_cfg = veto_cfg
        self.calib = calibration["levels"]
        self.ood = MahalanobisOOD.from_dict(calibration["ood"])

    def raw_metrics(self, v, src, pred, pred_alt=None) -> tuple[dict, np.ndarray, np.ndarray]:
        anchors = self.case["anchors"]
        pde, res_map = checks.pde_residual(pred, v, src, anchors, self.cfg)
        energy, curve = checks.energy_violation(pred, v, anchors, self.cfg)
        m = {
            "pde_residual": pde,
            "boundary_error": checks.boundary_error(pred, v, self.cfg),
            "energy_violation": energy,
            "ood_distance": self.ood.distance(input_features(v, src)),
        }
        if pred_alt is not None:
            m["ensemble_disagreement"] = checks.ensemble_disagreement(pred, pred_alt)
        return m, res_map, curve

    def evaluate(self, v, src, pred, pred_alt=None) -> VetoReport:
        m, res_map, curve = self.raw_metrics(v, src, pred, pred_alt)
        score, verdict, ratios, subs, reasons = score_metrics(m, self.calib, self.veto_cfg)
        return VetoReport(score, verdict, m, ratios, subs, reasons, res_map, curve)
