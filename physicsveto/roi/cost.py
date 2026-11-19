"""HPC cost and time-to-design model for a surrogate + physics-veto workflow.

The key honesty rule: every vetoed prediction still costs a full solver run (plus the
surrogate call that preceded it). Savings are computed after subtracting those runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass
class RoiInputs:
    sims_per_cycle: float = 2000          # simulations per design cycle
    cycles_per_year: float = 12           # design cycles per year
    core_hours_per_run: float = 400.0     # full solver cost per run (core-hours)
    usd_per_core_hour: float = 0.05       # cloud CPU price
    surrogate_speedup: float = 100.0      # ASSUMPTION at production scale (see results/scaling.json)
    veto_rate: float = 0.10               # share of predictions sent back to the solver (measured)
    wall_hours_per_run: float = 8.0       # solver wall-clock per run
    concurrent_runs: float = 50           # runs the cluster executes in parallel
    training_sims: float = 1500           # solver runs needed to build the training set (one-off)
    gpu_training_hours: float = 50        # one-off training cost
    usd_per_gpu_hour: float = 3.0
    engineer_usd_per_hour: float = 120.0
    engineer_setup_hours: float = 160     # one-off integration effort
    engineer_wait_share: float = 0.25     # share of solver wall time an engineer is blocked

    def to_dict(self) -> dict:
        return asdict(self)


def compute_roi(p: RoiInputs) -> dict[str, float]:
    runs = p.sims_per_cycle * p.cycles_per_year
    run_cost = p.core_hours_per_run * p.usd_per_core_hour
    baseline_compute = runs * run_cost

    vr = min(max(p.veto_rate, 0.0), 1.0)
    surrogate_cost = run_cost / max(p.surrogate_speedup, 1e-9)
    hybrid_compute = runs * (surrogate_cost + vr * run_cost)  # every run hits the surrogate; vetoed ones also hit the solver

    # wall-clock per design cycle (hours)
    base_cycle_h = p.sims_per_cycle * p.wall_hours_per_run / max(p.concurrent_runs, 1)
    hybrid_cycle_h = base_cycle_h * (vr + 1.0 / max(p.surrogate_speedup, 1e-9))
    eng_base = base_cycle_h * p.cycles_per_year * p.engineer_wait_share * p.engineer_usd_per_hour
    eng_hybrid = hybrid_cycle_h * p.cycles_per_year * p.engineer_wait_share * p.engineer_usd_per_hour

    upfront = (
        p.training_sims * run_cost
        + p.gpu_training_hours * p.usd_per_gpu_hour
        + p.engineer_setup_hours * p.engineer_usd_per_hour
    )
    annual_savings = (baseline_compute - hybrid_compute) + (eng_base - eng_hybrid)
    per_cycle = annual_savings / max(p.cycles_per_year, 1e-9)
    break_even_cycles = upfront / per_cycle if per_cycle > 0 else float("inf")
    return {
        "runs_per_year": runs,
        "baseline_compute_usd": baseline_compute,
        "hybrid_compute_usd": hybrid_compute,
        "compute_savings_usd": baseline_compute - hybrid_compute,
        "engineer_time_savings_usd": eng_base - eng_hybrid,
        "annual_savings_usd": annual_savings,
        "upfront_usd": upfront,
        "first_year_net_usd": annual_savings - upfront,
        "break_even_cycles": break_even_cycles,
        "break_even_months": break_even_cycles * 12.0 / max(p.cycles_per_year, 1e-9),
        "cycle_hours_baseline": base_cycle_h,
        "cycle_hours_hybrid": hybrid_cycle_h,
        "time_to_design_reduction": 1.0 - hybrid_cycle_h / base_cycle_h if base_cycle_h > 0 else 0.0,
        "effective_speedup": 1.0 / (vr + 1.0 / max(p.surrogate_speedup, 1e-9)),
    }


TORNADO_FACTORS = [
    ("Surrogate speedup", "surrogate_speedup", 0.5, 2.0),
    ("Veto rate", "veto_rate", 0.5, 2.0),
    ("Simulations per cycle", "sims_per_cycle", 0.5, 1.5),
    ("Core-hours per run", "core_hours_per_run", 0.5, 1.5),
    ("$ per core-hour", "usd_per_core_hour", 0.6, 1.6),
    ("Design cycles per year", "cycles_per_year", 0.5, 1.5),
    ("Engineer $/hour", "engineer_usd_per_hour", 0.7, 1.3),
]


def tornado(p: RoiInputs, metric: str = "first_year_net_usd") -> list[dict]:
    base = compute_roi(p)[metric]
    rows = []
    for label, key, lo, hi in TORNADO_FACTORS:
        v = getattr(p, key)
        lo_v, hi_v = v * lo, v * hi
        if key == "veto_rate":
            hi_v = min(hi_v, 1.0)
        a = compute_roi(replace(p, **{key: lo_v}))[metric]
        b = compute_roi(replace(p, **{key: hi_v}))[metric]
        rows.append({"factor": label, "low_input": lo_v, "high_input": hi_v,
                     "at_low": a, "at_high": b, "base": base, "swing": abs(b - a)})
    rows.sort(key=lambda r: r["swing"], reverse=True)
    return rows
