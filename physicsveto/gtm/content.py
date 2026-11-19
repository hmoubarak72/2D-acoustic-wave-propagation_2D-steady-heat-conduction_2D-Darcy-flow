"""Go-to-market content: personas, value propositions, objection handling, solution brief, talk track.

MOCK MODE: everything here is pre-written template text (PHYSICSVETO_MOCK=1, the default).
Numbers are injected from results/benchmark_summary.json, never typed in.
"""

from __future__ import annotations

import os

MOCK = os.environ.get("PHYSICSVETO_MOCK", "1") == "1"

VERTICALS: dict[str, dict] = {
    "Energy & subsurface": {
        "workload": "seismic wave propagation for survey design, imaging and full-waveform inversion (FWI); "
                    "reservoir and CO2-storage flow simulation",
        "personas": [
            ("Geophysicist / imaging specialist",
             "Waits days for forward models during survey design and FWI iterations; cannot test enough scenarios."),
            ("HPC / subsurface computing manager",
             "Owns a large cluster budget; every new imaging project competes for the same cores."),
            ("VP subsurface / CTO",
             "Needs faster decisions on wells and CCS sites without adding technical risk to reserves or permits."),
        ],
        "value": [
            "Screen thousands of survey or velocity-model scenarios with a surrogate, and spend solver time only "
            "where the physics veto says the surrogate cannot be trusted.",
            "Every accepted answer comes with evidence: wave-equation residual, boundary and energy checks, and "
            "distance from the training geology.",
            "Salt bodies and unusual geology are exactly where surrogates fail; the veto routes those to the solver.",
        ],
        "case": "Case A in this demo (2D acoustic wave) is a small version of this workload.",
    },
    "Automotive & aerospace": {
        "workload": "external aerodynamics CFD, crash and structural FEA, aeroacoustics",
        "personas": [
            ("CAE engineer", "Turnaround of hours to days per CFD run limits design iterations per program milestone."),
            ("CAE / simulation manager", "Must hit program gates with a fixed HPC allocation and headcount."),
            ("CTO / VP engineering", "Wants AI speed without a credibility problem in front of safety and certification."),
        ],
        "value": [
            "Explore far more shape variants in early design; keep the solver for designs that matter or that the "
            "veto flags.",
            "A per-prediction Trust Score gives reviewers something auditable, not a black box.",
            "Measured veto rate turns AI from a science project into a capacity plan.",
        ],
        "case": "Pattern transfers from Case A; residual checks become Navier-Stokes residuals and force balances.",
    },
    "Electronics & thermal": {
        "workload": "chip, board and data-center thermal simulation; electronics cooling",
        "personas": [
            ("Thermal engineer", "Hundreds of placement and power-map variants, each needing a conjugate heat solve."),
            ("Design-flow / EDA manager", "Thermal sign-off is a schedule bottleneck late in the design cycle."),
            ("CTO", "Needs faster sign-off without risking hot spots in silicon."),
        ],
        "value": [
            "Interactive what-if on power maps and cooling layouts; hot-spot predictions are checked against the "
            "heat equation and energy balance before anyone relies on them.",
            "Energy-balance violations are caught automatically, so a surrogate cannot invent or lose heat.",
            "Sign-off still uses the solver; the surrogate removes the exploratory runs.",
        ],
        "case": "Case B in this demo (steady heat conduction with hot spots) is a small version of this workload.",
    },
    "Manufacturing": {
        "workload": "process simulation (casting, additive, molding), digital twins of equipment",
        "personas": [
            ("Process / simulation engineer", "Process windows need many runs; solver time limits how many."),
            ("Plant digital-twin lead", "Real-time twins need answers in milliseconds, which full solvers cannot give."),
            ("Operations VP", "Wants yield gains without a twin that silently drifts from reality."),
        ],
        "value": [
            "Millisecond surrogates make real-time twins possible; the veto says when the twin is outside what it "
            "learned and must fall back to the solver.",
            "Out-of-distribution detection is a built-in drift alarm for changing materials and conditions.",
            "Net ROI already subtracts fallback runs, so the case survives finance review.",
        ],
        "case": "Case C in this demo (Darcy flow in porous media) covers filtration and porous-media processes.",
    },
}

OBJECTIONS = [
    ("Can I trust it?", "Don't trust it blindly. Every prediction is checked against the governing equation, the "
     "boundary conditions, energy balance and the training distribution. Failures go to the solver, and the demo "
     "shows how often the veto is right on planted failures."),
    ("Our cases are not like your training data.", "That is the out-of-distribution check's job. Unfamiliar inputs "
     "get flagged instead of answered. The fix is to add those cases to training, which the veto tells you to do."),
    ("The speedup claims sound inflated.", "They often are. This demo labels every speedup with its hardware and grid "
     "size, and at 64x64 on CPU the surrogate is not faster. The value grows with problem size and GPU inference, "
     "and the ROI model uses the measured veto rate rather than a headline speedup."),
    ("We already own solver licenses.", "Keep them. The surrogate removes exploratory runs; the solver handles the "
     "vetoed and sign-off cases. License spend shifts from exploration to decisions."),
    ("Who maintains the models?", "Retraining is triggered by the veto: a rising veto rate or OOD rate on new work is "
     "the signal. That makes model ops measurable instead of ad hoc."),
]

COMPETITIVE = [
    ("Traditional CAE vendors", "Established solvers adding AI/ML features to their suites",
     "Deep solver trust, installed base, validated physics", "AI features often tied to one suite; verification "
     "approach varies by product"),
    ("AI-physics startups", "Companies building learned surrogates for engineering design",
     "Fast iteration, strong ML teams, design-exploration UX", "Customer trust and per-prediction verification are "
     "the common adoption hurdle"),
    ("Open-source frameworks", "Libraries for neural operators, PINNs and graph networks (e.g. NVIDIA PhysicsNeMo, "
     "DeepXDE, neuraloperator)", "Free, flexible, research-grade models, GPU scaling",
     "Teams must build data pipelines, verification and ROI tooling themselves"),
    ("In-house ML teams", "Custom surrogates built by the customer's data science group",
     "Domain knowledge, data access", "Hard to maintain; verification is usually ad hoc"),
]


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def solution_brief(vertical: str, s: dict | None) -> str:
    v = VERTICALS[vertical]
    if s:
        evidence = (
            f"- On a planted-failure benchmark ({s['n_eval']} predictions), the veto stopped "
            f"{fmt_pct(s['flag_recall'])} of truly bad predictions (VETO or REVIEW); "
            f"{fmt_pct(s['veto_precision'])} of VETO calls were correct.\n"
            f"- Bad answers among accepted predictions: {fmt_pct(s['accepted_share_truly_bad'])} "
            f"(vs {fmt_pct(s['unfiltered_share_truly_bad'])} with no veto on the same stress-test mix).\n"
            f"- Veto rate on normal, in-distribution work: {100 * s['in_dist_veto_rate']:.1f}%.\n"
            f"- Measured at demo scale: {s['case']}, {s['hardware']}."
        )
    else:
        evidence = "- Run `make bench` to fill in measured evidence."
    return f"""# PhysicsVeto: solution brief for {vertical}
*{'Template text (mock mode).' if MOCK else ''} Prototype; not a commercial product.*

**AI surrogates make simulation much faster. Physics decides whether you can trust the answer.**

## The problem
Teams running {v['workload']} are limited by solver time. AI surrogates can answer in milliseconds, but
adoption stalls on one question: *how do I know the surrogate is right for this new case?*

## The solution
PhysicsVeto runs a surrogate, then checks every prediction against the physics:
1. **Governing-equation residual** on the predicted field.
2. **Boundary-condition** and **conservation** checks.
3. **Out-of-distribution** distance from the training data, plus **model disagreement**.

The result is a Trust Score and a verdict: ACCEPT, REVIEW, or VETO (route to the full solver).

## Who it is for
""" + "\n".join(f"- **{p}:** {pain}" for p, pain in v["personas"]) + """

## Value
""" + "\n".join(f"- {x}" for x in v["value"]) + f"""

## Evidence (measured, demo scale)
{evidence}

## Business case
Savings are computed **after** paying a full solve for every vetoed prediction. The effective speedup of the
workflow is 1 / (veto rate + 1/speedup), so a lower veto rate is worth more than a faster model.

## Scaling path
Same workflow on GPU: train and serve surrogates with an open-source physics-ML framework such as NVIDIA
PhysicsNeMo, compute residuals on GPU, and visualize in a 3D environment. See ARCHITECTURE.md.

*{v['case']}*
"""


def talk_track(vertical: str, s: dict | None) -> str:
    v = VERTICALS[vertical]
    persona = v["personas"][1][0]
    nums = ""
    if s:
        nums = (f" On our planted-failure benchmark it caught {fmt_pct(s['flag_recall'])} of bad predictions, and "
                f"on normal work it vetoes {100 * s['in_dist_veto_rate']:.1f}% of cases.")
    return f"""# 5-minute demo talk track: {vertical}
*{'Template text (mock mode).' if MOCK else ''} Audience: {persona}.*

**0:00-0:45 · Hook (Overview page)**
"Your team runs {v['workload']}. Surrogates promise answers in milliseconds. The question I hear every time is:
can I trust it for my next case? PhysicsVeto is built around that question."

**0:45-1:45 · Speed and accuracy (Simulate & Compare)**
Pick a model, run solver and surrogate side by side. Show the error map and the time. Say the honest part:
"At this toy scale the solver is cheap. The value shows up at production scale; the Benchmark Lab shows the
crossover."

**1:45-3:30 · The differentiator (Physics Veto)**
Show a normal case: ACCEPT. Press **Try to break it**. Watch the residual map light up and the verdict flip to
VETO. Reveal the true error: "The veto never saw the answer key, and it was right."{nums}
Mention one blind spot on purpose: a physically consistent but wrong answer can pass, which is why we also
check model disagreement.

**3:30-4:30 · Business case (ROI)**
Enter their numbers. Point at the effective-speedup line: "Every vetoed case still costs a full solve, so we
optimize veto rate, not headline speedup." Show the tornado: veto rate and speedup dominate.

**4:30-5:00 · Close**
"Next step: pick one workload, bring 1,000 historical runs, and we measure your veto rate in two weeks.
That number is your business case."
"""
