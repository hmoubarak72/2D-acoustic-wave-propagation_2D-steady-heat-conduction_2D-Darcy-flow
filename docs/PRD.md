# 🗺️ PhysicsVeto: product requirements (PRD)

*Owner: Dr Hesham Moubarak · Status: prototype (Case A live) · Last updated: November 2026*

## 1. Problem
Engineering and subsurface teams spend heavily on HPC to run CFD, structural, thermal and wave-propagation
solvers. AI surrogates (neural operators, graph networks, PINNs) can stand in for many of those runs during
design exploration, but adoption stalls on one question: **"How do I know the surrogate is right for this new
case?"** Average accuracy on a test set doesn't answer it. Engineers need a per-prediction answer.

## 2. Insight
The physics itself is a free, always-available reference. Even without the true solution, a prediction can be
checked against the governing equation, the boundary conditions and conservation laws, and against the
training distribution. A surrogate that is wrapped in those checks can be trusted where it passes and replaced
by the solver where it fails. The measured fallback rate then becomes the basis for an honest business case.

## 3. Users and jobs-to-be-done
| User | Job to be done | What they need from PhysicsVeto |
|---|---|---|
| Simulation / CAE engineer, geophysicist | "Explore many more design or survey scenarios without waiting days." | Fast answers, with a clear signal when not to trust one |
| CAE / HPC manager | "Deliver more simulation throughput on a fixed cluster and budget." | Measured veto rate, net savings, capacity plan |
| CTO / VP engineering | "Adopt AI in simulation without creating technical or safety risk." | Auditable evidence per prediction; failure modes stated up front |
| Partner / solutions team | "Show customers the value in 5 minutes and size the deal." | Demo, ROI model, one-pager, talk track |

## 4. Requirements (MVP, built)
- **R1** Reference solver per case, with correctness tests (convergence, CFL stability, absorbing boundaries).
- **R2** At least two surrogate architectures (FNO and U-Net) trained on generated data; weights shipped for
  inference-only hosting.
- **R3** Physics veto on every prediction: PDE residual, boundary condition, conservation, OOD distance, model
  disagreement. Combined into a 0-100 Trust Score and ACCEPT / REVIEW / VETO.
- **R4** The veto itself is validated on planted failures, with precision and recall reported per failure mode.
- **R5** ROI model that subtracts every vetoed run, with sensitivity analysis and every assumption visible.
- **R6** GTM assets generated per vertical: personas, value propositions, objection handling, brief, talk track.
- **R7** Every number in the app and README is produced by scripts into `results/`. No typed-in metrics.

## 5. Success metrics
| Metric | Why it matters | Target (pilot) |
|---|---|---|
| Veto recall on truly bad predictions | Trust: bad answers must not get through | ≥ 90% |
| Veto precision | Cost: unnecessary vetoes waste solver time | ≥ 80% |
| In-distribution veto rate | Sets effective speedup = 1 / (veto rate + 1/speedup) | ≤ 10% |
| Time-to-first-surrogate on a customer workload | Adoption speed | ≤ 2 weeks from 1,000 historical runs |
| Weekly active engineers using surrogate answers | Real adoption, not a demo | Growth month over month |
| App page load | Demo quality | < 3 s |

## 6. Non-goals (for now)
- Replacing the solver for sign-off or certification runs.
- Production-scale 3D meshes; this prototype is 2D, 64×64, synthetic.
- A general-purpose ML platform. PhysicsVeto is a verification and decision layer on top of existing solvers
  and surrogate frameworks.
- Claims of speedup beyond what is measured. Production speedups are inputs to the ROI model, labelled as
  assumptions.

## 7. Roadmap
**Now (built):** Case A (2D acoustic wave, seismic); FNO + U-Net; physics veto with calibrated Trust Score;
planted-failure evaluation; ROI with tornado; GTM Studio in template mode; CPU-only hosting.

**Next:**
- Case B (steady heat conduction, electronics cooling) and Case C (Darcy flow, reservoir / CCS) on the same
  pipeline.
- PINN baseline for Case B to show data-driven vs physics-informed trade-offs.
- Optional LLM generation in GTM Studio when an API key is present.
- Active learning: vetoed cases are queued as new training data.

**Later:**
- GPU training and serving with NVIDIA PhysicsNeMo (`physicsveto/backends/physicsnemo_backend.py`); GPU
  residual kernels; 3D volumes.
- Customer pilot harness: bring historical runs, measure veto rate and net ROI on their own workload.
- 3D visualization (for example Omniverse) for review of vetoed cases.

## 8. Risks and open questions
- **Blind spots:** physically consistent but wrong predictions (such as a uniform amplitude error) can pass
  residual checks. Mitigation: model disagreement, OOD detection, periodic solver spot-checks.
- **Calibration drift:** thresholds are calibrated on in-distribution validation data; they must be recalibrated
  per customer workload.
- **Speed at small scale:** at 64×64 on CPU the solver is faster than the surrogate. The value proposition
  depends on problem size and GPU inference, which the pilot must measure.
