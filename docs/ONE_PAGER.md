# PhysicsVeto: solution brief for Energy & subsurface
*Template text (mock mode). Prototype; not a commercial product.*

**AI surrogates make simulation much faster. Physics decides whether you can trust the answer.**

## The problem
Teams running seismic wave propagation for survey design, imaging and full-waveform inversion (FWI); reservoir and CO2-storage flow simulation are limited by solver time. AI surrogates can answer in milliseconds, but
adoption stalls on one question: *how do I know the surrogate is right for this new case?*

## The solution
PhysicsVeto runs a surrogate, then checks every prediction against the physics:
1. **Governing-equation residual** on the predicted field.
2. **Boundary-condition** and **conservation** checks.
3. **Out-of-distribution** distance from the training data, plus **model disagreement**.

The result is a Trust Score and a verdict: ACCEPT, REVIEW, or VETO (route to the full solver).

## Who it is for
- **Geophysicist / imaging specialist:** Waits days for forward models during survey design and FWI iterations; cannot test enough scenarios.
- **HPC / subsurface computing manager:** Owns a large cluster budget; every new imaging project competes for the same cores.
- **VP subsurface / CTO:** Needs faster decisions on wells and CCS sites without adding technical risk to reserves or permits.

## Value
- Screen thousands of survey or velocity-model scenarios with a surrogate, and spend solver time only where the physics veto says the surrogate cannot be trusted.
- Every accepted answer comes with evidence: wave-equation residual, boundary and energy checks, and distance from the training geology.
- Salt bodies and unusual geology are exactly where surrogates fail; the veto routes those to the solver.

## Evidence (measured, demo scale)
- On a planted-failure benchmark (430 predictions), the veto stopped 100% of truly bad predictions (VETO or REVIEW); 83% of VETO calls were correct.
- Bad answers among accepted predictions: 0% (vs 52% with no veto on the same stress-test mix).
- Veto rate on normal, in-distribution work: 0.0%.
- Measured at demo scale: A: 2D acoustic wave, 64x64, 252 steps, x86_64, 1 CPU thread.

## Business case
Savings are computed **after** paying a full solve for every vetoed prediction. The effective speedup of the
workflow is 1 / (veto rate + 1/speedup), so a lower veto rate is worth more than a faster model.

## Scaling path
Same workflow on GPU: train and serve surrogates with an open-source physics-ML framework such as NVIDIA
PhysicsNeMo, compute residuals on GPU, and visualize in a 3D environment. See ARCHITECTURE.md.

*Case A in this demo (2D acoustic wave) is a small version of this workload.*
