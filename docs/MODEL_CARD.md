# Model card: Case A surrogates (2D acoustic wave)

| | FNO (main) | U-Net (baseline / second opinion) |
|---|---|---|
| Code | `physicsveto/models/fno.py` | `physicsveto/models/unet.py` |
| Size | see `results/leaderboard.json` | see `results/leaderboard.json` |
| Training | AdamW, one-cycle LR, relative-L2 loss, CPU (2 threads) | same |
| Weights | `weights/case_a_fno.pt` | `weights/case_a_unet.pt` |

## Task
Input: a 64×64 P-wave velocity model (10 m cells) and a shot position. Output: the pressure wavefield at 8
times between 48 and 200 ms, each as (p, first and second time difference), 24 channels in total.

## Training data
1,200 synthetic velocity models from Gaussian random fields (smoothing σ 5-10 cells, velocities 2,000-3,500
m/s), shot at 160 m depth, lateral position cells 20-44. Reference wavefields from the finite-difference solver
in `physicsveto/solvers/wave.py` (2nd order, Mur absorbing edges, 15 Hz Ricker source, 0.8 ms step).
Validation 150, test 150, all generated with fixed seeds (`config/cases.yaml`).

## Intended use
Demonstrating surrogate + verification workflows. Not for any real survey design, imaging or engineering
decision.

## Known failure modes (measured in `results/benchmark_summary.json`, per mode)
- Salt bodies and velocities outside 2,000-3,500 m/s.
- Rough geology with correlation lengths far below training.
- Shots moved to the edges or deeper than 160 m.
- Accuracy degrades at later times as the wavefield becomes more complex.

## Limitations
2D, constant density, acoustic only, 64×64, synthetic data, single source wavelet. Timings are CPU-only. No
claims about production-scale accuracy or speed.


# Model card: Case B and C surrogates (steady heat conduction, Darcy flow)

| | FNO (main) | U-Net (second opinion) | Physics-loss FNO (Case B) |
|---|---|---|---|
| Code | `physicsveto/models/steady.py` | same | same, `--physics 0.2` |
| Training | AdamW, relative-L2 loss, CPU, `scripts/train_steady.py` | same | relative-L2 + 0.2 × discrete PDE residual |
| Weights | `weights/case_b_fno.pt`, `weights/case_c_fno.pt` | `weights/case_*_unet.pt` | `weights/case_b_fno_pino.pt` |
| Sizes, errors, training time | `results/case_*/leaderboard.json` | same | same |

## Task
Both cases solve −∇·(k∇u) = f on the unit square with u = 0 on the boundary, 64×64 nodes.
- **Case B (heat):** input log-conductivity (smooth random field, k in about 0.37-2.7) and a map of 1-3 Gaussian
  heat sources; output the temperature rise.
- **Case C (Darcy):** input a two-facies permeability map (k = 3 or 12, random channels); forcing f = 1
  everywhere; output the pressure. The network does not see f, so a different forcing is out of scope by design.

## Training data
1,000 train, 150 validation, 150 test per case, generated with fixed seeds (`config/cases.yaml`). References from
the finite-volume solver in `physicsveto/solvers/elliptic.py` (5-point stencil, harmonic face averaging, sparse
direct solve; second-order convergence and exact discrete flux balance are unit-tested). 100-sample variants
(`*_n100`) use the first 100 training samples.

## Known failure modes (measured, per mode, in `results/case_*/benchmark_summary.json`)
- Case B: heat sources next to the cold boundary, conductivity contrast far beyond training, many small sources,
  3-5× the training power.
- Case C: 1:100 permeability contrast, long thin channels, a concentrated injection well, very fine-scale rock.

## Classic PINN baseline (Case B)
`scripts/pinn_case_b.py`: 4×64 tanh MLP, hard Dirichlet boundary, residual loss at random collocation points,
Adam, no solver data. It is trained from scratch per instance; results in `results/case_b/pinn.json`.

## Limitations
2D, steady, 64×64, synthetic data, CPU only. At this size the sparse direct solver is fast, so these surrogates
do not save compute here; they exist to show that the veto transfers across physics.
