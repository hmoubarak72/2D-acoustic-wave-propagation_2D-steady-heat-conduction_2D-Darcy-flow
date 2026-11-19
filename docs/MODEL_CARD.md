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
