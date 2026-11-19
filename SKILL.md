---
name: physicsveto
description: Run, rebuild, benchmark and demo the PhysicsVeto repo (AI physics surrogates with a physics veto). Use when asked to regenerate results, retrain models, run the app, or prepare the interview demo.
---

# PhysicsVeto

## Layout
- `physicsveto/solvers/wave.py`: FD reference solver (2D acoustic, Mur absorbing edges).
- `physicsveto/data/fields.py`: velocity-model generators, planted failure modes, target encoding (p0, d1, d2 per anchor).
- `physicsveto/models/`: `fno.py`, `unet.py`, `surrogate.py` (normalisation + CPU inference).
- `physicsveto/verify/`: `checks.py` (residual, boundary, energy, disagreement), `ood.py`, `trust.py` (Trust Score + verdict).
- Cases B/C: `solvers/elliptic.py`, `data/steady.py`, `models/steady.py`, `verify/steady.py`, `config/veto_steady.yaml`; results in `results/case_b/`, `results/case_c/`.
- `physicsveto/roi/cost.py`, `physicsveto/gtm/content.py` (templates), `physicsveto/gtm/llm.py` (optional Claude API drafting, needs PHYSICSVETO_MOCK=0 + ANTHROPIC_API_KEY).
- `config/`: `cases.yaml`, `veto.yaml` (weights, hard limits, error tolerance), `roi_defaults.yaml`.
- `app/`: Streamlit (`Home.py` + 7 pages). `docs/templates/` are filled by `scripts/make_report.py`.

## Rebuild everything
```bash
pip install -r requirements-dev.txt && pip install -e .
make data      # ~1 min, writes data/*.npz (git-ignored)
make train     # ~45 min CPU, writes weights/ and results/train_history_*.json
make bench     # calibration, planted-failure eval, timing, scaling, README/docs
make data-steady train-steady pinn bench-steady   # Cases B, C, PINN (~1 h CPU)
make test lint
```

## Rules
- Never type a metric into README, docs or app. Change `docs/templates/*.tmpl` and run `make report`.
- Label every speedup with hardware and grid size. At 64×64 on CPU the solver is faster than the FNO.
- After changing `config/cases.yaml` or the solver, regenerate data and retrain: the encoded targets change.
- After changing `config/veto.yaml`, rerun `scripts/benchmark.py` (calibration lives in `results/calibration.json`).
- After changing `config/veto_steady.yaml`, rerun `scripts/benchmark_steady.py --case case_b` and `--case case_c`.

## Demo
`streamlit run app/Home.py`, then follow `docs/DEMO_SCRIPT.md`. The Physics Veto page's "Try to break it"
button picks a random planted failure. "Amplitude drift" is the deliberate blind-spot example.
