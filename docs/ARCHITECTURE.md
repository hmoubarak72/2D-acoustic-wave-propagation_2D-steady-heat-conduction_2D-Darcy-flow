# Architecture

```mermaid
flowchart LR
    subgraph Offline
        G[Dataset generator<br/>Gaussian random fields<br/>+ shot positions] --> S[FD reference solver<br/>2D acoustic wave, Mur edges]
        S --> D[(data/*.npz)]
        D --> T[Train FNO + U-Net<br/>CPU, scripts/train.py]
        T --> W[(weights/)]
        D --> C[Calibrate veto<br/>95th pct of in-dist. checks]
        W --> C
        C --> R[(results/)]
        R --> B[Benchmark + planted failures<br/>scripts/benchmark.py]
    end
    subgraph Online["Online (Streamlit app)"]
        I[New input] --> M[Surrogate inference]
        M --> V{Physics veto<br/>residual · boundary · energy · OOD · disagreement}
        V -- ACCEPT --> U[Use surrogate answer]
        V -- REVIEW --> E[Engineer reviews evidence]
        V -- VETO --> F[Run full solver]
        V --> O[ROI model<br/>net of vetoed runs]
    end
    W --> M
    R --> V
```

## Data encoding
For each of 8 anchor time steps *n*, the surrogate predicts three channels:
`p0 = p[n]`, `d1 = (p[n+1] − p[n−1]) / 2`, `d2 = p[n+1] − 2p[n] + p[n−1]`.
Predicting the time differences directly, instead of three raw snapshots, keeps the network's errors from being
amplified when the PDE residual takes a second time derivative. It also gives the energy check a velocity
field (`d1/dt`) and the boundary check the `p[n+1]` it needs.

## The checks (`physicsveto/verify/`)
| Check | What it computes | Needs the reference? |
|---|---|---|
| PDE residual | `d2 − (vΔt)² ∇²p0 − source` on interior cells, relative to `(vΔt)² ∇²p0` | No |
| Absorbing boundary | Mismatch of Mur's one-way wave condition along all four edges | No |
| Energy conservation | Growth of discrete acoustic energy after the source stops | No |
| OOD distance | Mahalanobis distance of input features (velocity stats, roughness, max gradient, shot position) from training | No |
| Model disagreement | Relative L2 difference between FNO and U-Net | No |

Each raw metric is divided by its 95th percentile on in-distribution validation predictions
(`results/calibration.json`). A ratio of 1 means "as good as normal". Sub-scores fall linearly to zero at a
per-check hard limit, and any hard-limit breach forces a VETO. Weights and limits are in `config/veto.yaml`.

## Scaling on the NVIDIA stack
The demo runs on CPU at 64×64. The same workflow maps onto GPU infrastructure without changing the veto
interface:

| Layer | Demo (built) | Production path (not run here) |
|---|---|---|
| Surrogate model | `FNO2d` in plain PyTorch | `physicsnemo.models.fno.FNO(dimension=3)` from NVIDIA PhysicsNeMo (signature checked against v2.2.2) |
| Training | Single-process CPU loop | `physicsnemo.distributed.DistributedManager` + PyTorch DDP across GPUs |
| Checkpoints | `torch.save` state dict | PhysicsNeMo `Module.save` / `Module.from_checkpoint` |
| Residual checks | NumPy stencils | Same stencils as batched GPU convolutions (torch or CuPy) |
| Data | `.npz` files | Sharded arrays; GPU dataframes (cuDF) for run metadata and OOD features (TODO: verify fit) |
| Visualization | Plotly heatmaps | 3D review of vetoed cases, e.g. in Omniverse (TODO: verify connector) |

`physicsveto/backends/physicsnemo_backend.py` holds the adapter stub. It is not executed in this repo.
