"""Scaling path on the NVIDIA stack: adapter from PhysicsVeto to NVIDIA PhysicsNeMo.

STATUS: interface stub. It is NOT run in this demo (no GPU, PhysicsNeMo not installed in the app image).
The API names below were checked against the nvidia-physicsnemo 2.2.2 wheel on PyPI
(physicsnemo.models.fno.FNO, physicsnemo.distributed.DistributedManager, Module.save / Module.from_checkpoint).
Anything not checked that way is marked TODO: verify.

What changes when scaling from this demo to production:
  - model:     physicsveto.models.fno.FNO2d  ->  physicsnemo.models.fno.FNO(dimension=3) for 3D volumes
  - training:  single-process CPU loop        ->  DistributedManager + torch DDP across GPUs
  - residuals: NumPy stencils                  ->  the same stencils on GPU tensors (torch or CuPy), batched
  - viz:       Plotly heatmaps                 ->  3D scenes, e.g. Omniverse (TODO: verify connector choice)
The physics veto interface (PhysicsVeto.evaluate) stays the same: that is the point of the design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PhysicsNeMoConfig:
    in_channels: int = 2          # velocity + source map (grid coordinates are added by PhysicsNeMo FNO)
    out_channels: int = 24        # 8 anchors x (p0, d1, d2), same encoding as the CPU demo
    dimension: int = 2            # 3 for volumetric seismic
    latent_channels: int = 40
    num_fno_layers: int = 4
    num_fno_modes: int = 16
    padding: int = 8
    decoder_layers: int = 1
    decoder_layer_size: int = 128


def physicsnemo_available() -> bool:
    try:
        import physicsnemo  # noqa: F401
    except ImportError:
        return False
    return True


class PhysicsNeMoBackend:
    """Same contract as physicsveto.models.surrogate.Surrogate: predict(v, srcs) -> encoded wavefields."""

    def __init__(self, cfg: PhysicsNeMoConfig | None = None):
        self.cfg = cfg or PhysicsNeMoConfig()
        self.model: Any = None

    def build_model(self) -> Any:
        if not physicsnemo_available():
            raise RuntimeError("PhysicsNeMo is not installed. Use physicsveto.models.fno.FNO2d on CPU.")
        from physicsnemo.models.fno import FNO  # verified in nvidia-physicsnemo 2.2.2

        c = self.cfg
        self.model = FNO(
            in_channels=c.in_channels,
            out_channels=c.out_channels,
            decoder_layers=c.decoder_layers,
            decoder_layer_size=c.decoder_layer_size,
            dimension=c.dimension,
            latent_channels=c.latent_channels,
            num_fno_layers=c.num_fno_layers,
            num_fno_modes=c.num_fno_modes,
            padding=c.padding,
        )
        return self.model

    def train_distributed(self, dataset: Any) -> None:
        """Multi-GPU training outline (not executed in this repo).

        from physicsnemo.distributed import DistributedManager   # verified
        DistributedManager.initialize(); dm = DistributedManager()
        model = self.build_model().to(dm.device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dm.local_rank])
        # TODO: verify the recommended PhysicsNeMo datapipe for sharded .npz / zarr inputs
        # loss: same relative-L2 as scripts/train.py
        model.module.save("case_a_fno.mdlus")                     # Module.save, verified
        """
        raise NotImplementedError("Outline only; requires GPUs and PhysicsNeMo.")

    def gpu_residual(self, pred: Any, v: Any) -> Any:
        """Port of physicsveto.verify.checks.pde_residual to GPU tensors.

        The 5-point Laplacian is a fixed convolution, so on GPU it becomes one conv2d (conv3d in 3D)
        over the whole batch. TODO: verify throughput vs. inference cost on target hardware.
        """
        raise NotImplementedError("Outline only; requires a CUDA device.")
