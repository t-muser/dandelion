"""GAOT baseline wrapper.

Bridges this project's dense-grid model contract onto the upstream GAOT's
coordinate/point-cloud API.


We treat the Sphere as a 2D lat-lon plane (coord_dim=2), upstream GAOT unmodified.
Stock GAOT couples ``len(latent_tokens_size) == coord_dim``, so true 3D-sphere
geometry would require forking the model; for a comparison baseline we run GAOT
the way its own configs run it (all upstream examples are 2D). This matches the
planar treatment of the Euclidean baselines.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ._upstream.gaot import GAOT
from ._upstream.layers.attn import TransformerConfig
from ._upstream.layers.magno import MAGNOConfig


@dataclass
class _ModelArgs:
    magno: MAGNOConfig
    transformer: TransformerConfig


@dataclass
class _ModelConfig:
    """Minimal stand-in for GAOT's ``core.default_configs.ModelConfig``."""
    latent_tokens_size: tuple
    args: _ModelArgs
    use_conditional_norm: bool = False
    name: str = "gaot"


def _latlon_grid_coords(nlat: int, nlon: int, grid: str) -> torch.Tensor:
    if grid == "legendre-gauss":
        from torch_harmonics.quadrature import precompute_latitudes

        colat, _ = precompute_latitudes(nlat, grid="legendre-gauss")
        lat = math.pi / 2 - torch.as_tensor(colat, dtype=torch.float32)
    else:  # equiangular: poles inclusive
        lat = torch.linspace(math.pi / 2, -math.pi / 2, nlat)
    lon = torch.linspace(0.0, 2 * math.pi, nlon + 1)[:-1]  # [0, 2pi)

    lat_n = (lat / (math.pi / 2)).float()          # [-1, 1]
    lon_n = (lon / math.pi - 1.0).float()          # [0, 2pi) -> [-1, 1)
    lat_g, lon_g = torch.meshgrid(lat_n, lon_n, indexing="ij")
    return torch.stack([lat_g.reshape(-1), lon_g.reshape(-1)], dim=-1)  # [nlat*nlon, 2]


class GAOTWrapper(nn.Module):
    """Adapt GAOT to the ``(inp_shape, inp_chans, out_chans)`` grid contract."""

    def __init__(
        self,
        inp_shape,
        inp_chans: int,
        out_chans: int,
        latent_tokens_size=(64, 64),
        patch_size: int = 8,
        hidden_size: int = 256,
        num_layers: int = 3,
        positional_embedding: str = "absolute",
        grid: str = "equiangular",
        magno: Optional[dict] = None,
    ):
        super().__init__()
        # dandelion.train sets torch.backends.cudnn.benchmark=True globally, which
        # makes cuDNN pick large persistent algorithm workspaces for GAOT's
        # Conv1d ChannelMLPs. Those are 1x1 (pointwise) convs, so a benchmark buys
        # ~nothing but inflated steady-state memory ~2.5x over a  single-iteration
        # profile. This is enough to OOM the high-channel datasets
        # (dry-atmosphere, ocean) at otherwise-fitting batch sizes.
        torch.backends.cudnn.benchmark = False
        # cuSOLVER's batched symmetric eigensolver (cusolverDnXsyevBatched) raises
        # CUSOLVER_STATUS_INTERNAL_ERROR on this torch build (2.11+cu130) for the
        # near-degenerate 2x2 covariance batches GAOT's geometry embedding
        # (gemb.py) feeds to torch.linalg.eigvalsh at 256x512 (other CUDA builds
        # are fine). MAGMA handles them fine, so we prefer it for GAOT's CUDA
        # linear algebra.
        torch.backends.cuda.preferred_linalg_library("magma")
        H, W = int(inp_shape[0]), int(inp_shape[1])
        Hl, Wl = int(latent_tokens_size[0]), int(latent_tokens_size[1])
        if Hl % patch_size or Wl % patch_size:
            raise ValueError(
                f"latent_tokens_size {(Hl, Wl)} must be divisible by patch_size {patch_size}"
            )

        magno_kwargs = dict(magno) if magno else {}
        magno_kwargs.setdefault("coord_dim", 2)
        magno_kwargs.setdefault("neighbor_search_method", "native")
        if int(magno_kwargs["coord_dim"]) != 2:
            raise ValueError(
                "GAOTWrapper treats the sphere as a 2D lat-lon plane; coord_dim must be 2."
            )
        magno_cfg = MAGNOConfig(**magno_kwargs)
        transformer_cfg = TransformerConfig(
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            positional_embedding=positional_embedding,
        )
        cfg = _ModelConfig(
            latent_tokens_size=(Hl, Wl),
            args=_ModelArgs(magno=magno_cfg, transformer=transformer_cfg),
        )

        self.H, self.W = H, W
        self.out_chans = out_chans
        self.model = GAOT(input_size=inp_chans, output_size=out_chans, config=cfg)

        self.register_buffer("xcoord", _latlon_grid_coords(H, W, grid), persistent=False)
        self.register_buffer(
            "latent_coord", _latlon_grid_coords(Hl, Wl, "equiangular"), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # [B, C, H, W] -> [B, H*W, C], row-major over (H, W) to match xcoord ordering.
        pndata = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        out = self.model(self.latent_coord, self.xcoord, pndata)  # [B, H*W, out_chans]
        n_query = out.shape[1]
        assert n_query == H * W, f"GAOT returned {n_query} query nodes, expected {H * W}"
        return out.reshape(B, H, W, self.out_chans).permute(0, 3, 1, 2).contiguous()
