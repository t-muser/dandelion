"""Validation and diagnostic metrics for spherical PDE training.

All metrics operate on channels-first tensors ``(B, C, H, W)``. ``H``
indexes latitude; spherical averages weight by ``cos(lat)`` (normalized
to mean 1) so equal-area regions contribute equally regardless of the
grid oversampling near the poles.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import torch
from torch import nn


def latitude_weights(
    nlat: int, grid: str = "equiangular", device: Optional[torch.device] = None
) -> torch.Tensor:
    """``cos(lat)`` weights on the grid's latitude nodes, normalized to mean 1.

    Returns a 1-D tensor of shape ``(nlat,)`` suitable for broadcasting
    against ``(B, C, H, W)`` along ``H``.
    """
    if grid == "legendre-gauss":
        from torch_harmonics.quadrature import precompute_latitudes
        colat, _ = precompute_latitudes(nlat, grid="legendre-gauss")
        colat = torch.as_tensor(colat, dtype=torch.float32)
        lat = math.pi / 2 - colat
    else:  # equiangular
        lat = torch.linspace(math.pi / 2, -math.pi / 2, nlat)
    w = torch.cos(lat).clamp(min=0.0)
    w = w / w.mean()
    if device is not None:
        w = w.to(device)
    return w


def _spatial_mean(
    x: torch.Tensor,
    lat_weights: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted mean over ``(H, W)`` → ``(B, C)``.

    With ``lat_weights`` this is the cos-latitude-weighted spherical
    average; without, a plain mean over spatial dims. With ``mask``
    (broadcastable to ``x``), the average is over the *unmasked* cells
    only — i.e. land cells do not bias the mean toward zero.
    """
    if lat_weights is None and mask is None:
        return x.mean(dim=(-2, -1))
    if lat_weights is not None:
        w = lat_weights.view(1, 1, -1, 1)
    else:
        w = torch.ones((), dtype=x.dtype, device=x.device)
    if mask is None:
        return (x * w).mean(dim=(-2, -1))
    weighted = (x * w * mask).sum(dim=(-2, -1))
    norm = (w * mask).sum(dim=(-2, -1)).clamp(min=1.0)
    return weighted / norm


def param_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    with torch.no_grad():
        for p in parameters:
            total += p.detach().float().pow(2).sum().item()
    return total ** 0.5


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    with torch.no_grad():
        for p in parameters:
            if p.grad is None:
                continue
            total += p.grad.detach().float().pow(2).sum().item()
    return total ** 0.5


def compute_loss_metrics(
    y_pred: torch.Tensor,
    y_target: torch.Tensor,
    lat_weights: Optional[torch.Tensor] = None,
    field_names: Optional[list[str]] = None,
    eps: float = 1e-7,
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Full metric suite on a single batch.

    Spatial means are sphere-area-weighted by ``lat_weights`` when
    provided. Reported scalars:

    - ``mse``: plain (unweighted) MSE — kept for backwards compatibility.
    - ``mse_sphere`` / ``rmse_sphere``: cos-lat-weighted MSE / RMSE.
    - ``vrmse``: variance-scaled RMSE per field (The Well's primary
      metric, ``sqrt(MSE / Var(y))``), averaged over the batch and then
      over fields. Cross-field comparable. Lat-weighted in both numerator
      and denominator to match the loss.
    - ``vrmse_unweighted``: same shape as ``vrmse`` but with *unweighted* spatial
      mean / variance (uniform per-pixel) — the form The Well reports, so
      its leaderboard numbers are directly comparable.
    - ``nrmse``: ``sqrt(MSE / mean(y²))``, the norm-scaled variant.
    - ``rel_l2``: per-sample ``||ŷ − y||₂ / ||y||₂``, batch-mean.
    - ``pearson``: spatial Pearson correlation per field (1.0 is
      perfect), batch-mean.
    - ``linf``: max absolute residual, batch-mean.

    Per-field variants are emitted under ``<metric>/<field>``.
    """
    assert y_pred.shape == y_target.shape, (
        f"shape mismatch: pred {tuple(y_pred.shape)} vs target {tuple(y_target.shape)}"
    )
    B, C = y_pred.shape[:2]
    if mask is not None:
        # Drop land contributions from both inputs so residuals/centerings
        # are computed on ocean cells only.
        y_pred = y_pred * mask
        y_target = y_target * mask
    diff = y_pred - y_target
    sq = diff.pow(2)

    # Flat MSE (no spherical weighting).
    if mask is not None:
        flat_mse = (sq.sum() / mask.sum().clamp(min=1.0)).item()
    else:
        flat_mse = sq.mean().item()

    # Sphere-weighted per-(B, C) statistics.
    per_sample_mse = _spatial_mean(sq, lat_weights, mask)                       # (B, C)
    mean_y = _spatial_mean(y_target, lat_weights, mask)                         # (B, C)
    mean_x = _spatial_mean(y_pred, lat_weights, mask)                           # (B, C)
    y_centered = y_target - mean_y[..., None, None]
    x_centered = y_pred - mean_x[..., None, None]
    if mask is not None:
        # Centering re-introduces non-zero values on land; re-mask before var.
        y_centered = y_centered * mask
        x_centered = x_centered * mask
    per_sample_var_y = _spatial_mean(y_centered.pow(2), lat_weights, mask)      # (B, C)
    per_sample_var_x = _spatial_mean(x_centered.pow(2), lat_weights, mask)      # (B, C)
    per_sample_norm_y = _spatial_mean(y_target.pow(2), lat_weights, mask)       # (B, C)
    per_sample_cov = _spatial_mean(x_centered * y_centered, lat_weights, mask)  # (B, C)

    # Aggregated (per-field, then mean over fields).
    per_field_mse = per_sample_mse.mean(dim=0)
    per_field_rmse = per_field_mse.sqrt()
    sphere_mse = per_field_mse.mean().item()
    sphere_rmse = math.sqrt(max(sphere_mse, 0.0))

    # VRMSE: per-sample ratio first, then average. Matches The Well's
    # behaviour where each sample is normalised by its own per-channel
    # variance before being averaged.
    per_sample_vrmse = (per_sample_mse / (per_sample_var_y + eps)).sqrt()
    per_field_vrmse = per_sample_vrmse.mean(dim=0)
    vrmse_mean = per_field_vrmse.mean().item()

    # Unweighted (Well-style) VRMSE for direct leaderboard comparison.
    per_sample_mse_uw = _spatial_mean(sq, None, mask)                       # (B, C)
    mean_y_uw = _spatial_mean(y_target, None, mask)                         # (B, C)
    y_target_centered_uw = y_target - mean_y_uw[..., None, None]
    if mask is not None:
        y_target_centered_uw = y_target_centered_uw * mask
    per_sample_var_y_uw = _spatial_mean(y_target_centered_uw.pow(2), None, mask)
    per_sample_vrmse_unweighted = (per_sample_mse_uw / (per_sample_var_y_uw + eps)).sqrt()
    per_field_vrmse_unweighted = per_sample_vrmse_unweighted.mean(dim=0)
    vrmse_unweighted_mean = per_field_vrmse_unweighted.mean().item()

    per_sample_nrmse = (per_sample_mse / (per_sample_norm_y + eps)).sqrt()
    per_field_nrmse = per_sample_nrmse.mean(dim=0)
    nrmse_mean = per_field_nrmse.mean().item()

    # Spatial Pearson R per (B, C), then averaged.
    std_x = per_sample_var_x.clamp(min=eps).sqrt()
    std_y = per_sample_var_y.clamp(min=eps).sqrt()
    per_sample_pearson = per_sample_cov / (std_x * std_y + eps)
    per_field_pearson = per_sample_pearson.mean(dim=0)
    pearson_mean = per_field_pearson.mean().item()

    # L∞: per-sample max abs residual (no spherical weighting — it's a
    # max, weighting doesn't apply).
    per_sample_linf = diff.abs().reshape(B, C, -1).amax(dim=-1)
    per_field_linf = per_sample_linf.mean(dim=0)
    linf_mean = per_field_linf.mean().item()

    # Variable-relative L2 per field, averaged over batch.
    pred_flat = y_pred.reshape(B, C, -1)
    targ_flat = y_target.reshape(B, C, -1)
    num = (pred_flat - targ_flat).norm(dim=-1)
    den = targ_flat.norm(dim=-1).clamp(min=1e-12)
    rel_l2_per_sample = num / den  # (B, C)
    rel_l2_per_field = rel_l2_per_sample.mean(dim=0)
    rel_l2_mean = rel_l2_per_field.mean().item()

    metrics: Dict[str, float] = {
        "mse": flat_mse,
        "mse_sphere": sphere_mse,
        "rmse_sphere": sphere_rmse,
        "vrmse": vrmse_mean,
        "vrmse_unweighted": vrmse_unweighted_mean,
        "nrmse": nrmse_mean,
        "rel_l2": rel_l2_mean,
        "pearson": pearson_mean,
        "linf": linf_mean,
    }
    names = field_names or [f"field_{i}" for i in range(C)]
    for i, name in enumerate(names[:C]):
        metrics[f"rmse_sphere/{name}"] = per_field_rmse[i].item()
        metrics[f"vrmse/{name}"] = per_field_vrmse[i].item()
        metrics[f"vrmse_unweighted/{name}"] = per_field_vrmse_unweighted[i].item()
        metrics[f"nrmse/{name}"] = per_field_nrmse[i].item()
        metrics[f"rel_l2/{name}"] = rel_l2_per_field[i].item()
        metrics[f"pearson/{name}"] = per_field_pearson[i].item()
        metrics[f"linf/{name}"] = per_field_linf[i].item()
    return metrics


class LatitudeWeightedMSELoss(nn.Module):
    """Cos-latitude-weighted MSE for spherical fields on (B, C, H, W).

    Plain MSE on equiangular grids overweights polar errors because the
    grid oversamples near the poles. This loss applies cos(lat) weights
    along the H axis (normalized to mean 1) so each unit of solid angle
    contributes equally to the loss — matching the convention used by
    the validation metrics and standard in numerical-weather literature
    (GraphCast, FourCastNet, etc.).

    Latitude weights are computed lazily on the first forward call from
    the input's H dimension. Pass ``nlat`` and ``grid`` to compute them
    eagerly (e.g. when constructing from a Hydra config), or call
    :meth:`set_lat_weights` to inject pre-computed weights.
    """

    def __init__(
        self,
        nlat: Optional[int] = None,
        grid: str = "equiangular",
    ):
        super().__init__()
        self.grid = grid
        if nlat is not None:
            self.register_buffer("lat_weights", latitude_weights(nlat, grid=grid))

    def set_lat_weights(self, weights: torch.Tensor) -> None:
        """Inject pre-computed latitude weights (1-D tensor of length H)."""
        if "lat_weights" in self._buffers:
            self.lat_weights = weights.to(self.lat_weights.device)
        else:
            self.register_buffer("lat_weights", weights)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if "lat_weights" not in self._buffers:
            self.register_buffer(
                "lat_weights",
                latitude_weights(y_pred.shape[-2], grid=self.grid).to(y_pred.device),
            )
        w = self.lat_weights.view(1, 1, -1, 1)
        sq = (y_pred - y_target).pow(2) * w
        if mask is None:
            return sq.mean()
        # mask broadcasts as (B, C, H, W) or (1, C, H, W); zero out invalid
        # cells and divide by the *weighted* count of valid cells so each
        # unit of solid-angle ocean contributes equally.
        sq = sq * mask
        denom = (w * mask).sum().clamp(min=1.0)
        return sq.sum() / denom
