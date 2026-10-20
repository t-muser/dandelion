"""The autoregressive slide keeps the per-frame input layout.

Each input frame is ``dim_in`` channels: ``dim_out`` predicted fields followed
by ``dim_in - dim_out`` input-only (forcing) channels (global-ocean-3D: depth +
sin/cos day of year). Over a rollout, every history frame must stay
``[fields, forcing]`` with the forcing of that frame's own time.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from dandelion.trainer import Trainer


def _trainer(dim_in, dim_out):
    md = SimpleNamespace(dim_in=dim_in, dim_out=dim_out, spatial_resolution=(4, 8),
                         grid="equiangular", field_names=())
    return Trainer(
        checkpoint_folder="", artifact_folder="", viz_folder="",
        model=torch.nn.Identity(), datamodule=SimpleNamespace(metadata=md),
        optimizer=None, loss_fn=torch.nn.MSELoss(), epochs=1,
    )


def _frame(t, C, F, H=4, W=8):
    """Frame at time t: fields tagged 100*t + c, forcing tagged -(100*t + f)."""
    fields = torch.stack([torch.full((H, W), 100.0 * t + c) for c in range(C)])
    forcing = torch.stack([torch.full((H, W), -(100.0 * t + f)) for f in range(F)])
    return fields, forcing


def test_slide_with_forcing_keeps_frame_layout():
    C, F, T_in, steps = 3, 2, 4, 6
    tr = _trainer(dim_in=C + F, dim_out=C)
    frames = [_frame(t, C, F) for t in range(T_in + steps)]
    hist = torch.cat([torch.cat(frames[t]) for t in range(T_in)])[None]       # (1, T_in*(C+F), H, W)
    forcing = torch.stack([frames[T_in + k][1] for k in range(steps)])[None]   # (1, steps, F, H, W)
    for k in range(steps):
        y_pred = frames[T_in + k][0][None]                                     # "perfect" prediction
        hist = tr._slide(hist, y_pred, forcing, k)
        expected = torch.cat([torch.cat(frames[t]) for t in range(k + 1, k + 1 + T_in)])[None]
        assert torch.equal(hist, expected), f"layout drifted after {k + 1} steps"


def test_slide_without_loader_forcing_repeats_newest():
    C, F, T_in = 3, 2, 4
    tr = _trainer(dim_in=C + F, dim_out=C)
    frames = [_frame(t, C, F) for t in range(T_in + 1)]
    hist = torch.cat([torch.cat(frames[t]) for t in range(T_in)])[None]
    out = tr._slide(hist, frames[T_in][0][None], forcing=None, k=0)
    newest = torch.cat([frames[T_in][0], frames[T_in - 1][1]])                 # fields(t=4), forcing(t=3)
    assert out.shape == hist.shape
    assert torch.equal(out[0, -(C + F):], newest)
    assert torch.equal(out[0, :-(C + F)], hist[0, C + F:])


def test_slide_without_extra_channels_unchanged():
    C, T_in = 3, 4
    tr = _trainer(dim_in=C, dim_out=C)
    hist = torch.randn(2, T_in * C, 4, 8)
    y_pred = torch.randn(2, C, 4, 8)
    assert torch.equal(tr._slide(hist, y_pred), torch.cat([hist[:, C:], y_pred], dim=1))
