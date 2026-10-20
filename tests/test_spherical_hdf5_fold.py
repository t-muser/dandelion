"""Tests for SphericalHDF5Dataset's level->channel fold.

Two layers:
  * ``test_fold_*`` exercise the pure tensor reshape (``_fold_level_into_channel``)
    with synthetic data -- no HDF5 or dataset on disk needed, runs in CI.
  * ``test_dry_atmosphere_fold_smoke`` builds the real datamodule from the
    shipped config and checks the folded metadata + a sample. Skips unless
    dry-atmosphere-3D has been downloaded to ``$DANDELION_DATA``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from dandelion.data.spherical_hdf5 import SphericalHDF5Dataset

_HS_HDF5 = Path(os.environ.get("DANDELION_DATA", "data")) / "dry-atmosphere-3D"
_CONFIG = Path(__file__).resolve().parents[1] / "configs/data/dry-atmosphere-3D.yaml"


def test_fold_collapses_levels_field_major():
    """[T(leveled), ps(surface), u(leveled), v(leveled)] over 3 levels folds to
    field-major channels with surface ps reduced to a single (level-0) channel."""
    nsteps, nchan, nlev, h, w = 2, 4, 3, 2, 2
    x = torch.zeros(nsteps, nchan, nlev, h, w)
    for c in range(nchan):
        for lev in range(nlev):
            x[:, c, lev] = c * 10 + lev  # unique per (channel, level)

    surface_mask = [False, True, False, False]  # ps is surface-only
    out = SphericalHDF5Dataset._fold_level_into_channel(x, 1, surface_mask)

    # T_l0..2 (0,1,2), ps_l0 (10), u_l0..2 (20,21,22), v_l0..2 (30,31,32)
    expected = [0, 1, 2, 10, 20, 21, 22, 30, 31, 32]
    assert out.shape == (nsteps, len(expected), h, w)
    assert out[0, :, 0, 0].tolist() == expected


def test_fold_all_leveled_is_plain_reshape():
    """With no surface fields every channel expands to nlev, preserving order."""
    x = torch.arange(1 * 2 * 3 * 1 * 1, dtype=torch.float32).reshape(1, 2, 3, 1, 1)
    out = SphericalHDF5Dataset._fold_level_into_channel(x, 1, [False, False])
    assert out.shape == (1, 6, 1, 1)
    assert out[0, :, 0, 0].tolist() == [0, 1, 2, 3, 4, 5]


def test_fold_constant_field_axis():
    """Fold also works on the (C, *spatial) constant-field layout (channel 0)."""
    x = torch.zeros(2, 3, 1, 1)  # (C=2, level=3, h, w)
    for c in range(2):
        for lev in range(3):
            x[c, lev] = c * 10 + lev
    out = SphericalHDF5Dataset._fold_level_into_channel(x, 0, [True, False])
    # c0 surface -> level 0 only (0); c1 leveled -> 10,11,12
    assert out.shape == (4, 1, 1)
    assert out[:, 0, 0].tolist() == [0, 10, 11, 12]


@pytest.mark.skipif(not _HS_HDF5.is_dir(), reason="dry-atmosphere-3D not downloaded")
def test_dry_atmosphere_fold_smoke():
    pytest.importorskip("h5py")
    pytest.importorskip("hydra")
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(_CONFIG)
    cfg.data.path = str(_HS_HDF5)
    dm = instantiate(cfg.data)  # NotWellDataModule builds datasets in __init__

    md = dm.metadata
    # 4 fields x 8 levels with surface ps collapsed -> 25 channels, 2-D grid.
    assert md.spatial_resolution == (144, 288), md.spatial_resolution
    assert md.dim_out == 25, md.dim_out
    assert md.n_spatial_dims == 2
    assert md.field_names[8] == "ps", md.field_names

    sample = dm.train_dataset[0]
    x = sample["input_fields"]
    assert x.shape[-3:] == (25, 144, 288), x.shape  # (Ti, 25, 144, 288)

    denorm = dm.denormalize_fn
    if denorm is not None:
        assert dm.train_dataset.folded_denorm_stats is not None
        mean, std = dm.train_dataset.folded_denorm_stats
        assert mean.numel() == 25 and std.numel() == 25
