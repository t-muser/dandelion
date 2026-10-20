"""Shape smoke tests for Dandelion on tiny CPU grids."""
from __future__ import annotations

import pytest
import torch

from dandelion.models.dandelion import Dandelion


@pytest.mark.parametrize("H", [32, 33])  # 33: south-pole row dropped and rebuilt
def test_dandelion_forward_shape(H):
    torch.manual_seed(0)
    model = Dandelion(
        inp_shape=(H, 64), inp_chans=3, out_chans=2, lifting_dim=8, n_levels=4,
        channel_multipliers=[1, 2, 4, 8], num_heads=2, groups=2,
    ).eval()
    with torch.no_grad():
        y = model(torch.randn(2, 3, H, 64))
    assert y.shape == (2, 2, H, 64)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("H", [16, 17])
def test_dandelion2p5d_forward_shape(H):
    from dandelion.models.dandelion2p5d import Dandelion2p5D

    torch.manual_seed(0)
    layout = [3, 1, 3, 3]  # [A x 3 levels, surface, B x 3, C x 3] = 10 channels
    model = Dandelion2p5D(
        inp_shape=(H, 32), inp_chans=2 * 10, out_chans=10, level_layout=layout,
        lifting_dim=8, n_levels=3, channel_multipliers=[1, 2, 4], num_heads=2, groups=2,
    ).eval()
    with torch.no_grad():
        y = model(torch.randn(2, 20, H, 32))
    assert y.shape == (2, 10, H, 32)
    assert torch.isfinite(y).all()


def _p5d(H, layout, n_steps=2, **kw):
    from dandelion.models.dandelion2p5d import Dandelion2p5D

    n_chans = sum(layout)
    return Dandelion2p5D(
        inp_shape=(H, 32), inp_chans=n_steps * (n_chans + kw.get("n_forcing", 0)),
        out_chans=n_chans, level_layout=layout, lifting_dim=8, n_levels=3,
        channel_multipliers=[1, 2, 4], num_heads=2, groups=2, **kw,
    )


def test_dandelion2p5d_default_level_schedule():
    from dandelion.models.dandelion2p5d import DenseLevelMix, default_level_schedule

    assert default_level_schedule(9, 4) == [9, 5, 3, 2]   # dry-atmosphere-3D
    assert default_level_schedule(7, 4) == [7, 4, 2, 1]   # global-ocean-3D, 6 levels
    model = _p5d(16, [3, 1, 3, 3])                         # 4 levels -> [4, 2, 1]
    assert model.model.level_schedule == [4, 2, 1]
    mixes = {m.levels for m in model.modules() if isinstance(m, DenseLevelMix)}
    assert mixes == {4, 2, 1}


def test_dandelion2p5d_constant_schedule_keeps_all_levels():
    from dandelion.models.dandelion2p5d import DenseLevelMix, LevelResample

    model = _p5d(16, [3, 1, 3, 3], level_schedule=[4, 4, 4])
    assert not any(isinstance(m, LevelResample) for m in model.modules())
    assert {m.levels for m in model.modules() if isinstance(m, DenseLevelMix)} == {4}
    with pytest.raises(ValueError):
        _p5d(16, [3, 1, 3, 3], level_schedule=[3, 2, 1])  # must start at levels + 1


def test_dandelion2p5d_forcing_channels_and_grads():
    """Ocean-style layout: surface field last, input-only forcing channels per step."""
    torch.manual_seed(0)
    layout = [3, 3, 1]                                     # [A x 3, B x 3, surface]
    model = _p5d(16, layout, n_forcing=2)
    x = torch.randn(2, 2 * (7 + 2), 16, 32, requires_grad=True)
    y = model(x)
    assert y.shape == (2, 7, 16, 32)
    y.square().mean().backward()
    # Forcing channels (last 2 of each step) reach the output.
    g = x.grad.reshape(2, 2, 9, 16, 32)
    assert g[:, :, 7:].abs().sum() > 0
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_level_resample_layout():
    from dandelion.models.dandelion2p5d import LevelResample

    C = 3
    rs = LevelResample(C, 9, C, 5)
    with torch.no_grad():
        rs.mix.weight.zero_()
        rs.mix.bias.zero_()
        for c in range(C):
            for d in range(5):
                rs.mix.weight[c * 5 + d, c * 9 + 2 * d, 0, 0] = 1.0
    x = torch.randn(2, C, 9, 4, 6)
    assert torch.equal(rs(x), x[:, :, ::2])
