"""Dandelion2.5D: Dandelion on a stack of vertical levels.

The atmospheric levels and the surface are stacked into one
``(B, C, D+1, H, W)`` volume (surface = bottom slice). Every FlowerBlock warps
each level independently with the tangent-plane warp
(:class:`~dandelion.models.warps.TangentSelfWarp2DPerLevel`) and then couples
all (channel, level) pairs with a dense 1x1 mix over the folded ``C*D``
channels. Resolution changes are the SHT 2x down/up of Dandelion, applied per
level. All other convolutions are 1x1x1.

The U-Net is also a U-Net in the vertical: ``level_schedule`` sets the number
of stacked levels at each stage (default: halve per stage, rounding up, e.g.
9 -> 5 -> 3 -> 2). The down/up blocks change the level count with a dense
fold-to-channels conv (:class:`LevelResample`), so the ``(C*D)^2`` level mixes
stay cheap at the wide, coarse stages. A constant schedule (e.g. ``[9, 9, 9,
9]``) keeps every level at every stage.

:class:`Dandelion2p5D` takes the same folded ``(B, T*C, H, W)`` input as the
2-D models (levels folded into channels, e.g. ``dry-atmosphere-3D``) and splits
it into surface and atmospheric fields with ``level_layout``.
"""

import math

import torch
import torch.nn as nn

import torch_harmonics as th

from dandelion.models.dandelion import make_lat_lon_grid, sphere_coord_grid
from dandelion.models.warps import TangentSelfWarp2DPerLevel


def make_norm3d(norm_type: str, num_channels: int, groups: int = 32) -> nn.Module:
    if norm_type == "none":
        return nn.Identity()
    if norm_type == "group":
        return nn.GroupNorm(num_groups=groups, num_channels=num_channels, affine=True)
    raise ValueError(f"Unknown norm_type={norm_type!r}; expected one of: group, none")


def default_level_schedule(levels: int, n_stages: int) -> list[int]:
    """Halve the number of levels per U-Net stage, rounding up: 9 -> 5 -> 3 -> 2,
    7 -> 4 -> 2 -> 1. For ``levels = 2^k + 1`` both end levels (top and
    surface) stay aligned at every stage."""
    schedule = [levels]
    for _ in range(n_stages - 1):
        schedule.append(max(1, math.ceil(schedule[-1] / 2)))
    return schedule


# =============================================================================
# Dense level mix / level resampling + FlowerBlock
# =============================================================================

class DenseLevelMix(nn.Module):
    """Joint channel+level mix: one 1x1 ``Conv2d`` over the ``C*D`` channels of
    the level-folded volume, so every (channel, level) pair mixes with every
    other."""

    def __init__(self, channels, levels, **kwargs):
        super().__init__()
        self.channels = channels
        self.levels = levels
        self.mix = nn.Conv2d(channels * levels, channels * levels, kernel_size=1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        z = self.mix(x.reshape(B, C * D, H, W))
        return z.reshape(B, C, D, H, W)


class LevelResample(nn.Module):
    """Dense channel+level change ``(B, c_in, d_in, H, W) -> (B, c_out, d_out, H, W)``:
    one 1x1 ``Conv2d`` from the folded ``c_in*d_in`` to ``c_out*d_out`` channels,
    so every output (channel, level) is learned from every input one."""

    def __init__(self, c_in, d_in, c_out, d_out):
        super().__init__()
        self.c_out, self.d_out = c_out, d_out
        self.mix = nn.Conv2d(c_in * d_in, c_out * d_out, kernel_size=1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        z = self.mix(x.reshape(B, C * D, H, W))
        return z.reshape(B, self.c_out, self.d_out, H, W)


def channel_proj(dim_in, dim_out, levels_in, levels_out):
    """1x1x1 channel conv; a dense :class:`LevelResample` if the level count changes."""
    if levels_in == levels_out:
        return nn.Conv3d(dim_in, dim_out, kernel_size=1)
    return LevelResample(dim_in, levels_in, dim_out, levels_out)


class FlowerBlock(nn.Module):
    """Pre-norm per-level warp + dense level mix. Returns f(norm(x)); caller adds + x."""

    def __init__(self, channels, spatial_resolution, num_heads,
                 lat_lon_grid, lat_top, lat_range, levels,
                 groups=32, dropout_rate=0.0):
        super().__init__()
        self.norm = make_norm3d("group", channels, groups=groups)
        self.warp = TangentSelfWarp2DPerLevel(
            channels, channels, spatial_resolution, num_heads,
            lat_lon_grid, lat_top, lat_range,
        )
        self.w = DenseLevelMix(channels, levels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x, meta=None):
        out = self.norm(x)
        out = self.w(self.warp(out))
        out = self.act(out)
        out = self.drop(out)
        return out


# =============================================================================
# Spectral 2x down / up, per level (+ level-count change)
# =============================================================================

class Spectral2xDown(nn.Module):
    """SHT-based 2x downsampler on S², applied to every level; the projection
    then maps ``levels_in`` to ``levels_out`` levels."""

    def __init__(self, dim_in, dim_out, high_res, low_res, levels_in, levels_out):
        super().__init__()
        H_hi, W_hi = high_res
        H_lo, W_lo = low_res
        self.H_lo, self.W_lo = H_lo, W_lo
        lmax = H_lo
        mmax = W_lo // 2 + 1
        self.sht_hi = th.RealSHT(H_hi, W_hi, lmax=lmax, mmax=mmax, grid="equiangular").float()
        self.isht_lo = th.InverseRealSHT(H_lo, W_lo, lmax=lmax, mmax=mmax, grid="equiangular").float()
        self.proj = nn.Sequential(
            channel_proj(dim_in, dim_out, levels_in, levels_out),
            nn.Identity(),
            nn.GELU(),
        )

    def forward(self, x):
        # Fold (B, D) into the batch so the SHT is a 2-D transform over (lat, lon).
        B, C, D, H, W = x.shape
        dtype = x.dtype
        xf = x.float().permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        # The SHT uses complex ops without bf16 support: keep it in fp32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = self.isht_lo(self.sht_hi(xf)).reshape(B, D, C, self.H_lo, self.W_lo)
        y = y.permute(0, 2, 1, 3, 4).to(dtype)
        return self.proj(y)


class Spectral2xUp(nn.Module):
    """SHT-based 2x upsampler on S², applied to every level; the input
    projection first maps ``levels_in`` to ``levels_out`` levels (at the
    coarse resolution, where it is cheap)."""

    def __init__(self, dim_in, dim_out, low_res, high_res, levels_in, levels_out):
        super().__init__()
        H_lo, W_lo = low_res
        H_hi, W_hi = high_res
        self.H_hi, self.W_hi = H_hi, W_hi
        lmax = H_lo
        mmax = W_lo // 2 + 1
        self.pre = nn.Sequential(
            channel_proj(dim_in, dim_out, levels_in, levels_out),
            nn.Identity(),
            nn.GELU(),
        )
        self.sht_lo = th.RealSHT(H_lo, W_lo, lmax=lmax, mmax=mmax, grid="equiangular").float()
        self.isht_hi = th.InverseRealSHT(H_hi, W_hi, lmax=lmax, mmax=mmax, grid="equiangular").float()
        self.post = nn.Conv3d(dim_out, dim_out, kernel_size=1)

    def forward(self, x):
        x = self.pre(x)
        B, C, D, H, W = x.shape
        dtype = x.dtype
        xf = x.float().permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = self.isht_hi(self.sht_lo(xf)).reshape(B, D, C, self.H_hi, self.W_hi)
        y = y.permute(0, 2, 1, 3, 4).to(dtype)
        return self.post(y)


class SpectralDownBlock(nn.Module):
    """Encoder stage: FlowerBlock at ``levels``, then spectral 2x down to ``levels_out``."""

    def __init__(self, in_channels, out_channels, spatial_resolution, low_res,
                 num_heads, groups, dropout_rate, lat_lon_grid, lat_top, lat_range,
                 levels, levels_out):
        super().__init__()
        self.flower = FlowerBlock(
            in_channels, spatial_resolution, num_heads,
            lat_lon_grid, lat_top, lat_range, levels,
            groups=groups, dropout_rate=dropout_rate,
        )
        self.spectral_down = Spectral2xDown(
            in_channels, out_channels,
            high_res=tuple(spatial_resolution), low_res=tuple(low_res),
            levels_in=levels, levels_out=levels_out,
        )

    def forward(self, x, meta=None):
        x = self.flower(x, meta) + x
        return x, self.spectral_down(x)


class SpectralUpBlock(nn.Module):
    """Decoder stage: spectral 2x up from ``levels_in`` to ``levels``,
    concatenate the skip, FlowerBlock at ``levels``."""

    def __init__(self, in_channels, splat_out_channels, total_channels,
                 out_spatial_resolution, low_res, num_heads, groups, dropout_rate,
                 lat_lon_grid, lat_top, lat_range, levels_in, levels):
        super().__init__()
        self.spectral_up = Spectral2xUp(
            in_channels, splat_out_channels,
            low_res=tuple(low_res), high_res=tuple(out_spatial_resolution),
            levels_in=levels_in, levels_out=levels,
        )
        self.flower = FlowerBlock(
            total_channels, out_spatial_resolution, num_heads,
            lat_lon_grid, lat_top, lat_range, levels,
            groups=groups, dropout_rate=dropout_rate,
        )

    def forward(self, x, skip, meta=None):
        x = torch.cat([self.spectral_up(x), skip], dim=1)
        return self.flower(x, meta) + x


# =============================================================================
# U-Net on the stacked (atmosphere + surface) volume
# =============================================================================

class FlowerUNet(nn.Module):
    """U-Net with SHT resolution changes, per-level tangent warps and a
    per-stage number of levels (``level_schedule``)."""

    def __init__(self, dim_in_surface, dim_in_atmos, dim_out_surface, dim_out_atmos,
                 n_atmos_levels, spatial_resolution, lifting_dim=128, n_levels=4,
                 channel_multipliers=None, num_heads=32, groups=32,
                 dropout_rate=0.0, coord_dim=3, drop_south_pole=False,
                 bottleneck_blocks=2, level_schedule=None):
        super().__init__()
        if channel_multipliers is None:
            channel_multipliers = [2**i for i in range(n_levels)]
        assert len(channel_multipliers) == n_levels
        encoder_channels = [lifting_dim * m for m in channel_multipliers]

        depth = n_atmos_levels + 1  # + surface slice
        if level_schedule is None:
            level_schedule = default_level_schedule(depth, n_levels)
        level_schedule = [int(d) for d in level_schedule]
        if len(level_schedule) != n_levels or level_schedule[0] != depth:
            raise ValueError(
                f"level_schedule {level_schedule} must have n_levels={n_levels} "
                f"entries starting with n_atmos_levels + 1 = {depth}"
            )
        if any(d < 1 for d in level_schedule):
            raise ValueError(f"level_schedule {level_schedule}: entries must be >= 1")
        self.level_schedule = level_schedule

        min_divisor = 2 ** (n_levels - 1)
        for i, d in enumerate(spatial_resolution):
            assert d % min_divisor == 0, (
                f"Dimension {i} ({d}) must be divisible by {min_divisor}"
            )
        schedule = [tuple(spatial_resolution)]
        for _ in range(n_levels - 1):
            schedule.append(tuple(d // 2 for d in schedule[-1]))

        def grid(res):
            lat_lon = make_lat_lon_grid(*res, drop_south_pole=drop_south_pole)
            lat_top = float(lat_lon[0, 0, 0, 1])
            lat_range = lat_top - float(lat_lon[0, -1, 0, 1])
            return lat_lon, lat_top, lat_range

        # Inputs get the (x, y, z) unit-sphere position and a level coordinate,
        # so every level (and the surface) has an explicit identity.
        self.lift_surface = nn.Conv2d(dim_in_surface + coord_dim + 1, encoder_channels[0], kernel_size=1)
        self.lift_atmos = nn.Conv3d(dim_in_atmos + coord_dim + 1, encoder_channels[0], kernel_size=1)
        self.register_buffer(
            "coord_grid", sphere_coord_grid(*spatial_resolution, drop_south_pole=drop_south_pole)
        )

        self.encoder_blocks = nn.ModuleList()
        for i in range(n_levels - 1):
            self.encoder_blocks.append(SpectralDownBlock(
                encoder_channels[i], encoder_channels[i + 1], schedule[i], schedule[i + 1],
                num_heads, groups, dropout_rate, *grid(schedule[i]),
                level_schedule[i], level_schedule[i + 1],
            ))

        bottleneck_ch = encoder_channels[-1]
        self.bottleneck = nn.ModuleList([
            FlowerBlock(bottleneck_ch, schedule[-1], num_heads, *grid(schedule[-1]),
                        level_schedule[-1], groups=groups, dropout_rate=dropout_rate)
            for _ in range(bottleneck_blocks)
        ])

        self.decoder_blocks = nn.ModuleList()
        decoder_in_ch = bottleneck_ch
        for i in range(n_levels - 1):
            encoder_idx = n_levels - 2 - i
            skip_ch = encoder_channels[encoder_idx]
            total_ch = 2 * skip_ch
            self.decoder_blocks.append(SpectralUpBlock(
                decoder_in_ch, skip_ch, total_ch,
                schedule[encoder_idx], schedule[encoder_idx + 1],
                num_heads, groups, dropout_rate, *grid(schedule[encoder_idx]),
                level_schedule[encoder_idx + 1], level_schedule[encoder_idx],
            ))
            decoder_in_ch = total_ch

        self.project_surface = nn.Sequential(
            nn.Conv2d(decoder_in_ch, lifting_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(lifting_dim, dim_out_surface, kernel_size=1),
        )
        self.project_atmos = nn.Sequential(
            nn.Conv3d(decoder_in_ch, lifting_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(lifting_dim, dim_out_atmos, kernel_size=1),
        )

    def forward(self, x_surface, x_atmos, meta=None):
        B = x_surface.shape[0]
        D = x_atmos.shape[2]
        H, W = x_surface.shape[-2:]
        dev, dt = x_atmos.device, x_atmos.dtype

        surf_coord = self.coord_grid.expand(B, -1, -1, -1)
        atmos_coord = self.coord_grid.unsqueeze(2).expand(B, -1, D, -1, -1)
        # Level coordinate over the stacked volume: atmosphere first, surface last.
        zcoord = torch.linspace(-1.0, 1.0, D + 1, device=dev, dtype=dt)
        atmos_z = zcoord[:D].view(1, 1, D, 1, 1).expand(B, 1, D, H, W)
        surf_z = zcoord[D].view(1, 1, 1, 1).expand(B, 1, H, W)

        x_surface = self.lift_surface(torch.cat([x_surface, surf_coord, surf_z], dim=1))
        x_atmos = self.lift_atmos(torch.cat([x_atmos, atmos_coord, atmos_z], dim=1))
        x = torch.cat([x_atmos, x_surface.unsqueeze(2)], dim=2)  # (B, C0, D+1, H, W)

        skips = []
        for block in self.encoder_blocks:
            skip, x = block(x, meta)
            skips.append(skip)
        for block in self.bottleneck:
            x = block(x, meta) + x
        for block, skip in zip(self.decoder_blocks, reversed(skips)):
            x = block(x, skip, meta)

        return self.project_surface(x[:, :, -1]), self.project_atmos(x[:, :, :-1])


# =============================================================================
# Dandelion2.5D on folded (B, T*C, H, W) inputs
# =============================================================================

class Dandelion2p5D(nn.Module):
    """Dandelion2.5D with the folded 2-D model interface.

    ``level_layout`` lists, in channel order, how many channels each field
    occupies in one folded time step: ``1`` for a surface field, ``D`` for an
    atmospheric field on ``D`` levels. For ``dry-atmosphere-3D``
    (``[T x 8, ps, u x 8, v x 8]``) it is ``[8, 1, 8, 8]``; for
    ``global-ocean-3D`` (``[theta x 6, salt x 6, u x 6, v x 6, eta]``) it is
    ``[6, 6, 6, 6, 1]``. ``n_forcing`` input-only channels may follow each input
    step (``global-ocean-3D``: depth, sin/cos day of year -> 3); they are fed to
    the surface slice and broadcast to every level. Inputs are
    ``(B, T_in*(C + n_forcing), H, W)`` with time-major channels, outputs
    ``(B, C, H, W)`` in the ``level_layout`` channel order.

    ``level_schedule``: levels per U-Net stage (first entry = atmospheric levels
    + 1); ``None`` halves per stage (see :func:`default_level_schedule`).
    """

    def __init__(self, inp_shape, inp_chans, out_chans, level_layout,
                 lifting_dim=70, n_levels=4, channel_multipliers=None,
                 num_heads=35, groups=35, dropout_rate=0.0, coord_dim=3,
                 level_schedule=None, n_forcing=0, **kwargs):
        super().__init__()
        layout = [int(n) for n in level_layout]
        n_chans = sum(layout)
        n_atmos_levels = max(layout)
        n_forcing = int(n_forcing)
        if any(n not in (1, n_atmos_levels) for n in layout):
            raise ValueError(f"level_layout {layout}: fields must span 1 or {n_atmos_levels} levels")
        if out_chans != n_chans or inp_chans % (n_chans + n_forcing):
            raise ValueError(
                f"inp_chans={inp_chans} / out_chans={out_chans} do not match "
                f"level_layout {layout} ({n_chans} channels per step) + "
                f"n_forcing={n_forcing}"
            )
        self.n_chans = n_chans
        self.n_forcing = n_forcing
        self.n_steps = inp_chans // (n_chans + n_forcing)

        starts = [sum(layout[:i]) for i in range(len(layout))]
        surface_idx = [s for s, n in zip(starts, layout) if n == 1]
        atmos_idx = [list(range(s, s + n)) for s, n in zip(starts, layout) if n > 1]
        self.n_surface, self.n_atmos = len(surface_idx), len(atmos_idx)
        self.register_buffer("surface_idx", torch.tensor(surface_idx), persistent=False)
        self.register_buffer("atmos_idx", torch.tensor(atmos_idx), persistent=False)
        order = torch.cat([self.surface_idx, self.atmos_idx.flatten()])
        self.register_buffer("inverse_order", torch.argsort(order), persistent=False)

        H, W = inp_shape
        divisor = 2 ** (n_levels - 1)
        self.drop_south_pole = False
        if H % divisor != 0:
            assert (H - 1) % divisor == 0, (
                f"H={H}: dropping one row gives H-1={H-1} which is still not "
                f"divisible by {divisor}."
            )
            self.drop_south_pole = True
            H = H - 1
        assert W % divisor == 0, f"Width {W} must be divisible by {divisor}"

        self.model = FlowerUNet(
            dim_in_surface=(self.n_surface + n_forcing) * self.n_steps,
            dim_in_atmos=(self.n_atmos + n_forcing) * self.n_steps,
            dim_out_surface=self.n_surface,
            dim_out_atmos=self.n_atmos,
            n_atmos_levels=n_atmos_levels,
            spatial_resolution=(H, W),
            lifting_dim=lifting_dim,
            n_levels=n_levels,
            channel_multipliers=channel_multipliers,
            num_heads=num_heads,
            groups=groups,
            dropout_rate=dropout_rate,
            coord_dim=coord_dim,
            drop_south_pole=self.drop_south_pole,
            level_schedule=level_schedule,
        )

    def forward(self, x, meta=None):
        B, _, H, W = x.shape
        x = x.reshape(B, self.n_steps, self.n_chans + self.n_forcing, H, W)
        forcing = x[:, :, self.n_chans:]                       # (B, T, F, H, W)
        x = x[:, :, :self.n_chans]
        # (B, T, C, H, W) -> surface (B, T*(Cs+F), H, W), atmosphere (B, T*(Ca+F), D, H, W)
        x_surface = torch.cat([x[:, :, self.surface_idx], forcing], dim=2).reshape(B, -1, H, W)
        x_atmos = x[:, :, self.atmos_idx]                      # (B, T, Ca, D, H, W)
        D = x_atmos.shape[3]
        f_atmos = forcing.unsqueeze(3).expand(-1, -1, -1, D, -1, -1)
        x_atmos = torch.cat([x_atmos, f_atmos], dim=2).reshape(B, -1, D, H, W)

        if self.drop_south_pole:
            x_surface = x_surface[..., :-1, :]
            x_atmos = x_atmos[..., :-1, :]

        y_surface, y_atmos = self.model(x_surface, x_atmos, meta)
        y = torch.cat([y_surface, y_atmos.flatten(1, 2)], dim=1)

        if self.drop_south_pole:
            south_pole = y[:, :, -1:, :].mean(dim=3, keepdim=True).expand(-1, -1, -1, W)
            y = torch.cat([y, south_pole], dim=2)

        return y[:, self.inverse_order]
