"""Dandelion: warp-based U-Net neural PDE solver on the sphere.

Equiangular internal grid throughout; FlowerBlocks use TangentSelfWarp (exact
Rodrigues rotation on S²); resolution changes use SHT-based spectral 2x
down/upsampling with no normalization on the resampling projections. If the
grid height is not divisible by 2^(n_levels-1), the south-pole row is dropped
on entry and reconstructed (zonal mean of row H-1) on egress.

This module is self-contained: the only external primitive it relies on is
``TangentSelfWarp`` from ``dandelion.models.warps``.
"""

import math

import torch
import torch.nn as nn

import torch_harmonics as th

from dandelion.models.warps import TangentSelfWarp


# =============================================================================
# Normalization
# =============================================================================

class LayerNorm2d(nn.Module):
    """LayerNorm across channels for (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class RMSNorm2d(nn.Module):
    """RMSNorm across channels for (B, C, H, W) — no mean subtraction."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms.to(x.dtype)) * self.weight.view(1, -1, 1, 1)


def make_norm2d(norm_type: str, num_channels: int, groups: int = 32) -> nn.Module:
    if norm_type == "none":
        return nn.Identity()
    if norm_type == "group":
        return nn.GroupNorm(num_groups=groups, num_channels=num_channels, affine=True)
    if norm_type == "layer":
        return LayerNorm2d(num_channels)
    if norm_type == "rms":
        return RMSNorm2d(num_channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True, track_running_stats=False)
    raise ValueError(
        f"Unknown norm_type={norm_type!r}; "
        f"expected one of: group, layer, rms, instance, none"
    )


# =============================================================================
# Lat/lon grid helpers (equiangular)
# =============================================================================

def _grid_lats(H, drop_south_pole: bool = False):
    """Latitude nodes (radians, descending from +π/2 toward -π/2)."""
    if drop_south_pole:
        true_H = H + 1
        delta = math.pi / (true_H - 1)
        return (math.pi / 2) - torch.arange(H, dtype=torch.float32) * delta
    return torch.linspace(math.pi / 2, -math.pi / 2, H)


def make_lat_lon_grid(H, W, drop_south_pole: bool = False):
    """(1, H, W, 2) with [..., 0]=lon, [..., 1]=lat (radians)."""
    lat = _grid_lats(H, drop_south_pole=drop_south_pole)
    lon = torch.linspace(0, 2 * math.pi * (1 - 1 / W), W)
    lat_g = lat[:, None].expand(H, W)
    lon_g = lon[None, :].expand(H, W)
    return torch.stack([lon_g, lat_g], dim=-1).unsqueeze(0).contiguous()


def sphere_coord_grid(H, W, drop_south_pole: bool = False):
    """3D Cartesian unit-sphere embedding: (1, 3, H, W)."""
    lat = _grid_lats(H, drop_south_pole=drop_south_pole)
    lon = torch.linspace(0, 2 * math.pi * (1 - 1 / W), W)
    cos_lat, sin_lat = torch.cos(lat), torch.sin(lat)
    cos_lon, sin_lon = torch.cos(lon), torch.sin(lon)
    x = cos_lat[:, None] * cos_lon[None, :]
    y = cos_lat[:, None] * sin_lon[None, :]
    z = sin_lat[:, None].expand(-1, W)
    return torch.stack([x, y, z], dim=0).unsqueeze(0).contiguous()


# =============================================================================
# FlowerBlock — pre-norm residual block with a tangent-plane spherical warp
# =============================================================================

class FlowerBlock(nn.Module):
    """Pre-norm TangentSelfWarp block. Returns f(norm(x)); caller adds + x."""

    def __init__(self, channels, spatial_resolution, num_heads,
                 lat_lon_grid, lat_top, lat_range,
                 groups=32, dropout_rate=0.0, norm_type: str = "group"):
        super().__init__()
        self.norm = make_norm2d(norm_type, channels, groups=groups)
        self.warp = TangentSelfWarp(
            channels, channels, spatial_resolution, num_heads,
            lat_lon_grid, lat_top, lat_range,
        )
        self.w = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x, meta=None):
        out = self.norm(x)
        out = self.w(self.warp(out))
        out = self.act(out)
        out = self.drop(out)
        return out


# =============================================================================
# Spectral 2x down / up — spherical-harmonic resolution change (no resampling norm)
# =============================================================================

class Spectral2xDown(nn.Module):
    """SHT-based 2x downsampler on S². Exact up to quadrature, pole-aware."""

    def __init__(self, dim_in, dim_out, high_res, low_res):
        super().__init__()
        H_hi, W_hi = high_res
        H_lo, W_lo = low_res

        lmax = H_lo
        mmax = W_lo // 2 + 1
        self.sht_hi = th.RealSHT(H_hi, W_hi, lmax=lmax, mmax=mmax, grid="equiangular").float()
        self.isht_lo = th.InverseRealSHT(H_lo, W_lo, lmax=lmax, mmax=mmax, grid="equiangular").float()

        self.proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x):
        dtype = x.dtype
        coef = self.sht_hi(x.float())
        y = self.isht_lo(coef).to(dtype)
        return self.proj(y)


class Spectral2xUp(nn.Module):
    """SHT-based 2x upsampler on S²."""

    def __init__(self, dim_in, dim_out, low_res, high_res):
        super().__init__()
        H_lo, W_lo = low_res
        H_hi, W_hi = high_res

        lmax = H_lo
        mmax = W_lo // 2 + 1
        self.pre = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1),
            nn.GELU(),
        )
        self.sht_lo = th.RealSHT(H_lo, W_lo, lmax=lmax, mmax=mmax, grid="equiangular").float()
        self.isht_hi = th.InverseRealSHT(H_hi, W_hi, lmax=lmax, mmax=mmax, grid="equiangular").float()
        self.post = nn.Conv2d(dim_out, dim_out, kernel_size=1)

    def forward(self, x):
        x = self.pre(x)
        dtype = x.dtype
        coef = self.sht_lo(x.float())
        y = self.isht_hi(coef).to(dtype)
        return self.post(y)


# =============================================================================
# Encoder / Decoder stages
# =============================================================================

class SpectralDownBlock(nn.Module):
    """Encoder stage: FlowerBlock, then Spectral 2x down."""

    def __init__(self, in_channels, out_channels, spatial_resolution,
                 num_heads, groups, dropout_rate,
                 lat_lon_grid, lat_top, lat_range, low_res,
                 norm_type: str = "group"):
        super().__init__()
        self.flower = FlowerBlock(
            in_channels, spatial_resolution, num_heads,
            lat_lon_grid, lat_top, lat_range,
            groups=groups, dropout_rate=dropout_rate, norm_type=norm_type,
        )
        self.spectral_down = Spectral2xDown(
            in_channels, out_channels,
            high_res=tuple(spatial_resolution), low_res=tuple(low_res),
        )

    def forward(self, x, meta=None):
        x = self.flower(x, meta) + x
        skip = x
        x_down = self.spectral_down(x)
        return skip, x_down


class SpectralUpBlock(nn.Module):
    """Decoder stage: Spectral 2x up, cat skip, FlowerBlock."""

    def __init__(self, in_channels, splat_out_channels, total_channels,
                 out_spatial_resolution, num_heads, groups, dropout_rate,
                 lat_lon_grid, lat_top, lat_range, low_res,
                 norm_type: str = "group"):
        super().__init__()
        self.spectral_up = Spectral2xUp(
            in_channels, splat_out_channels,
            low_res=tuple(low_res), high_res=tuple(out_spatial_resolution),
        )
        self.flower = FlowerBlock(
            total_channels, out_spatial_resolution, num_heads,
            lat_lon_grid, lat_top, lat_range,
            groups=groups, dropout_rate=dropout_rate, norm_type=norm_type,
        )

    def forward(self, x, skip, meta=None):
        x = self.spectral_up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.flower(x, meta) + x
        return x


# =============================================================================
# FlowerUNet — spectral U-Net scaffold on an equiangular grid
# =============================================================================

class FlowerUNet(nn.Module):
    """U-Net with SHT spectral resolution-change blocks and tangent warps."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        spatial_resolution: tuple[int, int],
        lifting_dim: int = 128,
        n_levels: int = 4,
        channel_multipliers: list[int] | None = None,
        num_heads: int = 32,
        groups: int = 32,
        dropout_rate: float = 0.0,
        coord_dim: int = 3,
        drop_south_pole: bool = False,
        bottleneck_blocks: int = 2,
        blocks_per_stage: int = 1,
        norm_type: str = "group",
    ):
        super().__init__()
        self.norm_type = norm_type

        if channel_multipliers is None:
            channel_multipliers = [2**i for i in range(n_levels)]
        assert len(channel_multipliers) == n_levels

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.spatial_resolution = list(spatial_resolution)
        self.n_levels = n_levels
        self.coord_dim = coord_dim

        encoder_channels = [lifting_dim * m for m in channel_multipliers]

        min_divisor = 2 ** (n_levels - 1)
        for i, d in enumerate(spatial_resolution):
            assert d % min_divisor == 0, (
                f"Dimension {i} ({d}) must be divisible by {min_divisor}"
            )
        schedule = [tuple(spatial_resolution)]
        for _ in range(n_levels - 1):
            schedule.append(tuple(d // 2 for d in schedule[-1]))
        self.resolution_schedule = schedule

        self.lift = nn.Conv2d(dim_in + coord_dim, encoder_channels[0], kernel_size=1)

        coord_grid = sphere_coord_grid(*spatial_resolution,
                                       drop_south_pole=drop_south_pole)
        self.register_buffer("coord_grid", coord_grid)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_extra_blocks = nn.ModuleList()
        current_res = list(schedule[0])
        for i in range(n_levels - 1):
            lat_lon = make_lat_lon_grid(*current_res, drop_south_pole=drop_south_pole)
            lat_top = float(lat_lon[0, 0, 0, 1])
            lat_bot = float(lat_lon[0, -1, 0, 1])
            lat_range = lat_top - lat_bot

            low_res = schedule[i + 1]
            self.encoder_blocks.append(SpectralDownBlock(
                encoder_channels[i], encoder_channels[i + 1], current_res,
                num_heads, groups, dropout_rate,
                lat_lon, lat_top, lat_range, low_res,
                norm_type=norm_type,
            ))
            n_extra = (blocks_per_stage - 1) if i > 0 else 0
            self.encoder_extra_blocks.append(nn.ModuleList([
                FlowerBlock(encoder_channels[i], current_res, num_heads,
                            lat_lon, lat_top, lat_range,
                            groups=groups, dropout_rate=dropout_rate,
                            norm_type=norm_type)
                for _ in range(n_extra)
            ]))
            current_res = list(schedule[i + 1])

        # Bottleneck
        bottleneck_ch = encoder_channels[-1]
        bn_lat_lon = make_lat_lon_grid(*current_res, drop_south_pole=drop_south_pole)
        bn_lat_top = float(bn_lat_lon[0, 0, 0, 1])
        bn_lat_bot = float(bn_lat_lon[0, -1, 0, 1])
        bn_lat_range = bn_lat_top - bn_lat_bot
        self.bottleneck = nn.ModuleList([
            FlowerBlock(bottleneck_ch, current_res, num_heads,
                        bn_lat_lon, bn_lat_top, bn_lat_range,
                        groups=groups, dropout_rate=dropout_rate,
                        norm_type=norm_type)
            for _ in range(bottleneck_blocks)
        ])

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_extra_blocks = nn.ModuleList()
        decoder_in_ch = bottleneck_ch
        for i in range(n_levels - 1):
            encoder_idx = n_levels - 2 - i
            skip_ch = encoder_channels[encoder_idx]
            splat_out_ch = skip_ch
            total_ch = splat_out_ch + skip_ch

            out_res = list(schedule[encoder_idx])
            low_res = tuple(current_res)
            lat_lon = make_lat_lon_grid(*out_res, drop_south_pole=drop_south_pole)
            lat_top = float(lat_lon[0, 0, 0, 1])
            lat_bot = float(lat_lon[0, -1, 0, 1])
            lat_range = lat_top - lat_bot

            self.decoder_blocks.append(SpectralUpBlock(
                decoder_in_ch, splat_out_ch, total_ch,
                out_res, num_heads, groups, dropout_rate,
                lat_lon, lat_top, lat_range, low_res,
                norm_type=norm_type,
            ))
            n_extra = (blocks_per_stage - 1) if i < n_levels - 2 else 0
            self.decoder_extra_blocks.append(nn.ModuleList([
                FlowerBlock(total_ch, out_res, num_heads,
                            lat_lon, lat_top, lat_range,
                            groups=groups, dropout_rate=dropout_rate,
                            norm_type=norm_type)
                for _ in range(n_extra)
            ]))
            decoder_in_ch = total_ch
            current_res = out_res

        self.project = nn.Sequential(
            nn.Conv2d(decoder_in_ch, lifting_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(lifting_dim, dim_out, kernel_size=1),
        )

    def forward(self, x, meta=None):
        B = x.shape[0]

        grid = self.coord_grid.expand(B, -1, -1, -1)
        x = torch.cat([x, grid], dim=1)

        x = self.lift(x)

        skips = []
        for block, extra_blocks in zip(self.encoder_blocks, self.encoder_extra_blocks):
            skip, x = block(x, meta)
            for fb in extra_blocks:
                skip = fb(skip, meta) + skip
            skips.append(skip)

        for block in self.bottleneck:
            x = block(x, meta) + x

        for block, extra_blocks, skip in zip(self.decoder_blocks, self.decoder_extra_blocks, reversed(skips)):
            x = block(x, skip, meta)
            for fb in extra_blocks:
                x = fb(x, meta) + x

        return self.project(x)


# =============================================================================
# Dandelion — equiangular grid, SHT spectral down/up, tangent warp
# =============================================================================

class Dandelion(nn.Module):
    """U-Net on S² with tangent-warp FlowerBlocks and SHT spectral down/up.

    Runs entirely on an equiangular grid. If the grid height is not divisible
    by 2^(n_levels-1), the south-pole row is dropped on entry and reconstructed
    (zonal mean of row H-1) on egress.
    """

    def __init__(self, inp_shape=(121, 240), out_shape=None,
                 inp_chans=69, out_chans=69,
                 lifting_dim=150, n_levels=4,
                 channel_multipliers=None,
                 num_heads=50, groups=50, dropout_rate=0.0,
                 coord_dim=3, norm_type: str = "group",
                 **kwargs):
        super().__init__()

        H, W = inp_shape
        divisor = 2 ** (n_levels - 1)

        self.orig_H = H
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
            dim_in=inp_chans,
            dim_out=out_chans,
            spatial_resolution=(H, W),
            lifting_dim=lifting_dim,
            n_levels=n_levels,
            channel_multipliers=channel_multipliers,
            num_heads=num_heads,
            groups=groups,
            dropout_rate=dropout_rate,
            coord_dim=coord_dim,
            drop_south_pole=self.drop_south_pole,
            norm_type=norm_type,
        )

    def forward(self, x, meta=None):
        if self.drop_south_pole:
            x = x[:, :, :-1, :]

        x = self.model(x, meta)

        if self.drop_south_pole:
            south_pole = x[:, :, -1:, :].mean(dim=3, keepdim=True)
            south_pole = south_pole.expand(-1, -1, -1, x.shape[3])
            x = torch.cat([x, south_pole], dim=2)

        return x
