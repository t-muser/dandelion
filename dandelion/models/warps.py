"""Tangent-plane warp on the sphere, used by Dandelion's FlowerBlocks.

- ``grid_sample_periodic_lon``: grid_sample with periodic longitude and
  border-clamped latitude.
- ``TangentSelfWarp``: flow in the local tangent plane, projected via Rodrigues.
- ``TangentSelfWarp2DPerLevel``: the same warp per level of a 3-D volume
  (Dandelion2.5D).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


FLOW_FOLD_HEADS = 'B (heads dir) H W -> (B heads) H W dir'
VALUE_FOLD_HEADS = 'B (heads C_i) H W -> (B heads) C_i H W'
VALUE_UNFOLD_HEADS = '(B heads) C_i H W -> B (heads C_i) H W'


@torch.compile
def grid_sample_periodic_lon(input, grid):
    """Bilinear ``grid_sample`` (align_corners=True) that wraps in W.

    The first longitude column is appended on the right so samples between the
    last and first column interpolate across the seam; latitude is clamped
    with ``border`` padding.
    """
    H, W = input.shape[2], input.shape[3]
    padded_input = torch.cat([input, input[:, :, :, 0:1]], dim=3)

    gx, gy = grid[..., 0], grid[..., 1]
    px = (gx + 1) * (W - 1) / 2
    py = (gy + 1) * (H - 1) / 2
    gx_new = 2 * (px % W) / W - 1
    gy_new = 2 * py / (H - 1) - 1 if H > 1 else gy

    return F.grid_sample(
        padded_input, torch.stack([gx_new, gy_new], dim=-1), mode="bilinear",
        padding_mode="border", align_corners=True,
    )


def _lat_to_gy(lat, lat_asc, gy_asc):
    """Monotonic lat → gy lookup."""
    idx = torch.searchsorted(lat_asc, lat, right=False).clamp(1, lat_asc.numel() - 1)
    lo = lat_asc[idx - 1]
    hi = lat_asc[idx]
    g_lo = gy_asc[idx - 1]
    g_hi = gy_asc[idx]
    t = (lat - lo) / (hi - lo).clamp_min(1e-12)
    return g_lo + t * (g_hi - g_lo)


@torch.compile
def _tangent_warp_coords(du, dv, e_east, e_north, base_pos_3d, lat_asc, gy_asc):
    """Rodrigues rotation + spherical→grid_sample conversion."""
    d_3d = du * e_east + dv * e_north
    theta = torch.sqrt(du ** 2 + dv ** 2 + 1e-12)
    sinc_t = torch.sin(theta) / theta
    pos_3d_new = base_pos_3d * torch.cos(theta) + d_3d * sinc_t

    r = torch.sqrt(pos_3d_new[..., 0] ** 2 + pos_3d_new[..., 1] ** 2)
    lat_new = torch.atan2(pos_3d_new[..., 2], r)
    lon_new = torch.atan2(pos_3d_new[..., 1], pos_3d_new[..., 0]) % (2 * math.pi)

    gx = lon_new / math.pi - 1
    gy = _lat_to_gy(lat_new, lat_asc, gy_asc)
    return torch.stack([gx, gy], dim=-1)


# Angular radius (radians) of the polar cap over which the longitude gradient is
# damped in _tangent_warp_coords_stable.
POLE_EPS = 0.05


@torch.compile
def _tangent_warp_coords_stable(du, dv, e_east, e_north, base_pos_3d, lat_asc, gy_asc):
    """:func:`_tangent_warp_coords` with stabilized gradients (Dandelion2.5D).

    Same forward; the backward is stable as theta -> 0 and damped at the poles.
    """
    d_3d = du * e_east + dv * e_north
    theta = torch.sqrt(du ** 2 + dv ** 2 + 1e-12)
    # Numerically stable sinc(theta) = sin(theta)/theta. The forward quotient is
    # fine, but autograd's derivative of it cancels as t->0 so we use a Taylor
    # series 1 - t^2/6 + t^4/120.
    small = theta < 1e-2
    theta_safe = torch.where(small, torch.ones_like(theta), theta)
    sinc_t = torch.where(
        small,
        1 - theta ** 2 / 6 + theta ** 4 / 120,
        torch.sin(theta) / theta_safe,
    )
    pos_3d_new = base_pos_3d * torch.cos(theta) + d_3d * sinc_t

    r = torch.sqrt(pos_3d_new[..., 0] ** 2 + pos_3d_new[..., 1] ** 2)
    lat_new = torch.atan2(pos_3d_new[..., 2], r)

    # Prevent numerical issues. forward is unchanged, but backward scales the longitude
    # gradient by r^2/(r^2+eps^2) so near-pole pixels don't dominate the grad clipping
    _lon_raw = torch.atan2(pos_3d_new[..., 1], pos_3d_new[..., 0])
    _lon_scale = (r ** 2 / (r ** 2 + POLE_EPS ** 2)).detach()
    lon_new = (_lon_raw.detach() + _lon_scale * (_lon_raw - _lon_raw.detach())) % (2 * math.pi)

    gx = lon_new / math.pi - 1
    gy = _lat_to_gy(lat_new, lat_asc, gy_asc)
    return torch.stack([gx, gy], dim=-1)


class TangentSelfWarp(nn.Module):
    """SelfWarp with flow in the local tangent plane, projected via 3D Rodrigues.

    Displacement heads predict (du, dv) = (East, North) shifts. These are
    projected into 3D Cartesian space using precomputed tangent basis vectors,
    added to the 3D grid positions, and normalised back onto the unit sphere.
    Pole crossing is handled naturally by the 3D projection.
    """

    def __init__(self, in_channels, out_channels, spatial_resolution, num_heads,
                 lat_lon_grid, lat_top, lat_range, **kwargs):
        super().__init__()
        self.num_heads = num_heads

        self.flow_head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 2 * num_heads, kernel_size=1),
        )
        self.value_head = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # lat_lon_grid: (1, H_p, W_p, 2) with [..., 0]=lon, [..., 1]=lat in radians
        lon = lat_lon_grid[..., 0]
        lat = lat_lon_grid[..., 1]

        base_pos_3d = torch.stack([
            torch.cos(lat) * torch.cos(lon),
            torch.cos(lat) * torch.sin(lon),
            torch.sin(lat),
        ], dim=-1)
        self.register_buffer("base_pos_3d", base_pos_3d)

        e_east = torch.stack([
            -torch.sin(lon),
             torch.cos(lon),
             torch.zeros_like(lon),
        ], dim=-1)
        self.register_buffer("e_east", e_east)

        e_north = torch.stack([
            -torch.sin(lat) * torch.cos(lon),
            -torch.sin(lat) * torch.sin(lon),
             torch.cos(lat),
        ], dim=-1)
        self.register_buffer("e_north", e_north)

        # Per-row lat → gy lookup (ascending-lat order). For equiangular grids
        # this reduces to gy = 2*(lat_top - lat)/lat_range - 1 to FP precision;
        # for Legendre–Gauss nodes it gives correct per-row coordinates.
        lat_nodes_desc = lat_lon_grid[0, :, 0, 1].contiguous().float()
        H_rows = lat_nodes_desc.numel()
        gy_nodes_desc = torch.linspace(-1.0, 1.0, H_rows, dtype=torch.float32)
        lat_asc = lat_nodes_desc.flip(0).contiguous()
        gy_asc = gy_nodes_desc.flip(0).contiguous()
        self.register_buffer("lat_asc", lat_asc)
        self.register_buffer("gy_asc", gy_asc)

    def forward(self, u, meta=None):
        flow  = self.flow_head(u)
        value = self.value_head(u)

        flow = rearrange(flow, FLOW_FOLD_HEADS, dir=2, heads=self.num_heads)

        du = flow[..., 0:1]
        dv = flow[..., 1:2]

        value = rearrange(value, VALUE_FOLD_HEADS, heads=self.num_heads)

        grid = _tangent_warp_coords(du, dv, self.e_east, self.e_north,
                                    self.base_pos_3d, self.lat_asc, self.gy_asc)
        u_warp = grid_sample_periodic_lon(value, grid)

        return rearrange(u_warp, VALUE_UNFOLD_HEADS, heads=self.num_heads)


PL_FLOW_FOLD = 'B (heads dir) D H W -> (B heads D) H W dir'
PL_VALUE_FOLD = 'B (heads C_i) D H W -> (B heads D) C_i H W'
PL_VALUE_UNFOLD = '(B heads D) C_i H W -> B (heads C_i) D H W'


class TangentSelfWarp2DPerLevel(TangentSelfWarp):
    """:class:`TangentSelfWarp` applied independently to every level of a
    ``(B, C, D, H, W)`` volume (Dandelion2.5D).

    You probably want another way for the model to handle vertical mixing if you
    use this warp.
    """

    def __init__(self, in_channels, out_channels, spatial_resolution, num_heads,
                 lat_lon_grid, lat_top, lat_range, **kwargs):
        super().__init__(in_channels, out_channels, spatial_resolution, num_heads,
                         lat_lon_grid, lat_top, lat_range)
        self.flow_head = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, 2 * num_heads, kernel_size=1),
        )
        self.value_head = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, u, meta=None):
        D = u.shape[2]
        flow = self.flow_head(u)
        value = self.value_head(u)

        flow = rearrange(flow, PL_FLOW_FOLD, dir=2, heads=self.num_heads)
        du = flow[..., 0:1]
        dv = flow[..., 1:2]
        value = rearrange(value, PL_VALUE_FOLD, heads=self.num_heads)

        grid = _tangent_warp_coords_stable(du, dv, self.e_east, self.e_north,
                                    self.base_pos_3d, self.lat_asc, self.gy_asc)
        u_warp = grid_sample_periodic_lon(value, grid)

        return rearrange(u_warp, PL_VALUE_UNFOLD, heads=self.num_heads, D=D)
