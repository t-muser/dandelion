"""cs32 cubed-sphere → equiangular lat/lon regridding for global-ocean-3D.

This code was largely written by Claude.

The HDF5 files store the MITgcm ``cs32x15`` output on its native cubed-sphere
grid ``(face=6, y=32, x=32)``. The data loader regrids every window on the fly:

* **Weights**: k-nearest-neighbour inverse-distance weights from the cs32 cell
  centres to the target grid cell centres, computed in 3-D Cartesian
  coordinates on the unit sphere (correct at the poles and the dateline).
* **Vector rotation**: ``(u, v)`` are rotated from face-aligned to geographic
  ``(east, north)`` on the native grid (``angle_cs`` / ``angle_sn``) before the
  scalar regrid.
* **Land imputation**: MITgcm's 0.0 land fill is replaced with the per-level
  wet-cell mean before regridding so it does not bleed into wet cells; land is
  masked out again after regridding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

DEFAULT_NLAT: int = 64

DEFAULT_NLON: int = 128


def _lonlat_to_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Convert (lon, lat) in degrees to unit-sphere Cartesian coords.

    Returns an array with shape ``input_shape + (3,)``. Operating in
    xyz space lets a Euclidean k-NN query stand in for a great-circle
    nearest-neighbour search (correct everywhere including poles, no
    seams at the dateline).
    """
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def _build_target_grid(
    nlat: int, nlon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Equispaced target grid (cell centres) covering the full sphere.

    lat ∈ (-90 + δ/2, 90 − δ/2) with δ = 180/nlat;
    lon ∈ ( -180 + δ/2,  180 − δ/2) with δ = 360/nlon.

    Both 1-D arrays (cell centres, not edges).
    """
    dlat = 180.0 / nlat
    dlon = 360.0 / nlon
    lat = np.linspace(-90.0 + dlat / 2, 90.0 - dlat / 2, nlat)
    lon = np.linspace(-180.0 + dlon / 2, 180.0 - dlon / 2, nlon)
    return lat, lon


@dataclass(frozen=True)
class RegridWeights:
    """Precomputed neighbour indices + weights mapping cs32 → (lat, lon).

    ``indices`` shape: ``(nlat, nlon, k)`` — flat-index into the cs32
    source array (so a 6×32×32 grid is indexed 0..6143 in
    face-major C order).

    ``weights`` shape: ``(nlat, nlon, k)`` — sums to 1 along axis -1.
    For ``method="nearest"`` the weights are ``[1.0]`` and ``k=1``.
    """
    indices: np.ndarray   # int64
    weights: np.ndarray   # float64
    nlat: int
    nlon: int
    method: str

    @property
    def k(self) -> int:
        return int(self.weights.shape[-1])


def build_weights(
    xc_face_j_i: np.ndarray,
    yc_face_j_i: np.ndarray,
    *,
    nlat: int = DEFAULT_NLAT,
    nlon: int = DEFAULT_NLON,
    method: Literal["nearest", "idw"] = "nearest",
    k: int = 4,
) -> RegridWeights:
    """Build the index + weight arrays for the cs32 → lat/lon mapping.

    Parameters
    ----------
    xc_face_j_i, yc_face_j_i
        cs32 cell-centre lon, lat in degrees, shape ``(face=6, j=32, i=32)``.
        These are the ``xc``, ``yc`` arrays in ``grid.h5``.
    nlat, nlon
        Target grid resolution. Default 64 × 128 (~2.8°).
    method
        ``"nearest"`` (default): single-neighbour assignment.
        ``"idw"``: inverse-distance-weighted average over ``k`` nearest.
    k
        Number of neighbours for IDW. Ignored for ``method="nearest"``.

    Returns
    -------
    A :class:`RegridWeights` instance, reusable across variables and
    time steps via :func:`apply_weights`.
    """
    if method not in ("nearest", "idw"):
        raise ValueError(f"Unknown regrid method: {method!r}")

    from scipy.spatial import cKDTree

    src_xc = np.asarray(xc_face_j_i, dtype=np.float64).reshape(-1)
    src_yc = np.asarray(yc_face_j_i, dtype=np.float64).reshape(-1)
    if src_xc.shape != src_yc.shape:
        raise ValueError(
            f"XC and YC must have matching shape (got {src_xc.shape} vs {src_yc.shape})"
        )

    src_xyz = _lonlat_to_xyz(src_xc, src_yc)
    tree = cKDTree(src_xyz)

    lat, lon = _build_target_grid(nlat, nlon)
    lon_grid, lat_grid = np.meshgrid(lon, lat)            # both (nlat, nlon)
    tgt_xyz = _lonlat_to_xyz(lon_grid, lat_grid).reshape(-1, 3)

    if method == "nearest":
        _, idx = tree.query(tgt_xyz, k=1)
        idx = idx.reshape(nlat, nlon, 1).astype(np.int64)
        w = np.ones_like(idx, dtype=np.float64)
        return RegridWeights(indices=idx, weights=w,
                             nlat=nlat, nlon=nlon, method="nearest")

    # IDW
    if k < 2:
        raise ValueError(f"IDW requires k ≥ 2; got {k}")
    dist, idx = tree.query(tgt_xyz, k=k)
    # Guard against zero distance (target == source) — set weight to 1.
    zero_mask = dist <= 1e-12
    eps = 1e-12
    w = 1.0 / (dist + eps)
    w[zero_mask] = 0.0
    # If any row had a zero-distance hit, give that neighbour all the weight.
    has_zero = zero_mask.any(axis=-1)
    w[has_zero] = zero_mask[has_zero].astype(np.float64)
    # Normalise rows to sum to 1.
    w = w / w.sum(axis=-1, keepdims=True)

    idx = idx.reshape(nlat, nlon, k).astype(np.int64)
    w = w.reshape(nlat, nlon, k).astype(np.float64)
    return RegridWeights(indices=idx, weights=w,
                         nlat=nlat, nlon=nlon, method="idw")


def apply_weights(
    field_face_j_i: np.ndarray,
    weights: RegridWeights,
) -> np.ndarray:
    """Regrid a single source field to the target lat/lon grid.

    ``field_face_j_i`` may be shape ``(face, j, i)`` or have any number
    of leading dims (e.g. ``(time, face, j, i)`` or
    ``(time, sigma, face, j, i)``); the trailing three are the cs32
    horizontal dims and are flattened together internally. The output
    has the same leading dims with the trailing ``(face, j, i)``
    replaced by ``(nlat, nlon)``.
    """
    src = np.asarray(field_face_j_i)
    if src.shape[-3:-2] == (1,):
        # In case the caller has already flattened face into a singleton
        # somehow — the algorithm only cares about the flattened source
        # length matching the weights.
        pass
    n_horiz = int(np.prod(src.shape[-3:]))
    expected = weights.indices.max() + 1
    if n_horiz < expected:
        raise ValueError(
            f"Field horizontal size {n_horiz} smaller than expected "
            f"(weights index up to {expected})"
        )
    leading_shape = src.shape[:-3]
    src_flat = src.reshape(*leading_shape, n_horiz)        # (..., n_horiz)
    # Gather the k-neighbour values: (..., nlat, nlon, k)
    gathered = src_flat[..., weights.indices]
    # Multiply by weights and sum over k.
    return (gathered * weights.weights).sum(axis=-1)


def rotate_uv_to_geographic(
    u: np.ndarray,
    v: np.ndarray,
    angle_cs: np.ndarray,
    angle_sn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate face-aligned ``(u, v)`` into geographic ``(east, north)``.

    MITgcm's ``AngleCS = cos(α)``, ``AngleSN = sin(α)`` where ``α`` is the
    rotation angle from local face axes to geographic east/north. Inverse
    of MITgcm's own ``rotate_uv2en.F`` convention used by the diagnostics
    package.

    Inputs may be (..., face, y, x); ``angle_cs`` and ``angle_sn`` are
    ``(face, y, x)`` and broadcast over the leading dims.
    """
    u_east  = angle_cs * u - angle_sn * v
    v_north = angle_sn * u + angle_cs * v
    return u_east, v_north


@dataclass(frozen=True)
class GlobalOceanLatLon:
    """Static lat/lon-side grid info needed by the dataloader.

    All arrays are on the lat/lon target grid built by :func:`build`.
    """
    nlat: int
    nlon: int
    weights: RegridWeights
    angle_cs: np.ndarray   # (face, y, x), source-side, used pre-regrid
    angle_sn: np.ndarray   # (face, y, x), source-side
    depth_ll: np.ndarray   # (nlat, nlon)
    mask_k1_ll: np.ndarray # (nlat, nlon) bool
    mask_k2_ll: np.ndarray # (nlat, nlon) bool
    mask_eta_ll: np.ndarray# (nlat, nlon) bool
    # Per-level masks (present in global-ocean-3D's grid.h5).
    mask_c_3d_ll: np.ndarray | None = None  # (Nr, nlat, nlon) bool
    mask_w_3d_ll: np.ndarray | None = None  # (Nr, nlat, nlon) bool
    # Source-side per-level masks (cs grid). Used by ``apply_dynamic_3d`` to
    # impute land cells before regridding so the IDW kernel doesn't bleed
    # MITgcm's 0.0 land-fill into wet cells across continental margins.
    mask_c_3d_src: np.ndarray | None = None  # (Nr, face, y, x) bool
    mask_w_3d_src: np.ndarray | None = None  # (Nr, face, y, x) bool


def build(
    grid: dict,
    *,
    nlat: int = DEFAULT_NLAT,
    nlon: int = DEFAULT_NLON,
    method: Literal["nearest", "idw"] = "idw",
    k: int = 4,
    mask_threshold: float = 0.5,
) -> GlobalOceanLatLon:
    """Build cs → lat/lon weights and regrid the static fields (depth, masks).

    ``grid`` is a mapping of name -> ndarray (the arrays of ``grid.h5``).
    Cheap; the dataloader calls this once.
    """
    g = {k: np.asarray(v) for k, v in dict(grid).items()}
    xc = g["xc"]
    yc = g["yc"]
    weights = build_weights(xc, yc, nlat=nlat, nlon=nlon,
                                    method=method, k=k)

    depth_ll = apply_weights(g["depth"], weights)
    # Regrid bool masks as floats in [0,1] then threshold.
    mask_k1_ll  = apply_weights(g["mask_k1"].astype(np.float32),
                                weights) > mask_threshold
    mask_k2_ll  = apply_weights(g["mask_k2"].astype(np.float32),
                                weights) > mask_threshold
    mask_eta_ll = apply_weights(g["mask_eta"].astype(np.float32),
                                weights) > mask_threshold

    # Optional 3-D per-level masks (only present in grid files written
    # after the 3-D extract_grid extension).
    mask_c_3d_ll: np.ndarray | None = None
    mask_w_3d_ll: np.ndarray | None = None
    mask_c_3d_src: np.ndarray | None = None
    mask_w_3d_src: np.ndarray | None = None
    if "mask_c_3d" in g:
        mask_c_3d_src = g["mask_c_3d"].astype(bool)
        mask_c_3d_ll = apply_weights(
            mask_c_3d_src.astype(np.float32), weights
        ) > mask_threshold
    if "mask_w_3d" in g:
        mask_w_3d_src = g["mask_w_3d"].astype(bool)
        mask_w_3d_ll = apply_weights(
            mask_w_3d_src.astype(np.float32), weights
        ) > mask_threshold

    return GlobalOceanLatLon(
        nlat=nlat, nlon=nlon, weights=weights,
        angle_cs=g["angle_cs"].astype(np.float32),
        angle_sn=g["angle_sn"].astype(np.float32),
        depth_ll=depth_ll.astype(np.float32),
        mask_k1_ll=mask_k1_ll, mask_k2_ll=mask_k2_ll, mask_eta_ll=mask_eta_ll,
        mask_c_3d_ll=mask_c_3d_ll, mask_w_3d_ll=mask_w_3d_ll,
        mask_c_3d_src=mask_c_3d_src, mask_w_3d_src=mask_w_3d_src,
    )


FIELDS_3D: tuple[str, ...] = ("theta", "salt", "u", "v")

FIELDS_2D: tuple[str, ...] = ("eta",)


def field_names_3d(level_idx: np.ndarray) -> tuple[str, ...]:
    """Channel names for the level-squashed 3-D output.

    Order: every 3-D variable's levels in ``FIELDS_3D`` order, followed by
    each 2-D field. E.g. for levels ``[1, 5]``:
    ``('theta_k01', 'theta_k05', 'salt_k01', 'salt_k05',
       'u_k01',     'u_k05',     'v_k01',    'v_k05', 'eta')``.
    """
    names: list[str] = []
    for var in FIELDS_3D:
        for lvl in level_idx:
            names.append(f"{var}_k{int(lvl):02d}")
    names.extend(FIELDS_2D)
    return tuple(names)


def apply_dynamic_3d(
    fields_3d: dict[str, np.ndarray],
    fields_2d: dict[str, np.ndarray],
    grid_ll: GlobalOceanLatLon,
    *,
    level_idx: np.ndarray | None = None,
    impute_land: bool = True,
) -> np.ndarray:
    """Regrid the per-variable 3-D global-ocean output to a level-squashed
    lat/lon tensor ``(time, C, nlat, nlon)``.

    Depth levels are folded into the channel axis so downstream models see
    a flat channel stack. Channel order is given by :func:`field_names_3d`.

    Inputs match the HDF5 field layout:

    * ``fields_3d``: ``{'theta', 'salt', 'u', 'v'}``, each
      ``(time, level, face, y, x)``.
    * ``fields_2d``: ``{'eta'}``, ``(time, face, y, x)``.

    The ``(u, v)`` pair is rotated to geographic east/north on the native
    cs grid at every level before regrid; ``theta`` / ``salt`` / ``eta``
    are treated as scalars.

    When ``impute_land`` is True (default) and the source-side per-level
    masks are available in ``grid_ll``, MITgcm's 0.0 land-fill is
    replaced with each variable's per-level wet-cell mean before regrid.
    This stops the IDW kernel from bleeding 0-valued land into wet cells
    near continental margins — particularly visible at deep levels
    where the wet-cell variance is tiny (e.g. salt_k15 ≈ 0.04 psu).
    Land cells are zeroed back out post-regrid via the mask, so this
    only affects values *inside* the wet region.

    ``level_idx`` is the 1-indexed model level for each entry along the
    ``level`` axis of the inputs. When omitted, defaults to
    ``range(1, Nlevel+1)`` (i.e. the inputs are assumed to be the
    first ``Nlevel`` levels).
    """
    if "u" not in fields_3d or "v" not in fields_3d:
        raise ValueError(
            f"fields_3d must contain 'u' and 'v'; got {sorted(fields_3d)}"
        )

    src_3d = {name: np.asarray(arr) for name, arr in fields_3d.items()}
    for name, arr in src_3d.items():
        if arr.ndim != 5:
            raise ValueError(
                f"fields_3d[{name!r}] must be (time, level, face, y, x); "
                f"got shape {arr.shape}"
            )
    Nt, Nlevel = next(iter(src_3d.values())).shape[:2]
    for name, arr in src_3d.items():
        if arr.shape[:2] != (Nt, Nlevel):
            raise ValueError(
                f"fields_3d[{name!r}] has shape {arr.shape}; "
                f"expected leading dims ({Nt}, {Nlevel})"
            )

    if level_idx is None:
        level_idx = np.arange(1, Nlevel + 1, dtype=np.int64)
    levels0 = np.asarray(level_idx, dtype=int) - 1

    # Rotate u/v at every (time, level) — angle_{cs,sn} have shape (face, y, x)
    # and broadcast across the leading two axes.
    u_east, v_north = rotate_uv_to_geographic(
        src_3d["u"], src_3d["v"], grid_ll.angle_cs, grid_ll.angle_sn,
    )
    src_3d = {**src_3d, "u": u_east, "v": v_north}

    # Land-imputation: replace 0-fill with per-variable per-level wet-cell
    # mean before the IDW regrid. We take the mean over wet cells AND time,
    # which is robust when a single window has near-uniform values (e.g. an
    # identity batch). Only runs when the source-side per-level masks are
    # present in the grid file.
    have_3d_src = (
        grid_ll.mask_c_3d_src is not None
        and grid_ll.mask_w_3d_src is not None
    )
    if impute_land and have_3d_src:
        for var in FIELDS_3D:
            src_mask_full = (
                grid_ll.mask_c_3d_src if var in ("theta", "salt")
                else grid_ll.mask_w_3d_src
            )
            # Copy before mutating so callers' input tensors are never
            # touched. ``u`` / ``v`` came back from ``rotate_uv_to_geographic``
            # as fresh arrays already; ``theta`` / ``salt`` may still alias.
            arr = src_3d[var].copy()
            for k in range(Nlevel):
                wet = src_mask_full[levels0[k]]  # (face, y, x) bool
                if not wet.any():
                    continue
                slab = arr[:, k]  # (time, face, y, x); writable view of arr
                mu = float(slab[..., wet].mean())
                slab[..., ~wet] = mu
            src_3d[var] = arr

    nlat, nlon = grid_ll.nlat, grid_ll.nlon
    n_3d_chans = len(FIELDS_3D) * Nlevel
    n_2d_chans = len(FIELDS_2D)
    out = np.empty((Nt, n_3d_chans + n_2d_chans, nlat, nlon), dtype=np.float32)

    for var_idx, var in enumerate(FIELDS_3D):
        regridded = apply_weights(src_3d[var], grid_ll.weights)  # (Nt, Nlevel, nlat, nlon)
        out[:, var_idx * Nlevel:(var_idx + 1) * Nlevel] = regridded.astype(np.float32)

    for off, name in enumerate(FIELDS_2D):
        arr2 = np.asarray(fields_2d[name])
        # eta wet mask on cs grid = "any level has water" = OR over levels.
        if impute_land and have_3d_src and name == "eta":
            wet_eta = grid_ll.mask_c_3d_src.any(axis=0)  # (face, y, x)
            if wet_eta.any():
                arr2 = arr2.copy()
                mu = float(arr2[..., wet_eta].mean())
                arr2[..., ~wet_eta] = mu
        regridded = apply_weights(arr2, grid_ll.weights)
        out[:, n_3d_chans + off] = regridded.astype(np.float32)

    return out


def field_masks_3d_ll(
    grid_ll: GlobalOceanLatLon,
    level_idx: np.ndarray,
) -> np.ndarray:
    """Build a per-channel mask stack ``(C, nlat, nlon)`` matching
    :func:`apply_dynamic_3d`'s output.

    Channel order follows :func:`field_names_3d`:
    ``theta`` and ``salt`` use the C-grid mask at each level;
    ``u`` and ``v`` use the W-grid mask at each level (the same
    "both-edges-wet" convention as the 2-D ``mask_k2``);
    ``eta`` uses ``mask_eta`` (column-has-water).

    Requires ``grid_ll.mask_c_3d_ll`` / ``mask_w_3d_ll`` to be present.
    """
    if grid_ll.mask_c_3d_ll is None or grid_ll.mask_w_3d_ll is None:
        raise ValueError(
            "field_masks_3d_ll requires per-level masks (mask_c_3d/mask_w_3d) "
            "in the grid file."
        )

    nlat, nlon = grid_ll.nlat, grid_ll.nlon
    Nlevel = int(level_idx.size)
    levels0 = np.asarray(level_idx, dtype=int) - 1
    if levels0.min() < 0 or levels0.max() >= grid_ll.mask_c_3d_ll.shape[0]:
        raise ValueError(
            f"level_idx {level_idx!r} out of range "
            f"1..{grid_ll.mask_c_3d_ll.shape[0]}"
        )

    n_3d_chans = len(FIELDS_3D) * Nlevel
    n_2d_chans = len(FIELDS_2D)
    out = np.empty((n_3d_chans + n_2d_chans, nlat, nlon), dtype=bool)

    mask_c = grid_ll.mask_c_3d_ll[levels0]  # (Nlevel, nlat, nlon)
    mask_w = grid_ll.mask_w_3d_ll[levels0]
    var_to_mask = {"theta": mask_c, "salt": mask_c, "u": mask_w, "v": mask_w}
    for var_idx, var in enumerate(FIELDS_3D):
        out[var_idx * Nlevel:(var_idx + 1) * Nlevel] = var_to_mask[var]

    out[n_3d_chans] = grid_ll.mask_eta_ll  # eta
    return out

