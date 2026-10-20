"""HDF5 + on-the-fly regrid DataModule for global-ocean-3D.

This code was largely written by Claude.

Reads the cubed-sphere HDF5 store (per-run ``run_*.h5`` under
``data/{train,valid,test}/``, fields on the native cs32 grid) and regrids each
window to ``(nlat, nlon)`` equiangular on the fly with
:mod:`dandelion.data.ocean_regrid` (uv rotation + IDW regrid + land
imputation + level→channel fold). Batches are ``(B, T, C, nlat, nlon)`` with
``C = len(FIELDS_3D)*Nlevel + len(FIELDS_2D)`` dynamic channels plus
``depth + sin/cos(doy)`` statics.

The static cs32 grid geometry (xc/yc, angle_cs/sn, per-level masks, depth) is
read once from the dataset's ``grid.h5``.

HDF5 field layout (per file, one trajectory):
    t0_fields/{theta,salt,u,v}: (1, T, level, face, y, x)
    t0_fields/eta:              (1, T, 1,     face, y, x)   # surface, singleton level
    dimensions/{time,level,...}, aux_coords/{xc,yc,depth}, scalars/*
"""
from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path
from typing import Mapping, Optional

import h5py
import hdf5plugin  # noqa: F401  (LZ4 filter for the dataset files)
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from dandelion.data.ocean_regrid import (
    FIELDS_2D,
    FIELDS_3D,
    GlobalOceanLatLon,
    apply_dynamic_3d,
    build as build_lat_lon,
    field_masks_3d_ll,
    field_names_3d,
)
from dandelion.data.datamodule import AbstractDataModule

logger = logging.getLogger(__name__)

_RUN_RE = re.compile(r"run_(\d+)\.h5")

DEFAULT_NLAT = 64
DEFAULT_NLON = 128
N_STATIC_INPUT = 1   # depth
N_TIME_INPUT = 2     # sin(doy), cos(doy)

_SECONDS_PER_DAY = 86400.0
_DAYS_PER_YEAR = 365.0

# HDF5 split-dir names (the_well convention uses "valid", not "val").
_SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}


@dataclasses.dataclass
class OceanMetadata:
    """Metadata surface consumed by ``dandelion.train.build_model``."""

    dataset_name: str
    dim_in: int
    dim_out: int
    spatial_resolution: tuple[int, int]
    grid: str = "equiangular"
    field_names: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    n_spatial_dims: int = 2


def _scan_run_ids(split_dir: Path) -> list[int]:
    if not split_dir.is_dir():
        return []
    ids = []
    for p in split_dir.iterdir():
        m = _RUN_RE.fullmatch(p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def _resolve_split_run_ids(root: Path) -> dict[str, list[int]]:
    data = root / "data"
    return {k: _scan_run_ids(data / _SPLIT_DIRS[k]) for k in ("train", "val", "test")}


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    out = np.log10(np.clip(depth, 1.0, None)) / 4.0
    return out.astype(np.float32)


def _doy_channels(time_seconds: np.ndarray) -> np.ndarray:
    days = time_seconds / _SECONDS_PER_DAY
    phase = 2 * np.pi * (days % _DAYS_PER_YEAR) / _DAYS_PER_YEAR
    return np.stack([np.sin(phase), np.cos(phase)], axis=-1).astype(np.float32)


def _load_grid_h5(path: Path) -> dict:
    """Read every array in the flat grid.h5 into a name -> ndarray dict."""
    out: dict[str, np.ndarray] = {}
    with h5py.File(str(path), "r") as g:
        for k in g.keys():
            out[k] = g[k][()]
    return out


def _run_path(split_dir: Path, run_id: int) -> Path:
    return split_dir / f"run_{run_id:04d}.h5"


class _GlobalOcean3DLatLonHDF5WindowDataset(Dataset):
    """Sliding-window sampler over per-run 3-D global-ocean HDF5 files, with
    on-the-fly cs32 -> (nlat, nlon) regrid."""

    def __init__(
        self,
        split_dir: Path,
        run_ids: list[int],
        *,
        n_steps_input: int,
        n_steps_output: int,
        time_steps_per_run: int,
        grid_ll: GlobalOceanLatLon,
        level_idx: np.ndarray,
        level_pos: np.ndarray,
        full_trajectory: bool = False,
        max_rollout_steps: Optional[int] = None,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
        impute_land: bool = True,
    ):
        self.split_dir = Path(split_dir)
        self.run_ids = list(run_ids)
        self.n_steps_input = int(n_steps_input)
        self.n_steps_output = int(n_steps_output)
        self.time_steps_per_run = int(time_steps_per_run)
        self.full_trajectory = bool(full_trajectory)
        self._grid_ll = grid_ll
        self._level_idx = np.asarray(level_idx, dtype=np.int64)   # 1-indexed
        self._level_pos = np.asarray(level_pos, dtype=np.int64)   # 0-indexed on disk
        self._impute_land = bool(impute_land)

        if self.full_trajectory:
            if max_rollout_steps is None:
                max_rollout_steps = self.time_steps_per_run - self.n_steps_input
            self.n_steps_output = int(max_rollout_steps)

        total_window = self.n_steps_input + self.n_steps_output
        if total_window > self.time_steps_per_run:
            raise ValueError(
                f"window {total_window} exceeds trajectory length "
                f"{self.time_steps_per_run}"
            )
        self._windows_per_run = (
            1 if self.full_trajectory
            else self.time_steps_per_run - total_window + 1
        )

        self._norm_mean = (
            norm_mean.reshape(1, -1, 1, 1).astype(np.float32)
            if norm_mean is not None else None
        )
        self._norm_std = (
            norm_std.reshape(1, -1, 1, 1).astype(np.float32)
            if norm_std is not None else None
        )

        depth_norm = _normalize_depth(grid_ll.depth_ll)
        self._static_depth = depth_norm[None, None, :, :].astype(np.float32)

        mask_stack = field_masks_3d_ll(grid_ll, self._level_idx).astype(np.float32)
        self._valid_mask_np = mask_stack
        self._mask_dyn_btch = mask_stack[None, :, :, :]  # (1, C, H, W)

        # Per-worker file-handle cache; dropped on pickle so each DataLoader
        # worker opens its own read-only handles.
        self._h5_cache: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.run_ids) * self._windows_per_run

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_cache"] = {}
        return state

    def _open_run(self, run_id: int) -> h5py.File:
        h = self._h5_cache.get(run_id)
        if h is None:
            h = h5py.File(str(_run_path(self.split_dir, run_id)), "r")
            self._h5_cache[run_id] = h
        return h

    def _window_to_arrays(
        self, h: h5py.File, t0: int, T: int
    ) -> tuple[np.ndarray, np.ndarray]:
        grp = h["t0_fields"]
        # 3-D fields: (1, T_all, level, face, y, x) -> (T, Nlevel, face, y, x),
        # reading only the configured levels/time window.
        fields_3d = {}
        for name in FIELDS_3D:
            a = grp[name][0, t0:t0 + T]      # (T, level_all, face, y, x)
            fields_3d[name] = a[:, self._level_pos]
        # 2-D field: (1, T_all, 1, face, y, x) -> (T, face, y, x); drop the
        # singleton level axis.
        fields_2d = {}
        for name in FIELDS_2D:
            a = grp[name][0, t0:t0 + T]      # (T, 1, face, y, x)
            fields_2d[name] = a[:, 0]
        latlon = apply_dynamic_3d(
            fields_3d, fields_2d, self._grid_ll,
            level_idx=self._level_idx, impute_land=self._impute_land,
        )
        time_seconds = h["dimensions"]["time"][t0:t0 + T].astype(np.float64)
        return latlon, time_seconds

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        run_idx, window_idx = divmod(idx, self._windows_per_run)
        run_id = self.run_ids[run_idx]
        t0 = 0 if self.full_trajectory else window_idx
        h = self._open_run(run_id)

        x_dyn, x_time = self._window_to_arrays(h, t0, self.n_steps_input)
        y_dyn, y_time = self._window_to_arrays(
            h, t0 + self.n_steps_input, self.n_steps_output,
        )

        x_dyn = x_dyn * self._mask_dyn_btch
        y_dyn = y_dyn * self._mask_dyn_btch

        if self._norm_mean is not None:
            x_dyn = (x_dyn - self._norm_mean) / self._norm_std
            y_dyn = (y_dyn - self._norm_mean) / self._norm_std
            x_dyn = x_dyn * self._mask_dyn_btch
            y_dyn = y_dyn * self._mask_dyn_btch

        T_in = x_dyn.shape[0]
        H, W = x_dyn.shape[-2:]
        depth_t = np.broadcast_to(self._static_depth, (T_in, 1, H, W))

        doy = _doy_channels(x_time)
        doy_b = np.broadcast_to(doy[:, :, None, None], (T_in, 2, H, W))

        x_input = np.concatenate([x_dyn, depth_t, doy_b], axis=1).astype(np.float32)

        # The input-only channels at the OUTPUT frames' times, in the same
        # layout as the trailing channels of each input frame. The trainer's
        # autoregressive slide appends output_forcing[:, k] after each
        # prediction so every history frame stays (dynamic, depth, doy).
        T_out = y_dyn.shape[0]
        depth_y = np.broadcast_to(self._static_depth, (T_out, 1, H, W))
        doy_y = np.broadcast_to(_doy_channels(y_time)[:, :, None, None], (T_out, 2, H, W))
        y_forcing = np.concatenate([depth_y, doy_y], axis=1).astype(np.float32)

        return {
            "input_fields":  torch.tensor(x_input),
            "output_fields": torch.tensor(y_dyn.astype(np.float32)),
            "output_forcing": torch.tensor(y_forcing),
            "valid_mask":    torch.tensor(self._valid_mask_np),
        }


def _build_norm_from_per_channel_stats(
    stats: dict, field_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Read per-channel stats keyed by ``field_names_3d`` names (``theta_k01``
    … ``eta``), computed over wet cells only."""
    means = np.asarray([float(stats[n]["mean"]) for n in field_names], dtype=np.float32)
    stds = np.asarray([float(stats[n]["std"]) for n in field_names], dtype=np.float32)
    return means, stds


class GlobalOcean3DLatLonHDF5DataModule(AbstractDataModule):
    """On-the-fly-regrid 3-D global-ocean datamodule.

    ``root`` is the HDF5 store (``data/{train,valid,test}/run_*.h5``,
    ``grid.h5``). ``grid_h5`` defaults to ``root/grid.h5``.

    Normalization stats come from the inline ``stats`` mapping: wet-masked,
    per-channel, keyed by ``field_names_3d``. It must cover every selected
    channel or construction fails.
    """

    def __init__(
        self,
        root: str,
        grid_h5: Optional[str] = None,
        dim_in: Optional[int] = None,
        dim_out: Optional[int] = None,
        spatial_resolution: tuple[int, int] = (DEFAULT_NLAT, DEFAULT_NLON),
        grid: str = "equiangular",
        batch_size: int = 2,
        n_steps_input: int = 4,
        n_steps_output: int = 1,
        max_rollout_steps: Optional[int] = None,
        num_workers: int = 8,
        dataset_name: str = "global_ocean_3d_latlon",
        rank: int = 0,
        world_size: int = 1,
        regrid_method: str = "idw",
        regrid_k: int = 4,
        levels: Optional[list[int]] = None,
        impute_land: bool = True,
        time_steps_per_run: Optional[int] = None,
        stats: Optional[Mapping] = None,
    ):
        self.root = Path(root)
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.n_steps_input = int(n_steps_input)
        self.n_steps_output = int(n_steps_output)
        self.max_rollout_steps = max_rollout_steps
        self.num_workers = int(num_workers)
        self.dataset_name = dataset_name

        nlat, nlon = int(spatial_resolution[0]), int(spatial_resolution[1])

        split_run_ids = _resolve_split_run_ids(self.root)
        logger.info(
            "%s splits from %s: train=%d val=%d test=%d",
            dataset_name, self.root,
            len(split_run_ids["train"]),
            len(split_run_ids["val"]),
            len(split_run_ids["test"]),
        )
        first_split = next(s for s in ("train", "val", "test") if split_run_ids[s])
        first_id = split_run_ids[first_split][0]
        peek_path = _run_path(self.root / "data" / _SPLIT_DIRS[first_split], first_id)

        # Discover on-disk trajectory length and level count from a real file.
        with h5py.File(str(peek_path), "r") as h:
            theta = h["t0_fields"]["theta"]
            n_time_on_disk = int(theta.shape[1])
            n_level_on_disk = int(theta.shape[2])
        self.time_steps_per_run = (
            int(time_steps_per_run) if time_steps_per_run is not None
            else n_time_on_disk
        )

        # 1-indexed model levels (what apply_dynamic_3d / field_names_3d /
        # field_masks_3d_ll expect) + the 0-indexed positions to read on disk.
        if levels is None:
            self._level_idx = np.arange(1, n_level_on_disk + 1, dtype=np.int64)
            self._level_pos = np.arange(n_level_on_disk, dtype=np.int64)
        else:
            self._level_idx = np.asarray(levels, dtype=np.int64)
            self._level_pos = self._level_idx - 1
            if self._level_pos.min() < 0 or self._level_pos.max() >= n_level_on_disk:
                raise ValueError(
                    f"levels {levels} out of range for {n_level_on_disk} on-disk "
                    f"levels (use 1-indexed level ids)"
                )
        Nlevel = int(self._level_idx.size)
        self._impute_land = bool(impute_land)
        n_dynamic = len(FIELDS_3D) * Nlevel + len(FIELDS_2D)
        n_input = n_dynamic + N_STATIC_INPUT + N_TIME_INPUT

        self._field_names = field_names_3d(self._level_idx)
        self._metadata = OceanMetadata(
            dataset_name=dataset_name,
            dim_in=int(dim_in) if dim_in is not None else n_input,
            dim_out=int(dim_out) if dim_out is not None else n_dynamic,
            spatial_resolution=(nlat, nlon),
            grid=grid,
            field_names=self._field_names,
        )

        grid_h5_path = Path(grid_h5) if grid_h5 else (self.root / "grid.h5")
        if not grid_h5_path.is_file():
            raise FileNotFoundError(
                f"{grid_h5_path} not found — download the dataset with "
                f"scripts/download_data.py"
            )
        self._grid_ll = build_lat_lon(
            _load_grid_h5(grid_h5_path),
            nlat=nlat, nlon=nlon, method=regrid_method, k=regrid_k,
        )
        if self._grid_ll.mask_c_3d_ll is None or self._grid_ll.mask_w_3d_ll is None:
            raise RuntimeError(
                f"{grid_h5_path} has no per-level masks (mask_c_3d/mask_w_3d); "
                "is this the global-ocean-3D grid file?"
            )
        logger.info(
            "%s regrid: cs32 → (%d, %d) method=%s k=%d  Nlevel=%d  "
            "C_dyn=%d  T=%d  grid=%s",
            dataset_name, nlat, nlon, regrid_method, self._grid_ll.weights.k,
            Nlevel, n_dynamic, self.time_steps_per_run, grid_h5_path,
        )

        # Per-channel z-score from the inline, wet-masked stats. Stats that
        # average MITgcm's 0.0 land fill in with the ocean inflate the deep
        # salt/theta stds by up to ~300x, so require full per-channel coverage.
        if stats is None:
            raise ValueError(f"{dataset_name}: an inline `stats` block is required")
        missing = [n for n in self._field_names if n not in stats]
        if missing:
            raise ValueError(
                f"{dataset_name}: inline `stats` covers "
                f"{len(self._field_names) - len(missing)}/"
                f"{len(self._field_names)} channels; missing e.g. "
                f"{missing[:3]}. Inline stats must be the wet-masked "
                "per-channel form keyed by field_names_3d names "
                "(theta_k01 … eta)."
            )
        means, stds = _build_norm_from_per_channel_stats(stats, self._field_names)
        self._norm_mean_np: Optional[np.ndarray] = means
        self._norm_std_np: Optional[np.ndarray] = stds
        self._norm_mean_t: Optional[torch.Tensor] = torch.from_numpy(means).reshape(1, -1, 1, 1)
        self._norm_std_t: Optional[torch.Tensor] = torch.from_numpy(stds).reshape(1, -1, 1, 1)

        def _make(split: str, full_trajectory: bool = False):
            return _GlobalOcean3DLatLonHDF5WindowDataset(
                split_dir=self.root / "data" / _SPLIT_DIRS[split],
                run_ids=split_run_ids[split],
                n_steps_input=self.n_steps_input,
                n_steps_output=self.n_steps_output,
                time_steps_per_run=self.time_steps_per_run,
                grid_ll=self._grid_ll,
                level_idx=self._level_idx,
                level_pos=self._level_pos,
                impute_land=self._impute_land,
                full_trajectory=full_trajectory,
                max_rollout_steps=self.max_rollout_steps,
                norm_mean=self._norm_mean_np,
                norm_std=self._norm_std_np,
            )

        self.train_dataset = _make("train")
        self.val_dataset = _make("val")
        self.test_dataset = _make("test")
        self.rollout_val_dataset = _make("val", full_trajectory=True)
        self.rollout_test_dataset = _make("test", full_trajectory=True)

    def denormalize_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self._norm_mean_t is None:
            return x
        mean = self._norm_mean_t.to(x.device, x.dtype)
        std = self._norm_std_t.to(x.device, x.dtype)
        if x.dim() == 5:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return x * std + mean

    @property
    def metadata(self) -> OceanMetadata:
        return self._metadata

    def _loader(
        self, dataset: Dataset, *,
        shuffle: bool, batch_size: Optional[int] = None, distribute: bool = True,
        drop_last: bool = True,
    ) -> DataLoader:
        bs = batch_size if batch_size is not None else self.batch_size
        if distribute and self.world_size > 1:
            sampler = DistributedSampler(
                dataset, num_replicas=self.world_size, rank=self.rank,
                shuffle=shuffle, drop_last=drop_last,
            )
            return DataLoader(
                dataset, batch_size=bs, sampler=sampler,
                num_workers=self.num_workers, pin_memory=False, drop_last=drop_last,
                persistent_workers=self.num_workers > 0,
            )
        return DataLoader(
            dataset, batch_size=bs, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=False, drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        # Full split on every rank. rollout_loop/validation_loop do not
        # all_reduce and only rank 0 logs, so a DistributedSampler here would
        # score 1/world_size of the set — and, worse, drive best.pt selection
        # off that shard. Mirrors the WellDataModule fix in c51fd89.
        return self._loader(self.val_dataset, shuffle=False,
                            distribute=False, drop_last=False)

    def rollout_val_dataloader(self) -> DataLoader:
        return self._loader(self.rollout_val_dataset, shuffle=False,
                            batch_size=1, distribute=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        # Full split on every rank: see val_dataloader.
        return self._loader(self.test_dataset, shuffle=False,
                            distribute=False, drop_last=False)

    def rollout_test_dataloader(self) -> DataLoader:
        return self._loader(self.rollout_test_dataset, shuffle=False,
                            batch_size=1, distribute=False)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.root}>"
