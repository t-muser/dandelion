"""Dataset for the spherical-PDE HDF5 files published on HuggingFace.

Adaptation of WellDataset to spherical geometries.

In particular we have:
- no boundary conditions returned
- explicit vector components (since the 3D datasets sometimes only have
  horizontal (u/v) but no vertical (z) components)
- per-level normalization (for 3D datasets)
- `fold_levels_into_channels`, which, if `True` collapses the level spatial
  axis to additional channels. Note that this naively flattens everything,
  in particular for e.g. dry-atmosphere-3D you get a channel ordering of
  [T_lev0..N, ps, u_lev0..N, v_lev0..N], i.e. not spatially coherent,
  but grouped by quantity.
"""
from __future__ import annotations

import dataclasses

import h5py
import torch

# Registers the LZ4 HDF5 filter the dataset files are compressed with.
import hdf5plugin  # noqa: F401

from dandelion.data.well import WellDataset


class SphericalHDF5Dataset(WellDataset):
    def __init__(self, *args, fold_levels_into_channels: bool = False, **kwargs):
        # Closed sphere: there is no boundary group to read or pad.
        kwargs["boundary_return_type"] = None
        super().__init__(*args, **kwargs)
        if self.norm is not None:
            self._reshape_per_level_stats()
        # Fold plan + folded metadata/denorm stats; a no-op (self._fold False)
        # for 2-D datasets or when the flag is off.
        self._fold = False
        self._surface_mask: list[bool] = []
        self.folded_denorm_stats: tuple[torch.Tensor, torch.Tensor] | None = None
        if fold_levels_into_channels:
            self._build_fold_plan()

    def _reshape_per_level_stats(self):
        """Reshape per-level field stats so they broadcast over the level axis.

        The normalizer applies ``(x - mean) / std`` per field on ``x`` of shape
        ``(n_steps, *spatial[, comp])``. A leveled field's stat arrives as a
        1-D ``(nlev,)`` (scalar) or 2-D ``(nlev, ncomp)`` (vector) tensor, which
        would broadcast against the wrong (trailing) axis. We reshape it to put
        ``nlev`` at the level position among the spatial dims and 1 elsewhere
        (component axis last), so each level normalizes independently."""
        with h5py.File(self.files_paths[0], "r") as f:
            spatial_dims = [str(d) for d in f["dimensions"].attrs["spatial_dims"]]
            if "level" not in spatial_dims:
                return  # 2-D dataset: nothing per-level
            level_pos = spatial_dims.index("level")
            n_spatial = len(spatial_dims)
            for grp in ("t0_fields", "t1_fields", "t2_fields"):
                if grp not in f:
                    continue
                for name in f[grp].attrs.get("field_names", []):
                    name = str(name)
                    dset = f[grp][name]  # (traj, time, *spatial[, comp])
                    nlev = dset.shape[2 + level_pos]  # +traj +time
                    if nlev <= 1:  # surface field (size-1 level placeholder)
                        continue
                    is_vec = grp != "t0_fields"
                    ncomp = dset.shape[-1] if is_vec else 1
                    target = [1] * n_spatial
                    target[level_pos] = nlev
                    if is_vec:
                        target = target + [ncomp]
                    for store in (self.norm.means, self.norm.stds,
                                  self.norm.means_delta, self.norm.stds_delta):
                        if name in store:
                            store[name] = store[name].reshape(*target)
        # Keep the flattened views consistent with the reshaped per-field stats.
        self.norm._precompute_flattened_stats()

    def _build_fold_plan(self):
        """Plan the level->channel fold and rebuild metadata + denorm stats.

        Walks the t0/t1/t2 field groups in the exact order ``_postprocess_data``
        concatenates them into the channel axis, so the per-assembled-channel
        ``surface_mask`` (and the folded field names / denorm stats) line up
        with the sample tensor produced by the base class."""
        with h5py.File(self.files_paths[0], "r") as f:
            spatial_dims = [str(d) for d in f["dimensions"].attrs["spatial_dims"]]
            if "level" not in spatial_dims:
                return  # 2-D dataset: nothing to fold
            level_pos = spatial_dims.index("level")
            # The fold merges the channel axis with the adjacent level axis; we
            # rely on level being the first spatial dim
            if level_pos != 0:
                raise NotImplementedError(
                    f"fold_levels_into_channels expects 'level' as the first "
                    f"spatial dim, got spatial_dims={spatial_dims}"
                )
            nlev = self.size_tuple[level_pos]

            # constant (time-invariant) leveled fields would need their own fold
            # + metadata bookkeeping; none exist in our leveled datasets today.
            if self.metadata.n_constant_fields:
                raise NotImplementedError(
                    "fold_levels_into_channels does not support constant "
                    "(time-invariant) spatial fields yet"
                )

            # Per-assembled-channel plan, in channel concatenation order.
            channels = []  # (field_name, comp_idx|None, ncomp, is_surface, label)
            for grp in ("t0_fields", "t1_fields", "t2_fields"):
                if grp not in f:
                    continue
                for name in f[grp].attrs.get("field_names", []):
                    name = str(name)
                    dset = f[grp][name]
                    is_surface = not bool(dset.attrs["dim_varying"][level_pos])
                    if grp == "t0_fields":
                        channels.append((name, None, 1, is_surface, name))
                    else:
                        ncomp = dset.shape[-1]
                        comps = dset.attrs.get("component_names")
                        labels = ([str(c) for c in comps] if comps is not None
                                  else [f"{name}_c{j}" for j in range(ncomp)])
                        for j in range(ncomp):
                            channels.append((name, j, ncomp, is_surface, labels[j]))

        self._fold = True
        self._level_count = nlev
        self._surface_mask = [c[3] for c in channels]
        folded_spatial = tuple(
            s for i, s in enumerate(self.size_tuple) if i != level_pos
        )

        # Folded channel names: surface field -> single channel; leveled field
        # -> one channel per level, contiguous.
        folded_names: list[str] = []
        for _, _, _, is_surface, label in channels:
            if is_surface:
                folded_names.append(label)
            else:
                folded_names.extend(f"{label}_lev{i}" for i in range(nlev))

        # Rebuild metadata for the 2-D, per-output-channel view. All folded
        # channels are flat scalars now, so n_fields == len(folded_names).
        self.metadata = dataclasses.replace(
            self.metadata,
            n_spatial_dims=self.metadata.n_spatial_dims - 1,
            spatial_resolution=folded_spatial,
            field_names={0: folded_names, 1: [], 2: []},
        )

        if self.norm is not None:
            self.folded_denorm_stats = self._build_folded_denorm_stats(channels, nlev)

    def _build_folded_denorm_stats(self, channels, nlev):
        """Per-output-channel (mean, std) vectors in folded channel order.

        Mirrors the data fold exactly: each leveled channel contributes ``nlev``
        per-level values, each surface channel a single value. The base
        normalizer already standardized the data per level on read, so this only
        feeds the datamodule's ``denormalize_fn`` for physical-unit logging."""
        means, stds = [], []
        for field_name, comp, ncomp, is_surface, _ in channels:
            m = self.norm.means[field_name].reshape(-1, ncomp)  # (nlev|1, ncomp)
            s = self.norm.stds[field_name].reshape(-1, ncomp)
            col = 0 if comp is None else comp
            mv, sv = m[:, col], s[:, col]
            if is_surface:
                mv, sv = mv[:1], sv[:1]
            means.append(mv)
            stds.append(sv)
        return torch.cat(means), torch.cat(stds)

    @staticmethod
    def _fold_level_into_channel(x, channel_axis, surface_mask):
        """Collapse the level axis (immediately after ``channel_axis``) into the
        channel axis. Leveled channels expand to one channel per level; surface
        channels (``surface_mask[c]`` True) keep only level 0."""
        level_axis = channel_axis + 1
        chunks = []
        for c, is_surface in enumerate(surface_mask):
            xc = x.narrow(channel_axis, c, 1)
            if is_surface:
                xc = xc.narrow(level_axis, 0, 1)
            merged = xc.shape[:channel_axis] + (-1,) + xc.shape[level_axis + 1:]
            chunks.append(xc.reshape(*merged))
        return torch.cat(chunks, dim=channel_axis)

    def _construct_sample(self, data, traj_metadata):
        sample = super()._construct_sample(data, traj_metadata)
        if not self._fold:
            return sample
        # Fields are (T, C, level, *rest); fold level (axis 2) into channels.
        sample["input_fields"] = self._fold_level_into_channel(
            sample["input_fields"], 1, self._surface_mask
        )
        sample["output_fields"] = self._fold_level_into_channel(
            sample["output_fields"], 1, self._surface_mask
        )
        if self.return_grid and "space_grid" in sample:
            # (level, lat, lon, D) -> (lat, lon, D-1): drop the level slice and
            # the now-meaningless level coordinate column.
            sample["space_grid"] = sample["space_grid"][0, ..., 1:]
        return sample

    def _scan_bc_types(self, _f, spatial_dims, file):
        # No boundary on a sphere; files carry no `boundary_conditions` group.
        return []

    def _field_component_names(self, _f, ti, field, order):
        # Prefer explicit per-component channel labels when present (already the
        # physical variable names, e.g. u_phi / u_theta); fall back to the Well
        # spatial-dim-product convention otherwise.
        comp = _f[ti][field].attrs.get("component_names")
        if comp is not None:
            return [str(c) for c in comp]
        return super()._field_component_names(_f, ti, field, order)
