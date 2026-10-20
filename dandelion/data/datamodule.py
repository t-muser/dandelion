"""Modified from the Well.
Copyright (c) 2024 Polymathic AI. BSD 3-Clause License
"""
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import torch

from torch.utils.data import DataLoader, DistributedSampler

from dandelion.data.normalization import ZScoreNormalization

# The Well integration is dormant until a PlanetSWE datamodule lands.
# Guard the heavy imports so `AbstractDataModule` remains importable even
# if the-well or dandelion.data.well are unavailable at import time — only
# callers that actually instantiate `WellDataModule` need those deps.
try:
    from the_well.data.augmentation import Augmentation
    from dandelion.data.well import LeadTimeDataset, WellDataset
    _WELL_AVAILABLE = True
    _WELL_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _e:  # pragma: no cover - optional integration
    Augmentation = Any  # type: ignore[assignment,misc]
    LeadTimeDataset = None  # type: ignore[assignment]
    WellDataset = None  # type: ignore[assignment]
    _WELL_AVAILABLE = False
    _WELL_IMPORT_ERROR = _e

logger = logging.getLogger(__name__)


class AbstractDataModule(ABC):
    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def rollout_val_dataloader(self) -> Union[DataLoader, Dict[int, DataLoader]]:
        raise NotImplementedError

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def rollout_test_dataloader(self) -> Union[DataLoader, Dict[int, DataLoader]]:
        raise NotImplementedError


class WellDataModule(AbstractDataModule):
    """Data module class to yield batches of samples.

    Args:
        well_base_path:
            Path to the data folder containing the splits (train, validation, and test).
        well_dataset_name:
            Name of the well dataset to use.
        batch_size:
            Size of the batches yielded by the dataloaders
        ---
        include_filters:
            Only file names containing any of these strings will be included.
        exclude_filters:
            File names containing any of these strings will be excluded.
        use_normalization:
            Whether to use normalization on the data.
        normalization_type:
            What kind of normalization to use if use_normalization is True. Currently supports zscore and rms.
        train_dataset:
            What type of training dataset type. WellDataset or DeltaWellDataset options.
        max_rollout_steps:
            Maximum number of steps to use for the rollout dataset. Mostly for memory reasons.
        n_steps_input:
            Number of steps to use as input.
        n_steps_output:
            Number of steps to use as output.
        min_dt_stride:
            Minimum stride in time to use for the dataset.
        max_dt_stride:
            Maximum stride in time to use for the dataset. If this is greater than min, randomly choose between them.
                Note that this is unused for validation/test which uses "min_dt_stride" for both the min and max.
        restrict_num_trajectories:
            Whether to restrict the number of trajectories in the training dataset. Integer inputs restrict to a number. Float to a percentage.
        restrict_num_samples:
            Whether to restrict the number of samples in the training dataset. Integer inputs restrict to a number. Float to a percentage.
        restriction_seed:
            Seed for restricting the training dataset.
        world_size:
            Number of GPUs in use for distributed training.
        data_workers:
            Number of workers to use for data loading.
        rank:
            Rank of the current process in distributed training.
        transform:
            Augmentation to apply to the data. If None, no augmentation is applied.
        dataset_kws:
            Additional keyword arguments to pass to each dataset, as a dict of dicts.
        storage_kwargs:
            Storage options passed to fsspec for accessing the raw data.
    """

    def __init__(
        self,
        well_base_path: str,
        well_dataset_name: str,
        batch_size: int,
        include_filters: List[str] = [],
        exclude_filters: List[str] = [],
        use_normalization: bool = False,
        normalization_type: Optional[Callable[..., Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
        meta_scalars: Optional[List[Dict[str, str]]] = None,
        train_dataset: Callable[..., Any] = WellDataset,
        eval_dataset: Callable[..., Any] = WellDataset,
        max_rollout_steps: int = 100,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        min_dt_stride: int = 1,
        max_dt_stride: int = 1,
        restrict_num_trajectories: Optional[float | int] = None,
        restrict_num_samples: Optional[float | int] = None,
        restriction_seed: int = 0,
        world_size: int = 1,
        data_workers: int = 4,
        rank: int = 1,
        boundary_return_type: Literal["padding", None] = "padding",
        transform: Optional[Augmentation] = None,
        dataset_kws: Optional[
            Dict[
                Literal["train", "val", "rollout_val", "test", "rollout_test"],
                Dict[str, Any],
            ]
        ] = None,
        storage_kwargs: Optional[Dict] = None,
        lead_time_mode: bool = False,
        max_lead_time: int = 20,
        curriculum_learning: bool = False,
    ):
        if not _WELL_AVAILABLE:
            raise ImportError(
                "Well-backed DataModule requires the_well and dandelion.data.well; "
                f"import failed at module load: {_WELL_IMPORT_ERROR!r}"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("always")  # Ensure warnings are always displayed

            if use_normalization:
                warnings.warn(
                    "`use_normalization` parameter will be removed in a future version. "
                    "For proper normalizing, set both use_normalization=True and normalization_type to either ZScoreNormalization or RMSNormalization."
                    "Default behavior is `normalization_type=ZScoreNormalization` and `use_normalization=True`."
                    "To switch off normalization instead, please set use_normalization=False in the config.yaml file",
                    DeprecationWarning,
                )
                if normalization_type is None:
                    warnings.warn(
                        "use_normalization=True, but normalization_type is None. "
                        "Defaulting to ZScoreNormalization.",
                        UserWarning,
                    )
                    normalization_type = ZScoreNormalization  # Default fallback

            elif normalization_type is not None:
                warnings.warn(
                    "Inconsistent normalization settings: `use_normalization=False`, but `normalization_type` is set. "
                    "Defaulting `normalization_type=None` and `use_normalization=False`.",
                    UserWarning,
                )
                normalization_type = None

        self.lead_time_mode = lead_time_mode
        self.max_lead_time = max_lead_time
        self.curriculum_learning = curriculum_learning

        # Common path kwargs for Well-style datasets
        path_kwargs = dict(
            well_base_path=well_base_path,
            well_dataset_name=well_dataset_name,
        )

        if lead_time_mode:
            # Lead-time mode: use LeadTimeDataset
            self.train_dataset = LeadTimeDataset(
                **path_kwargs,
                well_split_name="train",
                include_filters=include_filters,
                exclude_filters=exclude_filters,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                stats=stats,
                meta_scalars=meta_scalars,
                n_steps_input=n_steps_input,
                storage_options=storage_kwargs,
                restrict_num_trajectories=restrict_num_trajectories,
                restrict_num_samples=restrict_num_samples,
                restriction_seed=restriction_seed,
                boundary_return_type=boundary_return_type,
                transform=transform,
                max_lead_time=max_lead_time,
                fixed_lead_time=None,  # Random sampling during training
                **(
                    dataset_kws["train"]
                    if dataset_kws is not None and "train" in dataset_kws
                    else {}
                ),
            )
            self.val_dataset = LeadTimeDataset(
                **path_kwargs,
                well_split_name="valid",
                include_filters=include_filters,
                exclude_filters=exclude_filters,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                stats=stats,
                meta_scalars=meta_scalars,
                n_steps_input=n_steps_input,
                storage_options=storage_kwargs,
                boundary_return_type=boundary_return_type,
                max_lead_time=max_lead_time,
                fixed_lead_time=1,  # Single-step for normal validation
                **(
                    dataset_kws["val"]
                    if dataset_kws is not None and "val" in dataset_kws
                    else {}
                ),
            )
            # Rollout datasets: one for each k from 1 to max_lead_time
            self._rollout_val_datasets: Dict[int, LeadTimeDataset] = {}
            for k in range(1, max_lead_time + 1):
                self._rollout_val_datasets[k] = LeadTimeDataset(
                    **path_kwargs,
                    well_split_name="valid",
                    include_filters=include_filters,
                    exclude_filters=exclude_filters,
                    use_normalization=use_normalization,
                    normalization_type=normalization_type,
                    stats=stats,
                    meta_scalars=meta_scalars,
                    n_steps_input=n_steps_input,
                    storage_options=storage_kwargs,
                    boundary_return_type=boundary_return_type,
                    max_lead_time=max_lead_time,
                    fixed_lead_time=k,
                    **(
                        dataset_kws["rollout_val"]
                        if dataset_kws is not None and "rollout_val" in dataset_kws
                        else {}
                    ),
                )
            self.test_dataset = LeadTimeDataset(
                **path_kwargs,
                well_split_name="test",
                include_filters=include_filters,
                exclude_filters=exclude_filters,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                stats=stats,
                meta_scalars=meta_scalars,
                n_steps_input=n_steps_input,
                storage_options=storage_kwargs,
                boundary_return_type=boundary_return_type,
                max_lead_time=max_lead_time,
                fixed_lead_time=1,
                **(
                    dataset_kws["test"]
                    if dataset_kws is not None and "test" in dataset_kws
                    else {}
                ),
            )
            self._rollout_test_datasets: Dict[int, LeadTimeDataset] = {}
            for k in range(1, max_lead_time + 1):
                self._rollout_test_datasets[k] = LeadTimeDataset(
                    **path_kwargs,
                    well_split_name="test",
                    include_filters=include_filters,
                    exclude_filters=exclude_filters,
                    use_normalization=use_normalization,
                    normalization_type=normalization_type,
                    stats=stats,
                    meta_scalars=meta_scalars,
                    n_steps_input=n_steps_input,
                    storage_options=storage_kwargs,
                    boundary_return_type=boundary_return_type,
                    max_lead_time=max_lead_time,
                    fixed_lead_time=k,
                    **(
                        dataset_kws["rollout_test"]
                        if dataset_kws is not None and "rollout_test" in dataset_kws
                        else {}
                    ),
                )
            # Set single rollout datasets to None (not used in lead_time_mode)
            self.rollout_val_dataset = None
            self.rollout_test_dataset = None
        else:
            # Regular mode: use WellDataset (original behavior)
            # DeltaWellDataset only for training for delta case, WellDataset for everything else
            self.train_dataset = train_dataset(
                **path_kwargs,
                well_split_name="train",
                include_filters=include_filters,
                exclude_filters=exclude_filters,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                stats=stats,
                meta_scalars=meta_scalars,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                storage_options=storage_kwargs,
                min_dt_stride=min_dt_stride,
                max_dt_stride=max_dt_stride,
                restrict_num_trajectories=restrict_num_trajectories,
                restrict_num_samples=restrict_num_samples,
                restriction_seed=restriction_seed,
                boundary_return_type=boundary_return_type,
                transform=transform,
                **(
                    dataset_kws["train"]
                    if dataset_kws is not None and "train" in dataset_kws
                    else {}
                ),
            )
            self.val_dataset = eval_dataset(
                **path_kwargs,
                well_split_name="valid",
                include_filters=include_filters,
                exclude_filters=exclude_filters,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                stats=stats,
                meta_scalars=meta_scalars,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                storage_options=storage_kwargs,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride,
                boundary_return_type=boundary_return_type,
                **(
                    dataset_kws["val"]
                    if dataset_kws is not None and "val" in dataset_kws
                    else {}
                ),
            )
            self.rollout_val_dataset = eval_dataset(
                **path_kwargs,
                well_split_name="valid",
                include_filters=include_filters,
                exclude_filters=exclude_filters,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                stats=stats,
                meta_scalars=meta_scalars,
                max_rollout_steps=max_rollout_steps,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                full_trajectory_mode=True,
                storage_options=storage_kwargs,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride,
                boundary_return_type=boundary_return_type,
                **(
                    dataset_kws["rollout_val"]
                    if dataset_kws is not None and "rollout_val" in dataset_kws
                    else {}
                ),
            )
            self.test_dataset = eval_dataset(
                **path_kwargs,
                well_split_name="test",
                include_filters=include_filters,
                exclude_filters=exclude_filters,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                stats=stats,
                meta_scalars=meta_scalars,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                storage_options=storage_kwargs,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride,
                boundary_return_type=boundary_return_type,
                **(
                    dataset_kws["test"]
                    if dataset_kws is not None and "test" in dataset_kws
                    else {}
                ),
            )
            self.rollout_test_dataset = eval_dataset(
                **path_kwargs,
                well_split_name="test",
                include_filters=include_filters,
                exclude_filters=exclude_filters,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                stats=stats,
                meta_scalars=meta_scalars,
                max_rollout_steps=max_rollout_steps,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                full_trajectory_mode=True,
                storage_options=storage_kwargs,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride,
                boundary_return_type=boundary_return_type,
                **(
                    dataset_kws["rollout_test"]
                    if dataset_kws is not None and "rollout_test" in dataset_kws
                    else {}
                ),
            )
            # Set dict rollout datasets to empty (not used in regular mode)
            self._rollout_val_datasets = {}
            self._rollout_test_datasets = {}

        self.well_base_path = well_base_path
        self.well_dataset_name = well_dataset_name
        self.batch_size = batch_size
        self.world_size = world_size
        self.data_workers = data_workers
        self.rank = rank

    @property
    def metadata(self):
        """Adapt inner WellDataset's WellMetadata to the SweThMetadata-like
        surface consumed by dandelion.train.build_model and dandelion.trainer.Trainer.
        """
        from types import SimpleNamespace
        md = self.train_dataset.metadata  # WellMetadata
        # Flatten {order: [names]} → flat tuple, in tensor-order order.
        flat_field_names: list[str] = []
        for order in sorted(md.field_names.keys()):
            flat_field_names.extend(md.field_names[order])
        # NOTE: dim_in sums time-varying and constant fields because constant
        # fields are concatenated once as extra input channels. dandelion.train's
        # `dim_in * n_steps_input` over-counts constants for n_steps_input > 1;
        # fine for planetswe where n_constant_fields == 0.
        return SimpleNamespace(
            dataset_name=md.dataset_name,
            dim_in=md.n_fields + md.n_constant_fields,
            dim_out=md.n_fields,
            spatial_resolution=md.spatial_resolution,
            grid=md.grid_type,
            field_names=tuple(flat_field_names),
            n_spatial_dims=md.n_spatial_dims,
        )

    @property
    def denormalize_fn(self):
        """Return a callable that denormalizes (B,C,H,W) tensors back to physical units.

        Uses the variable-field stats from the training dataset's normalizer.
        Returns None when normalization is not active.
        """
        norm = getattr(self.train_dataset, "norm", None)
        if norm is None:
            return None
        # When the dataset folds levels into channels, the per-channel stats no
        # longer match norm.flattened_* (which is per-field, level-interleaved);
        # use the fold-order vectors the dataset rebuilt instead.
        folded = getattr(self.train_dataset, "folded_denorm_stats", None)
        if folded is not None:
            mean, std = folded
        else:
            std = norm.flattened_stds["variable"]   # (C,)
            mean = norm.flattened_means["variable"]  # (C,)

        def _denorm(x: torch.Tensor) -> torch.Tensor:
            s = std.to(x.device).view(1, -1, 1, 1)
            m = mean.to(x.device).view(1, -1, 1, 1)
            return x * s + m

        return _denorm

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def set_epoch(self, epoch: int):
        """Set the current epoch for curriculum learning.

        When curriculum_learning=True and lead_time_mode=True, this updates
        the training dataset's max lead time to min(epoch, max_lead_time).
        This allows the model to learn progressively longer horizons.

        Args:
            epoch: Current training epoch (1-indexed)
        """
        if self.curriculum_learning and self.lead_time_mode:
            current_max_k = min(epoch, self.max_lead_time)
            self.train_dataset.set_current_max_lead_time(current_max_k)
            logger.info(f"Curriculum learning: epoch {epoch}, max lead time = {current_max_k}")

    def train_dataloader(self) -> DataLoader:
        """Generate a dataloader for training data.

        Returns:
            A dataloader
        """
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            logger.debug(
                f"Use {sampler.__class__.__name__} "
                f"({self.rank}/{self.world_size}) for training data"
            )
        shuffle = sampler is None

        return DataLoader(
            self.train_dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=True,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        """Generate a dataloader for validation data.

        Returns:
            A dataloader
        """
        # Evaluate the FULL split on every rank (no DistributedSampler).
        # validation_loop / rollout_loop do not all-reduce, and only rank 0
        # logs, so a sharded sampler would report metrics over just
        # 1/world_size of the split (and drive best.pt selection off that
        # shard). Redundant across ranks but correct.
        return DataLoader(
            self.val_dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            sampler=None,
        )

    def _make_dataloader(
        self, dataset, shuffle: bool = False, batch_size: int = None
    ) -> DataLoader:
        """Helper to create a dataloader with optional distributed sampling."""
        batch_size = batch_size if batch_size is not None else self.batch_size
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
            )
            shuffle = False  # Sampler handles shuffling
        return DataLoader(
            dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            sampler=sampler,
        )

    def rollout_val_dataloader(self) -> Union[DataLoader, Dict[int, DataLoader]]:
        """Generate dataloader(s) for rollout validation data.

        Returns:
            In lead_time_mode: Dict mapping lead-time k to DataLoader
            Otherwise: Single DataLoader for autoregressive rollout
        """
        if self.lead_time_mode:
            return {
                k: self._make_dataloader(dataset, shuffle=True, batch_size=1)
                for k, dataset in self._rollout_val_datasets.items()
            }
        else:
            # Full rollout-val set, unsharded (see val_dataloader). Only rank 0
            # runs rollout_val, and rollout_loop does not all-reduce, so a
            # DistributedSampler would score just 1/world_size of the set.
            return DataLoader(
                self.rollout_val_dataset,
                num_workers=self.data_workers,
                pin_memory=True,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                sampler=None,
            )

    def test_dataloader(self) -> DataLoader:
        """Generate a dataloader for test data.

        Returns:
            A dataloader
        """
        # Full split on every rank (no DistributedSampler): see val_dataloader.
        # Sharding here would make the reported test/ metrics cover only
        # 1/world_size of the test set.
        return DataLoader(
            self.test_dataset,
            num_workers=self.data_workers,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            sampler=None,
        )

    def rollout_test_dataloader(self) -> Union[DataLoader, Dict[int, DataLoader]]:
        """Generate dataloader(s) for rollout test data.

        Returns:
            In lead_time_mode: Dict mapping lead-time k to DataLoader
            Otherwise: Single DataLoader for autoregressive rollout
        """
        if self.lead_time_mode:
            return {
                k: self._make_dataloader(dataset, shuffle=False, batch_size=1)
                for k, dataset in self._rollout_test_datasets.items()
            }
        else:
            # Full rollout-test set, unsharded (see val_dataloader). test()
            # runs rollout_loop on all ranks with no all-reduce and only rank 0
            # logs, so a DistributedSampler would report the rollout metric
            # over just 1/world_size (= 15 of 57 on 4 GPUs) of the test set.
            return DataLoader(
                self.rollout_test_dataset,
                num_workers=self.data_workers,
                pin_memory=True,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                sampler=None,
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.well_dataset_name} on {self.well_base_path}>"


class NotWellDataModule(WellDataModule):
    """Data module class to yield batches of samples using path-style datasets.

    Args:
        path: Path to the dataset
    """

    def __init__(
        self,
        path: str,
        batch_size: int,
        dataset_name: str,  # only used to generate experiment name
        include_filters: List[str] = [],
        exclude_filters: List[str] = [],
        use_normalization: bool = False,
        normalization_type: Optional[Callable[..., Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
        meta_scalars: Optional[List[Dict[str, str]]] = None,
        train_dataset: Callable[..., Any] = WellDataset,
        eval_dataset: Callable[..., Any] = WellDataset,
        max_rollout_steps: int = 100,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        min_dt_stride: int = 1,
        max_dt_stride: int = 1,
        restrict_num_trajectories: Optional[float | int] = None,
        restrict_num_samples: Optional[float | int] = None,
        restriction_seed: int = 0,
        world_size: int = 1,
        data_workers: int = 4,
        rank: int = 1,
        boundary_return_type: Literal["padding", None] = "padding",
        transform: Optional[Augmentation] = None,
        dataset_kws: Optional[
            Dict[
                Literal["train", "val", "rollout_val", "test", "rollout_test"],
                Dict[str, Any],
            ]
        ] = None,
        storage_kwargs: Optional[Dict] = None,
        lead_time_mode: bool = False,
        max_lead_time: int = 20,
        curriculum_learning: bool = False,
    ):
        if not _WELL_AVAILABLE:
            raise ImportError(
                "Well-backed DataModule requires the_well and dandelion.data.well; "
                f"import failed at module load: {_WELL_IMPORT_ERROR!r}"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("always")  # Ensure warnings are always displayed

            if use_normalization:
                warnings.warn(
                    "`use_normalization` parameter will be removed in a future version. "
                    "For proper normalizing, set both use_normalization=True and normalization_type to either ZScoreNormalization or RMSNormalization."
                    "Default behavior is `normalization_type=ZScoreNormalization` and `use_normalization=True`."
                    "To switch off normalization instead, please set use_normalization=False in the config.yaml file",
                    DeprecationWarning,
                )
                if normalization_type is None:
                    warnings.warn(
                        "use_normalization=True, but normalization_type is None. "
                        "Defaulting to ZScoreNormalization.",
                        UserWarning,
                    )
                    normalization_type = ZScoreNormalization  # Default fallback

            elif normalization_type is not None:
                warnings.warn(
                    "Inconsistent normalization settings: `use_normalization=False`, but `normalization_type` is set. "
                    "Defaulting `normalization_type=None` and `use_normalization=False`.",
                    UserWarning,
                )
                normalization_type = None

        self.lead_time_mode = lead_time_mode
        self.max_lead_time = max_lead_time
        self.curriculum_learning = curriculum_learning

        # Path kwargs for NotWell-style datasets
        path_kwargs = dict(path=path)

        if lead_time_mode:
            # Lead-time mode: use LeadTimeDataset
            self.train_dataset = LeadTimeDataset(
                **path_kwargs,
                well_split_name="train",
                include_filters=include_filters,
                exclude_filters=exclude_filters,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                stats=stats,
                meta_scalars=meta_scalars,
                n_steps_input=n_steps_input,
                storage_options=storage_kwargs,
                restrict_num_trajectories=restrict_num_trajectories,
                restrict_num_samples=restrict_num_samples,
                restriction_seed=restriction_seed,
                boundary_return_type=boundary_return_type,
                transform=transform,
                max_lead_time=max_lead_time,
                fixed_lead_time=None,  # Random sampling during training
                **(
                    dataset_kws["train"]
                    if dataset_kws is not None and "train" in dataset_kws
                    else {}
                ),
            )
            self.val_dataset = LeadTimeDataset(
                **path_kwargs,
                well_split_name="valid",
                include_filters=include_filters,
                exclude_filters=exclude_filters,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                stats=stats,
                meta_scalars=meta_scalars,
                n_steps_input=n_steps_input,
                storage_options=storage_kwargs,
                boundary_return_type=boundary_return_type,
                max_lead_time=max_lead_time,
                fixed_lead_time=1,  # Single-step for normal validation
                **(
                    dataset_kws["val"]
                    if dataset_kws is not None and "val" in dataset_kws
                    else {}
                ),
            )
            # Rollout datasets: one for each k from 1 to max_lead_time
            self._rollout_val_datasets: Dict[int, LeadTimeDataset] = {}
            for k in range(1, max_lead_time + 1):
                self._rollout_val_datasets[k] = LeadTimeDataset(
                    **path_kwargs,
                    well_split_name="valid",
                    include_filters=include_filters,
                    exclude_filters=exclude_filters,
                    use_normalization=use_normalization,
                    normalization_type=normalization_type,
                    stats=stats,
                    meta_scalars=meta_scalars,
                    n_steps_input=n_steps_input,
                    storage_options=storage_kwargs,
                    boundary_return_type=boundary_return_type,
                    max_lead_time=max_lead_time,
                    fixed_lead_time=k,
                    **(
                        dataset_kws["rollout_val"]
                        if dataset_kws is not None and "rollout_val" in dataset_kws
                        else {}
                    ),
                )
            self.test_dataset = LeadTimeDataset(
                **path_kwargs,
                well_split_name="test",
                include_filters=include_filters,
                exclude_filters=exclude_filters,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                stats=stats,
                meta_scalars=meta_scalars,
                n_steps_input=n_steps_input,
                storage_options=storage_kwargs,
                boundary_return_type=boundary_return_type,
                max_lead_time=max_lead_time,
                fixed_lead_time=1,
                **(
                    dataset_kws["test"]
                    if dataset_kws is not None and "test" in dataset_kws
                    else {}
                ),
            )
            self._rollout_test_datasets: Dict[int, LeadTimeDataset] = {}
            for k in range(1, max_lead_time + 1):
                self._rollout_test_datasets[k] = LeadTimeDataset(
                    **path_kwargs,
                    well_split_name="test",
                    include_filters=include_filters,
                    exclude_filters=exclude_filters,
                    use_normalization=use_normalization,
                    normalization_type=normalization_type,
                    stats=stats,
                    meta_scalars=meta_scalars,
                    n_steps_input=n_steps_input,
                    storage_options=storage_kwargs,
                    boundary_return_type=boundary_return_type,
                    max_lead_time=max_lead_time,
                    fixed_lead_time=k,
                    **(
                        dataset_kws["rollout_test"]
                        if dataset_kws is not None and "rollout_test" in dataset_kws
                        else {}
                    ),
                )
            # Set single rollout datasets to None (not used in lead_time_mode)
            self.rollout_val_dataset = None
            self.rollout_test_dataset = None
        else:
            # Regular mode: use WellDataset (original behavior)
            self.train_dataset = train_dataset(
                **path_kwargs,
                well_split_name="train",
                include_filters=include_filters,
                exclude_filters=exclude_filters,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                stats=stats,
                meta_scalars=meta_scalars,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                storage_options=storage_kwargs,
                min_dt_stride=min_dt_stride,
                max_dt_stride=max_dt_stride,
                restrict_num_trajectories=restrict_num_trajectories,
                restrict_num_samples=restrict_num_samples,
                restriction_seed=restriction_seed,
                boundary_return_type=boundary_return_type,
                transform=transform,
                **(
                    dataset_kws["train"]
                    if dataset_kws is not None and "train" in dataset_kws
                    else {}
                ),
            )
            self.val_dataset = eval_dataset(
                **path_kwargs,
                well_split_name="valid",
                include_filters=include_filters,
                exclude_filters=exclude_filters,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                stats=stats,
                meta_scalars=meta_scalars,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                storage_options=storage_kwargs,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride,
                boundary_return_type=boundary_return_type,
                **(
                    dataset_kws["val"]
                    if dataset_kws is not None and "val" in dataset_kws
                    else {}
                ),
            )
            self.rollout_val_dataset = eval_dataset(
                **path_kwargs,
                well_split_name="valid",
                include_filters=include_filters,
                exclude_filters=exclude_filters,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                stats=stats,
                meta_scalars=meta_scalars,
                max_rollout_steps=max_rollout_steps,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                full_trajectory_mode=True,
                storage_options=storage_kwargs,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride,
                boundary_return_type=boundary_return_type,
                **(
                    dataset_kws["rollout_val"]
                    if dataset_kws is not None and "rollout_val" in dataset_kws
                    else {}
                ),
            )
            self.test_dataset = eval_dataset(
                **path_kwargs,
                well_split_name="test",
                include_filters=include_filters,
                exclude_filters=exclude_filters,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                stats=stats,
                meta_scalars=meta_scalars,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                storage_options=storage_kwargs,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride,
                boundary_return_type=boundary_return_type,
                **(
                    dataset_kws["test"]
                    if dataset_kws is not None and "test" in dataset_kws
                    else {}
                ),
            )
            self.rollout_test_dataset = eval_dataset(
                **path_kwargs,
                well_split_name="test",
                include_filters=include_filters,
                exclude_filters=exclude_filters,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
                stats=stats,
                meta_scalars=meta_scalars,
                max_rollout_steps=max_rollout_steps,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                full_trajectory_mode=True,
                storage_options=storage_kwargs,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride,
                boundary_return_type=boundary_return_type,
                **(
                    dataset_kws["rollout_test"]
                    if dataset_kws is not None and "rollout_test" in dataset_kws
                    else {}
                ),
            )
            # Set dict rollout datasets to empty (not used in regular mode)
            self._rollout_val_datasets = {}
            self._rollout_test_datasets = {}

        self.path = path
        # These are not Well datasets, so the naming key is `dataset_name`
        # (not `well_dataset_name`); used only for experiment/run naming.
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.world_size = world_size
        self.data_workers = data_workers
        self.rank = rank

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.path}>"
