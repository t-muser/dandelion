"""Training loop: train, validation, rollout and test loops on
``(input, target)`` batches from an ``AbstractDataModule``, single-GPU or DDP.

Metrics are computed in ``dandelion.metrics`` and pushed through an optional
``log_fn`` callback (``(dict, step) -> None``) so ``dandelion.train`` can route
them to wandb without this module depending on wandb.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Optional

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from dandelion.data.datamodule import AbstractDataModule
from dandelion.dist import barrier, is_main_process
from dandelion.metrics import (
    compute_loss_metrics,
    grad_norm,
    latitude_weights,
    param_norm,
)

logger = logging.getLogger("dandelion")

LogFn = Callable[[dict, int], None]


def _noop_log(metrics: dict, step: int) -> None:
    pass


class Trainer:
    """Single-device train/val loop on ``(input, target)`` tensors.

    Batches may be ``(x, y)`` tuples or dicts with ``input_fields`` /
    ``output_fields``. Tensors may be ``(B, C, H, W)`` (single-step) or
    ``(B, T, C, H, W)`` (time-folded into channels before forward).

    The trainer logs a compact set of training scalars every epoch
    (``train/loss``, ``train/grad_norm``, ``train/param_norm``,
    ``train/lr``, ``train/samples_per_sec``, ``epoch_time_s``) and a
    broader spherical metric suite at validation (``valid/loss``,
    ``valid/mse`` / ``mse_sphere`` / ``rmse_sphere`` / ``rel_l2`` plus
    per-field variants). Every metric flows through ``log_fn``.
    """

    def __init__(
        self,
        checkpoint_folder: str,
        artifact_folder: str,
        viz_folder: str,
        model: torch.nn.Module,
        datamodule: AbstractDataModule,
        optimizer: Optional[torch.optim.Optimizer],
        loss_fn: Callable,
        epochs: int,
        ar_epochs: int = 0,
        ar_steps: int = 1,
        checkpoint_frequency: int = 1,
        val_frequency: int = 1,
        rollout_val_frequency: int = 1,
        max_rollout_steps: int = 1,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device("cpu"),
        checkpoint_path: str = "",
        log_fn: LogFn = _noop_log,
        grad_clip: Optional[float] = None,
    ):
        self.checkpoint_folder = checkpoint_folder
        self.artifact_folder = artifact_folder
        self.viz_folder = viz_folder
        self.model = model
        self.datamodule = datamodule
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.device = device

        self.epochs = epochs
        self.ar_epochs = ar_epochs
        self.ar_steps = ar_steps
        self.max_epoch = epochs + ar_epochs
        self.checkpoint_frequency = checkpoint_frequency
        self.val_frequency = val_frequency
        self.rollout_val_frequency = rollout_val_frequency
        self.max_rollout_steps = max_rollout_steps

        self.log_fn = log_fn
        self.grad_clip = grad_clip

        self.starting_epoch = 1
        self.best_val_loss: Optional[float] = None
        self.starting_val_loss = float("inf")
        self._global_step = 0

        md = getattr(datamodule, "metadata", None)
        nlat = md.spatial_resolution[0] if md is not None else None
        grid = getattr(md, "grid", "equiangular") if md is not None else "equiangular"
        self.field_names = list(getattr(md, "field_names", ())) if md is not None else []
        # Per-frame channel count for the time-varying field stream — used
        # by AR / rollout loops to slide a 4-frame buffer through the
        # channel dim. Matches `md.dim_out` in the WellDataModule adapter.
        self._n_field_channels = int(getattr(md, "dim_out", 0)) if md is not None else 0
        # Per-frame INPUT channel count. Exceeds dim_out when every input frame
        # carries input-only channels after the predicted fields (global-ocean-3D:
        # depth + sin/cos day of year); the slide then works in whole input
        # frames (see _slide).
        self._n_input_channels = int(getattr(md, "dim_in", 0)) if md is not None else 0
        self.lat_weights = (
            latitude_weights(nlat, grid=grid, device=device) if nlat else None
        )
        self.denormalize_fn = getattr(datamodule, "denormalize_fn", None)

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    @staticmethod
    def _split_batch(batch: Any):
        """Return ``(x, y, mask)`` where ``mask`` is None unless the
        dataloader supplied a ``valid_mask`` (per-channel ocean mask)."""
        if isinstance(batch, dict):
            x = batch.get("input_fields")
            y = batch.get("output_fields")
            mask = batch.get("valid_mask")
            if x is None or y is None:
                raise KeyError(
                    "dict batch must contain 'input_fields' and 'output_fields'"
                )
            return x, y, mask
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1], None
        raise TypeError(f"unsupported batch type: {type(batch).__name__}")

    def _slide(self, hist, y_pred, forcing=None, k=0):
        """Advance the folded ``(B, T_in*C_in, H, W)`` history by one frame.

        Without input-only channels (``dim_in == dim_out``) this drops the
        oldest frame and appends ``y_pred``. With them, each input frame is
        ``dim_in`` channels wide: drop one whole frame and append
        ``[y_pred, forcing_k]``, where ``forcing_k`` is the loader's
        ``output_forcing[:, k]`` (the forcing at the predicted frame's time) or,
        if the loader does not supply it, the newest input frame's forcing.
        """
        C, S = self._n_field_channels, self._n_input_channels
        if S <= C:
            return torch.cat([hist[:, C:], y_pred], dim=1)
        f_k = forcing[:, k] if forcing is not None else hist[:, -(S - C):]
        return torch.cat([hist[:, S:], y_pred, f_k], dim=1)

    @staticmethod
    def _batch_forcing(batch: Any):
        """``output_forcing`` (B, T_out, F, H, W) if the loader supplies it."""
        return batch.get("output_forcing") if isinstance(batch, dict) else None

    def _compute_loss(self, y_pred, y_target, mask=None):
        """Call the configured loss with an optional mask kwarg.

        Falls back to a plain call when the loss does not accept ``mask``
        (e.g. ``torch.nn.MSELoss``)."""
        if mask is None:
            return self.loss_fn(y_pred, y_target)
        try:
            return self.loss_fn(y_pred, y_target, mask=mask)
        except TypeError:
            sq = (y_pred - y_target).pow(2) * mask
            return sq.sum() / mask.sum().clamp(min=1.0)

    @staticmethod
    def _fold_time_into_channels(x: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) → (B, T*C, H, W); (B, C, H, W) passes through."""
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B, T * C, H, W)
        return x

    def _unwrap_model(self) -> torch.nn.Module:
        """Strip the DDP wrapper so save/load operates on the inner module
        (DDP state dicts have an extra "module." key prefix that won't
        match a freshly instantiated bare model on test_mode reloads)."""
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module
        return self.model

    def save_model(self, epoch: int, validation_loss: float, output_path: str):
        if not is_main_process():
            return
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self._unwrap_model().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "validation_loss": validation_loss,
            "best_validation_loss": self.best_val_loss,
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        torch.save(checkpoint, output_path)

    def load_checkpoint(self, checkpoint_path: str):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self._unwrap_model().load_state_dict(ckpt["model_state_dict"])
        if self.optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.lr_scheduler is not None and ckpt.get("lr_scheduler_state_dict") is not None:
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
        self.best_val_loss = ckpt.get("best_validation_loss")
        self.starting_val_loss = ckpt.get("validation_loss", float("inf"))
        self.starting_epoch = ckpt["epoch"] + 1

    def train_one_epoch(self, epoch: int, dataloader: DataLoader) -> dict:
        self.model.train()
        running_loss = 0.0
        running_grad = 0.0
        n_batches = len(dataloader)
        n_samples = 0
        t0 = time.time()
        for i, batch in enumerate(dataloader):
            x, y, mask = self._split_batch(batch)
            x = self._fold_time_into_channels(x.to(self.device))
            y = y.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)
            # Single-step training predicts one frame. With n_steps_output>1
            # (e.g. when phase-2 AR also runs in this trainer) ignore the
            # extra GT frames here.
            if y.dim() == 5:
                y = y[:, 0]
            y_pred = self.model(x)
            loss = self._compute_loss(y_pred, y, mask=mask)
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            gn = grad_norm(self.model.parameters())
            self.optimizer.step()
            running_loss += loss.item() / max(n_batches, 1)
            running_grad += gn / max(n_batches, 1)
            n_samples += x.shape[0]
            self._global_step += 1
            logger.info(
                f"Epoch {epoch}, batch {i + 1}/{n_batches}: "
                f"loss {loss.item():.6g} grad {gn:.4g}"
            )
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        dt = time.time() - t0
        lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0
        logs = {
            "train/loss": running_loss,
            "train/grad_norm": running_grad,
            "train/param_norm": param_norm(self.model.parameters()),
            "train/lr": lr,
            "train/samples_per_sec": n_samples / max(dt, 1e-9),
            "epoch_time_s": dt,
            "epoch": epoch,
        }
        logger.info(
            f"Epoch {epoch}: avg train loss {running_loss:.6g} grad {running_grad:.4g} "
            f"in {dt:.2f}s ({logs['train/samples_per_sec']:.2f} samples/s)"
        )
        return logs

    @torch.inference_mode()
    def validation_loop(self, dataloader: DataLoader, tag: str = "valid") -> dict:
        self.model.eval()
        n_batches = len(dataloader)
        agg: dict[str, float] = {}
        loss_total = 0.0
        for batch in dataloader:
            x, y, mask = self._split_batch(batch)
            x = self._fold_time_into_channels(x.to(self.device))
            y = y.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)
            if y.dim() == 5:
                y = y[:, 0]
            y_pred = self.model(x)
            loss_total += float(self._compute_loss(y_pred, y, mask=mask).item())
            if self.denormalize_fn is not None:
                y_pred_m = self.denormalize_fn(y_pred)
                y_m = self.denormalize_fn(y)
            else:
                y_pred_m, y_m = y_pred, y
            m = compute_loss_metrics(
                y_pred_m, y_m,
                lat_weights=self.lat_weights,
                field_names=self.field_names or None,
                mask=mask,
            )
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
        denom = max(n_batches, 1)
        logs = {f"{tag}/{k}": v / denom for k, v in agg.items()}
        logs[f"{tag}/loss"] = loss_total / denom
        logger.info(
            f"{tag}: loss {logs[f'{tag}/loss']:.6g} "
            f"vrmse {logs.get(f'{tag}/vrmse', float('nan')):.6g} "
            f"rmse_sphere {logs.get(f'{tag}/rmse_sphere', float('nan')):.6g} "
            f"rel_l2 {logs.get(f'{tag}/rel_l2', float('nan')):.6g}"
        )
        return logs

    def train_one_epoch_ar(
        self, epoch: int, dataloader: DataLoader, ar_steps: int
    ) -> dict:
        """Autoregressive training: forward the model `ar_steps` times,
        feeding each prediction back as the next-step input. Loss is the
        mean across steps. Requires the loader to yield ``ar_steps`` GT
        output frames (i.e. ``data.n_steps_output >= ar_steps``).
        """
        self.model.train()
        running_loss = 0.0
        running_grad = 0.0
        n_batches = len(dataloader)
        n_samples = 0
        t0 = time.time()
        C = self._n_field_channels
        for i, batch in enumerate(dataloader):
            x, y, mask = self._split_batch(batch)
            forcing = self._batch_forcing(batch)
            if forcing is not None:
                forcing = forcing.to(self.device)
            hist = self._fold_time_into_channels(x.to(self.device))  # (B, T_in*C, H, W)
            y = y.to(self.device)  # (B, T_out, C, H, W)
            if mask is not None:
                mask = mask.to(self.device)
            if y.dim() != 5 or y.shape[1] < ar_steps:
                raise ValueError(
                    f"AR training needs y of shape (B, T>=ar_steps, C, H, W); "
                    f"got {tuple(y.shape)} with ar_steps={ar_steps}"
                )
            loss = 0.0
            for k in range(ar_steps):
                y_pred = self.model(hist)  # (B, C, H, W)
                loss = loss + self._compute_loss(y_pred, y[:, k], mask=mask)
                if k < ar_steps - 1:
                    hist = self._slide(hist, y_pred, forcing, k)
            loss = loss / ar_steps
            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            gn = grad_norm(self.model.parameters())
            self.optimizer.step()
            running_loss += loss.item() / max(n_batches, 1)
            running_grad += gn / max(n_batches, 1)
            n_samples += hist.shape[0]
            self._global_step += 1
            logger.info(
                f"Epoch {epoch} (AR x{ar_steps}), batch {i + 1}/{n_batches}: "
                f"loss {loss.item():.6g} grad {gn:.4g}"
            )
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        dt = time.time() - t0
        lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0
        logs = {
            "train/loss": running_loss,
            "train/grad_norm": running_grad,
            "train/param_norm": param_norm(self.model.parameters()),
            "train/lr": lr,
            "train/samples_per_sec": n_samples / max(dt, 1e-9),
            "train/ar_steps": ar_steps,
            "epoch_time_s": dt,
            "epoch": epoch,
        }
        logger.info(
            f"Epoch {epoch} (AR x{ar_steps}): avg train loss {running_loss:.6g} "
            f"grad {running_grad:.4g} in {dt:.2f}s "
            f"({logs['train/samples_per_sec']:.2f} samples/s)"
        )
        return logs

    @torch.inference_mode()
    def rollout_loop(
        self, dataloader: DataLoader, max_steps: int, tag: str = "rollout_val"
    ) -> dict:
        """Autoregressive rollout on full-trajectory batches.

        Expects ``input_fields`` of shape ``(B, T_in, C, H, W)`` (warm-up
        history) and ``output_fields`` of shape ``(B, T_out, C, H, W)``
        (ground truth). Rolls the model forward up to
        ``min(max_steps, T_out)`` and logs per-step metrics under
        ``{tag}/step{k}/<metric>`` plus an aggregated ``{tag}/<metric>``.
        """
        self.model.eval()
        C = self._n_field_channels
        agg: dict[str, float] = {}
        per_step_agg: dict[int, dict[str, float]] = {}
        n_batches = 0
        for batch in dataloader:
            x, y, mask = self._split_batch(batch)
            forcing = self._batch_forcing(batch)
            if forcing is not None:
                forcing = forcing.to(self.device)
            hist = self._fold_time_into_channels(x.to(self.device))
            y = y.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)
            if y.dim() != 5:
                raise ValueError(
                    f"rollout_loop expects y of shape (B, T, C, H, W); "
                    f"got {tuple(y.shape)}"
                )
            T_out = y.shape[1]
            steps = min(max_steps, T_out)
            for k in range(steps):
                y_pred = self.model(hist)
                if self.denormalize_fn is not None:
                    y_pred_m = self.denormalize_fn(y_pred)
                    y_k_m = self.denormalize_fn(y[:, k])
                else:
                    y_pred_m, y_k_m = y_pred, y[:, k]
                m = compute_loss_metrics(
                    y_pred_m,
                    y_k_m,
                    lat_weights=self.lat_weights,
                    field_names=self.field_names or None,
                    mask=mask,
                )
                step_acc = per_step_agg.setdefault(k, {})
                for kk, vv in m.items():
                    step_acc[kk] = step_acc.get(kk, 0.0) + vv
                    agg[kk] = agg.get(kk, 0.0) + vv
                if k < steps - 1:
                    hist = self._slide(hist, y_pred, forcing, k)
            n_batches += 1
        denom = max(n_batches, 1)
        total_obs = denom * max(len(per_step_agg), 1)
        logs: dict[str, float] = {}
        for kk, vv in agg.items():
            logs[f"{tag}/{kk}"] = vv / max(total_obs, 1)
        for k, step_acc in per_step_agg.items():
            for kk, vv in step_acc.items():
                logs[f"{tag}/step{k + 1}/{kk}"] = vv / denom
        logger.info(
            f"{tag}: vrmse {logs.get(f'{tag}/vrmse', float('nan')):.6g} "
            f"rmse_sphere {logs.get(f'{tag}/rmse_sphere', float('nan')):.6g} "
            f"rel_l2 {logs.get(f'{tag}/rel_l2', float('nan')):.6g} "
            f"over {len(per_step_agg)} steps × {denom} batches"
        )
        return logs

    def train(self):
        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()
        # Rollout loaders are not distributed; only rank 0 runs them.
        rollout_val_loader = (
            self.datamodule.rollout_val_dataloader() if is_main_process() else None
        )
        val_loss = self.starting_val_loss
        for epoch in range(self.starting_epoch, self.max_epoch + 1):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            if epoch <= self.epochs:
                train_logs = self.train_one_epoch(epoch, train_loader)
            else:
                train_logs = self.train_one_epoch_ar(
                    epoch, train_loader, self.ar_steps
                )
            self.log_fn(train_logs, epoch)
            if epoch % self.val_frequency == 0 or epoch == self.max_epoch:
                val_logs = self.validation_loop(val_loader, tag="valid")
                val_loss = val_logs["valid/loss"]
                self.log_fn({**val_logs, "epoch": epoch}, epoch)
                if self.best_val_loss is None or val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model(epoch, val_loss, os.path.join(self.checkpoint_folder, "best.pt"))
            if (epoch % self.rollout_val_frequency == 0 or epoch == self.max_epoch) and is_main_process():
                rollout_logs = self.rollout_loop(
                    rollout_val_loader,
                    max_steps=self.max_rollout_steps,
                    tag="rollout_val",
                )
                self.log_fn({**rollout_logs, "epoch": epoch}, epoch)
            if epoch % self.checkpoint_frequency == 0 or epoch == self.max_epoch:
                self.save_model(epoch, val_loss, os.path.join(self.checkpoint_folder, "recent.pt"))
            # rollout_loop and save_model run on rank 0 only; resync before
            # the next epoch so the other ranks don't race ahead into a
            # collective and trip the NCCL watchdog.
            barrier()
        logger.info(
            f"Training complete. Final param norm: {param_norm(self.model.parameters()):.3f}"
        )

    def test(self):
        """Evaluate on the held-out test split.

        Assumes the caller already loaded ``best.pt`` (via the
        ``checkpoint_path`` constructor arg). Logs single-step metrics
        under ``test/`` and rollout metrics under ``rollout_test/``.
        """
        test_loader = self.datamodule.test_dataloader()
        rollout_test_loader = self.datamodule.rollout_test_dataloader()
        single_logs = self.validation_loop(test_loader, tag="test")
        self.log_fn({**single_logs, "epoch": self.max_epoch}, self.max_epoch)
        rollout_logs = self.rollout_loop(
            rollout_test_loader,
            max_steps=self.max_rollout_steps,
            tag="rollout_test",
        )
        self.log_fn({**rollout_logs, "epoch": self.max_epoch}, self.max_epoch)
        logger.info("Test complete.")
