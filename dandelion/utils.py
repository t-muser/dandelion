"""Experiment directory setup and LR schedule helpers."""
from __future__ import annotations

import logging
import os
import os.path as osp

import torch
from omegaconf import DictConfig, OmegaConf


def build_warmup_cosine(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    T_max: int,
    eta_min: float = 0.0,
    start_factor: float = 1.0e-3,
) -> torch.optim.lr_scheduler.SequentialLR:
    """Linear warmup for ``warmup_epochs`` then cosine anneal.

    ``T_max`` is the total epoch count (warmup + cosine); the cosine phase
    runs for ``T_max - warmup_epochs`` epochs so callers can keep passing
    the run's total epoch count, matching the prior CosineAnnealingLR setup.
    """
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(T_max - warmup_epochs, 1), eta_min=eta_min
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def configure_paths(experiment_folder: str):
    checkpoint_folder = osp.join(experiment_folder, "checkpoints")
    artifact_folder = osp.join(experiment_folder, "artifacts")
    viz_folder = osp.join(experiment_folder, "viz")
    for p in (checkpoint_folder, artifact_folder, viz_folder):
        os.makedirs(p, exist_ok=True)
    return checkpoint_folder, artifact_folder, viz_folder


def get_experiment_name(cfg: DictConfig) -> str:
    model_target = OmegaConf.select(cfg, "model._target_") or "model"
    model_name = model_target.split(".")[-1]
    data_name = (
        OmegaConf.select(cfg, "data.dataset_name")
        or OmegaConf.select(cfg, "data.well_dataset_name")
        or (OmegaConf.select(cfg, "data._target_") or "data").split(".")[-1]
    )
    cfg_name = OmegaConf.select(cfg, "name") or "run"
    lr = OmegaConf.select(cfg, "optimizer.lr") or "na"
    return f"{data_name}-{cfg_name}-{model_name}-{lr}"


def configure_experiment(
    cfg: DictConfig, logger: logging.Logger
) -> tuple[DictConfig, str, str, str, str, str]:
    """Resolve paths for a new run (or resume, if a checkpoint exists).

    Honors ``cfg.folder_override`` / ``cfg.checkpoint_override`` /
    ``cfg.config_override`` if present; otherwise builds a fresh folder
    under ``cfg.experiment_dir``.
    """
    experiment_name = get_experiment_name(cfg)
    experiment_dir = OmegaConf.select(cfg, "experiment_dir") or "runs"
    base_experiment_folder = osp.join(experiment_dir, experiment_name)

    folder_override = OmegaConf.select(cfg, "folder_override") or ""
    checkpoint_override = OmegaConf.select(cfg, "checkpoint_override") or ""
    config_override = OmegaConf.select(cfg, "config_override") or ""
    auto_resume = bool(OmegaConf.select(cfg, "auto_resume"))

    experiment_folder = folder_override
    if not experiment_folder:
        if osp.exists(base_experiment_folder):
            prev_runs = sorted(
                (d for d in os.listdir(base_experiment_folder) if d.isdigit()),
                key=lambda x: int(x),
            )
        else:
            prev_runs = []
        if auto_resume and prev_runs:
            experiment_folder = osp.join(base_experiment_folder, prev_runs[-1])
        else:
            experiment_folder = osp.join(base_experiment_folder, str(len(prev_runs)))
    os.makedirs(experiment_folder, exist_ok=True)

    checkpoint_file = checkpoint_override
    if not checkpoint_file:
        last = osp.join(experiment_folder, "checkpoints", "recent.pt")
        if osp.isfile(last):
            checkpoint_file = last
    if checkpoint_file:
        logger.info(f"Using checkpoint {checkpoint_file}")

    if config_override:
        cfg = OmegaConf.load(config_override)
    if "trainer" in cfg:
        cfg.trainer.checkpoint_path = checkpoint_file

    checkpoint_folder, artifact_folder, viz_folder = configure_paths(experiment_folder)
    return cfg, experiment_name, experiment_folder, checkpoint_folder, artifact_folder, viz_folder
