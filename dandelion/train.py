"""Train / test entry point.

Usage:
    uv run python -m dandelion.train \\
        --data configs/data/shock-caps.yaml \\
        --model configs/models/dandelion.yaml \\
        --train configs/train.yaml \\
        trainer.epochs=2

Configs merge in order: data (base) < train (defaults) < model (highest
priority). CLI overrides are merged last.

wandb is optional. It is gated on ``cfg.wandb.enabled`` (default false)
so smoke runs on offline nodes don't hang trying to reach the cloud.
When enabled, ``cfg.wandb.mode`` ("online" | "offline" | "disabled")
and ``cfg.wandb.project`` apply. Offline runs accumulate locally in
``<experiment_folder>/wandb/`` and can be synced later with
``wandb sync``.
"""
from __future__ import annotations

import argparse
import logging
import os
import os.path as osp
from typing import Callable

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from dandelion.data.datamodule import AbstractDataModule
from dandelion.dist import (
    cleanup_distributed,
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    setup_distributed,
)
from dandelion.metrics import LatitudeWeightedMSELoss, latitude_weights
from dandelion.trainer import Trainer
from dandelion.utils import configure_experiment


def _patch_torch_harmonics_compile() -> None:
    # SpectralConvS2._contract_lwise is decorated with @torch.compile in
    # torch_harmonics. Under DDP + AR (multi-forward, single-backward) the
    # compiled callable's saved-for-backward references to self.weight end
    # up at a version one ahead of what autograd captured during the first
    # forward, raising "variable modified by inplace op: ... version V+1;
    # expected V". Replacing the method with the bare einsum (same math,
    # no compile) avoids the issue. Single-GPU never tripped this.
    from torch_harmonics import spectral_convolution as _sc

    def _contract_lwise_uncompiled(self, ac, bc):
        return torch.einsum("bgixy,giox->bgoxy", ac, bc)

    _sc.SpectralConvS2._contract_lwise = _contract_lwise_uncompiled


_patch_torch_harmonics_compile()


logger = logging.getLogger("dandelion")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(_h)


def load_modular_config(data_config: str, model_config: str, train_config: str) -> DictConfig:
    logger.info(f"Loading configs: data={data_config} model={model_config} train={train_config}")
    data_cfg = OmegaConf.load(data_config)
    model_cfg = OmegaConf.load(model_config)
    train_cfg = OmegaConf.load(train_config)
    return OmegaConf.merge(data_cfg, train_cfg, model_cfg)


def build_model(cfg: DictConfig, datamodule: AbstractDataModule) -> torch.nn.Module:
    md = datamodule.metadata
    # `model_predict_steps` decouples model out_chans from the number of GT
    # frames returned by the loader: AR training calls the model multiple
    # times with `n_steps_output` GT targets but the architecture itself
    # always emits one frame.
    n_predict = cfg.get("model_predict_steps", cfg.data.get("n_steps_output", 1))
    n_input_fields = md.dim_in * cfg.data.get("n_steps_input", 1)
    n_output_fields = md.dim_out * n_predict
    logger.info(
        f"Model in/out: dim_in={n_input_fields}, dim_out={n_output_fields}, "
        f"res={md.spatial_resolution}"
    )
    return instantiate(
        cfg.model,
        inp_shape=tuple(md.spatial_resolution),
        inp_chans=n_input_fields,
        out_chans=n_output_fields,
    )


def setup_wandb(cfg: DictConfig, experiment_folder: str, experiment_name: str) -> Callable[[dict, int], None]:
    """Initialise wandb if enabled, otherwise return a no-op logger.

    Returns a ``(metrics_dict, step) -> None`` callback the trainer uses
    to push metrics per epoch. Under DDP only rank 0 ever talks to wandb
    so we don't get N parallel offline run dirs / N login attempts.
    """
    if not is_main_process():
        return lambda metrics, step: None
    wandb_cfg = OmegaConf.select(cfg, "wandb") or OmegaConf.create({})
    enabled = bool(OmegaConf.select(wandb_cfg, "enabled"))
    if not enabled:
        logger.info("wandb: disabled")
        return lambda metrics, step: None

    import wandb

    mode = OmegaConf.select(wandb_cfg, "mode") or "online"
    project = OmegaConf.select(wandb_cfg, "project") or "dandelion"
    entity = OmegaConf.select(wandb_cfg, "entity") or None
    # Default group = dataset name so all runs on the same dataset land
    # together on the wandb project page. Fall back to well_dataset_name for
    # the HDF5/the_well datamodules, which expose that instead of
    # dataset_name (mirrors get_experiment_name in dandelion.utils).
    dataset_name = OmegaConf.select(cfg, "data.dataset_name") or OmegaConf.select(
        cfg, "data.well_dataset_name"
    )
    group = OmegaConf.select(wandb_cfg, "group") or dataset_name
    tags = list(OmegaConf.select(wandb_cfg, "tags") or [])
    if dataset_name and dataset_name not in tags:
        tags.append(dataset_name)

    wandb_dir = osp.join(experiment_folder, "wandb")
    os.makedirs(wandb_dir, exist_ok=True)
    logger.info(f"wandb: init mode={mode} project={project} dir={wandb_dir}")
    wandb.init(
        dir=experiment_folder,
        project=project,
        entity=entity,
        group=group,
        tags=tags,
        name=experiment_name,
        mode=mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    def _log(metrics: dict, step: int) -> None:
        wandb.log(metrics, step=step)

    return _log


def run_main(cfg: DictConfig):
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()
    if not is_main_process():
        # Demote non-zero ranks to WARNING so the console is one rank's
        # voice; they'd otherwise interleave 4× duplicate per-batch lines.
        logger.setLevel(logging.WARNING)

    # Folder allocation must happen on a single rank — otherwise each rank
    # races os.listdir + makedirs and they pick different sequence numbers,
    # so train writes ckpt to /N while test ends up looking at /N+1. Rank 0
    # decides; we broadcast the resolved (immutable) paths to every rank.
    if is_main_process():
        cfg, experiment_name, experiment_folder, ckpt_folder, art_folder, viz_folder = (
            configure_experiment(cfg, logger)
        )
        bcast_payload = [experiment_name, experiment_folder, ckpt_folder, art_folder, viz_folder, cfg.trainer.get("checkpoint_path", "")]
    else:
        bcast_payload = [None, None, None, None, None, None]
    if is_distributed():
        import torch.distributed as dist
        dist.broadcast_object_list(bcast_payload, src=0)
        if not is_main_process():
            experiment_name, experiment_folder, ckpt_folder, art_folder, viz_folder, ckpt_path = bcast_payload
            if "trainer" in cfg:
                cfg.trainer.checkpoint_path = ckpt_path
    logger.info(f"Experiment: {experiment_name} at {experiment_folder}")
    if is_main_process():
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    if is_distributed():
        logger.info(f"DDP: rank {rank}/{world_size} on {device}")

    if bool(OmegaConf.select(cfg, "test_mode")):
        recent_ckpt = osp.join(ckpt_folder, "recent.pt")
        if not osp.isfile(recent_ckpt):
            raise FileNotFoundError(
                f"test_mode=true but no recent.pt at {recent_ckpt}. "
                "Make sure the train job ran to completion in this folder."
            )
        cfg.trainer.checkpoint_path = recent_ckpt
        logger.info(f"test_mode: using final checkpoint {recent_ckpt}")

    logger.info(f"Instantiate datamodule {cfg.data._target_}")
    datamodule: AbstractDataModule = instantiate(cfg.data)
    # Inject rank/world_size after construction; the datamodules read them
    # when building their DistributedSamplers, so a post-hoc set is effective.
    if hasattr(datamodule, "rank"):
        datamodule.rank = rank
    if hasattr(datamodule, "world_size"):
        datamodule.world_size = world_size

    model = build_model(cfg, datamodule)
    model = model.to(device)
    if is_distributed():
        # broadcast_buffers=False: DDP's default per-iteration buffer sync
        # writes into register_buffer tensors in-place. torch_harmonics SHT
        # layers store their precomputed Legendre weights as buffers and
        # capture them in the autograd graph via einsum, so the broadcast
        # bumps the buffer's version *between* AR steps and backward fails
        # with "variable modified by inplace operation: ... version V+1;
        # expected V". Buffers here are deterministic from layer __init__
        # args, so skipping the per-iter broadcast is correct.
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[get_local_rank()],
            find_unused_parameters=False, broadcast_buffers=False,
        )

    logger.info(f"Instantiate optimizer {cfg.optimizer._target_}")
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    lr_scheduler = None
    if "lr_scheduler" in cfg:
        logger.info(f"Instantiate lr_scheduler {cfg.lr_scheduler._target_}")
        lr_scheduler = instantiate(cfg.lr_scheduler, optimizer=optimizer)

    loss_fn = instantiate(cfg.loss) if "loss" in cfg else torch.nn.MSELoss()
    if isinstance(loss_fn, LatitudeWeightedMSELoss) and "lat_weights" not in loss_fn._buffers:
        md = datamodule.metadata
        nlat = md.spatial_resolution[0]
        grid = getattr(md, "grid", "equiangular")
        loss_fn.set_lat_weights(latitude_weights(nlat, grid=grid))
        logger.info(f"loss: lat-weighted MSE on {nlat}-pt {grid} grid")
    loss_fn = loss_fn.to(device) if isinstance(loss_fn, torch.nn.Module) else loss_fn

    if is_main_process():
        with open(osp.join(experiment_folder, "extended_config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

    log_fn = setup_wandb(cfg, experiment_folder, experiment_name)

    trainer: Trainer = instantiate(
        cfg.trainer,
        checkpoint_folder=ckpt_folder,
        artifact_folder=art_folder,
        viz_folder=viz_folder,
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_fn=loss_fn,
        device=device,
        log_fn=log_fn,
    )
    if bool(OmegaConf.select(cfg, "test_mode")):
        trainer.test()
    else:
        trainer.train()

    if is_main_process() and bool(OmegaConf.select(cfg, "wandb.enabled")):
        import wandb
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train / test dandelion models", add_help=False)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--train", type=str, required=True)
    args, unknown = parser.parse_known_args()

    cfg = load_modular_config(args.data, args.model, args.train)
    if unknown:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(unknown))
    run_main(cfg)
