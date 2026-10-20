"""Lightweight DDP helpers.

Reads `RANK` / `LOCAL_RANK` / `WORLD_SIZE` from the env (set by torchrun) and
initialises NCCL when world_size > 1. On a single process, all helpers
degrade to no-ops so single-GPU code paths keep working unchanged.
"""
from __future__ import annotations

import datetime
import logging
import os

import torch
import torch.distributed as dist


logger = logging.getLogger("dandelion")


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_distributed() -> bool:
    return get_world_size() > 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> torch.device:
    """Init NCCL on multi-process launches; pin this rank to its GPU.

    Returns the per-rank ``torch.device``. Safe to call once at the top of
    main() — a no-op when launched outside torchrun.
    """
    local_rank = get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_distributed() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # Collective timeout in minutes (default 10, torch's NCCL default). A long
        # data-loader stall on one rank (e.g. a slow shared filesystem) otherwise
        # trips the watchdog and kills the run; raise it via the env var.
        timeout_min = float(os.environ.get("DANDELION_DIST_TIMEOUT_MIN", "10"))
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=timeout_min))
        logger.info(
            "ddp: initialised %s backend (rank %d / %d, local_rank %d, device %s)",
            backend, get_rank(), get_world_size(), local_rank, device,
        )
    return device


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()
