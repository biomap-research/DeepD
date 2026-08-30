"""Learning rate schedulers (warmup + cosine decay)."""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR

from src.config import Config


def lr_multiplier_at_step(step: int, cfg: Config) -> float:
    """Return LR multiplier relative to cfg.train.lr at global step (0-indexed)."""
    warmup = cfg.train.warmup_steps
    max_steps = cfg.train.max_steps
    min_ratio = cfg.train.min_lr_ratio

    if cfg.train.lr_schedule == "constant":
        return 1.0

    if warmup > 0 and step < warmup:
        return (step + 1) / warmup

    if max_steps <= warmup:
        return min_ratio

    progress = (step - warmup) / max(max_steps - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Config,
) -> LambdaLR | None:
    if cfg.train.lr_schedule == "constant":
        return None

    if cfg.train.lr_schedule != "warmup_cosine":
        raise ValueError(f"Unknown lr_schedule: {cfg.train.lr_schedule}")

    def lr_lambda(step: int) -> float:
        return lr_multiplier_at_step(step, cfg)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
