"""Loss functions with mask support."""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn.functional as F

from src.config import Config

logger = logging.getLogger(__name__)

LossFn = Callable[..., torch.Tensor]

_REGISTRY: dict[str, LossFn] = {}


def register_loss(name: str):
    def decorator(fn: LossFn) -> LossFn:
        _REGISTRY[name] = fn
        return fn
    return decorator


def build_loss_mask(
    padding_mask: torch.Tensor | None,
    targets: torch.Tensor,
    cfg: Config,
    is_training: bool,
) -> torch.Tensor:
    """Combine padding mask with optional stochastic zero-label mask (train only)."""
    if padding_mask is None:
        mask = torch.ones_like(targets, dtype=torch.bool)
    else:
        mask = padding_mask.bool()

    if not is_training or not cfg.loss.zero_label_mask:
        return mask

    is_zero = targets.abs() < cfg.loss.zero_label_eps
    rand = torch.rand_like(targets.float())
    drop = is_zero & (rand < cfg.loss.zero_label_drop_prob)
    return mask & ~drop


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.float().sum()
    if denom < 1:
        logger.warning("No tokens in loss mask; returning zero loss")
        return loss.sum() * 0.0
    return (loss * mask.float()).sum() / denom


def _classification_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
) -> torch.Tensor:
    if cfg.task.mode == "multi_label":
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)
        if preds.dim() == 1:
            preds = preds.unsqueeze(0)
        return F.binary_cross_entropy_with_logits(preds.float(), targets.float())

    if targets.dim() > 1:
        targets = targets.squeeze(-1)
    return F.cross_entropy(preds, targets.long())


@register_loss("cross_entropy")
def cross_entropy_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    cfg: Config,
    is_training: bool = False,
) -> torch.Tensor:
    return _classification_loss(preds, targets, cfg)


@register_loss("bce")
def bce_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    cfg: Config,
    is_training: bool = False,
) -> torch.Tensor:
    if targets.dim() == 1:
        targets = targets.unsqueeze(0)
    if preds.dim() == 1:
        preds = preds.unsqueeze(0)
    return F.binary_cross_entropy_with_logits(preds.float(), targets.float())


def _nonzero_label_mask(targets: torch.Tensor, eps: float) -> torch.Tensor:
    return targets > eps


@register_loss("log_masked_mse")
def log_masked_mse_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    cfg: Config,
    is_training: bool = False,
) -> torch.Tensor:
    eps = cfg.loss.zero_label_eps
    nonzero = _nonzero_label_mask(targets, eps)
    if not nonzero.any():
        logger.warning("No nonzero label positions in batch; returning zero loss")
        return preds.sum() * 0.0

    log_pred = torch.log(preds.float().clamp(min=eps))
    log_tgt = torch.log(targets.float().clamp(min=eps))
    loss = (log_pred - log_tgt) ** 2
    return loss[nonzero].mean()


@register_loss("mse")
def mse_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    cfg: Config,
    is_training: bool = False,
) -> torch.Tensor:
    if cfg.model.output_mode in ("sequence", "vector"):
        if cfg.model.output_mode == "vector":
            eps = cfg.loss.zero_label_eps
            nonzero = _nonzero_label_mask(targets, eps)
            if not nonzero.any():
                return preds.sum() * 0.0
            loss = F.mse_loss(preds.float(), targets.float(), reduction="none")
            return loss[nonzero].mean()
        if preds.dim() == 2 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)
        if targets.dim() == 2:
            targets = targets.squeeze(-1)
        return F.mse_loss(preds.float(), targets.float())

    effective_mask = build_loss_mask(mask, targets, cfg, is_training)
    loss = F.mse_loss(preds.float(), targets.float(), reduction="none")
    return _masked_mean(loss, effective_mask)


@register_loss("mae")
def mae_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    cfg: Config,
    is_training: bool = False,
) -> torch.Tensor:
    if cfg.model.output_mode == "sequence":
        if preds.dim() == 2 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)
        if targets.dim() == 2:
            targets = targets.squeeze(-1)
        return F.l1_loss(preds.float(), targets.float())

    effective_mask = build_loss_mask(mask, targets, cfg, is_training)
    loss = F.l1_loss(preds.float(), targets.float(), reduction="none")
    return _masked_mean(loss, effective_mask)


@register_loss("huber")
def huber_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    cfg: Config,
    is_training: bool = False,
) -> torch.Tensor:
    if cfg.model.output_mode == "sequence":
        if preds.dim() == 2 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)
        if targets.dim() == 2:
            targets = targets.squeeze(-1)
        return F.huber_loss(preds.float(), targets.float())

    effective_mask = build_loss_mask(mask, targets, cfg, is_training)
    loss = F.huber_loss(preds.float(), targets.float(), reduction="none")
    return _masked_mean(loss, effective_mask)


def default_loss_name(cfg: Config) -> str:
    if cfg.task.type == "classification":
        return "bce" if cfg.task.mode == "multi_label" else "cross_entropy"
    return "mse"


def build_loss(cfg: Config) -> LossFn:
    name = (cfg.loss.name or default_loss_name(cfg)).lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown loss '{name}', registered: {list(_REGISTRY.keys())}")

    def wrapper(
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
        cfg: Config,
        is_training: bool = False,
    ) -> torch.Tensor:
        if cfg.task.type == "classification":
            if name in ("cross_entropy", "bce"):
                return _REGISTRY[name](preds, targets, mask, cfg, is_training)
            return _classification_loss(preds, targets, cfg)
        return _REGISTRY[name](preds, targets, mask, cfg, is_training)

    return wrapper
