"""Evaluation metrics for regression and classification (MCC)."""

from __future__ import annotations

import numpy as np
import torch

from src.config import Config
from src.predictions import predictions_to_numpy


def _binary_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(int).reshape(-1)
    y_pred = y_pred.astype(int).reshape(-1)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / np.sqrt(denom))


def _multiclass_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews correlation for single-label (binary or multi-class)."""
    try:
        from sklearn.metrics import matthews_corrcoef
        return float(matthews_corrcoef(y_true.astype(int), y_pred.astype(int)))
    except ImportError:
        classes = np.unique(np.concatenate([y_true, y_pred]))
        if len(classes) <= 2:
            return _binary_mcc((y_true > 0).astype(int), (y_pred > 0).astype(int))
        mccs = []
        for c in classes:
            yt = (y_true == c).astype(int)
            yp = (y_pred == c).astype(int)
            mccs.append(_binary_mcc(yt, yp))
        return float(np.mean(mccs))


def compute_mcc(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    cfg: Config,
) -> float:
    y_pred = predictions_to_numpy(preds, cfg)
    y_true = targets.detach().cpu().numpy()

    if cfg.task.type != "classification":
        yt = (y_true.reshape(-1) > 0.5).astype(int)
        yp = (y_pred.reshape(-1) > 0.5).astype(int)
        return _binary_mcc(yt, yp)

    if cfg.task.mode == "multi_label":
        if y_true.ndim == 1:
            y_true = y_true.reshape(1, -1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(1, -1)
        mccs = [
            _binary_mcc(y_true[:, j], y_pred[:, j])
            for j in range(y_true.shape[1])
        ]
        return float(np.mean(mccs))

    return _multiclass_mcc(y_true.reshape(-1), y_pred.reshape(-1))


def compute_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    cfg: Config,
) -> float:
    if cfg.task.type != "classification" or cfg.task.mode == "multi_label":
        return float("nan")
    y_pred = predictions_to_numpy(preds, cfg).reshape(-1)
    y_true = targets.detach().cpu().numpy().reshape(-1).astype(int)
    return float((y_pred == y_true).mean())


def _vector_nonzero_mask(targets: torch.Tensor, cfg: Config) -> torch.Tensor:
    return targets > cfg.loss.zero_label_eps


def compute_log_mse(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    cfg: Config,
) -> float:
    eps = cfg.loss.zero_label_eps
    nonzero = _vector_nonzero_mask(targets, cfg)
    if not nonzero.any():
        return float("nan")
    log_pred = torch.log(preds.float().clamp(min=eps))
    log_tgt = torch.log(targets.float().clamp(min=eps))
    loss = (log_pred - log_tgt) ** 2
    return loss[nonzero].mean().item()


def _masked_values(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg.model.output_mode == "vector":
        nonzero = _vector_nonzero_mask(targets, cfg)
        return preds[nonzero], targets[nonzero]

    if cfg.model.output_mode == "sequence":
        if preds.dim() == 2 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)
        if targets.dim() == 2 and targets.shape[-1] == 1:
            targets = targets.squeeze(-1)
        return preds.reshape(-1), targets.reshape(-1)

    if mask is None:
        return preds.reshape(-1), targets.reshape(-1)

    valid = mask.bool()
    return preds[valid], targets[valid]


def compute_mse(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    cfg: Config,
) -> float:
    p, t = _masked_values(preds, targets, mask, cfg)
    if p.numel() == 0:
        return float("nan")
    return torch.mean((p.float() - t.float()) ** 2).item()


def compute_mae(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    cfg: Config,
) -> float:
    p, t = _masked_values(preds, targets, mask, cfg)
    if p.numel() == 0:
        return float("nan")
    return torch.mean(torch.abs(p.float() - t.float())).item()


def compute_pearson(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    cfg: Config,
) -> float:
    if cfg.task.type == "classification":
        return float("nan")
    if cfg.model.output_mode == "sequence":
        p, t = _masked_values(preds, targets, mask, cfg)
        if p.numel() < 2:
            return float("nan")
        p = p.float() - p.mean()
        t = t.float() - t.mean()
        denom = p.norm() * t.norm()
        if denom < 1e-8:
            return float("nan")
        return (p * t).sum().item() / denom.item()
    return float("nan")


class RegressionAccumulator:
    """Online accumulation for masked regression metrics (avoids storing all preds)."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg
        self.n = 0
        self.sum_sq = 0.0
        self.sum_abs = 0.0
        self.sum_p = 0.0
        self.sum_t = 0.0
        self.sum_pp = 0.0
        self.sum_tt = 0.0
        self.sum_pt = 0.0
        self.n_log = 0
        self.sum_log_sq = 0.0

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
        cfg: Config,
    ) -> None:
        if cfg.model.output_mode == "vector":
            eps = cfg.loss.zero_label_eps
            nonzero = _vector_nonzero_mask(targets, cfg)
            if not nonzero.any():
                return
            log_pred = torch.log(preds.float().clamp(min=eps))
            log_tgt = torch.log(targets.float().clamp(min=eps))
            diff_log = (log_pred - log_tgt)[nonzero]
            self.n_log += diff_log.numel()
            self.sum_log_sq += (diff_log * diff_log).sum().item()
            diff = (preds.float() - targets.float())[nonzero]
            self.n += diff.numel()
            self.sum_sq += (diff * diff).sum().item()
            return

        p, t = _masked_values(preds, targets, mask, cfg)
        if p.numel() == 0:
            return
        p = p.float()
        t = t.float()
        n = p.numel()
        self.n += n
        diff = p - t
        self.sum_sq += (diff * diff).sum().item()
        self.sum_abs += diff.abs().sum().item()
        self.sum_p += p.sum().item()
        self.sum_t += t.sum().item()
        self.sum_pp += (p * p).sum().item()
        self.sum_tt += (t * t).sum().item()
        self.sum_pt += (p * t).sum().item()

    def finalize(self) -> dict[str, float]:
        if self.cfg and self.cfg.model.output_mode == "vector":
            return {
                "log_mse": self.sum_log_sq / self.n_log if self.n_log else float("nan"),
                "mse": self.sum_sq / self.n if self.n else float("nan"),
            }
        if self.n == 0:
            return {"mse": float("nan"), "mae": float("nan"), "pearson": float("nan")}
        mse = self.sum_sq / self.n
        mae = self.sum_abs / self.n
        num = self.n * self.sum_pt - self.sum_p * self.sum_t
        den = (
            (self.n * self.sum_pp - self.sum_p ** 2)
            * (self.n * self.sum_tt - self.sum_t ** 2)
        )
        pearson = float(num / np.sqrt(den)) if den > 1e-16 else float("nan")
        return {"mse": mse, "mae": mae, "pearson": pearson}


def get_metric_names(cfg: Config) -> list[str]:
    if cfg.metrics.names:
        return list(cfg.metrics.names)
    if cfg.task.type == "classification":
        names = ["mcc"]
        if cfg.task.mode == "single_label":
            names.append("accuracy")
        return names
    return ["mse", "mae", "pearson"]


def compute_all_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    cfg: Config,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in get_metric_names(cfg):
        if name == "mcc":
            result["mcc"] = compute_mcc(preds, targets, mask, cfg)
        elif name == "accuracy":
            result["accuracy"] = compute_accuracy(preds, targets, cfg)
        elif name == "log_mse":
            result["log_mse"] = compute_log_mse(preds, targets, mask, cfg)
        elif name == "mse":
            result["mse"] = compute_mse(preds, targets, mask, cfg)
        elif name == "mae":
            result["mae"] = compute_mae(preds, targets, mask, cfg)
        elif name == "pearson":
            result["pearson"] = compute_pearson(preds, targets, mask, cfg)
        else:
            raise ValueError(f"Unknown metric: {name}")
    return result
