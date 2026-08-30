"""Decode model outputs and format inference results."""

from __future__ import annotations

import numpy as np
import torch

from src.config import Config


def get_output_dim(cfg: Config) -> int:
    if cfg.task.type != "classification":
        if cfg.model.output_mode == "vector":
            return cfg.task.num_outputs
        return 1
    if cfg.task.mode == "multi_label":
        return cfg.task.num_labels
    return cfg.task.num_classes


def decode_predictions(preds: torch.Tensor, cfg: Config) -> list[str]:
    """Convert logits to infer_output strings for CSV."""
    if cfg.task.type != "classification":
        probs = preds.detach().cpu().float()
        if cfg.model.output_mode == "vector":
            if probs.dim() == 1:
                probs = probs.unsqueeze(0)
            return [
                ",".join(f"{v:.6g}" for v in row.tolist())
                for row in probs
            ]
        flat = probs.reshape(-1)
        return [str(v.item()) for v in flat]

    if cfg.task.mode == "multi_label":
        probs = torch.sigmoid(preds).cpu().numpy()
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        outputs = []
        for row in probs:
            bits = (row >= cfg.task.threshold).astype(int)
            outputs.append(",".join(str(int(b)) for b in bits))
        return outputs

    logits = preds.cpu()
    if logits.dim() == 1:
        class_ids = logits.long().tolist()
        if not isinstance(class_ids, list):
            class_ids = [class_ids]
    else:
        class_ids = logits.argmax(dim=-1).tolist()
    return [str(int(c)) for c in class_ids]


def predictions_to_numpy(preds: torch.Tensor, cfg: Config) -> np.ndarray:
    """Discrete predictions for metric computation."""
    if cfg.task.type != "classification":
        return preds.detach().cpu().float().numpy().reshape(-1)

    if cfg.task.mode == "multi_label":
        return (torch.sigmoid(preds) >= cfg.task.threshold).cpu().numpy().astype(int)

    if preds.dim() == 1:
        return preds.cpu().numpy().astype(int)
    return preds.argmax(dim=-1).cpu().numpy().astype(int)
