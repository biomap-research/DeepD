"""MLP models for embedding finetuning."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import Config
from src.predictions import get_output_dim
from src.transforms import build_transform


class PerPositionMLP(nn.Module):
    """Shared MLP: [B, L, D] -> [B, L, out] or pooled [B, out]."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.output_mode = cfg.model.output_mode
        input_dim = cfg.model.input_dim
        hidden_dims = cfg.model.hidden_dims
        dropout = cfg.model.dropout
        out_dim = get_output_dim(cfg)

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            vector: [B, out_dim] probabilities (optional softmax)
            sequence: [B, out_dim] (regression [B,1] or classification logits)
            per_base: [B, L, out_dim]
        """
        if self.output_mode == "vector":
            out = self.mlp(embeddings)
            if self.cfg.model.output_activation == "softmax":
                out = torch.softmax(out, dim=-1)
            return out

        if self.output_mode == "sequence":
            if mask is None:
                pooled = embeddings.mean(dim=1)
            else:
                mask_f = mask.unsqueeze(-1).float()
                summed = (embeddings * mask_f).sum(dim=1)
                counts = mask_f.sum(dim=1).clamp(min=1.0)
                pooled = summed / counts
            return self.mlp(pooled)

        b, seq_len, d = embeddings.shape
        flat = embeddings.reshape(b * seq_len, d)
        out = self.mlp(flat).reshape(b, seq_len, -1)
        if out.shape[-1] == 1:
            return out.squeeze(-1)
        return out


class EmbeddingModel(nn.Module):
    """Optional frozen transform + trainable MLP head."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.transform = build_transform(cfg)
        self.head = PerPositionMLP(cfg)

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.transform is not None:
            embeddings = self.transform(embeddings, mask)
        return self.head(embeddings, mask)


def build_model(cfg: Config) -> nn.Module:
    if cfg.model.type != "mlp":
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return EmbeddingModel(cfg)
