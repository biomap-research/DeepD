"""Load precomputed embeddings from .pt files."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


def embedding_path(
    embedding_dir: str,
    unique_id: str,
    filename_fmt: str = "embedding_{unique_id}.pt",
) -> str:
    if "{unique_id}" not in filename_fmt:
        raise ValueError(
            f"embedding_filename_fmt must contain '{{unique_id}}', got: {filename_fmt}"
        )
    return os.path.join(embedding_dir, filename_fmt.format(unique_id=unique_id))


def _extract_embedding_tensor(
    data: dict | torch.Tensor,
    embedding_key: str,
    path: str,
) -> torch.Tensor:
    if isinstance(data, dict):
        if embedding_key not in data:
            raise KeyError(f"Key '{embedding_key}' not found in {path}, keys={list(data.keys())}")
        tensor = data[embedding_key]
    elif isinstance(data, torch.Tensor):
        tensor = data
    else:
        raise TypeError(f"Unsupported embedding format in {path}: {type(data)}")

    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    elif tensor.dim() == 1:
        pass  # vector mode: [D]
    elif tensor.dim() != 2:
        raise ValueError(f"Expected 1D, 2D or 3D embedding, got shape {tensor.shape} in {path}")

    return tensor.float()


def load_embedding(
    embedding_dir: str,
    unique_id: str,
    embedding_key: str = "layernorm_embedding",
    filename_fmt: str = "embedding_{unique_id}.pt",
) -> torch.Tensor | None:
    """Load embedding tensor of shape [L, D], [D], or return None if missing."""
    path = embedding_path(embedding_dir, unique_id, filename_fmt)
    if not os.path.isfile(path):
        return None

    data = torch.load(path, map_location="cpu", weights_only=False)
    return _extract_embedding_tensor(data, embedding_key, path)


def load_pt_sample(
    embedding_dir: str,
    sample_id: str,
    embedding_key: str = "embedding",
    label_key: str = "label",
    filename_fmt: str = "{unique_id}.pt",
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load embedding and label from one .pt file. Returns None if file missing."""
    path = embedding_path(embedding_dir, sample_id, filename_fmt)
    if not os.path.isfile(path):
        return None

    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"pt label_source expects dict in {path}, got {type(data)}")

    emb = _extract_embedding_tensor(data, embedding_key, path)
    if label_key not in data:
        raise KeyError(f"Key '{label_key}' not found in {path}, keys={list(data.keys())}")
    label = data[label_key].float().reshape(-1)
    return emb, label


def strip_special_tokens(
    embedding: torch.Tensor,
    skip_prefix: int = 0,
    skip_suffix: int = 0,
    unique_id: str = "",
) -> torch.Tensor:
    """Remove leading/trailing special-token positions from [L, D] embedding."""
    if skip_prefix == 0 and skip_suffix == 0:
        return embedding

    seq_len = embedding.shape[0]
    if skip_prefix + skip_suffix >= seq_len:
        raise ValueError(
            f"Cannot strip {skip_prefix}+{skip_suffix} tokens from length {seq_len} "
            f"for {unique_id}"
        )

    end = seq_len - skip_suffix if skip_suffix > 0 else seq_len
    stripped = embedding[skip_prefix:end]
    logger.debug(
        "Stripped embedding for %s: %d -> %d (prefix=%d, suffix=%d)",
        unique_id,
        seq_len,
        stripped.shape[0],
        skip_prefix,
        skip_suffix,
    )
    return stripped


def align_length(
    embedding: torch.Tensor,
    nt_seq: str | None,
    start: int | None,
    end: int | None,
    unique_id: str,
) -> tuple[torch.Tensor, int]:
    """Truncate embedding to expected sequence length."""
    seq_len = embedding.shape[0]
    expected = None

    if nt_seq is not None and len(nt_seq) > 0:
        expected = len(nt_seq)
    elif start is not None and end is not None:
        expected = end - start + 1

    if expected is not None and expected != seq_len:
        target = min(expected, seq_len)
        if target < seq_len:
            logger.warning(
                "Truncating embedding for %s: embedding_len=%d, expected=%d -> %d",
                unique_id,
                seq_len,
                expected,
                target,
            )
        embedding = embedding[:target]
        seq_len = target

    return embedding, seq_len
