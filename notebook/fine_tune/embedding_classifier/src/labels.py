"""Extensible label loading via get_label."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

if TYPE_CHECKING:
    from src.config import Config

LabelFn = Callable[[dict, "Config"], torch.Tensor]

_PROVIDERS: dict[str, LabelFn] = {}


def register_label_source(name: str):
    def decorator(fn: LabelFn) -> LabelFn:
        _PROVIDERS[name] = fn
        return fn
    return decorator


def _parse_multi_label(value: str, num_labels: int) -> torch.Tensor:
    text = str(value).strip()
    for sep in (",", ";", "|", " "):
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip() != ""]
            if len(parts) == num_labels:
                return torch.tensor([float(p) for p in parts], dtype=torch.float32)
    if len(text) == num_labels and text.isdigit():
        return torch.tensor([float(c) for c in text], dtype=torch.float32)
    raise ValueError(
        f"Cannot parse multi_label '{value}' into {num_labels} binary labels"
    )


@register_label_source("csv")
def _label_from_csv(row: dict, cfg: Config) -> torch.Tensor:
    if "label" not in row or row["label"] is None or (isinstance(row["label"], float) and np.isnan(row["label"])):
        raise ValueError(f"Missing label in row unique_id={row.get('unique_id')}")

    if cfg.task.type == "classification":
        if cfg.task.mode == "multi_label":
            return _parse_multi_label(row["label"], cfg.task.num_labels)
        return torch.tensor(int(row["label"]), dtype=torch.long)

    return torch.tensor([float(row["label"])], dtype=torch.float32)


@lru_cache(maxsize=4096)
def _load_npz_array(path: str, key: str) -> np.ndarray:
    """Load one window from NPZ via mmap (does not load entire chromosome into RAM)."""
    with np.load(path, mmap_mode="r") as data:
        if key not in data:
            raise KeyError(
                f"Key '{key}' not found in {path}, "
                f"available keys sample: {list(data.keys())[:5]}"
            )
        return np.asarray(data[key], dtype=np.float32)


@register_label_source("pt")
def _label_from_pt(row: dict, cfg: Config) -> torch.Tensor:
    """Label from pre-loaded row['_pt_label'] set by dataset (single torch.load)."""
    if "_pt_label" in row:
        return row["_pt_label"]
    raise ValueError(
        f"pt labels must be loaded with embedding in dataset; "
        f"missing _pt_label for id={row.get(cfg.data.id_column)}"
    )


@register_label_source("npz")
def _label_from_npz(row: dict, cfg: Config) -> torch.Tensor:
    if "chr" not in row or row["chr"] is None:
        raise ValueError(f"npz label requires 'chr' in row unique_id={row.get('unique_id')}")
    if "start" not in row or row["start"] is None:
        raise ValueError(f"npz label requires 'start' in row unique_id={row.get('unique_id')}")
    chr_name = str(row["chr"])
    start = int(row["start"] // 8192)
    path = os.path.join(cfg.paths.label_dir, f"{chr_name}_label.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Label file not found: {path}")

    key = str(start)
    label = _load_npz_array(path, key)
    return torch.from_numpy(label)


def get_label(row: dict, cfg: Config) -> torch.Tensor:
    """Unified label entry; dispatches by cfg.data.label_source."""
    source = cfg.data.label_source
    if source not in _PROVIDERS:
        raise ValueError(f"Unknown label_source '{source}', registered: {list(_PROVIDERS.keys())}")
    return _PROVIDERS[source](row, cfg)
