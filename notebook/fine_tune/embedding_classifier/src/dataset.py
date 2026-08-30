"""CSV dataset and padded batch collation."""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import Config
from src.embedding_io import (
    align_length,
    embedding_path,
    load_embedding,
    load_pt_sample,
    strip_special_tokens,
)
from src.labels import get_label

logger = logging.getLogger(__name__)

NPZ_LABEL_COLUMNS = {"chr", "start"}


def get_required_columns(cfg: Config, require_label: bool) -> set[str]:
    """Columns that must exist in the CSV."""
    required = {cfg.data.id_column}
    if cfg.data.label_source == "npz":
        required |= NPZ_LABEL_COLUMNS
    elif require_label and cfg.data.label_source == "csv":
        required.add("label")
    return required


def _row_sample_id(row_dict: dict, cfg: Config) -> str:
    col = cfg.data.id_column
    if col not in row_dict or pd.isna(row_dict.get(col)):
        raise ValueError(f"Missing {col} in row")
    return str(row_dict[col])


def _parse_row_fields(row_dict: dict) -> tuple[str | None, int | None, int | None, str | None]:
    nt_seq = row_dict.get("nt_seq")
    if isinstance(nt_seq, str):
        pass
    elif pd.isna(nt_seq):
        nt_seq = None
    else:
        nt_seq = str(nt_seq)

    start = (
        int(row_dict["start"])
        if "start" in row_dict and not pd.isna(row_dict.get("start"))
        else None
    )
    end = (
        int(row_dict["end"])
        if "end" in row_dict and not pd.isna(row_dict.get("end"))
        else None
    )
    chr_name = (
        str(row_dict["chr"])
        if "chr" in row_dict and not pd.isna(row_dict.get("chr"))
        else None
    )
    return nt_seq, start, end, chr_name


class EmbeddingDataset(Dataset):
    """Lazy-loads embeddings from disk in __getitem__ to avoid OOM on large datasets."""

    def __init__(
        self,
        csv_path: str,
        cfg: Config,
        require_label: bool = True,
        embedding_dir: str | None = None,
    ):
        self.cfg = cfg
        self.require_label = require_label
        if embedding_dir is None:
            raise ValueError("embedding_dir must be provided")
        self.embedding_dir = embedding_dir
        self.embedding_key = cfg.data.embedding_key
        self.label_key = cfg.data.label_key
        self.embedding_filename_fmt = cfg.data.embedding_filename_fmt
        self.output_mode = cfg.model.output_mode
        self.id_column = cfg.data.id_column

        df = pd.read_csv(csv_path)
        required_cols = get_required_columns(cfg, require_label)
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"CSV {csv_path} missing columns: {missing_cols}")

        self.rows: list[dict[str, Any]] = []
        skipped_missing_emb = 0
        skipped_error = 0

        for _, row in df.iterrows():
            row_dict = row.to_dict()
            try:
                sample_id = _row_sample_id(row_dict, cfg)
            except ValueError:
                skipped_error += 1
                if cfg.data.strict:
                    raise
                continue

            if not os.path.isfile(
                embedding_path(self.embedding_dir, sample_id, self.embedding_filename_fmt)
            ):
                skipped_missing_emb += 1
                if cfg.data.strict:
                    raise FileNotFoundError(
                        f"Embedding not found for {sample_id} in {self.embedding_dir}"
                    )
                continue

            nt_seq, start, end, chr_name = _parse_row_fields(row_dict)

            try:
                label_for_store = None
                if require_label or (
                    cfg.data.label_source == "npz"
                    or ("label" in row_dict and not pd.isna(row_dict.get("label")))
                ):
                    if cfg.data.label_source == "csv":
                        label_for_store = get_label(row_dict, cfg)
                    # pt / npz labels loaded in __getitem__
            except Exception as e:
                skipped_error += 1
                if cfg.data.strict:
                    raise
                logger.debug("Skipping row %s: %s", sample_id, e)
                continue

            self.rows.append(
                {
                    "row_dict": row_dict,
                    "label": label_for_store,
                    "sample_id": sample_id,
                    "nt_seq": nt_seq,
                    "start": start,
                    "end": end,
                    "chr": chr_name,
                }
            )

        total = len(df)
        kept = len(self.rows)
        logger.info(
            "Indexed %s: %d/%d samples (skipped missing_emb=%d, errors=%d) "
            "[lazy embedding load]",
            csv_path,
            kept,
            total,
            skipped_missing_emb,
            skipped_error,
        )
        if kept == 0:
            raise RuntimeError(f"No valid samples loaded from {csv_path}")

    def _load_pt_sample(self, meta: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample_id = meta["sample_id"]
        loaded = load_pt_sample(
            self.embedding_dir,
            sample_id,
            self.embedding_key,
            self.label_key,
            self.embedding_filename_fmt,
        )
        if loaded is None:
            raise FileNotFoundError(
                f"Embedding not found for {sample_id} in {self.embedding_dir}"
            )
        emb, label = loaded
        if emb.dim() == 1:
            return emb, label, 1
        return emb, label, emb.shape[0]

    def _load_embedding_tensor(self, meta: dict[str, Any]) -> tuple[torch.Tensor, int]:
        sample_id = meta["sample_id"]
        emb = load_embedding(
            self.embedding_dir,
            sample_id,
            self.embedding_key,
            self.embedding_filename_fmt,
        )
        if emb is None:
            raise FileNotFoundError(
                f"Embedding not found for {sample_id} in {self.embedding_dir}"
            )

        if emb.dim() == 1:
            return emb, 1

        cfg = self.cfg
        if cfg.data.skip_prefix_tokens > 0 or cfg.data.skip_suffix_tokens > 0:
            emb = strip_special_tokens(
                emb,
                skip_prefix=cfg.data.skip_prefix_tokens,
                skip_suffix=cfg.data.skip_suffix_tokens,
                unique_id=sample_id,
            )

        return align_length(
            emb, meta["nt_seq"], meta["start"], meta["end"], sample_id
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        meta = self.rows[idx]
        sample_id = meta["sample_id"]

        if self.cfg.data.label_source == "pt":
            emb, label, seq_len = self._load_pt_sample(meta)
        else:
            emb, seq_len = self._load_embedding_tensor(meta)
            label = meta["label"]
            if label is None and (
                self.require_label
                or self.cfg.data.label_source == "npz"
                or (
                    "label" in meta["row_dict"]
                    and not pd.isna(meta["row_dict"].get("label"))
                )
            ):
                label = get_label(meta["row_dict"], self.cfg)
                if self.output_mode == "per_base":
                    target_len = min(label.numel(), seq_len)
                    if target_len < seq_len:
                        emb = emb[:target_len]
                        seq_len = target_len
                    label = label[:target_len]

        return {
            "embedding": emb,
            "label": label,
            "sample_id": sample_id,
            "start": meta["start"],
            "end": meta["end"],
            "chr": meta["chr"],
            "seq_len": seq_len,
        }


def collate_batch(batch: list[dict[str, Any]], cfg: Config) -> dict[str, Any]:
    batch_size = len(batch)
    unique_ids = [item["sample_id"] for item in batch]
    starts = [item["start"] for item in batch]
    ends = [item["end"] for item in batch]
    chrs = [item["chr"] for item in batch]
    has_label = batch[0]["label"] is not None

    if cfg.model.output_mode == "vector":
        input_dim = batch[0]["embedding"].shape[-1]
        embeddings = torch.zeros(batch_size, input_dim, dtype=torch.float32)
        for i, item in enumerate(batch):
            emb = item["embedding"]
            if emb.dim() == 2:
                embeddings[i] = emb.mean(dim=0)
            else:
                embeddings[i] = emb

        labels = None
        if has_label:
            num_out = cfg.task.num_outputs
            labels = torch.zeros(batch_size, num_out, dtype=torch.float32)
            for i, item in enumerate(batch):
                lab = item["label"]
                labels[i] = lab.reshape(-1)[:num_out]

        result: dict[str, Any] = {
            "embeddings": embeddings,
            "mask": None,
            "unique_ids": unique_ids,
            "starts": starts,
            "ends": ends,
            "chrs": chrs,
        }
        if has_label:
            result["labels"] = labels
        return result

    max_len = max(item["seq_len"] for item in batch)
    input_dim = batch[0]["embedding"].shape[-1]

    embeddings = torch.zeros(batch_size, max_len, input_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    if has_label:
        if cfg.task.type == "classification" and cfg.task.mode == "multi_label":
            labels = torch.zeros(batch_size, cfg.task.num_labels, dtype=torch.float32)
        elif cfg.task.type == "classification":
            labels = torch.zeros(batch_size, dtype=torch.long)
        elif cfg.model.output_mode == "per_base":
            labels = torch.zeros(batch_size, max_len, dtype=torch.float32)
        else:
            labels = torch.zeros(batch_size, 1, dtype=torch.float32)
    else:
        labels = None

    for i, item in enumerate(batch):
        seq_len = item["seq_len"]
        emb = item["embedding"]
        if emb.dim() == 1:
            embeddings[i, 0] = emb
        else:
            embeddings[i, :seq_len] = emb
        mask[i, :seq_len] = True

        if has_label and item["label"] is not None:
            lab = item["label"]
            if cfg.task.type == "classification" and cfg.task.mode == "multi_label":
                labels[i] = lab.float()
            elif cfg.task.type == "classification":
                labels[i] = lab.long().squeeze()
            elif cfg.model.output_mode == "per_base":
                labels[i, :seq_len] = lab[:seq_len]
            else:
                labels[i, 0] = lab.squeeze()

    result = {
        "embeddings": embeddings,
        "mask": mask,
        "unique_ids": unique_ids,
        "starts": starts,
        "ends": ends,
        "chrs": chrs,
    }
    if has_label:
        result["labels"] = labels
    return result
