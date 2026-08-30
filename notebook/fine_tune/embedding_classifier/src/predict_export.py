"""Export model predictions to CSV (and optional NPZ for per-base)."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.predictions import decode_predictions

logger = logging.getLogger(__name__)


@torch.no_grad()
def export_predictions_csv(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    input_csv: str,
    output_csv: str,
    merge_to_input: bool = True,
    predictions_npz: str = "",
    desc: str = "Predictions",
) -> str:
    """Run model on loader and write CSV with infer_output column."""
    model.eval()
    df_input = pd.read_csv(input_csv)

    pred_by_uid: dict[str, str] = {}
    per_base_preds: dict[str, np.ndarray] = {}

    id_column = cfg.data.id_column

    for batch in tqdm(loader, desc=desc):
        embeddings = batch["embeddings"].to(device)
        mask = batch["mask"]
        if mask is not None:
            mask = mask.to(device)
        preds = model(embeddings, mask).cpu()

        decoded = decode_predictions(preds, cfg)
        for i, uid in enumerate(batch["unique_ids"]):
            pred_by_uid[uid] = decoded[i]

        if cfg.model.output_mode == "per_base" and predictions_npz and mask is not None:
            for i, uid in enumerate(batch["unique_ids"]):
                seq_len = int(mask[i].sum().item())
                per_base_preds[uid] = preds[i, :seq_len].numpy()

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    if merge_to_input:
        df_out = df_input.copy()
        df_out["infer_output"] = df_out[id_column].astype(str).map(pred_by_uid)
        missing = int(df_out["infer_output"].isna().sum())
        if missing > 0:
            logger.warning("%d rows have no prediction (missing embedding?)", missing)
        df_out.to_csv(output_csv, index=False)
    else:
        pd.DataFrame(
            [{"unique_id": uid, "infer_output": pred_by_uid[uid]} for uid in pred_by_uid]
        ).to_csv(output_csv, index=False)

    logger.info("Saved predictions CSV to %s", output_csv)

    if cfg.model.output_mode == "per_base" and predictions_npz:
        os.makedirs(os.path.dirname(predictions_npz) or ".", exist_ok=True)
        np.savez(predictions_npz, **per_base_preds)
        logger.info("Saved per-base predictions to %s", predictions_npz)

    return output_csv


def resolve_test_output_csv(cfg: Config) -> str:
    if cfg.train.test_output_csv:
        return cfg.train.test_output_csv
    return os.path.join(cfg.paths.output_dir, "test_predictions.csv")
