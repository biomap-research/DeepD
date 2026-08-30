"""Inference CLI entry point."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import get_embedding_dir, load_config
from src.dataset import EmbeddingDataset, collate_batch
from src.metrics import compute_all_metrics
from src.model import build_model
from src.transforms import apply_transform_init
from src.predict_export import export_predictions_csv


def _get_device(cfg) -> torch.device:
    if cfg.infer.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path: str, model: torch.nn.Module, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])


@torch.no_grad()
def run_inference(cfg) -> None:
    device = _get_device(cfg)
    logging.info("Using device: %s", device)

    input_csv = cfg.infer.input_csv
    df_head = pd.read_csv(input_csv, nrows=1)
    has_label_col = "label" in df_head.columns
    require_label = cfg.infer.metrics_if_label and (
        has_label_col or cfg.data.label_source == "pt"
    )

    dataset = EmbeddingDataset(
        input_csv,
        cfg,
        require_label=require_label,
        embedding_dir=get_embedding_dir(cfg, "infer"),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.infer.batch_size,
        shuffle=False,
        num_workers=cfg.infer.num_workers,
        collate_fn=lambda batch: collate_batch(batch, cfg),
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg).to(device)
    apply_transform_init(model, cfg, device)
    load_checkpoint(cfg.infer.checkpoint, model, device)

    output_csv = cfg.infer.output_csv or input_csv
    export_predictions_csv(
        model,
        loader,
        cfg,
        device,
        input_csv=input_csv,
        output_csv=output_csv,
        merge_to_input=cfg.infer.merge_to_input,
        predictions_npz=cfg.infer.predictions_npz,
        desc="Inference",
    )

    if require_label:
        all_preds = []
        all_targets = []
        all_masks = []
        for batch in tqdm(loader, desc="Metrics"):
            embeddings = batch["embeddings"].to(device)
            mask = batch["mask"]
            if mask is not None:
                mask = mask.to(device)
            labels = batch["labels"].to(device)
            preds = model(embeddings, mask).cpu()
            all_preds.append(preds)
            all_targets.append(labels.cpu())
            if mask is not None:
                all_masks.append(mask.cpu())

        mask_arg = torch.cat(all_masks, dim=0) if all_masks else None
        metrics = compute_all_metrics(
            torch.cat(all_preds, dim=0),
            torch.cat(all_targets, dim=0),
            mask_arg,
            cfg,
        )
        metrics_path = os.path.join(
            os.path.dirname(output_csv) or ".", "infer_metrics.json"
        )
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logging.info("Inference metrics: %s", metrics)
        logging.info("Saved metrics to %s", metrics_path)


def main():
    parser = argparse.ArgumentParser(description="Embedding finetune inference")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    cfg = load_config(args.config, mode="infer")
    if not cfg.infer.checkpoint:
        raise ValueError("infer.checkpoint must be set in config")
    if not cfg.infer.input_csv:
        raise ValueError("infer.input_csv must be set in config")

    run_inference(cfg)


if __name__ == "__main__":
    main()
