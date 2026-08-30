"""Training loop with validation, early stopping, and test evaluation."""

from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import Config, get_embedding_dir
from src.dataset import EmbeddingDataset, collate_batch
from src.losses import build_loss, build_loss_mask
from src.lr_scheduler import build_lr_scheduler
from src.metrics import RegressionAccumulator, compute_all_metrics, get_metric_names
from src.model import build_model
from src.transforms import apply_transform_init
from src.predict_export import export_predictions_csv, resolve_test_output_csv

logger = logging.getLogger(__name__)


@dataclass
class EarlyStopping:
    patience: int
    best_score: float = float("inf")
    counter: int = 0
    should_stop: bool = False
    best_step: int = 0

    def step(self, score: float, step: int, higher_is_better: bool = False) -> bool:
        improved = score > self.best_score if higher_is_better else score < self.best_score
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_step = step
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def _get_device(cfg: Config) -> torch.device:
    if cfg.train.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found for optimizer")
    if cfg.train.optimizer.lower() == "adam":
        return torch.optim.Adam(params, lr=cfg.train.lr)
    if cfg.train.optimizer.lower() == "adamw":
        return torch.optim.AdamW(params, lr=cfg.train.lr)
    if cfg.train.optimizer.lower() == "sgd":
        return torch.optim.SGD(params, lr=cfg.train.lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {cfg.train.optimizer}")


def _make_loader(
    dataset: EmbeddingDataset,
    cfg: Config,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
        collate_fn=lambda batch: collate_batch(batch, cfg),
        pin_memory=device.type == "cuda",
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn,
    cfg: Config,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    reg_acc = (
        RegressionAccumulator(cfg) if cfg.task.type == "regression" else None
    )
    cls_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for batch in loader:
        embeddings = batch["embeddings"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"]
        if mask is not None:
            mask = mask.to(device)

        preds = model(embeddings, mask)
        loss = loss_fn(preds, labels, mask, cfg, is_training=False)
        total_loss += loss.item()
        n_batches += 1

        preds_cpu = preds.cpu()
        labels_cpu = labels.cpu()
        mask_cpu = mask.cpu() if mask is not None else None
        if reg_acc is not None:
            reg_acc.update(preds_cpu, labels_cpu, mask_cpu, cfg)
        else:
            cls_batches.append((preds_cpu, labels_cpu, mask_cpu))

        del embeddings, labels, mask, preds, preds_cpu, labels_cpu, mask_cpu

    avg_loss = total_loss / max(n_batches, 1)
    if reg_acc is not None:
        finalized = reg_acc.finalize()
        metrics = {k: finalized[k] for k in get_metric_names(cfg) if k in finalized}
    else:
        metrics = compute_all_metrics(
            torch.cat([b[0] for b in cls_batches], dim=0),
            torch.cat([b[1] for b in cls_batches], dim=0),
            torch.cat([b[2] for b in cls_batches], dim=0),
            cfg,
        )
    metrics["loss"] = avg_loss

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: Config,
    metrics: dict[str, float],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "cfg": cfg.raw,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def train(cfg: Config) -> dict[str, float]:
    torch.manual_seed(cfg.train.seed)
    device = _get_device(cfg)
    logger.info("Using device: %s", device)

    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    os.makedirs(cfg.tensorboard_log_dir, exist_ok=True)

    train_ds = EmbeddingDataset(
        cfg.paths.train_csv, cfg, require_label=True,
        embedding_dir=get_embedding_dir(cfg, "train"),
    )
    val_ds = EmbeddingDataset(
        cfg.paths.val_csv, cfg, require_label=True,
        embedding_dir=get_embedding_dir(cfg, "val"),
    )
    test_ds = EmbeddingDataset(
        cfg.paths.test_csv, cfg, require_label=True,
        embedding_dir=get_embedding_dir(cfg, "test"),
    )

    train_loader = _make_loader(train_ds, cfg, shuffle=True, device=device)
    val_loader = _make_loader(val_ds, cfg, shuffle=False, device=device)
    test_loader = _make_loader(test_ds, cfg, shuffle=False, device=device)

    model = build_model(cfg).to(device)
    apply_transform_init(model, cfg, device)
    optimizer = _build_optimizer(model, cfg)
    scheduler = build_lr_scheduler(optimizer, cfg)
    loss_fn = build_loss(cfg)
    es_metric = cfg.train.early_stopping_metric
    higher_is_better = es_metric == "mcc"
    early_stopping = EarlyStopping(
        patience=cfg.train.early_stopping_patience,
        best_score=float("-inf") if higher_is_better else float("inf"),
    )

    writer = SummaryWriter(log_dir=cfg.tensorboard_log_dir)
    best_ckpt_path = os.path.join(cfg.paths.checkpoint_dir, "best.pt")

    global_step = 0
    train_loss_accum = 0.0
    train_batches = 0
    stop_training = False

    model.train()
    pbar = tqdm(total=cfg.train.max_steps, desc="Training")

    while global_step < cfg.train.max_steps and not stop_training:
        for batch in train_loader:
            if global_step >= cfg.train.max_steps or stop_training:
                break

            embeddings = batch["embeddings"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"]
            if mask is not None:
                mask = mask.to(device)

            optimizer.zero_grad()
            preds = model(embeddings, mask)
            loss = loss_fn(preds, labels, mask, cfg, is_training=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            train_loss_accum += loss.item()
            train_batches += 1
            pbar.update(1)
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            if cfg.loss.zero_label_mask and cfg.model.output_mode == "per_base":
                pad_count = mask.sum().item()
                if pad_count > 0:
                    loss_mask = build_loss_mask(mask, labels, cfg, is_training=True)
                    ratio = loss_mask.sum().item() / pad_count
                    writer.add_scalar("train/zero_token_loss_ratio", ratio, global_step)

            if global_step % cfg.train.eval_every_steps == 0:
                avg_train_loss = train_loss_accum / max(train_batches, 1)
                writer.add_scalar("train/loss_avg", avg_train_loss, global_step)
                train_loss_accum = 0.0
                train_batches = 0

                val_metrics = evaluate(model, val_loader, loss_fn, cfg, device)
                writer.add_scalar("val/loss", val_metrics["loss"], global_step)
                for k, v in val_metrics.items():
                    if k != "loss":
                        writer.add_scalar(f"val/{k}", v, global_step)

                metric_str = " | ".join(
                    f"val_{k}={v:.4f}"
                    if k in ("mcc", "accuracy", "pearson")
                    else f"val_{k}={v:.6f}"
                    for k, v in val_metrics.items()
                )
                logger.info("Step %d | %s", global_step, metric_str)

                es_score = val_metrics.get(es_metric, val_metrics["loss"])
                improved = early_stopping.step(es_score, global_step, higher_is_better)
                if improved:
                    save_checkpoint(
                        best_ckpt_path, model, optimizer, global_step, cfg, val_metrics
                    )
                    logger.info("Saved best checkpoint at step %d", global_step)

                if early_stopping.should_stop:
                    logger.info(
                        "Early stopping at step %d (best step %d, best val_%s=%.6f)",
                        global_step,
                        early_stopping.best_step,
                        es_metric,
                        early_stopping.best_score,
                    )
                    stop_training = True
                    break

                model.train()

    pbar.close()
    writer.close()

    if os.path.isfile(best_ckpt_path):
        load_checkpoint(best_ckpt_path, model, device)
        logger.info("Loaded best checkpoint from %s", best_ckpt_path)
    else:
        logger.warning("No best checkpoint found, using last model weights")
        save_checkpoint(best_ckpt_path, model, optimizer, global_step, cfg, {})

    test_metrics = evaluate(model, test_loader, loss_fn, cfg, device)
    test_metrics_path = os.path.join(cfg.paths.output_dir, "test_metrics.json")
    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info("Test metrics: %s", test_metrics)
    logger.info("Saved test metrics to %s", test_metrics_path)

    if cfg.train.save_test_predictions and cfg.paths.test_csv:
        test_out = resolve_test_output_csv(cfg)
        export_predictions_csv(
            model,
            test_loader,
            cfg,
            device,
            input_csv=cfg.paths.test_csv,
            output_csv=test_out,
            merge_to_input=cfg.train.merge_test_predictions,
            predictions_npz=cfg.train.test_predictions_npz,
            desc="Test predictions",
        )

    tb_final = SummaryWriter(log_dir=cfg.tensorboard_log_dir)
    for k, v in test_metrics.items():
        tb_final.add_scalar(f"test/{k}", v, global_step)
    tb_final.close()

    return test_metrics
