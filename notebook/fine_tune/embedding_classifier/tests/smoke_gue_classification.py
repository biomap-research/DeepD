"""Engineering smoke test for the generic GUE classification workflow.

Synthetic embeddings/labels are used only to test software execution. They are not
scientific data and must not be used for reported benchmark results.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
import yaml

HERE = Path(__file__).resolve()
BACKEND_ROOT = HERE.parent.parent
sys.path.insert(0, str(BACKEND_ROOT))


def make_split(root: Path, split: str, n: int = 8) -> tuple[Path, Path]:
    csv_path = root / f"{split}.csv"
    emb_dir = root / f"{split}_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n):
        uid = f"{split}_{i:03d}"
        label = i % 2
        rows.append({"unique_id": uid, "label": label})
        # 18 tokens: first 2 mimic special tokens, leaving 16 valid tokens.
        emb = torch.randn(1, 18, 2048)
        torch.save(
            {"layernorm_embedding": emb},
            emb_dir / f"embedding_{uid}.pt",
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path, emb_dir


def main() -> None:
    torch.manual_seed(42)
    tmp = Path(tempfile.mkdtemp(prefix="gue_cls_smoke_"))
    try:
        train_csv, train_emb = make_split(tmp, "train")
        val_csv, val_emb = make_split(tmp, "val")
        test_csv, test_emb = make_split(tmp, "test")

        cfg_dict = {
            "paths": {
                "train_csv": str(train_csv),
                "val_csv": str(val_csv),
                "test_csv": str(test_csv),
                "train_embedding_dir": str(train_emb),
                "val_embedding_dir": str(val_emb),
                "test_embedding_dir": str(test_emb),
                "checkpoint_dir": str(tmp / "checkpoints"),
                "output_dir": str(tmp / "outputs"),
            },
            "data": {
                "label_source": "csv",
                "embedding_key": "layernorm_embedding",
                "skip_prefix_tokens": 2,
                "skip_suffix_tokens": 0,
                "strict": True,
            },
            "task": {
                "type": "classification",
                "mode": "single_label",
                "num_classes": 2,
            },
            "model": {
                "type": "mlp",
                "hidden_dims": [512, 128],
                "dropout": 0.0,
                "output_mode": "sequence",
                "input_dim": 2048,
            },
            "metrics": {"names": ["mcc", "accuracy"]},
            "loss": {"name": "cross_entropy"},
            "train": {
                "batch_size": 4,
                "eval_every_steps": 1,
                "early_stopping_patience": 3,
                "early_stopping_metric": "mcc",
                "max_steps": 2,
                "lr": 1e-3,
                "optimizer": "adam",
                "seed": 42,
                "num_workers": 0,
                "device": "cpu",
                "save_test_predictions": True,
            },
            "tensorboard": {
                "log_dir": str(tmp / "tensorboard"),
                "task_name": "gue_smoke",
            },
        }
        cfg_path = tmp / "smoke.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")

        from src.config import load_config
        from src.dataset import EmbeddingDataset
        from src.trainer import train

        cfg = load_config(str(cfg_path))
        ds = EmbeddingDataset(str(train_csv), cfg, require_label=True, embedding_dir=str(train_emb))
        sample = ds[0]
        assert tuple(sample["embedding"].shape) == (16, 2048), sample["embedding"].shape

        metrics = train(cfg)
        assert "mcc" in metrics and "accuracy" in metrics
        assert (tmp / "checkpoints" / "best.pt").is_file()
        assert (tmp / "outputs" / "test_metrics.json").is_file()
        print("GUE classification smoke test passed.")
        print(metrics)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
