"""YAML configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class PathsConfig:
    train_csv: str = ""
    val_csv: str = ""
    test_csv: str = ""
    train_embedding_dir: str = ""
    val_embedding_dir: str = ""
    test_embedding_dir: str = ""
    embedding_dir: str = ""  # fallback when split-specific dirs are not set
    label_dir: str = ""
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"


@dataclass
class DataConfig:
    label_source: str = "csv"
    id_column: str = "unique_id"  # CSV column used to locate embedding files
    label_key: str = "label"  # key in .pt when label_source=pt
    embedding_key: str = "layernorm_embedding"
    # Filename template under embedding_dir; must contain {unique_id}
    # e.g. "embedding_{unique_id}.pt" (default) or "{unique_id}.pt"
    embedding_filename_fmt: str = "embedding_{unique_id}.pt"
    skip_prefix_tokens: int = 0  # drop leading special tokens, e.g. 2 for [CLS][SEP]
    skip_suffix_tokens: int = 0  # drop trailing special tokens
    strict: bool = False


@dataclass
class ModelConfig:
    type: str = "mlp"
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.0
    output_mode: str = "sequence"
    input_dim: int = 2048
    output_activation: str = "none"  # none | softmax (vector mode)


@dataclass
class TransformConfig:
    enabled: bool = False
    name: str = "identity"
    weights_path: str = ""
    freeze: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    type: str = "regression"  # regression | classification
    mode: str = "single_label"  # single_label | multi_label (binary per label)
    num_classes: int = 2
    num_labels: int = 1
    num_outputs: int = 1  # regression output dim (e.g. 4 for probability vector)
    threshold: float = 0.5  # multi-label decision threshold


@dataclass
class MetricsConfig:
    names: list[str] = field(default_factory=list)  # empty -> auto by task type


@dataclass
class TrainConfig:
    batch_size: int = 512
    eval_every_steps: int = 20
    early_stopping_patience: int = 10
    early_stopping_metric: str = "loss"  # loss | mcc (classification: higher mcc is better)
    max_steps: int = 10000
    lr: float = 1e-3
    optimizer: str = "adam"
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"
    lr_schedule: str = "constant"  # constant | warmup_cosine
    warmup_steps: int = 500
    min_lr_ratio: float = 0.01
    save_test_predictions: bool = True
    test_output_csv: str = ""  # default: {output_dir}/test_predictions.csv
    merge_test_predictions: bool = True
    test_predictions_npz: str = ""  # per_base only, optional


@dataclass
class LossConfig:
    name: str = "mse"
    zero_label_mask: bool = False
    zero_label_drop_prob: float = 0.99
    zero_label_eps: float = 1e-8


@dataclass
class TensorboardConfig:
    log_dir: str = "./tensorboard"
    task_name: str = "default"


@dataclass
class InferConfig:
    checkpoint: str = ""
    input_csv: str = ""
    embedding_dir: str = ""  # override; else uses paths.test_embedding_dir / paths.embedding_dir
    output_csv: str = ""  # default: overwrite/extend input_csv with infer_output column
    merge_to_input: bool = True  # add infer_output column to original CSV
    predictions_npz: str = ""
    metrics_if_label: bool = True
    batch_size: int = 512
    num_workers: int = 4
    device: str = "cuda"


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    tensorboard: TensorboardConfig = field(default_factory=TensorboardConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def tensorboard_log_dir(self) -> str:
        return os.path.join(self.tensorboard.log_dir, self.tensorboard.task_name)


_SPLIT_EMBEDDING_ATTRS = {
    "train": "train_embedding_dir",
    "val": "val_embedding_dir",
    "test": "test_embedding_dir",
}


def get_embedding_dir(cfg: Config, split: str) -> str:
    """Resolve embedding directory for train/val/test/infer."""
    if split == "infer":
        if cfg.infer.embedding_dir:
            return cfg.infer.embedding_dir
        split = "test"

    attr = _SPLIT_EMBEDDING_ATTRS.get(split)
    if attr:
        specific = getattr(cfg.paths, attr, "")
        if specific:
            return specific
    if cfg.paths.embedding_dir:
        return cfg.paths.embedding_dir
    raise ValueError(
        f"No embedding directory for split '{split}'. "
        f"Set paths.{attr} or paths.embedding_dir"
        if attr
        else f"Set paths.embedding_dir or infer.embedding_dir"
    )


def _merge_dict(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_dict(result[k], v)
        else:
            result[k] = v
    return result


def _dict_to_dataclass(cls, d: dict):
    if d is None:
        return cls()
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in d.items() if k in field_names}
    return cls(**filtered)


def load_config(path: str, mode: str = "train") -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    cfg = Config(
        paths=_dict_to_dataclass(PathsConfig, raw.get("paths")),
        data=_dict_to_dataclass(DataConfig, raw.get("data")),
        task=_dict_to_dataclass(TaskConfig, raw.get("task")),
        model=_dict_to_dataclass(ModelConfig, raw.get("model")),
        transform=_dict_to_dataclass(TransformConfig, raw.get("transform")),
        metrics=_dict_to_dataclass(MetricsConfig, raw.get("metrics")),
        train=_dict_to_dataclass(TrainConfig, raw.get("train")),
        loss=_dict_to_dataclass(LossConfig, raw.get("loss")),
        tensorboard=_dict_to_dataclass(TensorboardConfig, raw.get("tensorboard")),
        infer=_dict_to_dataclass(InferConfig, raw.get("infer")),
        raw=raw,
    )
    validate_config(cfg, for_infer=(mode == "infer"))
    return cfg


def validate_config(cfg: Config, for_infer: bool = False) -> None:
    if cfg.data.label_source not in ("csv", "npz", "pt"):
        raise ValueError(f"Unknown label_source: {cfg.data.label_source}")

    if cfg.model.output_mode not in ("per_base", "sequence", "vector"):
        raise ValueError(f"Unknown output_mode: {cfg.model.output_mode}")

    if cfg.data.label_source == "csv" and cfg.model.output_mode != "sequence":
        raise ValueError("csv label_source requires output_mode=sequence")

    if cfg.data.label_source == "npz" and cfg.model.output_mode != "per_base":
        raise ValueError("npz label_source requires output_mode=per_base")

    if cfg.data.label_source == "pt" and cfg.model.output_mode != "vector":
        raise ValueError("pt label_source requires output_mode=vector")

    if cfg.data.label_source == "npz" and not cfg.paths.label_dir:
        raise ValueError("label_dir is required when label_source=npz")

    if cfg.model.output_activation not in ("none", "softmax"):
        raise ValueError(f"Unknown output_activation: {cfg.model.output_activation}")

    if cfg.model.output_mode == "vector":
        if cfg.task.type != "regression":
            raise ValueError("vector output_mode requires task.type=regression")
        if cfg.task.num_outputs < 1:
            raise ValueError("task.num_outputs must be >= 1 for vector output_mode")
        if cfg.model.output_activation == "softmax" and cfg.task.num_outputs < 2:
            raise ValueError("softmax requires task.num_outputs >= 2")

    if cfg.task.type not in ("regression", "classification"):
        raise ValueError(f"Unknown task.type: {cfg.task.type}")

    _validate_transform_config(cfg)

    if cfg.train.lr_schedule not in ("constant", "warmup_cosine"):
        raise ValueError(f"Unknown lr_schedule: {cfg.train.lr_schedule}")

    if cfg.loss.zero_label_mask:
        if cfg.task.type != "regression" or cfg.model.output_mode != "per_base":
            raise ValueError(
                "zero_label_mask requires regression with output_mode=per_base"
            )
        if not 0.0 <= cfg.loss.zero_label_drop_prob < 1.0:
            raise ValueError("zero_label_drop_prob must be in [0, 1)")

    if cfg.task.type == "classification":
        if cfg.model.output_mode != "sequence":
            raise ValueError("classification requires model.output_mode=sequence")
        if cfg.task.mode == "single_label" and cfg.task.num_classes < 2:
            raise ValueError("task.num_classes must be >= 2 for single_label classification")
        if cfg.task.mode == "multi_label" and cfg.task.num_labels < 1:
            raise ValueError("task.num_labels must be >= 1 for multi_label classification")
        if cfg.train.early_stopping_metric not in ("loss", "mcc"):
            raise ValueError("early_stopping_metric must be loss or mcc")

    if cfg.task.type == "regression" and cfg.model.output_mode == "vector":
        allowed_es = ("loss", "log_mse", "mse", "mae", "pearson")
        if cfg.train.early_stopping_metric not in allowed_es:
            raise ValueError(
                f"early_stopping_metric must be one of {allowed_es} for vector regression"
            )

    if not for_infer:
        for path_attr in ("train_csv", "val_csv", "test_csv"):
            if not getattr(cfg.paths, path_attr):
                raise ValueError(f"paths.{path_attr} is required for training")
        for split in ("train", "val", "test"):
            try:
                get_embedding_dir(cfg, split)
            except ValueError as e:
                raise ValueError(f"Training requires embedding dir for {split}: {e}") from e

    if for_infer:
        if not cfg.infer.input_csv:
            raise ValueError("infer.input_csv is required for inference")
        try:
            get_embedding_dir(cfg, "infer")
        except ValueError as e:
            raise ValueError(f"Inference requires embedding dir: {e}") from e


def _validate_transform_config(cfg: Config) -> None:
    if not cfg.transform.enabled:
        return

    from src.transforms import get_registered_transform_names

    if cfg.transform.name not in get_registered_transform_names():
        names = ", ".join(sorted(get_registered_transform_names()))
        raise ValueError(
            f"Unknown transform.name: {cfg.transform.name}. "
            f"Registered transforms: {names}"
        )

    if cfg.transform.weights_path and not os.path.isfile(cfg.transform.weights_path):
        raise ValueError(f"transform.weights_path not found: {cfg.transform.weights_path}")

    params = cfg.transform.params
    out_dim = params.get("out_dim", cfg.model.input_dim)
    if out_dim != cfg.model.input_dim:
        raise ValueError(
            f"transform.params.out_dim ({out_dim}) must equal "
            f"model.input_dim ({cfg.model.input_dim})"
        )
