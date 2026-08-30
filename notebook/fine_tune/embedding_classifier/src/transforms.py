"""Optional embedding transforms applied before the MLP head."""

from __future__ import annotations

import logging
from typing import Type

import torch
import torch.nn as nn

from src.config import Config

logger = logging.getLogger(__name__)

TransformCls = Type[nn.Module]
_REGISTRY: dict[str, TransformCls] = {}


def register_transform(name: str):
    def decorator(cls: TransformCls) -> TransformCls:
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_registered_transform_names() -> set[str]:
    return set(_REGISTRY.keys())


def get_transform_output_dim(cfg: Config) -> int:
    """Output feature dim after transform (equals MLP input_dim)."""
    if not cfg.transform.enabled:
        return cfg.model.input_dim
    return int(cfg.transform.params.get("out_dim", cfg.model.input_dim))


def build_transform(cfg: Config) -> nn.Module | None:
    if not cfg.transform.enabled:
        return None
    name = cfg.transform.name
    if name not in _REGISTRY:
        raise ValueError(f"Unknown transform: {name}")
    return _REGISTRY[name](cfg)


def _extract_state_dict(loaded: object) -> dict[str, torch.Tensor]:
    if isinstance(loaded, dict):
        if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            return loaded["state_dict"]
        if "model_state_dict" in loaded and isinstance(loaded["model_state_dict"], dict):
            return loaded["model_state_dict"]
        tensor_keys = {k: v for k, v in loaded.items() if isinstance(v, torch.Tensor)}
        if tensor_keys:
            if "weight" in tensor_keys and len(tensor_keys) <= 3:
                state: dict[str, torch.Tensor] = {}
                if "weight" in tensor_keys:
                    state["proj.weight"] = tensor_keys["weight"]
                if "bias" in tensor_keys:
                    state["proj.bias"] = tensor_keys["bias"]
                return state
            return tensor_keys
    raise ValueError("Unsupported transform weights file format")


def _strip_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    plen = len(prefix)
    return {k[plen:]: v for k, v in state.items() if k.startswith(prefix)}


def load_transform_weights(
    module: nn.Module,
    path: str,
    device: torch.device,
) -> None:
    loaded = torch.load(path, map_location=device, weights_only=False)
    state = _extract_state_dict(loaded)

    candidates = [state]
    for prefix in ("transform.", "module.", "proj."):
        stripped = _strip_prefix(state, prefix)
        if stripped:
            candidates.append(stripped)

    last_incompatible = None
    for candidate in candidates:
        incompatible = module.load_state_dict(candidate, strict=False)
        if not incompatible.missing_keys and not incompatible.unexpected_keys:
            logger.info("Loaded transform weights from %s", path)
            return
        last_incompatible = incompatible

    if last_incompatible is not None:
        logger.warning(
            "Loaded transform weights from %s with missing=%s unexpected=%s",
            path,
            last_incompatible.missing_keys,
            last_incompatible.unexpected_keys,
        )


def _loads_weights_in_init(transform: nn.Module) -> bool:
    return bool(getattr(transform, "_loads_weights_in_init", False))


def apply_transform_init(model: nn.Module, cfg: Config, device: torch.device) -> None:
    """Load external transform weights and optionally freeze."""
    transform = getattr(model, "transform", None)
    if transform is None or not cfg.transform.enabled:
        return
    if cfg.transform.weights_path and not _loads_weights_in_init(transform):
        load_transform_weights(transform, cfg.transform.weights_path, device)
    if cfg.transform.freeze:
        for param in transform.parameters():
            param.requires_grad = False


@register_transform("identity")
class IdentityTransform(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return x


@register_transform("linear")
class LinearTransform(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        params = cfg.transform.params
        out_dim = int(params.get("out_dim", cfg.model.input_dim))
        in_dim = int(params.get("in_dim", out_dim))
        use_bias = bool(params.get("bias", True))
        self.proj = nn.Linear(in_dim, out_dim, bias=use_bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, seq_len, d = x.shape
        flat = x.reshape(b * seq_len, d)
        out = self.proj(flat).reshape(b, seq_len, -1)
        return out

@register_transform("rmsnorm")
class RMSNormTransform(nn.Module):
    """Loads scale tensor from weights_path in __init__; skips apply_transform_init loading."""

    _loads_weights_in_init = True

    def __init__(self, cfg: Config):
        super().__init__()
        path = cfg.transform.weights_path
        if not path:
            raise ValueError("rmsnorm transform requires transform.weights_path")

        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, torch.Tensor):
            raise ValueError(
                f"rmsnorm weights_path must be a Tensor, got {type(loaded)}"
            )

        dim = cfg.model.input_dim
        scale = loaded.float().reshape(-1)
        if scale.numel() != dim:
            raise ValueError(f"scale dim {scale.numel()} != model.input_dim {dim}")

        self.register_buffer("scale_tensor", scale)
        self.eps = float(cfg.transform.params.get("eps", 1e-5))
        logger.info("Loaded rmsnorm scale from %s", path)

    def forward(self, x, mask=None):
        orig_dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        out = xf * torch.rsqrt(var + self.eps) * self.scale_tensor
        return out.to(orig_dtype)