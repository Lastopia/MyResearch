from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def resolve_device(cfg: dict[str, Any]) -> torch.device:
    requested = str(cfg["run"]["device"])
    if requested == "auto":
        if bool(cfg["run"].get("require_cuda", False)) and not torch.cuda.is_available():
            raise RuntimeError(
                "This experiment requires CUDA, but torch.cuda.is_available() is false"
            )
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def resolve_dtype(cfg: dict[str, Any]) -> torch.dtype:
    name = str(cfg["run"]["dtype"]).lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def dataloader_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    workers = int(cfg.get("resources", {}).get("data_workers", 0))
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(
            cfg.get("resources", {}).get("resolved_pin_memory", False)
        ),
    }
    if workers > 0:
        kwargs.update(
            {
                "persistent_workers": bool(
                    cfg["resources"].get(
                        "resolved_persistent_workers",
                        True,
                    )
                ),
                "prefetch_factor": int(
                    cfg["resources"].get(
                        "resolved_prefetch_factor",
                        2,
                    )
                ),
            }
        )
    return kwargs
