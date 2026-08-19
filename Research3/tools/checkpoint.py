from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.io import read_json, write_json


def _checkpoint_kind(path: str | Path) -> str:
    return "adapt" if Path(path).name.startswith("adapt") else "pretrain"


def _config_fingerprint(cfg: dict[str, Any], kind: str) -> str:
    data_keys = (
        "source",
        "seed",
        "local_tokens_path",
        "tokenizer_name",
        "tokenizer_revision",
        "fineweb_dataset",
        "fineweb_config",
        "fineweb_revision",
        "streaming",
        "shuffle_buffer",
        "token_dtype",
        "vocab_size",
        "block_size",
        "train_tokens",
    )
    train_keys = (
        "token_budget",
        "micro_batch_size",
        "effective_batch_tokens",
        "learning_rate",
        "min_learning_rate",
        "warmup_fraction",
        "weight_decay",
        "beta1",
        "beta2",
        "grad_clip",
    )
    model_keys = (
        "n_layer",
        "n_head",
        "n_embd",
        "ffn_dim",
        "dropout",
        "bias",
    )
    method = str(cfg["run"]["method"])
    position_keys = {
        "rope": ("rope_base",),
        "alibi": (),
        "cable": (),
        "ra_cable": ("ra_gate_bias", "ra_sparsity_weight"),
        "ra_cable_lite": (
            "ra_gate_bias",
            "ra_sparsity_weight",
            "ra_lite_layers",
        ),
        "ra_cable_static": ("ra_gate_bias", "ra_sparsity_weight"),
        "dape_kerple": ("dape_mlp_width", "dape_kerple_epsilon"),
    }[method]
    relevant: dict[str, Any] = {
        "run": {
            "method": method,
            "seed": cfg["run"]["seed"],
            "dtype": cfg["run"]["dtype"],
        },
        "data": {key: cfg["data"][key] for key in data_keys},
        "model": {key: cfg["model"][key] for key in model_keys},
        "position": {
            key: cfg["position"][key] for key in position_keys
        },
        "train": {key: cfg["train"][key] for key in train_keys},
    }
    if kind == "adapt":
        adapt_keys = (
            "enabled",
            "max_seq_len",
            "steps",
            "batch_size",
            "micro_batch_size",
            "learning_rate",
        )
        relevant["adapt"] = {
            key: cfg["adapt"][key] for key in adapt_keys
        }
        relevant["data"].update(
            {
                "retrieval_train_samples": cfg["data"][
                    "retrieval_train_samples"
                ],
                "num_key_value_pairs": cfg["data"]["num_key_value_pairs"],
            }
        )
    encoded = json.dumps(
        relevant,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_suffix(f"{target.suffix}.meta.json")


def _cuda_device_index(device: Any) -> int | None:
    try:
        parsed = torch.device(str(device))
    except (RuntimeError, TypeError):
        return None
    if parsed.type != "cuda":
        return None
    return (
        int(parsed.index)
        if parsed.index is not None
        else int(torch.cuda.current_device())
    )


def capture_rng_state(device: Any = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        device_index = _cuda_device_index(device)
        if device_index is None:
            device_index = int(torch.cuda.current_device())
        state["cuda_device_index"] = device_index
        state["cuda_device"] = torch.cuda.get_rng_state(device_index)
    return state


def restore_rng_state(
    state: dict[str, Any] | None,
    *,
    saved_device: Any = None,
    target_device: Any = None,
) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if not torch.cuda.is_available():
        return

    target_index = _cuda_device_index(target_device)
    if target_index is None:
        return

    cuda_state = state.get("cuda_device")
    if cuda_state is None and "cuda" in state:
        # Historical checkpoints stored one RNG state for every GPU visible
        # to the old process. Select the state belonging to that job's old
        # cfg.run.device, then map it to the GPU assigned to the resumed job.
        historical_states = list(state["cuda"])
        saved_index = _cuda_device_index(saved_device)
        if saved_index is None:
            saved_index = state.get("cuda_device_index")
        if saved_index is not None and 0 <= int(saved_index) < len(
            historical_states
        ):
            cuda_state = historical_states[int(saved_index)]
        elif len(historical_states) == 1:
            cuda_state = historical_states[0]
        else:
            raise RuntimeError(
                "Cannot identify the CUDA RNG state for the saved training "
                "device. The checkpoint contains multiple GPU states but no "
                "valid saved device index."
            )
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device=target_index)


def latest_compatible_checkpoint(
    directory: str | Path,
    *,
    prefix: str,
    cfg: dict[str, Any],
) -> Path | None:
    root = Path(directory)
    candidates = list(root.glob(f"{prefix}_step*.pt"))
    compatible: list[tuple[int, Path]] = []
    for path in candidates:
        metadata_path = _metadata_path(path)
        if not metadata_path.exists():
            continue
        metadata = read_json(metadata_path)
        expected = _config_fingerprint(cfg, _checkpoint_kind(path))
        if metadata.get("config_fingerprint") == expected:
            compatible.append((int(metadata.get("step", -1)), path))
    if compatible:
        return max(compatible, key=lambda item: item[0])[1]
    if candidates:
        raise RuntimeError(
            f"Found {prefix} checkpoints, but none match the current CFG. "
            "Use the original CFG to resume, set run.force=True for a fresh "
            "run, or change run.task."
        )
    return None


def assert_checkpoint_compatible(
    path: str | Path,
    cfg: dict[str, Any],
) -> None:
    target = Path(path)
    metadata_path = _metadata_path(target)
    if not metadata_path.exists():
        raise RuntimeError(
            f"Checkpoint metadata is missing for {target}. "
            "Delete the checkpoint or set run.force=True to start over."
        )
    metadata = read_json(metadata_path)
    kind = _checkpoint_kind(target)
    expected = _config_fingerprint(cfg, kind)
    if metadata.get("config_fingerprint") != expected:
        raise RuntimeError(
            f"Configuration does not match existing checkpoint {target}. "
            "Set run.force=True only if you intend to overwrite it, or change "
            "run.task to preserve both experiments."
        )


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    step: int,
    tokens_seen: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    extra: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "cfg": cfg,
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "method": cfg["run"]["method"],
        "seed": cfg["run"]["seed"],
        "rng_state": capture_rng_state(cfg["run"].get("device")),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if extra:
        payload.update(extra)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.partial")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    kind = _checkpoint_kind(target)
    write_json(
        _metadata_path(target),
        {
            "kind": kind,
            "config_fingerprint": _config_fingerprint(cfg, kind),
            "method": cfg["run"]["method"],
            "seed": cfg["run"]["seed"],
            "step": int(step),
            "tokens_seen": int(tokens_seen),
        },
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = map_location
        saved_device = (
            payload.get("cfg", {}).get("run", {}).get("device")
        )
        restore_rng_state(
            payload.get("rng_state"),
            saved_device=saved_device,
            target_device=target_device,
        )
    return payload
