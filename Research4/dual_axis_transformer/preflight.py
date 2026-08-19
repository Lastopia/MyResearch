"""Fail-fast server and real-data checks before an unattended suite."""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .external_model import ByteTokenizer, CausalLanguageModel
from .external_tasks import _load_examples as load_clutrr_examples
from .external_tasks import _model_config, ensure_clutrr_dataset
from .language_model import (
    active_corpus,
    active_language_settings,
    ensure_active_language_dataset,
    formal_language_enabled,
    language_model_config,
)
from .formal_data import GPT2Tokenizer
from .research_model import (
    COLOR_CLASSES,
    COUNTRY_CLASSES,
    ResearchModelConfig,
    build_model,
    initialize_named_parameters,
)
from .resources import detect_resources
from .runner import _autocast, resolve_run_config
from .synthetic_data import ensure_synthetic_dataset, load_examples
from .storage import prepare_storage


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _check_writable(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    directory = output_root / "preflight"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, delete=True
    ) as handle:
        handle.write("write-probe")
        handle.flush()


def _training_probe(
    device: torch.device, cfg: dict[str, Any], size_name: str = "large"
) -> dict[str, Any]:
    length = min(32, int(active_language_settings(cfg, size_name)["sequence_length"]))
    config = (
        language_model_config(cfg, "concept_bus_v2", length, size_name)
        if formal_language_enabled(cfg, size_name)
        else _model_config(cfg, "concept_bus_v2", length, size_name)
    )
    model = CausalLanguageModel(config)
    initialize_named_parameters(model, 20260806)
    model.to(device).train()
    vocab_size = (
        GPT2Tokenizer.vocab_size
        if formal_language_enabled(cfg, size_name)
        else ByteTokenizer.vocab_size
    )
    inputs = torch.randint(3, vocab_size, (2, length), device=device)
    mask = torch.ones_like(inputs)
    targets = torch.randint(3, vocab_size, (2, length), device=device)
    with _autocast(device):
        output = model(inputs, None)
        loss = F.cross_entropy(
            output.logits.reshape(-1, output.logits.shape[-1]), targets.reshape(-1)
        )
    loss.backward()
    groups = {
        "projector": ".ffn.projector.",
        "attention": ".ffn.concept_attention.",
        "readout": ".ffn.concept_readout.",
    }
    gradient_norms = {
        group: sum(
            float(parameter.grad.float().norm())
            for name, parameter in model.named_parameters()
            if marker in name and parameter.grad is not None
        )
        for group, marker in groups.items()
    }
    bus_gradient = sum(gradient_norms.values())
    loss_value = float(loss.detach())
    bad_gradients = {
        name: value
        for name, value in gradient_norms.items()
        if not math.isfinite(value) or value <= 0.0
    }
    if not math.isfinite(loss_value) or bad_gradients:
        raise RuntimeError(
            f"training probe failed: loss={loss_value}, gradients={gradient_norms}"
        )

    # A platform-specific attention/mask regression must fail before a
    # multi-day run. Changing future tokens may not change prefix logits.
    model.eval()
    split = length // 2
    changed = inputs.clone()
    changed[:, split:] = torch.randint(
        3, vocab_size, changed[:, split:].shape, device=device
    )
    with torch.no_grad(), _autocast(device):
        original_prefix = model(inputs, None).logits[:, :split].float()
        changed_prefix = model(changed, None).logits[:, :split].float()
    future_leakage = float((original_prefix - changed_prefix).abs().max())
    if not math.isfinite(future_leakage) or future_leakage > 1e-5:
        raise RuntimeError(
            f"causal-mask probe failed: prefix max error={future_leakage}"
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "device": str(device),
        "loss": loss_value,
        "concept_bus_v2_gradient_norm": bus_gradient,
        "gradient_norms": gradient_norms,
        "future_leakage_max_abs": future_leakage,
        "peak_cuda_memory_bytes": float(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0.0,
    }


def _fair_synthetic_capacity_probe(
    device: torch.device,
    cfg: dict[str, Any],
    size_name: str,
    phase_name: str,
) -> dict[str, Any]:
    """Exercise the largest fair synthetic model at the exact scheduled shape."""

    phase = next(
        phase
        for phase in cfg["sizes"][size_name].get("suite", [])
        if phase.get("name") == phase_name
    )
    methods = list(phase.get("methods", cfg["sizes"][size_name]["methods"]))
    method = (
        "concept_bus_v2" if "concept_bus_v2" in methods else methods[-1]
    )
    resolved = resolve_run_config(
        cfg,
        size_name,
        method,
        int(cfg["sizes"][size_name]["seeds"][0]),
        phase_name,
    )
    values = resolved["model"]
    model_config = ResearchModelConfig(
        # A conservative synthetic vocabulary; embeddings are not the memory
        # bottleneck, while sequence/batch/optimizer state are exact.
        vocab_size=512,
        max_length=int(resolved["data"]["max_length"]),
        method=method,
        num_layers=int(values["num_layers"]),
        d_model=int(values["d_model"]),
        d_ff=int(values["d_ff"]),
        num_heads=int(values["num_heads"]),
        slot_dim=int(values["slot_dim"]),
        num_bus_slots=int(values["num_bus_slots"]),
        bus_heads=int(values["bus_heads"]),
        bus_layers=int(values["bus_layers"]),
        concept_residual_dim=int(values["concept_residual_dim"]),
        dropout=float(values["dropout"]),
        norm_eps=float(values["norm_eps"]),
        rope_theta=float(values["rope_theta"]),
        bias=bool(values["bias"]),
        keep_residual_attention=bool(values["keep_residual_attention"]),
    )
    micro_batch = int(phase["required_micro_batch_size"])
    length = int(resolved["data"]["max_length"])
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(model_config)
    initialize_named_parameters(model, 20260807)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(resolved["train"]["learning_rate"]),
        betas=tuple(float(value) for value in resolved["optimizer"]["betas"]),
    )
    inputs = torch.randint(
        0, model_config.vocab_size, (micro_batch, length), device=device
    )
    mask = torch.ones_like(inputs)
    country_targets = torch.randint(
        0, len(COUNTRY_CLASSES), (micro_batch,), device=device
    )
    color_targets = torch.randint(
        0, len(COLOR_CLASSES), (micro_batch,), device=device
    )
    concept_targets = torch.randint(
        0, 2, (micro_batch, 12), device=device, dtype=torch.float32
    )
    with _autocast(device):
        output = model(inputs, mask)
        loss = 0.5 * (
            F.cross_entropy(output.country_logits, country_targets)
            + F.cross_entropy(output.color_logits, color_targets)
        )
        if output.concept_logits is not None:
            loss = loss + F.binary_cross_entropy_with_logits(
                output.concept_logits, concept_targets
            )
        counterfactuals = (
            output.country_swap_country_logits,
            output.country_swap_color_logits,
            output.color_swap_country_logits,
            output.color_swap_color_logits,
        )
        for logits, targets in zip(
            counterfactuals,
            (country_targets, color_targets, country_targets, color_targets),
        ):
            if logits is not None:
                loss = loss + F.cross_entropy(logits, targets)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()  # forces allocation of the real AdamW moment tensors
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "device": str(device),
        "method": method,
        "micro_batch_size": micro_batch,
        "sequence_length": length,
        "loss": float(loss.detach()),
        "gradient_norm": gradient_norm,
        "peak_cuda_memory_bytes": (
            float(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0.0
        ),
    }
    del optimizer, model, inputs, mask, country_targets, color_targets
    del concept_targets, output, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_preflight(
    cfg: dict[str, Any],
    *,
    size_name: str,
    output_root: str | Path,
    prepare_data: bool,
) -> Path:
    output_root = Path(output_root).resolve()
    roots = prepare_storage(cfg, output_root)
    report_path = output_root / "preflight" / "report.json"
    checks: dict[str, Any] = {}
    errors: list[str] = []

    try:
        _check_writable(output_root)
        checks["output_writable"] = True
    except Exception as error:
        errors.append(f"output_writable: {type(error).__name__}: {error}")

    disk = shutil.disk_usage(output_root)
    free_gb = disk.free / 1024**3
    minimum_gb = float(cfg.get("preflight", {}).get("minimum_free_disk_gb", 10.0))
    checks["disk"] = {
        "free_gb": free_gb,
        "total_gb": disk.total / 1024**3,
        "required_free_gb": minimum_gb,
    }
    if free_gb < minimum_gb:
        errors.append(
            f"disk: only {free_gb:.1f} GiB free; {minimum_gb:.1f} GiB required"
        )

    snapshot = detect_resources(required_dtype=str(cfg["resources"]["required_dtype"]))
    checks["resources"] = snapshot.to_dict()
    usable = [gpu for gpu in snapshot.gpus if gpu.torch_usable]
    require_gpu = bool(cfg["sizes"][size_name].get("require_gpu", False))
    if require_gpu and not usable:
        errors.append("gpu: no device passed the real PyTorch CUDA kernel probe")
    device = torch.device("cuda", usable[0].index) if usable else torch.device("cpu")
    if not (require_gpu and not usable):
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            checks["training_probe"] = _training_probe(device, cfg, size_name)
        except Exception as error:
            errors.append(f"training_probe: {type(error).__name__}: {error}")
        capacity_phases = [
            phase
            for phase in cfg["sizes"][size_name].get("suite", [])
            if phase.get("runner") == "synthetic"
            and phase.get("required_micro_batch_size") is not None
        ]
        if usable and capacity_phases:
            try:
                checks["fair_synthetic_capacity_probe"] = [
                    _fair_synthetic_capacity_probe(
                        torch.device("cuda", gpu.index),
                        cfg,
                        size_name,
                        str(capacity_phases[0]["name"]),
                    )
                    for gpu in usable
                ]
            except Exception as error:
                errors.append(
                    "fair_synthetic_capacity_probe: "
                    f"{type(error).__name__}: {error}"
                )

    can_prepare_data = not (require_gpu and not usable)
    configured_suite = cfg["sizes"][size_name].get("suite", [])
    required_runners = (
        {str(phase["runner"]) for phase in configured_suite}
        if configured_suite
        else {"synthetic", "clutrr", "language_model"}
    )
    if prepare_data and not can_prepare_data:
        checks["data_preparation"] = "skipped because required GPU is unavailable"
    elif prepare_data:
        if "synthetic" in required_runners:
            try:
                synthetic = ensure_synthetic_dataset(
                    roots.data, cfg["sizes"][size_name]["data"]
                )
                example = load_examples(synthetic.split_paths["train"])[0]
                checks["synthetic_data"] = {
                    "manifest": str(synthetic.manifest_path),
                    "sample_id": example.example_id,
                }
            except Exception as error:
                errors.append(f"synthetic_data: {type(error).__name__}: {error}")
        if "clutrr" in required_runners:
            try:
                clutrr = ensure_clutrr_dataset(cfg, roots.data)
                train = load_clutrr_examples(clutrr.train_path)
                test = load_clutrr_examples(clutrr.test_path)
                lengths = sorted({example.length for example in test})
                required = sorted(
                    int(value) for value in cfg["external"]["clutrr"]["test_lengths"]
                )
                if not train or not test or not set(required).issubset(lengths):
                    raise RuntimeError(
                        f"normalized rows/lengths incomplete: train={len(train)}, "
                        f"test={len(test)}, lengths={lengths}"
                    )
                checks["clutrr_data"] = {
                    "manifest": str(clutrr.manifest_path),
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "test_lengths": lengths,
                }
            except Exception as error:
                errors.append(f"clutrr_data: {type(error).__name__}: {error}")
        # The 1B-token preparation is deliberately last: cheap structural and
        # CLUTRR failures should surface before any multi-GB language download.
        if "language_model" in required_runners:
            try:
                stories = ensure_active_language_dataset(cfg, roots.data, size_name)
                train_corpus = active_corpus(cfg, stories.train_path, size_name)
                validation_corpus = active_corpus(cfg, stories.validation_path, size_name)
                checks["language_data"] = {
                    "manifest": str(stories.manifest_path),
                    "train_tokens": len(train_corpus),
                    "validation_tokens": len(validation_corpus),
                }
            except Exception as error:
                errors.append(f"language_data: {type(error).__name__}: {error}")

    required_entrypoint_files = [
        Path(cfg["paths"]["project_root"]) / name
        for name in ("main.py", "requirements.txt")
    ]
    missing_entrypoint_files = [
        str(path) for path in required_entrypoint_files if not path.exists()
    ]
    checks["python_entrypoint"] = {
        "paths": [str(path) for path in required_entrypoint_files],
        "missing": missing_entrypoint_files,
    }
    if missing_entrypoint_files:
        errors.append(f"python_entrypoint: missing {missing_entrypoint_files}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "size": size_name,
        "prepare_data": prepare_data,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "pid": os.getpid(),
        },
        "checks": checks,
        "errors": errors,
    }
    _write_atomic(report_path, payload)
    if errors:
        raise RuntimeError(
            "preflight failed; inspect " + str(report_path) + ": " + " | ".join(errors)
        )
    return report_path
