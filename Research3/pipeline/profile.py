from __future__ import annotations

import statistics
import time
from typing import Any, Callable

import torch

from pipeline.train import load_pretrained_model, optimizer_parameter_groups
from tools.io import read_json, write_json
from tools.log import log_resources, stage_banner
from tools.memory import (
    hardware_metadata,
    phase_peak_rss_gb,
    peak_vram_gb,
    reset_peak_vram,
    synchronize,
)
from tools.paths import metric_dir, profile_dir
from tools.runtime import autocast_context, resolve_device, resolve_dtype


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered))))
    return {
        "median_seconds": statistics.median(ordered),
        "p95_seconds": ordered[p95_index],
    }


def _measure(
    callback: Callable[[], None],
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> tuple[list[float], float | None]:
    with phase_peak_rss_gb() as memory:
        for _ in range(warmup):
            callback()
        synchronize(device)
        samples: list[float] = []
        for _ in range(repeat):
            start = time.perf_counter()
            callback()
            synchronize(device)
            samples.append(time.perf_counter() - start)
    return samples, memory["peak_rss_gb"]


def _parameter_counts(model: torch.nn.Module) -> dict[str, float | int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    position = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if ".position." in name
    )
    return {
        "total_parameters": total,
        "position_parameters": position,
        "position_parameter_ratio": position / max(1, total),
    }


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    stage_banner("PROFILE", cfg=cfg)
    device = resolve_device(cfg)
    dtype = resolve_dtype(cfg)
    checkpoint_kind = str(cfg["profile"].get("checkpoint", "adapt")).lower()
    if checkpoint_kind not in {"pretrain", "adapt", "auto"}:
        raise ValueError("profile.checkpoint must be pretrain, adapt, or auto")
    model, checkpoint = load_pretrained_model(
        cfg,
        prefer_adapted=checkpoint_kind in {"adapt", "auto"},
        require_adapted=checkpoint_kind == "adapt",
    )
    batch_size = int(cfg["profile"]["batch_size"])
    train_length = int(cfg["profile"]["train_length"])
    warmup = int(cfg["profile"]["warmup"])
    repeat = max(1, int(cfg["profile"]["repeat"]))
    decode_tokens = max(1, int(cfg["profile"]["decode_tokens"]))
    vocab_size = int(cfg["data"]["vocab_size"])
    max_seq_len = int(cfg["model"]["max_seq_len"])
    use_kv_cache = bool(cfg["profile"]["use_kv_cache"])
    generator = torch.Generator(device="cpu").manual_seed(int(cfg["data"]["seed"]))

    result: dict[str, Any] = {
        "method": cfg["run"]["method"],
        "seed": cfg["run"]["seed"],
        "checkpoint": str(checkpoint),
        "checkpoint_kind": checkpoint_kind,
        "device": str(device),
        "physical_gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "hardware": hardware_metadata(),
        "dtype": str(dtype),
        "batch_size": batch_size,
        "inference_batch_size": batch_size,
        "attention_kernel": str(cfg["profile"]["attention_kernel"]),
        "measurement_protocol": {
            "fairness": (
                "all method-seed profiles are run sequentially on one "
                "physical GPU by the all/extensions orchestrator"
            ),
            "warmup_trials": warmup,
            "measurement_trials": repeat,
            "decode_tokens": decode_tokens,
            "host_ram_peak": (
                "phase-local current RSS sampler; peaks do not carry across "
                "train, prefill and decode"
            ),
        },
        "training_profile": {
            "length": train_length,
            "micro_batch_size": int(cfg["train"]["micro_batch_size"]),
            "effective_batch_tokens": int(
                cfg["train"]["effective_batch_tokens"]
            ),
        },
        "decode_mode": (
            "incremental_with_kv_cache"
            if use_kv_cache
            else "full_recompute_without_kv_cache"
        ),
        "kv_cache_requested": use_kv_cache,
        "kv_cache_supported": True,
        **_parameter_counts(model),
        "lengths": {},
    }

    for length_value in cfg["profile"]["lengths"]:
        length = int(length_value)
        if length > max_seq_len:
            continue
        input_ids = torch.randint(
            0,
            vocab_size,
            (batch_size, length),
            generator=generator,
        ).to(device)
        length_result: dict[str, Any] = {"status": "completed"}
        try:
            if length == train_length:
                train_micro_batch = int(cfg["train"]["micro_batch_size"])
                effective_batch_tokens = int(
                    cfg["train"]["effective_batch_tokens"]
                )
                micro_batch_tokens = train_micro_batch * length
                if effective_batch_tokens % micro_batch_tokens:
                    raise ValueError(
                        "profile train length and train micro-batch must divide "
                        "train.effective_batch_tokens"
                    )
                gradient_accumulation = (
                    effective_batch_tokens // micro_batch_tokens
                )
                train_input_ids = torch.randint(
                    0,
                    vocab_size,
                    (train_micro_batch, length),
                    generator=generator,
                ).to(device)
                train_targets = torch.randint(
                    0,
                    vocab_size,
                    (train_micro_batch, length),
                    generator=generator,
                ).to(device)
                # lr=0 includes AdamW bookkeeping without changing the loaded
                # checkpoint used by the subsequent inference measurements.
                optimizer = torch.optim.AdamW(
                    optimizer_parameter_groups(
                        model,
                        float(cfg["train"]["weight_decay"]),
                    ),
                    lr=0.0,
                    betas=(
                        float(cfg["train"]["beta1"]),
                        float(cfg["train"]["beta2"]),
                    ),
                )
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda _: 1.0,
                )
                model.train()

                def train_step() -> None:
                    optimizer.zero_grad(set_to_none=True)
                    for _ in range(gradient_accumulation):
                        with autocast_context(device, dtype):
                            output = model(train_input_ids, train_targets)
                            loss = output["loss"] / gradient_accumulation
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(cfg["train"]["grad_clip"]),
                    )
                    optimizer.step()
                    scheduler.step()

                reset_peak_vram(device)
                train_samples, train_peak_ram = _measure(
                    train_step,
                    warmup=warmup,
                    repeat=repeat,
                    device=device,
                )
                train_summary = _summary(train_samples)
                train_summary.update(
                    {
                        "tokens_per_second": effective_batch_tokens
                        / train_summary["median_seconds"],
                        "effective_batch_tokens": effective_batch_tokens,
                        "micro_batch_size": train_micro_batch,
                        "micro_batch_tokens": micro_batch_tokens,
                        "gradient_accumulation_steps": gradient_accumulation,
                        "gradient_clipping_included": True,
                        "optimizer_step_included": True,
                        "scheduler_step_included": True,
                        "peak_vram_gb": peak_vram_gb(device),
                        "peak_host_ram_gb": train_peak_ram,
                    }
                )
                length_result["train_step"] = train_summary
                optimizer.zero_grad(set_to_none=True)
                del optimizer, scheduler, train_input_ids, train_targets
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            model.eval()

            @torch.no_grad()
            def prefill() -> None:
                with autocast_context(device, dtype):
                    model(input_ids, use_cache=use_kv_cache)

            reset_peak_vram(device)
            prefill_samples, prefill_peak_ram = _measure(
                prefill,
                warmup=warmup,
                repeat=repeat,
                device=device,
            )
            prefill_summary = _summary(prefill_samples)
            prefill_summary.update(
                {
                    "tokens_per_second": (batch_size * length)
                    / prefill_summary["median_seconds"],
                    "peak_vram_gb": peak_vram_gb(device),
                    "peak_host_ram_gb": prefill_peak_ram,
                }
            )
            length_result["prefill"] = prefill_summary

            decode_context_length = min(
                length,
                max(1, max_seq_len - decode_tokens),
            )
            decode_input = input_ids[:, :decode_context_length]
            if use_kv_cache:
                with torch.no_grad(), autocast_context(device, dtype):
                    initial = model(decode_input, use_cache=True)
                initial_past = initial["past_key_values"]
                initial_token = initial["logits"][:, -1, :].argmax(
                    dim=-1,
                    keepdim=True,
                )

                @torch.no_grad()
                def decode() -> None:
                    past = initial_past
                    current_token = initial_token
                    for _ in range(decode_tokens):
                        with autocast_context(device, dtype):
                            decoded = model(
                                current_token,
                                past_key_values=past,
                                use_cache=True,
                            )
                        past = decoded["past_key_values"]
                        current_token = decoded["logits"][:, -1, :].argmax(
                            dim=-1,
                            keepdim=True,
                        )
            else:

                @torch.no_grad()
                def decode() -> None:
                    current = decode_input
                    for _ in range(decode_tokens):
                        with autocast_context(device, dtype):
                            logits = model(current)["logits"]
                        next_token = logits[:, -1, :].argmax(
                            dim=-1,
                            keepdim=True,
                        )
                        current = torch.cat((current, next_token), dim=1)

            reset_peak_vram(device)
            decode_samples, decode_peak_ram = _measure(
                decode,
                warmup=warmup,
                repeat=repeat,
                device=device,
            )
            decode_summary = _summary(decode_samples)
            decode_summary.update(
                {
                    "milliseconds_per_token": 1000.0
                    * decode_summary["median_seconds"]
                    / decode_tokens,
                    "tokens_per_second": batch_size
                    * decode_tokens
                    / decode_summary["median_seconds"],
                    "generated_tokens_per_trial": batch_size * decode_tokens,
                    "decode_context_length": decode_context_length,
                    "peak_vram_gb": peak_vram_gb(device),
                    "peak_host_ram_gb": decode_peak_ram,
                }
            )
            length_result["decode"] = decode_summary
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            length_result = {
                "status": "out_of_memory",
                "error": str(error),
                "peak_vram_gb": peak_vram_gb(device),
                "peak_host_ram_gb": None,
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result["lengths"][str(length)] = length_result
        log_resources(
            cfg,
            "profile",
            device=device,
            length=length,
            status=length_result["status"],
        )

    train_summary_path = metric_dir(cfg) / "train_summary.json"
    adapt_summary_path = metric_dir(cfg) / "adapt_summary.json"
    pretraining: dict[str, Any] | None = None
    adaptation: dict[str, Any] | None = None
    if train_summary_path.exists():
        pretraining = read_json(train_summary_path)
        result["pretraining"] = pretraining
    if adapt_summary_path.exists():
        adaptation = read_json(adapt_summary_path)
        result["adaptation"] = adaptation
    if pretraining is not None:
        time_basis = pretraining.get("time_basis")
        if adaptation is not None and adaptation.get("time_basis") != time_basis:
            raise RuntimeError(
                "Pretraining and adaptation use different time bases. "
                "Delete the outputs and rerun both stages."
            )
        result["full_training"] = {
            "time_basis": time_basis,
            "wall_clock_seconds": float(
                pretraining.get("wall_clock_seconds", 0.0)
            )
            + float(
                adaptation.get("wall_clock_seconds", 0.0)
                if adaptation
                else 0.0
            ),
            "gpu_hours": float(pretraining.get("gpu_hours", 0.0))
            + float(
                adaptation.get("gpu_hours", 0.0)
                if adaptation
                else 0.0
            ),
            "pretrain_wall_clock_seconds": float(
                pretraining.get("wall_clock_seconds", 0.0)
            ),
            "adapt_wall_clock_seconds": float(
                adaptation.get("wall_clock_seconds", 0.0)
                if adaptation
                else 0.0
            ),
            "pretrain_resume_count": int(
                pretraining.get("resume_count", 0)
            ),
            "adapt_resume_count": int(
                adaptation.get("resume_count", 0)
                if adaptation
                else 0
            ),
        }
    write_json(profile_dir(cfg) / "efficiency.json", result)
    stage_banner("PROFILE", "DONE", cfg=cfg)
    return result
