from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from model.factory import build_model
from pipeline.data import load_token_dataset
from tools.checkpoint import (
    assert_checkpoint_compatible,
    latest_compatible_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from tools.io import (
    append_jsonl,
    read_json,
    rollback_jsonl,
    write_json,
)
from tools.log import (
    gpu_display_name,
    log_fields,
    stage_banner,
    utc_timestamp,
)
from tools.memory import (
    peak_vram_gb,
    process_rss_gb,
    reset_peak_vram,
    system_ram_total_gb,
    vram_usage_gb,
)
from tools.paths import checkpoint_dir, metric_dir
from tools.runtime import (
    autocast_context,
    dataloader_kwargs,
    resolve_device,
    resolve_dtype,
)
from tools.sampler import ResumableBatchSampler
from tools.seed import seed_everything
from tools.training_time import TrainingTimer


def _learning_rate_lambda(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_ratio: float,
) -> float:
    if step < warmup_steps:
        return max(1e-8, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_ratio + (1.0 - min_ratio) * cosine


def optimizer_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict[str, Any]]:
    decay = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.ndim >= 2
    ]
    no_decay = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.ndim < 2
    ]
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _clear_pretrain_state(cfg: dict[str, Any]) -> None:
    checkpoint_root = checkpoint_dir(cfg)
    for path in (
        path
        for pattern in ("pretrain*.pt", "pretrain*.pt.meta.json")
        for path in checkpoint_root.glob(pattern)
    ):
        path.unlink()

    metrics_root = metric_dir(cfg)
    for path in (
        metrics_root / name
        for name in (
            "train.jsonl",
            "validation.jsonl",
            "train_summary.json",
            "training_time.json",
        )
        if (metrics_root / name).exists()
    ):
        path.unlink()


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3_600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{remaining_seconds:02d}s"
    if minutes:
        return f"{minutes}m{remaining_seconds:02d}s"
    return f"{remaining_seconds}s"


def _format_usage(used: float | None, total: float | None) -> str:
    if used is None or total is None:
        return "n/a"
    return f"{float(used):.2f}/{float(total):.2f}GB"


def _log_train(cfg: dict[str, Any], **fields: Any) -> None:
    identity = [str(cfg["run"]["method"])]
    gpu = gpu_display_name(cfg)
    if gpu is not None:
        identity.append(gpu)
    body = " | ".join(
        [*identity, *(f"{key}={value}" for key, value in fields.items())]
    )
    print(f"[TRAIN] | {body}", flush=True)


@torch.no_grad()
def _validation_metrics(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    *,
    batches: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    start = time.perf_counter()
    for batch_index, (batch_input_ids, batch_targets) in enumerate(loader):
        if batch_index >= batches:
            break
        batch_input_ids = batch_input_ids.to(device)
        batch_targets = batch_targets.to(device)
        with autocast_context(device, dtype):
            output = model(batch_input_ids, batch_targets)
        valid_tokens = int(batch_targets.ne(-100).sum().item())
        total_nll += float(output["lm_loss"].item()) * valid_tokens
        total_tokens += valid_tokens
    if was_training:
        model.train()
    if total_tokens == 0:
        raise RuntimeError("Periodic validation produced no valid tokens")
    mean_nll = total_nll / total_tokens
    return {
        "validation_loss": mean_nll,
        "validation_ppl": math.exp(min(mean_nll, 20.0)),
        "validation_tokens": float(total_tokens),
        "validation_seconds": time.perf_counter() - start,
    }


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    stage_banner("TRAIN", cfg=cfg)
    seed = int(cfg["run"]["seed"])
    seed_everything(seed)
    device = resolve_device(cfg)
    dtype = resolve_dtype(cfg)
    checkpoint_root = checkpoint_dir(cfg)
    final_path = checkpoint_root / "pretrain_final.pt"
    metrics_root = metric_dir(cfg)
    summary_path = metrics_root / "train_summary.json"
    force = bool(cfg["run"].get("force", False))

    if final_path.exists() and not force:
        assert_checkpoint_compatible(final_path, cfg)
        summary = (
            read_json(summary_path)
            if summary_path.exists()
            else {
                "status": "completed_checkpoint_reused",
                "checkpoint": str(final_path),
                "method": cfg["run"]["method"],
                "seed": seed,
            }
        )
        returned = dict(summary)
        returned["runtime_status"] = "reused"
        stage_banner("TRAIN", "REUSED", cfg=cfg)
        return returned

    if force:
        _clear_pretrain_state(cfg)

    resume_path = (
        None
        if force
        else latest_compatible_checkpoint(
            checkpoint_root,
            prefix="pretrain",
            cfg=cfg,
        )
    )
    train_log = metrics_root / "train.jsonl"
    validation_log = metrics_root / "validation.jsonl"
    timer: TrainingTimer | None = None

    step = 0
    tokens_seen = 0
    data_epoch = 0
    batch_in_epoch = 0
    resumed_from_step = 0
    payload: dict[str, Any] | None = None
    try:
        dataset = load_token_dataset(cfg, "train")
        validation_dataset = load_token_dataset(cfg, "valid")
        micro_batch_size = int(cfg["train"]["micro_batch_size"])
        block_size = int(cfg["data"]["block_size"])
        micro_batch_tokens = micro_batch_size * block_size
        effective_batch_tokens = int(cfg["train"]["effective_batch_tokens"])
        if effective_batch_tokens % micro_batch_tokens != 0:
            raise ValueError(
                "effective_batch_tokens must be divisible by "
                "micro_batch_size * block_size"
            )
        accumulation_steps = effective_batch_tokens // micro_batch_tokens
        token_budget = int(cfg["train"]["token_budget"])
        tokens_per_step = micro_batch_tokens * accumulation_steps
        total_steps = math.ceil(token_budget / tokens_per_step)
        peak_lr = float(cfg["train"]["learning_rate"])
        min_ratio = float(cfg["train"]["min_learning_rate"]) / peak_lr
        warmup_steps = max(
            1,
            int(total_steps * float(cfg["train"]["warmup_fraction"])),
        )

        model = build_model(cfg).to(device)
        optimizer = torch.optim.AdamW(
            optimizer_parameter_groups(
                model,
                float(cfg["train"]["weight_decay"]),
            ),
            lr=peak_lr,
            betas=(
                float(cfg["train"]["beta1"]),
                float(cfg["train"]["beta2"]),
            ),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda current: _learning_rate_lambda(
                current,
                total_steps,
                warmup_steps,
                min_ratio,
            ),
        )

        if resume_path is not None:
            payload = load_checkpoint(
                resume_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location="cpu",
                restore_rng=True,
            )
            step = int(payload["step"])
            tokens_seen = int(payload["tokens_seen"])
            resumed_from_step = step

        rollback_jsonl(train_log, step)
        rollback_jsonl(validation_log, step)
        summary_path.unlink(missing_ok=True)
        timer = TrainingTimer(
            metrics_root / "training_time.json",
            stage="pretrain",
            method=str(cfg["run"]["method"]),
            seed=seed,
            gpu_count=1 if device.type == "cuda" else 0,
            resumed_from=str(resume_path) if resume_path else None,
            resume_snapshot=(
                payload.get("time_accounting")
                if payload is not None
                else None
            ),
        )

        sampler = ResumableBatchSampler(
            dataset,
            batch_size=micro_batch_size,
            seed=seed,
        )
        batches_per_epoch = sampler.batches_per_epoch
        if resume_path is not None:
            consumed_micro_batches = step * accumulation_steps
            data_epoch = int(
                payload.get(
                    "data_epoch",
                    consumed_micro_batches // batches_per_epoch,
                )
            )
            batch_in_epoch = int(
                payload.get(
                    "batch_in_epoch",
                    consumed_micro_batches % batches_per_epoch,
                )
            )
            if batch_in_epoch >= batches_per_epoch:
                data_epoch += batch_in_epoch // batches_per_epoch
                batch_in_epoch %= batches_per_epoch
        sampler.set_position(data_epoch, batch_in_epoch)
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            **dataloader_kwargs(cfg),
        )
        iterator = iter(loader)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(cfg["eval"]["batch_size"]),
            shuffle=False,
            **dataloader_kwargs(cfg),
        )

        log_interval = int(cfg["train"]["log_interval"])
        eval_interval = int(cfg["train"]["eval_interval"])
        save_interval = int(cfg["train"]["save_interval"])
        eval_batches = int(cfg["train"]["eval_batches"])
        ram_total_gb = system_ram_total_gb()

        reset_peak_vram(device)
        model.train()
        start_timing = timer.update(
            step=step,
            tokens_seen=tokens_seen,
            peak_vram_gb=peak_vram_gb(device),
            peak_host_ram_gb=process_rss_gb(peak=True),
        )
        _log_train(
            cfg,
            state="resumed" if resume_path else "fresh",
            checkpoint=resume_path.name if resume_path else "none",
            step=f"{step:,}",
            total=_format_duration(start_timing["wall_clock_seconds"]),
        )
        last_log_training_seconds = float(
            start_timing["wall_clock_seconds"]
        )

        while tokens_seen < token_budget:
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            step_lm_loss = 0.0
            step_position_loss = 0.0
            step_tokens = 0
            for _ in range(accumulation_steps):
                if batch_in_epoch >= batches_per_epoch:
                    data_epoch += 1
                    batch_in_epoch = 0
                    sampler.set_position(data_epoch, batch_in_epoch)
                    iterator = iter(loader)
                input_ids, targets = next(iterator)
                batch_in_epoch += 1
                non_blocking = bool(
                    cfg["resources"].get("resolved_pin_memory", False)
                )
                input_ids = input_ids.to(device, non_blocking=non_blocking)
                targets = targets.to(device, non_blocking=non_blocking)
                with autocast_context(device, dtype):
                    output = model(input_ids, targets)
                    loss = output["loss"] / accumulation_steps
                loss.backward()
                step_lm_loss += float(output["lm_loss"].detach().item())
                step_position_loss += float(
                    output["position_loss"].detach().item()
                )
                step_tokens += input_ids.numel()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg["train"]["grad_clip"]),
            )
            optimizer.step()
            scheduler.step()
            step += 1
            tokens_seen += step_tokens
            step_seconds = time.perf_counter() - step_start
            finished = tokens_seen >= token_budget
            record = {
                "task": cfg["run"]["task"],
                "method": cfg["run"]["method"],
                "seed": seed,
                "stage": "train",
                "step": step,
                "tokens_seen": tokens_seen,
                "lm_loss": step_lm_loss / accumulation_steps,
                "position_loss": step_position_loss / accumulation_steps,
                "grad_norm": float(grad_norm),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "step_seconds": step_seconds,
                "tokens_per_second": step_tokens
                / max(step_seconds, 1e-9),
                "peak_vram_gb": peak_vram_gb(device),
                "host_ram_gb": process_rss_gb(),
                "data_epoch": data_epoch,
                "batch_in_epoch": batch_in_epoch,
                "timestamp": utc_timestamp(),
            }

            should_log = step % log_interval == 0 or finished
            should_eval = step % eval_interval == 0 or finished
            should_save = (
                step % save_interval == 0
                and not finished
            )

            # Persist cumulative cost every optimizer step. This happens
            # before the console line so resumed runs report the durable
            # cross-session total from the time ledger.
            timing = timer.update(
                step=step,
                tokens_seen=tokens_seen,
                add_seconds=step_seconds,
                peak_vram_gb=peak_vram_gb(device),
                peak_host_ram_gb=process_rss_gb(peak=True),
            )

            if should_log:
                interval_seconds = max(
                    0.0,
                    float(timing["wall_clock_seconds"])
                    - last_log_training_seconds,
                )
                last_log_training_seconds = float(
                    timing["wall_clock_seconds"]
                )
                current_ram_gb = process_rss_gb()
                current_vram_gb, total_vram_gb = vram_usage_gb(device)
                record.update(
                    {
                        "log_interval_seconds": interval_seconds,
                        "cumulative_wall_seconds": timing[
                            "wall_clock_seconds"
                        ],
                    }
                )
                append_jsonl(train_log, record)
                _log_train(
                    cfg,
                    step=f"{step:,}",
                    loss=f"{record['lm_loss']:.4f}",
                    lr=f"{record['learning_rate']:.3e}",
                    vram=_format_usage(
                        current_vram_gb,
                        total_vram_gb,
                    ),
                    ram=_format_usage(
                        current_ram_gb,
                        ram_total_gb,
                    ),
                    time=(
                        f"{_format_duration(interval_seconds)}/"
                        f"{_format_duration(timing['wall_clock_seconds'])}"
                    ),
                )

            if should_eval:
                validation = _validation_metrics(
                    model,
                    validation_loader,
                    batches=eval_batches,
                    device=device,
                    dtype=dtype,
                )
                validation_record = {
                    "task": cfg["run"]["task"],
                    "method": cfg["run"]["method"],
                    "seed": seed,
                    "stage": "validation",
                    "step": step,
                    "tokens_seen": tokens_seen,
                    **validation,
                    "timestamp": utc_timestamp(),
                }
                append_jsonl(validation_log, validation_record)
                log_fields(
                    "validation",
                    cfg=cfg,
                    method=cfg["run"]["method"],
                    step=step,
                    loss=f"{validation['validation_loss']:.4f}",
                    ppl=f"{validation['validation_ppl']:.3f}",
                )

            if should_save:
                checkpoint_path = (
                    checkpoint_root / f"pretrain_step{step}.pt"
                )
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cfg=cfg,
                    step=step,
                    tokens_seen=tokens_seen,
                    extra={
                        "data_epoch": data_epoch,
                        "batch_in_epoch": batch_in_epoch,
                        "time_accounting": timer.snapshot(),
                    },
                )
                timer.commit(
                    checkpoint_path,
                    step=step,
                    tokens_seen=tokens_seen,
                )

        timer.update(
            step=step,
            tokens_seen=tokens_seen,
            peak_vram_gb=peak_vram_gb(device),
            peak_host_ram_gb=process_rss_gb(peak=True),
        )
        save_checkpoint(
            final_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            step=step,
            tokens_seen=tokens_seen,
            extra={
                "data_epoch": data_epoch,
                "batch_in_epoch": batch_in_epoch,
                "time_accounting": timer.snapshot(),
            },
        )
        timer.commit(
            final_path,
            step=step,
            tokens_seen=tokens_seen,
        )
        timing = timer.update(
            step=step,
            tokens_seen=tokens_seen,
            status="completed",
            peak_vram_gb=peak_vram_gb(device),
            peak_host_ram_gb=process_rss_gb(peak=True),
        )
        summary = {
            "status": "completed",
            "checkpoint": str(final_path),
            "method": cfg["run"]["method"],
            "seed": seed,
            "steps": step,
            "tokens_seen": tokens_seen,
            "resumed": resume_path is not None,
            "resumed_from_checkpoint": (
                str(resume_path) if resume_path else None
            ),
            "resumed_from_step": resumed_from_step,
            **timing,
            "gpu_count": 1 if device.type == "cuda" else 0,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "effective_batch_tokens": effective_batch_tokens,
            "data_workers": int(
                cfg["resources"].get("data_workers", 0)
            ),
            "log_interval": log_interval,
            "eval_interval": eval_interval,
            "save_interval": save_interval,
        }
        write_json(summary_path, summary)
        stage_banner("TRAIN", "DONE", cfg=cfg)
        return summary
    except BaseException:
        if timer is not None:
            timer.rollback()
        raise


def load_pretrained_model(
    cfg: dict[str, Any],
    *,
    prefer_adapted: bool = False,
    require_adapted: bool = False,
) -> tuple[torch.nn.Module, Path]:
    device = resolve_device(cfg)
    model = build_model(cfg)
    candidates = []
    if prefer_adapted:
        candidates.append(checkpoint_dir(cfg) / "adapt_final.pt")
    candidates.append(checkpoint_dir(cfg) / "pretrain_final.pt")
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if require_adapted and (
        path is None or path.name != "adapt_final.pt"
    ):
        raise RuntimeError(
            "An adapted checkpoint was requested but adapt_final.pt is missing. "
            "Run `python main.py run adapt method=...` first."
        )
    if path is None:
        with torch.enable_grad():
            run(cfg)
        path = checkpoint_dir(cfg) / "pretrain_final.pt"
    assert_checkpoint_compatible(path, cfg)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.to(device)
    return model, path
