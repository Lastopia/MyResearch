from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from pipeline.data import RetrievalDataset, load_retrieval_adapt
from pipeline.train import load_pretrained_model, optimizer_parameter_groups
from tools.checkpoint import (
    assert_checkpoint_compatible,
    latest_compatible_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from tools.io import append_jsonl, read_json, rollback_jsonl, write_json
from tools.log import log_fields, log_resources, stage_banner, utc_timestamp
from tools.memory import peak_vram_gb, process_rss_gb, reset_peak_vram
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


def _clear_adapt_state(cfg: dict[str, Any]) -> None:
    checkpoint_root = checkpoint_dir(cfg)
    for path in (
        path
        for pattern in ("adapt*.pt", "adapt*.pt.meta.json")
        for path in checkpoint_root.glob(pattern)
    ):
        path.unlink()

    metrics_root = metric_dir(cfg)
    for path in (
        metrics_root / name
        for name in (
            "adapt.jsonl",
            "adapt_summary.json",
            "adaptation_time.json",
        )
        if (metrics_root / name).exists()
    ):
        path.unlink()


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    stage_banner("ADAPT", cfg=cfg)
    if not bool(cfg["adapt"]["enabled"]):
        summary = {"status": "disabled"}
        write_json(metric_dir(cfg) / "adapt_summary.json", summary)
        return summary

    checkpoint_root = checkpoint_dir(cfg)
    metrics_root = metric_dir(cfg)
    final_path = checkpoint_root / "adapt_final.pt"
    summary_path = metrics_root / "adapt_summary.json"
    adapt_log = metrics_root / "adapt.jsonl"
    force = bool(cfg["run"].get("force", False))
    if final_path.exists() and not force:
        assert_checkpoint_compatible(final_path, cfg)
        summary = (
            read_json(summary_path)
            if summary_path.exists()
            else {
                "status": "completed_checkpoint_reused",
                "checkpoint": str(final_path),
            }
        )
        returned = dict(summary)
        returned["runtime_status"] = "reused"
        stage_banner("ADAPT", "REUSED", cfg=cfg)
        return returned

    if force:
        _clear_adapt_state(cfg)

    resume_path = (
        None
        if force
        else latest_compatible_checkpoint(
            checkpoint_root,
            prefix="adapt",
            cfg=cfg,
        )
    )
    seed = int(cfg["run"]["seed"])
    seed_everything(seed)
    device = resolve_device(cfg)
    dtype = resolve_dtype(cfg)
    timer: TrainingTimer | None = None

    step = 0
    tokens_seen = 0
    data_epoch = 0
    batch_in_epoch = 0
    resumed_from_step = 0
    payload: dict[str, Any] | None = None
    try:
        model, _ = load_pretrained_model(cfg)
        model.train()
        retrieval = load_retrieval_adapt(cfg)
        dataset = RetrievalDataset(retrieval)
        effective_batch_size = int(cfg["adapt"]["batch_size"])
        micro_batch_size = int(cfg["adapt"]["micro_batch_size"])
        if effective_batch_size % micro_batch_size != 0:
            raise ValueError(
                "adapt.batch_size must be divisible by adapt.micro_batch_size"
            )
        accumulation_steps = effective_batch_size // micro_batch_size
        sampler = ResumableBatchSampler(
            dataset,
            batch_size=micro_batch_size,
            seed=seed,
        )
        batches_per_epoch = sampler.batches_per_epoch
        optimizer = torch.optim.AdamW(
            optimizer_parameter_groups(
                model,
                float(cfg["train"]["weight_decay"]),
            ),
            lr=float(cfg["adapt"]["learning_rate"]),
            betas=(
                float(cfg["train"]["beta1"]),
                float(cfg["train"]["beta2"]),
            ),
        )

        if resume_path is not None:
            payload = load_checkpoint(
                resume_path,
                model,
                optimizer=optimizer,
                map_location="cpu",
                restore_rng=True,
            )
            step = int(payload["step"])
            tokens_seen = int(payload["tokens_seen"])
            resumed_from_step = step
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

        rollback_jsonl(adapt_log, step)
        summary_path.unlink(missing_ok=True)
        timer = TrainingTimer(
            metrics_root / "adaptation_time.json",
            stage="adapt",
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

        sampler.set_position(data_epoch, batch_in_epoch)
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            **dataloader_kwargs(cfg),
        )
        iterator = iter(loader)
        total_steps = int(cfg["adapt"]["steps"])
        log_interval = int(cfg["adapt"]["log_interval"])
        save_interval = int(cfg["adapt"]["save_interval"])
        sequence_length = int(cfg["adapt"]["max_seq_len"])

        reset_peak_vram(device)
        log_fields(
            "adapt",
            cfg=cfg,
            state="resumed" if resume_path else "fresh",
            checkpoint=str(resume_path) if resume_path else "none",
            step=step,
            total_steps=total_steps,
        )
        while step < total_steps:
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            accumulated = {
                "loss": 0.0,
                "lm_loss": 0.0,
                "position_loss": 0.0,
            }
            for _ in range(accumulation_steps):
                if batch_in_epoch >= batches_per_epoch:
                    data_epoch += 1
                    batch_in_epoch = 0
                    sampler.set_position(data_epoch, batch_in_epoch)
                    iterator = iter(loader)
                batch = next(iterator)
                batch_in_epoch += 1
                non_blocking = bool(
                    cfg["resources"].get("resolved_pin_memory", False)
                )
                input_ids = batch["input_ids"].to(
                    device,
                    non_blocking=non_blocking,
                )
                labels = batch["label"].to(
                    device,
                    non_blocking=non_blocking,
                )
                targets = torch.full_like(input_ids, -100)
                targets[:, -1] = labels
                with autocast_context(device, dtype):
                    output = model(input_ids, targets)
                    scaled_loss = output["loss"] / accumulation_steps
                scaled_loss.backward()
                for key in accumulated:
                    accumulated[key] += float(output[key].detach().item())
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(cfg["train"]["grad_clip"]),
            )
            optimizer.step()
            step += 1
            step_tokens = effective_batch_size * sequence_length
            tokens_seen += step_tokens
            step_seconds = time.perf_counter() - step_start
            finished = step >= total_steps
            record = {
                "task": cfg["run"]["task"],
                "method": cfg["run"]["method"],
                "seed": seed,
                "stage": "adapt",
                "step": step,
                "tokens_seen": tokens_seen,
                "loss": accumulated["loss"] / accumulation_steps,
                "lm_loss": accumulated["lm_loss"] / accumulation_steps,
                "position_loss": accumulated["position_loss"]
                / accumulation_steps,
                "step_seconds": step_seconds,
                "tokens_per_second": step_tokens
                / max(step_seconds, 1e-9),
                "micro_batch_size": micro_batch_size,
                "effective_batch_size": effective_batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "data_epoch": data_epoch,
                "batch_in_epoch": batch_in_epoch,
                "timestamp": utc_timestamp(),
            }
            should_log = step % log_interval == 0 or finished
            should_save = step % save_interval == 0 and not finished
            if should_log:
                append_jsonl(adapt_log, record)
                log_fields(
                    "adapt",
                    cfg=cfg,
                    method=cfg["run"]["method"],
                    step=step,
                    loss=f"{record['loss']:.6f}",
                )
            if step % int(cfg["resources"]["monitor_interval_steps"]) == 0:
                log_resources(
                    cfg,
                    "adapt",
                    device=device,
                    step=step,
                    tokens=tokens_seen,
                )
            # Keep time accounting durable independently of the user-facing
            # log and checkpoint frequencies.
            timer.update(
                step=step,
                tokens_seen=tokens_seen,
                add_seconds=step_seconds,
                peak_vram_gb=peak_vram_gb(device),
                peak_host_ram_gb=process_rss_gb(peak=True),
            )
            if should_save:
                checkpoint_path = checkpoint_root / f"adapt_step{step}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
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
            cfg=cfg,
            step=step,
            tokens_seen=tokens_seen,
            extra={
                "adapt_steps": step,
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
            "effective_batch_size": effective_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "log_interval": log_interval,
            "save_interval": save_interval,
        }
        write_json(summary_path, summary)
        stage_banner("ADAPT", "DONE", cfg=cfg)
        return summary
    except BaseException:
        if timer is not None:
            timer.rollback()
        raise
