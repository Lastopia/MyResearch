"""Language-model comparison: tiny byte smoke path and formal GPT-2 path."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .data_download import download_file, sha256_file
from .external_model import ByteTokenizer, CausalLanguageModel
from .locking import RunLock
from .external_tasks import _model_config
from .formal_data import (
    FormalLanguageBundle,
    GPT2Tokenizer,
    TokenCorpus,
    ensure_formal_language_dataset,
)
from .reporting import build_report
from .research_model import (
    BUS_METHODS,
    estimated_model_macs,
    initialize_named_parameters,
    parameter_count,
)
from .resources import (
    ResourcePolicy,
    WorkloadSpec,
    detect_resources,
    plan_jobs,
    write_resource_outputs,
)
from .runner import (
    _autocast,
    _optimizer,
    _restore_rng,
    _rng_state,
    _scheduler,
    _source_fingerprint,
    select_device,
)
from .scheduler import ScheduledCommand, execute_schedule
from .training_log import (
    CumulativeTrainingTimer,
    FixedWidthTrainingLogger,
    checkpoint_due,
    current_device_log_status,
)
from .storage import prepare_run_checkpoint_dir, prepare_storage, storage_roots


@dataclass(frozen=True)
class TinyStoriesBundle:
    root: Path
    manifest_path: Path
    train_path: Path
    validation_path: Path


def formal_language_enabled(cfg: dict[str, Any], size_name: str = "large") -> bool:
    globally_enabled = bool(
        cfg.get("external", {}).get("formal_language", {}).get("enabled", False)
    )
    backend = str(cfg["sizes"].get(size_name, {}).get("language_backend", "formal"))
    return globally_enabled and backend == "formal"


def active_language_settings(
    cfg: dict[str, Any], size_name: str = "large"
) -> dict[str, Any]:
    key = "formal_language" if formal_language_enabled(cfg, size_name) else "tinystories"
    return cfg["external"][key]


def active_language_stage(cfg: dict[str, Any], size_name: str = "large") -> str:
    return (
        "formal_language_model"
        if formal_language_enabled(cfg, size_name)
        else "pilot_tinystories_lm" if size_name == "medium" else "tinystories_lm"
    )


def ensure_active_language_dataset(
    cfg: dict[str, Any], data_root: Path, size_name: str = "large"
) -> TinyStoriesBundle | FormalLanguageBundle:
    if formal_language_enabled(cfg, size_name):
        return ensure_formal_language_dataset(cfg, data_root)
    return ensure_tinystories_dataset(cfg, data_root)


def active_tokenizer(
    cfg: dict[str, Any], size_name: str = "large"
) -> ByteTokenizer | GPT2Tokenizer:
    if formal_language_enabled(cfg, size_name):
        cache = Path(cfg["paths"]["data_root"]) / "formal_language" / "tokenizer_cache"
        return GPT2Tokenizer(cache)
    return ByteTokenizer()


def active_corpus(
    cfg: dict[str, Any], path: Path, size_name: str = "large"
) -> Tensor | TokenCorpus:
    return TokenCorpus(path) if formal_language_enabled(cfg, size_name) else _corpus(path)


def language_model_config(
    cfg: dict[str, Any], method: str, sequence_length: int, size_name: str = "large"
) -> Any:
    if not formal_language_enabled(cfg, size_name):
        return _model_config(cfg, method, sequence_length, size_name)
    from .research_model import ResearchModelConfig

    values = cfg["external"]["formal_language"]["model"]
    defaults = cfg["model_defaults"]
    return ResearchModelConfig(
        vocab_size=GPT2Tokenizer.vocab_size,
        max_length=sequence_length,
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
        norm_eps=float(defaults["norm_eps"]),
        rope_theta=float(defaults["rope_theta"]),
        bias=bool(defaults["bias"]),
        keep_residual_attention=bool(defaults["keep_residual_attention"]),
    )


def ensure_tinystories_dataset(
    cfg: dict[str, Any], data_root: Path
) -> TinyStoriesBundle:
    settings = cfg["external"]["tinystories"]
    root = data_root / "tinystories"
    train_path = root / "train_prefix.txt"
    validation_path = root / "validation_prefix.txt"
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and train_path.exists() and validation_path.exists():
        return TinyStoriesBundle(root, manifest_path, train_path, validation_path)
    if not bool(cfg["external"].get("allow_download", True)):
        raise FileNotFoundError(
            f"TinyStories is not cached at {root} and external.allow_download is false"
        )
    timeout = float(cfg["external"]["download_timeout_seconds"])
    download_file(
        str(settings["train_url"]),
        train_path,
        timeout=timeout,
        max_bytes=int(settings["train_bytes"]),
    )
    download_file(
        str(settings["validation_url"]),
        validation_path,
        timeout=timeout,
        max_bytes=int(settings["validation_bytes"]),
    )
    for path in (train_path, validation_path):
        prefix = path.read_bytes()[:200]
        if b"version https://git-lfs.github.com/spec" in prefix:
            raise RuntimeError(f"received a Git LFS pointer instead of data: {path}")
    manifest = {
        "benchmark": "TinyStoriesV2-GPT4",
        "source": {
            "train": settings["train_url"],
            "validation": settings["validation_url"],
        },
        "license": "CDLA-Sharing-1.0",
        "tokenizer": "fixed UTF-8 byte tokenizer",
        "train_bytes": train_path.stat().st_size,
        "validation_bytes": validation_path.stat().st_size,
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return TinyStoriesBundle(root, manifest_path, train_path, validation_path)


def _corpus(path: Path) -> Tensor:
    payload = path.read_bytes()
    if len(payload) < 1024:
        raise RuntimeError(f"language-model corpus is unexpectedly small: {path}")
    return ByteTokenizer().encode_bytes(payload)


def _sample_batch(
    corpus: Tensor | TokenCorpus,
    *,
    batch_size: int,
    sequence_length: int,
    generator: torch.Generator,
    include_mask: bool = True,
) -> tuple[Tensor, Tensor | None, Tensor]:
    upper = len(corpus) - sequence_length - 1
    if upper <= 0:
        raise ValueError("corpus is shorter than sequence_length")
    starts = torch.randint(upper, (batch_size,), generator=generator)
    offsets = torch.arange(sequence_length + 1)
    sequences = corpus[starts[:, None] + offsets[None, :]]
    inputs = sequences[:, :-1]
    targets = sequences[:, 1:]
    mask = torch.ones_like(inputs) if include_mask else None
    return inputs, mask, targets


def _evaluation_batches(
    settings: dict[str, Any], batch_size: int, *, external: bool = False
) -> int:
    prefix = "external_" if external else ""
    sequence_key = prefix + "validation_sequences"
    if sequence_key in settings:
        count = int(settings[sequence_key])
        if count % batch_size:
            raise ValueError(
                f"{sequence_key}={count} must be divisible by eval batch {batch_size}"
            )
        return count // batch_size
    return int(settings[prefix + "validation_batches"])


@torch.no_grad()
def _evaluate_lm(
    model: CausalLanguageModel,
    corpus: Tensor | TokenCorpus,
    *,
    sequence_length: int,
    batch_size: int,
    batches: int,
    device: torch.device,
    intervention: str | None = None,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator().manual_seed(20260806)
    losses = []
    token_count = 0
    for _ in range(batches):
        inputs, _, targets = _sample_batch(
            corpus,
            batch_size=batch_size,
            sequence_length=sequence_length,
            generator=generator,
            include_mask=False,
        )
        inputs, targets = inputs.to(device), targets.to(device)
        with _autocast(device):
            output = model(inputs, None, intervention=intervention)
            loss = F.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]),
                targets.reshape(-1),
            )
        losses.append(float(loss))
        token_count += targets.numel()
    mean = sum(losses) / len(losses)
    model.train()
    return {
        "loss": mean,
        "perplexity": math.exp(min(20.0, mean)),
        "evaluated_tokens": float(token_count),
    }


def run_language_model_one(
    cfg: dict[str, Any], *, method: str, seed: int, output_root: Path,
    size_name: str = "large", stage: str | None = None,
) -> Path:
    roots = prepare_storage(cfg, output_root)
    bundle = ensure_active_language_dataset(cfg, roots.data, size_name)
    settings = active_language_settings(cfg, size_name)
    stage = stage or active_language_stage(cfg, size_name)
    sequence_length = int(settings["sequence_length"])
    global_batch = int(settings["batch_size"])
    micro_batch = int(os.environ.get("CONCEPT_BUS_MICRO_BATCH", global_batch))
    accumulation = int(
        os.environ.get("CONCEPT_BUS_GRAD_ACCUM", max(1, global_batch // micro_batch))
    )
    if micro_batch * accumulation != global_batch:
        raise RuntimeError("LM micro batch and accumulation do not match global batch")
    max_steps = math.ceil(
        int(settings["train_tokens"]) / (global_batch * sequence_length)
    )
    payload = {
        "task": stage,
        "size": size_name,
        "method": method,
        "seed": seed,
        "settings": settings,
        "model": (
            settings["model"]
            if formal_language_enabled(cfg, size_name)
            else cfg["sizes"][size_name]["model"]
        ),
        "manifest": sha256_file(bundle.manifest_path),
        "source": _source_fingerprint(),
        "runtime_batching": {
            "global_batch_size": global_batch,
            "micro_batch_size": micro_batch,
            "gradient_accumulation_steps": accumulation,
        },
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = output_root / "runs" / stage / method / f"seed{seed}" / digest / "attempt1"
    checkpoint_dir = prepare_run_checkpoint_dir(cfg, output_root, run_dir)
    final_path = run_dir / "metrics" / "final.json"
    if final_path.exists() and cfg["run"]["skip_completed"]:
        print(f"skip completed | {stage}/{method}/seed{seed}")
        return run_dir
    for name in ("metrics", "logs"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    run_lock = RunLock(run_dir, final_path)
    if not run_lock.acquire():
        return run_dir
    train_corpus = active_corpus(cfg, bundle.train_path, size_name)
    validation_corpus = active_corpus(cfg, bundle.validation_path, size_name)
    external_corpus = (
        active_corpus(cfg, bundle.external_test_path, size_name)
        if isinstance(bundle, FormalLanguageBundle)
        else None
    )
    random.seed(seed)
    torch.manual_seed(seed)
    device, diagnostics = select_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    config = language_model_config(cfg, method, sequence_length, size_name)
    model = CausalLanguageModel(config)
    initialize_named_parameters(model, seed)
    model.to(device)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                **payload,
                "model_resolved": model.config.to_dict(),
                "max_steps": max_steps,
                "global_batch_size": global_batch,
                "micro_batch_size": micro_batch,
                "gradient_accumulation_steps": accumulation,
                "data_manifest": str(bundle.manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "device": str(device),
                "device_diagnostics": diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if method == "parameter_matched":
        target_values = model.config.to_dict()
        target_values.update({"method": "concept_bus_v2", "matched_ffn_width": None})
        with torch.device("meta"):
            target = CausalLanguageModel(type(model.config)(**target_values))
        target_parameters = parameter_count(target)
        actual_parameters = parameter_count(model)
        target_macs = estimated_model_macs(target.config, sequence_length)
        actual_macs = estimated_model_macs(model.config, sequence_length)
        matching = {
            "target_parameters": target_parameters,
            "actual_parameters": actual_parameters,
            "parameter_difference_fraction": abs(actual_parameters - target_parameters)
            / target_parameters,
            "target_macs": target_macs,
            "actual_macs": actual_macs,
            "mac_difference_fraction": abs(actual_macs - target_macs) / target_macs,
        }
        if method == "parameter_matched" and matching["parameter_difference_fraction"] > 0.01:
            raise RuntimeError("language-model parameter match exceeds 1% tolerance")
        (run_dir / "matching_report.json").write_text(
            json.dumps(matching, indent=2), encoding="utf-8"
        )
        del target
    train_config = {
        "train": {
            "max_steps": max_steps,
            "learning_rate": float(settings.get("learning_rate", cfg["sizes"][size_name]["train"]["learning_rate"])),
            "weight_decay": float(settings.get("weight_decay", 0.1)),
            "warmup_fraction": float(settings.get("warmup_fraction", 0.02)),
        },
        "optimizer": cfg["optimizer"],
    }
    optimizer = _optimizer(model, train_config)
    scheduler = _scheduler(optimizer, train_config)
    generator = torch.Generator().manual_seed(30_000 + seed)
    timer = CumulativeTrainingTimer()
    start = 0
    latest = checkpoint_dir / "latest.pt"
    if latest.exists() and cfg["run"]["resume"]:
        state = torch.load(latest, map_location=device, weights_only=False)
        if state.get("config_hash") != digest:
            raise RuntimeError("language-model checkpoint config hash mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["generator"])
        timer.load_state_dict(state["timer"])
        if "rng" in state:
            _restore_rng(state["rng"])
        start = int(state["step"])
    logger_cfg = cfg["logging"]
    log_interval = int(settings["log_interval_steps"])
    eval_interval = int(settings["eval_interval_steps"])
    checkpoint_interval_minutes = float(settings["checkpoint_interval_minutes"])
    last_checkpoint_seconds = timer.elapsed_seconds

    def save(step: int) -> None:
        temporary = latest.with_suffix(".tmp")
        torch.save(
            {
                "step": step,
                "config_hash": digest,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "generator": generator.get_state(),
                "timer": timer.state_dict(),
                "rng": _rng_state(),
            },
            temporary,
        )
        temporary.replace(latest)

    with FixedWidthTrainingLogger(
        run_dir / "logs" / str(logger_cfg["file_name"]),
        timer=timer,
        device_provider=lambda: current_device_log_status(device),
        widths=logger_cfg["column_widths"],
        console_widths=logger_cfg["console_column_widths"],
        console_mode=str(logger_cfg["console_mode"]),
        flush_each_line=bool(logger_cfg["flush_each_line"]),
    ) as logger:
        task = f"{stage}/{method}"
        logger.log("system", step=start, total_steps=max_steps, seed=seed, task=task)
        model.train()
        last_validation_metrics: dict[str, float] | None = None
        for step in range(start + 1, max_steps + 1):
            timer.start()
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for _ in range(accumulation):
                inputs, _, targets = _sample_batch(
                    train_corpus,
                    batch_size=micro_batch,
                    sequence_length=sequence_length,
                    generator=generator,
                    include_mask=False,
                )
                inputs, targets = inputs.to(device), targets.to(device)
                with _autocast(device):
                    output = model(inputs, None)
                    loss = F.cross_entropy(
                        output.logits.reshape(-1, output.logits.shape[-1]),
                        targets.reshape(-1),
                    )
                (loss / accumulation).backward()
                loss_value += float(loss.detach()) / accumulation
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["optimizer"]["gradient_clip"]))
            optimizer.step()
            scheduler.step()
            timer.pause()
            if step == 1 or step % log_interval == 0 or step == max_steps:
                logger.log("train", step=step, total_steps=max_steps, loss=loss_value, seed=seed, task=task)
            if step % eval_interval == 0 or step == max_steps:
                metrics = _evaluate_lm(
                    model,
                    validation_corpus,
                    sequence_length=sequence_length,
                    batch_size=micro_batch,
                    batches=_evaluation_batches(settings, micro_batch),
                    device=device,
                )
                last_validation_metrics = metrics
                logger.log("valid", step=step, total_steps=max_steps, loss=metrics["loss"], seed=seed, task=task)
            if checkpoint_due(
                timer,
                last_checkpoint_seconds,
                checkpoint_interval_minutes,
                final_step=step == max_steps,
            ):
                save(step)
                last_checkpoint_seconds = timer.elapsed_seconds
                logger.log("checkpt", step=step, total_steps=max_steps, seed=seed, task=task)
        metrics = last_validation_metrics
        if metrics is None:
            # Handles a process resumed from an already-finished checkpoint.
            metrics = _evaluate_lm(
                model,
                validation_corpus,
                sequence_length=sequence_length,
                batch_size=micro_batch,
                batches=_evaluation_batches(settings, micro_batch),
                device=device,
            )
        final_metrics = {f"validation_{key}": value for key, value in metrics.items()}
        if external_corpus is not None:
            external_metrics = _evaluate_lm(
                model,
                external_corpus,
                sequence_length=sequence_length,
                batch_size=micro_batch,
                batches=_evaluation_batches(settings, micro_batch, external=True),
                device=device,
            )
            final_metrics.update(
                {f"wikitext103_{key}": value for key, value in external_metrics.items()}
            )
        if method in BUS_METHODS:
            zero = _evaluate_lm(
                model,
                validation_corpus,
                sequence_length=sequence_length,
                batch_size=micro_batch,
                batches=_evaluation_batches(settings, micro_batch),
                device=device,
                intervention="zero_bus",
            )
            final_metrics["zero_bus_loss_increase"] = zero["loss"] - metrics["loss"]
        final_metrics.update(
            {
                "parameters": parameter_count(model),
                "estimated_macs_per_token": estimated_model_macs(model.config, sequence_length) / sequence_length,
                "training_seconds": timer.elapsed_seconds,
                "training_tokens": float(max_steps * global_batch * sequence_length),
                "training_tokens_per_second": max_steps * global_batch * sequence_length / max(1e-9, timer.elapsed_seconds),
                "peak_cuda_memory_bytes": float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0,
            }
        )
        final_path.write_text(
            json.dumps(
                {
                    "stage": stage,
                    "method": method,
                    "seed": seed,
                    "config_hash": digest,
                    "source_fingerprint": payload["source"],
                    "metrics": final_metrics,
                    "data_manifest": str(bundle.manifest_path),
                    "device_diagnostics": diagnostics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        # The loop always saves the final step; avoid rewriting the same
        # optimizer/model checkpoint after evaluation.
        logger.log("final", step=max_steps, total_steps=max_steps, loss=metrics["loss"], seed=seed, task=task)
    run_lock.release()
    return run_dir


def run_language_model_phase(
    cfg: dict[str, Any],
    *,
    size_name: str,
    phase: dict[str, Any],
    output_root: Path,
    project_root: Path,
    monitor_interval: float,
) -> dict[str, Any]:
    try:
        bundle = ensure_active_language_dataset(
            cfg, storage_roots(cfg, output_root).data, size_name
        )
    except Exception as error:
        return {"status": "skipped", "reason": f"language data unavailable: {type(error).__name__}: {error}"}
    methods = list(phase["methods"])
    seeds = [
        int(value)
        for value in phase.get("seeds", cfg["sizes"][size_name]["seeds"])
    ]
    resources = cfg["resources"]
    phase_resources = {**resources, **phase.get("resources", {})}
    settings = active_language_settings(cfg, size_name)
    stage = str(phase.get("stage", active_language_stage(cfg, size_name)))
    jobs = []
    commands = []
    for method in methods:
        for seed in seeds:
            name = f"{stage}_{method}_seed{seed}"
            jobs.append(
                WorkloadSpec(
                    name=name,
                    global_batch_size=int(settings["batch_size"]),
                    max_micro_batch_size=max(
                        1,
                        int(
                            int(settings["batch_size"])
                            * float(phase.get("_micro_batch_scale", 1.0))
                        ),
                    ),
                    estimated_model_memory_gb=float(phase_resources["estimated_model_memory_gb"]),
                    estimated_activation_memory_per_sample_gb=float(phase_resources["estimated_activation_memory_per_sample_gb"]),
                )
            )
            commands.append(
                ScheduledCommand(
                    name=name,
                    command=(sys.executable, str(project_root / "main.py"), "external-one", "--task", "language", "--size", size_name, "--stage", stage, "--method", method, "--seed", str(seed), "--output-root", str(output_root)),
                    cwd=str(project_root),
                )
            )
    snapshot = detect_resources(required_dtype=str(resources["required_dtype"]))
    plans = plan_jobs(jobs, snapshot, ResourcePolicy.from_dict(phase_resources))
    write_resource_outputs(output_root, snapshot, plans)
    schedule = execute_schedule(plans, commands, output_root, monitor=True, monitor_interval_seconds=monitor_interval)
    statuses = json.loads((schedule / "status.json").read_text(encoding="utf-8"))
    failed = sum(item["status"] != "completed" for item in statuses)
    report = build_report(
        output_root,
        stage=stage,
        max_static_figures=int(cfg["report"]["max_static_figures"]),
        source_fingerprint=_source_fingerprint(),
    )
    return {
        "status": "completed" if not failed else "completed_with_failures",
        "runs": len(jobs),
        "failed_runs": failed,
        "data_manifest": str(bundle.manifest_path),
        "scheduler": str(schedule),
        "report": str(report),
    }
