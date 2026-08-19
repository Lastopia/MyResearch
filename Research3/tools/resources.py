from __future__ import annotations

import gc
import os
from copy import deepcopy
from typing import Any

import torch

from model.factory import build_model
from tools.io import write_json
from tools.log import log_fields, log_resources, stage_banner, utc_timestamp
from tools.memory import resource_snapshot
from tools.paths import resource_plan_path, workspace_root
from tools.runtime import autocast_context, resolve_dtype


def _model_parameter_estimate(cfg: dict[str, Any]) -> int:
    layers = int(cfg["model"]["n_layer"])
    hidden = int(cfg["model"]["n_embd"])
    ffn = int(cfg["model"]["ffn_dim"])
    vocab = int(cfg["data"]["vocab_size"])
    per_layer = 4 * hidden * hidden + 2 * hidden * ffn
    return vocab * hidden + layers * per_layer


def _ram_limited_jobs(cfg: dict[str, Any], available_ram_gb: float | None) -> int:
    if available_ram_gb is None:
        return 1
    parameters = _model_parameter_estimate(cfg)
    # Parameters, gradients, Adam states and Python/DataLoader overhead.
    estimated_job_gb = max(2.0, parameters * 20 / (1024**3) + 1.0)
    usable = available_ram_gb * float(cfg["resources"]["ram_safety_fraction"])
    return max(1, int(usable // estimated_job_gb))


def _gpu_inventory(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    resources = cfg["resources"]
    require_idle = bool(resources["require_idle_gpus"])
    maximum_used_fraction = float(
        resources["max_preexisting_vram_fraction"]
    )
    minimum_free_bytes = int(
        float(resources["min_free_vram_gb"]) * (1024**3)
    )
    inventory: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        used_bytes = total_bytes - free_bytes
        used_fraction = used_bytes / max(1, total_bytes)
        rejection_reasons: list[str] = []
        if free_bytes < minimum_free_bytes:
            rejection_reasons.append("insufficient_free_vram")
        if require_idle and used_fraction > maximum_used_fraction:
            rejection_reasons.append("gpu_not_idle")
        inventory.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "free_vram_gb": free_bytes / (1024**3),
                "total_vram_gb": total_bytes / (1024**3),
                "preexisting_used_vram_gb": used_bytes / (1024**3),
                "preexisting_used_fraction": used_fraction,
                "eligible": not rejection_reasons,
                "rejection_reasons": rejection_reasons,
            }
        )
    return inventory


def _candidate_gpu_ids(
    cfg: dict[str, Any],
    inventory: list[dict[str, Any]],
) -> list[int]:
    del cfg
    eligible = [item for item in inventory if bool(item["eligible"])]
    eligible.sort(
        key=lambda item: (
            float(item["free_vram_gb"]),
            float(item["total_vram_gb"]),
            -int(item["index"]),
        ),
        reverse=True,
    )
    return [int(item["index"]) for item in eligible]


def _calibration_method(methods: list[str]) -> str:
    priority = (
        "ra_cable",
        "cable",
        "alibi",
        "rope",
    )
    return next((method for method in priority if method in methods), methods[0])


def _trial_micro_batch(
    cfg: dict[str, Any],
    method: str,
    batch_size: int,
    device: torch.device,
    safety_fraction: float,
) -> dict[str, Any]:
    trial = deepcopy(cfg)
    trial["run"]["method"] = method
    dtype = resolve_dtype(trial)
    model = None
    optimizer = None
    input_ids = None
    output = None
    torch.cuda.empty_cache()
    free_before, total_bytes = torch.cuda.mem_get_info(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    result: dict[str, Any] = {
        "gpu": int(device.index or 0),
        "method": method,
        "micro_batch_size": batch_size,
        "free_before_gb": free_before / (1024**3),
        "total_vram_gb": total_bytes / (1024**3),
        "safety_fraction": safety_fraction,
        "safe_budget_gb": free_before * safety_fraction / (1024**3),
        "fit": False,
        "oom": False,
    }
    try:
        model = build_model(trial).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        input_ids = torch.zeros(
            (batch_size, int(trial["data"]["block_size"])),
            dtype=torch.long,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, dtype):
            output = model(input_ids, input_ids)
        output["loss"].backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        incremental_reserved = max(0, peak_reserved - baseline_reserved)
        result.update(
            {
                "peak_reserved_gb": peak_reserved / (1024**3),
                "incremental_reserved_gb": incremental_reserved
                / (1024**3),
                "fit": incremental_reserved
                <= free_before * safety_fraction,
                "rejection_reason": (
                    None
                    if incremental_reserved <= free_before * safety_fraction
                    else "exceeds_vram_safety_budget"
                ),
            }
        )
        return result
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        result.update(
            {
                "oom": True,
                "rejection_reason": "cuda_out_of_memory",
                "error": str(error),
            }
        )
        return result
    finally:
        del output, input_ids, optimizer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def calibrate_micro_batch(
    cfg: dict[str, Any],
    methods: list[str],
    gpu_id: int,
) -> dict[str, Any]:
    resources = cfg["resources"]
    minimum = int(resources["min_micro_batch_size"])
    maximum = int(resources["max_micro_batch_size"])
    effective_samples = max(
        1,
        int(cfg["train"]["effective_batch_tokens"])
        // int(cfg["data"]["block_size"]),
    )
    maximum = min(maximum, effective_samples)
    device = torch.device(f"cuda:{gpu_id}")
    method = _calibration_method(methods)
    candidates = [
        value
        for value in range(minimum, maximum + 1)
        if effective_samples % value == 0
    ]
    if not candidates:
        raise ValueError(
            "No micro-batch candidate divides the effective batch"
        )
    trials: list[dict[str, Any]] = []
    best = 0
    lower = 0
    upper = len(candidates) - 1
    safety_fraction = float(resources["vram_safety_fraction"])
    while lower <= upper:
        middle = (lower + upper) // 2
        candidate = candidates[middle]
        log_fields(
            "resource-plan",
            cfg=cfg,
            action="calibrate_micro_batch",
            method=method,
            gpu=gpu_id,
            candidate=candidate,
        )
        trial = _trial_micro_batch(
            cfg,
            method,
            candidate,
            device,
            safety_fraction,
        )
        trials.append(trial)
        if bool(trial["fit"]):
            best = candidate
            lower = middle + 1
        else:
            upper = middle - 1
    if best < minimum:
        raise RuntimeError(
            f"Even micro_batch_size={minimum} does not fit on cuda:{gpu_id}"
        )
    return {
        "gpu": gpu_id,
        "method": method,
        "micro_batch_size": best,
        "candidate_micro_batches": candidates,
        "trials": trials,
        "safety_fraction": safety_fraction,
    }


def build_resource_plan(
    cfg: dict[str, Any],
    *,
    methods: list[str] | None = None,
    job_count: int = 1,
) -> dict[str, Any]:
    methods = methods or [str(cfg["run"]["method"])]
    stage_banner("RESOURCES", cfg=cfg)
    snapshot = resource_snapshot(workspace=workspace_root(cfg))
    gpu_inventory = _gpu_inventory(cfg) if torch.cuda.is_available() else []
    candidate_gpu_ids = (
        _candidate_gpu_ids(cfg, gpu_inventory)
        if torch.cuda.is_available()
        else []
    )
    requested_parallel = cfg["resources"]["parallel_jobs"]
    required_parallel = (
        1
        if str(requested_parallel).lower() == "auto"
        else min(max(1, int(requested_parallel)), max(1, job_count))
    )
    if (
        bool(cfg["run"].get("require_cuda", False))
        and len(candidate_gpu_ids) < required_parallel
    ):
        raise RuntimeError(
            f"This run requires {required_parallel} available CUDA GPU(s), "
            f"but the resource planner found {len(candidate_gpu_ids)} eligible "
            "GPU(s). Check CUDA_VISIBLE_DEVICES, current GPU memory use, "
            "resources.min_free_vram_gb, and the idle-GPU threshold."
        )
    gpu_limit = max(1, len(candidate_gpu_ids)) if candidate_gpu_ids else 1
    ram_limit = _ram_limited_jobs(
        cfg,
        (
            float(snapshot["ram_available_gb"])
            if snapshot["ram_available_gb"] is not None
            else None
        ),
    )
    if str(requested_parallel).lower() == "auto":
        parallel_jobs = min(max(1, job_count), gpu_limit, ram_limit)
    else:
        parallel_jobs = min(
            max(1, int(requested_parallel)),
            max(1, job_count),
            gpu_limit,
            ram_limit,
        )

    selected_gpu_ids = candidate_gpu_ids[:parallel_jobs]
    micro_batch = int(cfg["train"]["micro_batch_size"])
    gpu_calibrations: dict[str, Any] = {}
    if (
        selected_gpu_ids
        and bool(cfg["resources"]["calibrate_micro_batch"])
        and bool(cfg["resources"]["auto_plan"])
    ):
        calibrated_gpu_ids: list[int] = []
        calibration_failures: dict[str, str] = {}
        for gpu_id in candidate_gpu_ids:
            try:
                calibration = calibrate_micro_batch(
                    cfg,
                    methods,
                    gpu_id,
                )
            except RuntimeError as error:
                calibration_failures[str(gpu_id)] = str(error)
                log_fields(
                    "resource-plan",
                    cfg=cfg,
                    action="reject_gpu_after_calibration",
                    gpu=gpu_id,
                    reason=type(error).__name__,
                )
                continue
            gpu_calibrations[str(gpu_id)] = calibration
            calibrated_gpu_ids.append(gpu_id)
            if len(calibrated_gpu_ids) >= parallel_jobs:
                break
        if len(calibrated_gpu_ids) < required_parallel:
            raise RuntimeError(
                "GPU calibration left only "
                f"{len(calibrated_gpu_ids)} usable GPU(s), but "
                f"{required_parallel} are required. Failures: "
                f"{calibration_failures}"
            )
        if not calibrated_gpu_ids:
            raise RuntimeError(
                f"No GPU passed micro-batch calibration: {calibration_failures}"
            )
        parallel_jobs = min(parallel_jobs, len(calibrated_gpu_ids))
        selected_gpu_ids = calibrated_gpu_ids[:parallel_jobs]
        micro_batch = min(
            int(gpu_calibrations[str(gpu_id)]["micro_batch_size"])
            for gpu_id in selected_gpu_ids
        )
    gpu_ids = selected_gpu_ids
    cpu_count = os.cpu_count() or 1
    worker_limit = max(0, cpu_count // max(1, parallel_jobs) - 1)
    data_workers = min(
        int(cfg["resources"]["max_data_workers"]),
        worker_limit,
    )
    if snapshot["ram_available_gb"] is not None:
        data_workers = min(
            data_workers,
            max(0, int(float(snapshot["ram_available_gb"]) // 2)),
        )
    effective = int(cfg["train"]["effective_batch_tokens"])
    block = int(cfg["data"]["block_size"])
    micro_tokens = micro_batch * block
    if effective % micro_tokens != 0:
        divisors = [
            value
            for value in range(micro_batch, 0, -1)
            if effective % (value * block) == 0
        ]
        if not divisors:
            raise ValueError(
                "effective_batch_tokens must be divisible by block_size"
            )
        micro_batch = divisors[0]
        micro_tokens = micro_batch * block
    accumulation = effective // micro_tokens
    adapt_effective_batch = int(cfg["adapt"]["batch_size"])
    adapt_candidates = [
        value
        for value in range(min(adapt_effective_batch, micro_batch), 0, -1)
        if adapt_effective_batch % value == 0
    ]
    adapt_micro_batch = adapt_candidates[0]
    adapt_accumulation = adapt_effective_batch // adapt_micro_batch

    plan = {
        "created_at": utc_timestamp(),
        "task": cfg["run"]["task"],
        "methods": methods,
        "job_count": job_count,
        "device_type": "cuda" if gpu_ids else "cpu",
        "gpu_ids": gpu_ids,
        "parallel_jobs": parallel_jobs,
        "data_workers": data_workers,
        "pin_memory": bool(cfg["resources"]["pin_memory"]) and bool(gpu_ids),
        "persistent_workers": (
            bool(cfg["resources"]["persistent_workers"]) and data_workers > 0
        ),
        "prefetch_factor": int(cfg["resources"]["prefetch_factor"]),
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_tokens": effective,
        "tokens_per_optimizer_step": micro_tokens * accumulation,
        "adapt_micro_batch_size": adapt_micro_batch,
        "adapt_gradient_accumulation_steps": adapt_accumulation,
        "hardware_at_planning": snapshot,
        "gpu_inventory": gpu_inventory,
        "gpu_calibrations": gpu_calibrations,
        "vram_safety_fraction": float(
            cfg["resources"]["vram_safety_fraction"]
        ),
        "common_micro_batch_policy": (
            "minimum calibrated safe micro-batch across all selected GPUs"
        ),
        "fairness_constraints": {
            "shared_across_methods": True,
            "model_structure_unchanged": True,
            "token_budget_unchanged": True,
            "effective_batch_tokens_unchanged": True,
        },
    }
    write_json(resource_plan_path(cfg), plan)
    log_resources(
        cfg,
        "resources",
        parallel_jobs=parallel_jobs,
        data_workers=data_workers,
        micro_batch=micro_batch,
        grad_accumulation=accumulation,
    )
    stage_banner("RESOURCES", "DONE", cfg=cfg)
    return plan


def apply_resource_plan(
    cfg: dict[str, Any],
    plan: dict[str, Any],
    *,
    gpu_id: int | None = None,
) -> dict[str, Any]:
    cfg["train"]["micro_batch_size"] = int(plan["micro_batch_size"])
    cfg["adapt"]["micro_batch_size"] = int(plan["adapt_micro_batch_size"])
    cfg["resources"]["data_workers"] = int(plan["data_workers"])
    cfg["resources"]["resolved_pin_memory"] = bool(plan["pin_memory"])
    cfg["resources"]["resolved_persistent_workers"] = bool(
        plan["persistent_workers"]
    )
    cfg["resources"]["resolved_prefetch_factor"] = int(plan["prefetch_factor"])
    cfg["resources"]["resolved_parallel_jobs"] = int(plan["parallel_jobs"])
    if plan["device_type"] == "cuda":
        selected = int(plan["gpu_ids"][0] if gpu_id is None else gpu_id)
        cfg["run"]["device"] = f"cuda:{selected}"
    elif str(cfg["run"]["device"]) == "auto":
        cfg["run"]["device"] = "cpu"
    return cfg
