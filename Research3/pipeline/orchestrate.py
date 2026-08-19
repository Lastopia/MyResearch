from __future__ import annotations

import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from copy import deepcopy
from typing import Any

from tools.io import write_json
from tools.log import log_fields, log_resources
from tools.paths import config_path
from tools.resources import apply_resource_plan
from tools.reproducibility import write_reproducibility_manifest


MODEL_STAGES = (
    "train",
    "adapt",
    "evaluate",
    "attention_audit",
)


def _runner(stage: str):
    if stage == "train":
        from pipeline.train import run
    elif stage == "adapt":
        from pipeline.adapt import run
    elif stage == "evaluate":
        from pipeline.evaluate import run
    elif stage == "attention_audit":
        from pipeline.attention_audit import run
    elif stage == "profile":
        from pipeline.profile import run
    else:
        raise ValueError(f"Unsupported per-job stage: {stage}")
    return run


def run_experiment_job(
    cfg: dict[str, Any],
    stages: tuple[str, ...],
) -> dict[str, Any]:
    import os

    from tools.console import install_console_log

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    install_console_log(cfg)
    write_json(config_path(cfg), cfg)
    write_reproducibility_manifest(cfg)
    log_resources(cfg, "job", device=cfg["run"]["device"], state="start")
    result: dict[str, Any] = {}
    try:
        for stage in stages:
            result[stage] = _runner(stage)(cfg)
    except Exception as error:
        log_resources(
            cfg,
            "job",
            device=cfg["run"]["device"],
            state="failed",
            error_type=type(error).__name__,
        )
        raise
    log_resources(cfg, "job", device=cfg["run"]["device"], state="done")
    return {
        "method": cfg["run"]["method"],
        "seed": cfg["run"]["seed"],
        "stages": list(result),
    }


def run_jobs(
    configs: list[dict[str, Any]],
    stages: tuple[str, ...],
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    if not configs:
        return []
    parallel = min(int(plan["parallel_jobs"]), len(configs))
    gpu_ids = [int(value) for value in plan["gpu_ids"]]
    if parallel <= 1:
        gpu_id = gpu_ids[0] if gpu_ids else None
        return [
            run_experiment_job(
                apply_resource_plan(deepcopy(cfg), plan, gpu_id=gpu_id),
                stages,
            )
            for cfg in configs
        ]

    assigned_gpus = gpu_ids[:parallel]
    context = multiprocessing.get_context("spawn")
    executors = {
        gpu_id: ProcessPoolExecutor(max_workers=1, mp_context=context)
        for gpu_id in assigned_gpus
    }
    futures: dict[Future[dict[str, Any]], tuple[str, int, int]] = {}
    try:
        for index, cfg in enumerate(configs):
            gpu_id = assigned_gpus[index % len(assigned_gpus)]
            prepared = apply_resource_plan(deepcopy(cfg), plan, gpu_id=gpu_id)
            future = executors[gpu_id].submit(
                run_experiment_job,
                prepared,
                stages,
            )
            futures[future] = (
                str(cfg["run"]["method"]),
                int(cfg["run"]["seed"]),
                gpu_id,
            )
        results: list[dict[str, Any]] = []
        for future in as_completed(futures):
            method, seed, gpu_id = futures[future]
            result = future.result()
            results.append(result)
            log_fields(
                "scheduler",
                cfg=configs[0],
                method=method,
                seed=seed,
                gpu=gpu_id,
                state="done",
            )
        return results
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=False)
