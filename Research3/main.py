from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from cfg import get_config, set_by_path, validate_config
from tools.io import write_json
from tools.paths import config_path, output_dir


STAGES = (
    "bootstrap",
    "resources",
    "data",
    "train",
    "adapt",
    "evaluate",
    "attention_audit",
    "profile",
    "stats",
    "report",
    "extensions",
    "pilot",
)


def _parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _arguments(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    aliases = {
        "method": "run.method",
        "seed": "run.seed",
        "task": "run.task",
        "force": "run.force",
        "device": "run.device",
        "dtype": "run.dtype",
        "bootstrap": "run.bootstrap",
    }
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must use key=value: {item}")
        key, raw = item.split("=", 1)
        overrides[aliases.get(key, key)] = _parse_value(raw)
    return overrides


def _runner(stage: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if stage == "bootstrap":
        from pipeline.bootstrap import run
    elif stage == "data":
        from pipeline.data import run
    elif stage == "train":
        from pipeline.train import run
    elif stage == "adapt":
        from pipeline.adapt import run
    elif stage == "evaluate":
        from pipeline.evaluate import run
    elif stage == "attention_audit":
        from pipeline.attention_audit import run
    elif stage == "profile":
        from pipeline.profile import run
    elif stage == "stats":
        from pipeline.stats import run
    elif stage == "report":
        from pipeline.report import run
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return run


def _build_config(
    overrides: dict[str, Any],
    *,
    method: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    cfg = get_config()
    for key, value in overrides.items():
        set_by_path(cfg, key, value)
    selected_method = method or str(cfg["run"]["method"])
    cfg["run"]["method"] = selected_method
    if seed is not None:
        cfg["run"]["seed"] = int(seed)
    project_root = cfg.get("paths", {}).get("project_root")
    if project_root:
        resolved_root = Path(str(project_root)).expanduser().resolve()
        cfg["paths"]["project_root"] = str(resolved_root)
        local_tokens = cfg["data"].get("local_tokens_path")
        if local_tokens and not Path(str(local_tokens)).expanduser().is_absolute():
            cfg["data"]["local_tokens_path"] = str(
                (resolved_root / str(local_tokens)).resolve()
            )
    validate_config(cfg)
    return cfg


def _usage() -> str:
    return (
        "Usage: python main.py run <stage|all> "
        "[method=rope|alibi|cable|ra_cable|ra_cable_lite|"
        "ra_cable_static|dape_kerple] "
        "[paths.project_root=/srv/project] [bootstrap=off|verify|install]"
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2 or args[0] != "run":
        print(_usage())
        return 2
    stage = args[1]
    if stage not in STAGES and stage != "all":
        raise ValueError(f"Unknown stage: {stage}. Available: {STAGES} and all")

    overrides = _arguments(args[2:])
    base_cfg = _build_config(overrides)
    from tools.console import install_console_log

    console_path = install_console_log(base_cfg)
    print(f"[CONSOLE LOG] path={console_path}", flush=True)
    methods = list(base_cfg["run"]["methods"])
    extension_methods = list(base_cfg["run"]["extension_methods"])
    seeds = [int(seed) for seed in base_cfg["run"]["seeds"]]

    bootstrap_mode = str(base_cfg["run"]["bootstrap"]).lower()
    if bootstrap_mode == "install":
        base_cfg["bootstrap"]["install_missing_dependencies"] = True
        base_cfg["bootstrap"]["install_cuda_torch_if_gpu_detected"] = True
    write_json(config_path(base_cfg), base_cfg)
    if bootstrap_mode != "off":
        from pipeline.bootstrap import run as bootstrap

        bootstrap(base_cfg)
    if stage == "bootstrap":
        return 0

    from model.factory import METHODS

    unknown_methods = sorted(
        (set(methods) | set(extension_methods)) - METHODS
    )
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")

    if stage in {"all", "extensions", "pilot"}:
        from pipeline.orchestrate import MODEL_STAGES, run_jobs
        from tools.resources import build_resource_plan

        _runner("data")(base_cfg)
        selected_methods = (
            extension_methods if stage == "extensions" else methods
        )
        selected_seeds = (
            [int(base_cfg["run"]["seed"])]
            if stage == "pilot"
            else seeds
        )
        job_configs = []
        for seed_index, seed in enumerate(selected_seeds):
            offset = seed_index % len(selected_methods)
            rotated_methods = (
                selected_methods[offset:] + selected_methods[:offset]
            )
            job_configs.extend(
                _build_config(overrides, method=method, seed=seed)
                for method in rotated_methods
            )
        plan = build_resource_plan(
            base_cfg,
            methods=selected_methods,
            job_count=len(job_configs),
        )
        write_json(
            output_dir(base_cfg) / "resources" / "training_plan.json",
            plan,
        )
        # Complete each stage for every method/seed before advancing. This
        # prevents long evaluations for reused checkpoints from occupying all
        # GPUs while unfinished training jobs wait in the executor queues.
        for model_stage in MODEL_STAGES:
            run_jobs(job_configs, (model_stage,), plan)
        fair_profile_cfg = deepcopy(base_cfg)
        fair_profile_cfg["resources"]["parallel_jobs"] = 1
        fair_profile_plan = build_resource_plan(
            fair_profile_cfg,
            methods=selected_methods,
            job_count=len(job_configs),
        )
        write_json(
            output_dir(base_cfg) / "resources" / "profile_plan.json",
            fair_profile_plan,
        )
        run_jobs(job_configs, ("profile",), fair_profile_plan)
        _runner("stats")(base_cfg)
        _runner("report")(base_cfg)
        return 0

    if stage == "resources":
        from tools.resources import build_resource_plan

        build_resource_plan(
            base_cfg,
            methods=methods,
            job_count=len(methods),
        )
        return 0

    if stage in {"train", "adapt", "evaluate", "attention_audit", "profile"}:
        from pipeline.orchestrate import run_jobs
        from tools.resources import build_resource_plan

        # A single-stage command runs run.method only. Edit run.method in cfg.py
        # or pass method=... to select it.
        selected = str(base_cfg["run"]["method"])
        cfg = _build_config(overrides, method=selected)
        plan = build_resource_plan(cfg, methods=[selected], job_count=1)
        run_jobs([cfg], (stage,), plan)
        return 0

    _runner(stage)(deepcopy(base_cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
