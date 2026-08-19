"""Unattended, resumable paper-suite orchestration."""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .reporting import build_report, discover_final_records
from .resources import (
    ResourcePolicy,
    WorkloadSpec,
    detect_resources,
    plan_jobs,
    write_resource_outputs,
)
from .runner import _source_fingerprint, resolve_run_config
from .research_model import ALL_METHODS
from .scheduler import ScheduledCommand, execute_schedule
from .synthetic_data import ensure_synthetic_dataset
from .storage import (
    consume_legacy_compatibility,
    legacy_compatibility_pending,
    prepare_storage,
    storage_roots,
)
from .verdict import build_paper_verdict


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": _now(), **payload}, ensure_ascii=False) + "\n")


def _effective_phase(
    base: dict[str, Any],
    phase: dict[str, Any],
    method_override: list[str] | None,
    seed_override: list[int] | None,
) -> dict[str, Any] | None:
    effective = dict(phase)
    methods = list(phase.get("methods", base.get("methods", [])))
    seeds = [int(value) for value in phase.get("seeds", base.get("seeds", []))]
    unknown = sorted(set(methods) - ALL_METHODS)
    if unknown:
        raise ValueError(f"unregistered methods in phase {phase.get('name')}: {unknown}")
    if len(methods) != len(set(methods)):
        raise ValueError(f"duplicate methods in phase {phase.get('name')}")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate seeds in phase {phase.get('name')}")
    if phase.get("runner") not in {"synthetic", "clutrr", "language_model"}:
        raise ValueError(
            f"unknown runner in phase {phase.get('name')}: {phase.get('runner')}"
        )
    if method_override is not None:
        methods = [value for value in methods if value in method_override]
    if seed_override is not None:
        seeds = [value for value in seeds if value in seed_override]
    if not methods or not seeds:
        return None
    effective["methods"] = methods
    effective["seeds"] = seeds
    return effective


def describe_suite(
    cfg: dict[str, Any],
    size_name: str,
    *,
    method_override: list[str] | None = None,
    seed_override: list[int] | None = None,
) -> list[dict[str, Any]]:
    size = cfg["sizes"][size_name]
    description = []
    for phase in size.get("suite", []):
        effective = _effective_phase(
            size, phase, method_override, seed_override
        )
        if effective is None:
            continue
        methods = effective["methods"]
        seeds = effective["seeds"]
        description.append(
            {
                "name": effective["name"],
                "runner": effective["runner"],
                "methods": list(methods),
                "seeds": list(seeds),
                "runs": len(methods) * len(seeds),
            }
        )
    return description


def _phase_signature(
    cfg: dict[str, Any], size_name: str, phase: dict[str, Any]
) -> str:
    base = cfg["sizes"][size_name]
    payload = {
        "name": phase["name"],
        "runner": phase["runner"],
        "methods": list(phase.get("methods", base.get("methods", []))),
        "seeds": [int(value) for value in phase.get("seeds", base.get("seeds", []))],
        "source": _source_fingerprint(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]


def _phase_has_all_requested_finals(
    cfg: dict[str, Any], size_name: str, phase: dict[str, Any], output_root: Path
) -> bool:
    """Never let a storage migration hide newly added methods or seeds."""

    base = cfg["sizes"][size_name]
    methods = list(phase.get("methods", base.get("methods", [])))
    seeds = [int(value) for value in phase.get("seeds", base.get("seeds", []))]
    runner = str(phase["runner"])
    defaults = {
        "synthetic": str(base.get("stage", phase["name"])),
        "clutrr": "clutrr",
        "language_model": "formal_language_model",
    }
    stage = str(phase.get("stage", defaults.get(runner, ""))) or None
    if stage is None:
        return False
    completed = {
        (record["method"], int(record["seed"]))
        for record in discover_final_records(
            output_root,
            stage=stage,
            source_fingerprint=_source_fingerprint(),
        )
    }
    return all((method, seed) in completed for method in methods for seed in seeds)


def _synthetic_phase(
    cfg: dict[str, Any],
    *,
    size_name: str,
    phase: dict[str, Any],
    output_root: Path,
    project_root: Path,
    monitor_interval: float,
    method_override: list[str] | None,
    seed_override: list[int] | None,
) -> dict[str, Any]:
    base = cfg["sizes"][size_name]
    phase_name = str(phase["name"])
    methods = list(phase.get("methods", base["methods"]))
    seeds = [int(value) for value in phase.get("seeds", base["seeds"])]
    if method_override:
        methods = [value for value in methods if value in method_override]
    if seed_override:
        seeds = [value for value in seeds if value in seed_override]
    if not methods or not seeds:
        return {"status": "skipped", "reason": "empty method/seed intersection"}
    resolved = resolve_run_config(cfg, size_name, methods[0], seeds[0], phase_name)
    bundle = ensure_synthetic_dataset(
        storage_roots(cfg, output_root).data, resolved["data"]
    )
    settings = {**cfg["resources"], **phase.get("resources", {})}
    snapshot = detect_resources(required_dtype=str(settings["required_dtype"]))
    jobs: list[WorkloadSpec] = []
    commands: list[ScheduledCommand] = []
    for method in methods:
        for seed in seeds:
            resolve_run_config(cfg, size_name, method, seed, phase_name)
            name = f"{phase_name}_{method}_seed{seed}"
            jobs.append(
                WorkloadSpec(
                    name=name,
                    global_batch_size=int(resolved["train"]["batch_size"]),
                    max_micro_batch_size=max(
                        1,
                        int(
                            int(resolved["train"]["batch_size"])
                            * float(phase.get("_micro_batch_scale", 1.0))
                        ),
                    ),
                    estimated_model_memory_gb=float(
                        settings["estimated_model_memory_gb"]
                    ),
                    estimated_activation_memory_per_sample_gb=float(
                        settings["estimated_activation_memory_per_sample_gb"]
                    ),
                    required_micro_batch_size=(
                        int(phase["required_micro_batch_size"])
                        if phase.get("required_micro_batch_size") is not None
                        else None
                    ),
                    profiling=bool(phase.get("profiling", False)),
                )
            )
            commands.append(
                ScheduledCommand(
                    name=name,
                    command=(
                        sys.executable,
                        str(project_root / "main.py"),
                        "run-one",
                        "--size",
                        size_name,
                        "--phase",
                        phase_name,
                        "--method",
                        method,
                        "--seed",
                        str(seed),
                        "--output-root",
                        str(output_root),
                    ),
                    cwd=str(project_root),
                )
            )
    plans = plan_jobs(jobs, snapshot, ResourcePolicy.from_dict(settings))
    write_resource_outputs(output_root, snapshot, plans)
    schedule_dir = execute_schedule(
        plans,
        commands,
        output_root,
        monitor=True,
        monitor_interval_seconds=monitor_interval,
    )
    statuses = json.loads((schedule_dir / "status.json").read_text(encoding="utf-8"))
    failed = [item for item in statuses if item["status"] != "completed"]
    report_dir = build_report(
        output_root,
        stage=str(resolved["stage"]),
        max_static_figures=int(cfg["report"]["max_static_figures"]),
        source_fingerprint=_source_fingerprint(),
    )
    return {
        "status": "completed" if not failed else "completed_with_failures",
        "runs": len(jobs),
        "failed_runs": len(failed),
        "data_manifest": str(bundle.manifest_path),
        "scheduler": str(schedule_dir),
        "report": str(report_dir),
    }


def _external_phase(
    cfg: dict[str, Any],
    *,
    size_name: str,
    phase: dict[str, Any],
    output_root: Path,
    project_root: Path,
    monitor_interval: float,
) -> dict[str, Any]:
    runner = str(phase["runner"])
    if runner == "clutrr":
        from .external_tasks import run_clutrr_phase

        function = run_clutrr_phase
    elif runner == "language_model":
        from .language_model import run_language_model_phase

        function = run_language_model_phase
    else:
        raise ValueError(f"unknown suite runner: {runner}")
    return function(
        cfg,
        size_name=size_name,
        phase=phase,
        output_root=output_root,
        project_root=project_root,
        monitor_interval=monitor_interval,
    )


def run_suite(
    cfg: dict[str, Any],
    *,
    size_name: str,
    output_root: str | Path,
    project_root: str | Path,
    dry_run: bool,
    monitor_interval: float,
    method_override: list[str] | None = None,
    seed_override: list[int] | None = None,
) -> Path:
    output_root = Path(output_root).resolve()
    project_root = Path(project_root).resolve()
    prepare_storage(cfg, output_root)
    suite_dir = output_root / "suites" / size_name
    state_path = suite_dir / "state.json"
    events_path = suite_dir / "events.jsonl"
    base = cfg["sizes"][size_name]
    effective_phases = [
        effective
        for phase in base.get("suite", [])
        if (
            effective := _effective_phase(
                base, phase, method_override, seed_override
            )
        )
        is not None
    ]
    description = describe_suite(
        cfg,
        size_name,
        method_override=method_override,
        seed_override=seed_override,
    )
    print(f"suite | size {size_name} | phases {len(description)}")
    for item in description:
        print(
            f"  {item['name']:<24} | {item['runner']:<14}"
            f" | {item['runs']} run(s)"
        )
    if dry_run:
        _atomic_json(
            suite_dir / "dry_run_plan.json",
            {"generated_at": _now(), "size": size_name, "phases": description},
        )
        print("dry run only; no download or training started")
        return suite_dir

    if bool(cfg["sizes"][size_name].get("require_gpu", False)):
        snapshot = detect_resources(
            required_dtype=str(cfg["resources"]["required_dtype"])
        )
        usable = [gpu for gpu in snapshot.gpus if gpu.torch_usable]
        if not usable:
            raise RuntimeError(
                f"{size_name} requires at least one GPU that passes a real PyTorch "
                "CUDA kernel probe; run small on CPU or repair the server CUDA stack"
            )

    previous: dict[str, Any] = {}
    if state_path.exists():
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
    previous_phases = previous.get("phases", {})
    active_phase_names = {str(phase["name"]) for phase in effective_phases}
    state = {
        "size": size_name,
        "started_at": previous.get("started_at", _now()),
        "updated_at": _now(),
        "status": "running",
        # Retired phases from an older suite definition must not affect the
        # success status of the current experiment matrix.
        "phases": {
            name: value
            for name, value in previous_phases.items()
            if name in active_phase_names
        },
    }
    if legacy_compatibility_pending(output_root):
        accepted = []
        for phase in effective_phases:
            name = str(phase["name"])
            existing = state["phases"].get(name, {})
            if existing.get("status") != "completed":
                continue
            if not _phase_has_all_requested_finals(
                cfg, size_name, phase, output_root
            ):
                print(
                    f"storage migration | {name} has missing new methods/seeds; "
                    "phase will resume"
                )
                continue
            existing["signature"] = _phase_signature(cfg, size_name, phase)
            existing["storage_layout_migrated"] = True
            state["phases"][name] = existing
            accepted.append(name)
        source = _source_fingerprint()
        consume_legacy_compatibility(output_root, source)
        if accepted:
            print(
                "storage migration | reused completed phases | "
                + ", ".join(accepted)
            )
    _atomic_json(state_path, state)
    policy = str(cfg.get("external", {}).get("failure_policy", "continue"))
    for phase in effective_phases:
        name = str(phase["name"])
        signature = _phase_signature(cfg, size_name, phase)
        existing = state["phases"].get(name, {})
        if (
            existing.get("status") == "completed"
            and existing.get("signature") == signature
            and _phase_has_all_requested_finals(
                cfg, size_name, phase, output_root
            )
        ):
            print(f"phase | {name} | already completed")
            continue
        state["phases"][name] = {
            "status": "running",
            "signature": signature,
            "started_at": _now(),
        }
        state["updated_at"] = _now()
        _atomic_json(state_path, state)
        _event(events_path, {"event": "phase_started", "phase": name})
        print(f"phase | {name} | start")
        attempts = []
        max_attempts = int(cfg["run"].get("max_phase_attempts", 1))
        for attempt in range(1, max_attempts + 1):
            attempt_phase = dict(phase)
            attempt_phase["_micro_batch_scale"] = 0.5 ** (attempt - 1)
            try:
                if phase["runner"] == "synthetic":
                    result = _synthetic_phase(
                        cfg,
                        size_name=size_name,
                        phase=attempt_phase,
                        output_root=output_root,
                        project_root=project_root,
                        monitor_interval=monitor_interval,
                        method_override=None,
                        seed_override=None,
                    )
                else:
                    result = _external_phase(
                        cfg,
                        size_name=size_name,
                        phase=attempt_phase,
                        output_root=output_root,
                        project_root=project_root,
                        monitor_interval=monitor_interval,
                    )
                phase_state = {
                    **result,
                    "signature": signature,
                    "finished_at": _now(),
                }
                if (
                    phase_state["status"] == "completed"
                    and not _phase_has_all_requested_finals(
                        cfg, size_name, attempt_phase, output_root
                    )
                ):
                    phase_state["status"] = "completed_with_failures"
                    phase_state["error"] = (
                        "one or more requested final.json artifacts are missing"
                    )
            except Exception as error:  # preserve all other multi-day phases
                phase_state = {
                    "status": "failed",
                    "signature": signature,
                    "finished_at": _now(),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            attempts.append(
                {
                    "attempt": attempt,
                    "status": phase_state["status"],
                    "finished_at": phase_state["finished_at"],
                    "error": phase_state.get("error"),
                }
            )
            if phase_state["status"] not in {"failed", "completed_with_failures"}:
                break
            if attempt < max_attempts:
                print(f"phase | {name} | retry {attempt + 1}/{max_attempts}")
        phase_state["attempts"] = attempts
        state["phases"][name] = phase_state
        state["updated_at"] = _now()
        _atomic_json(state_path, state)
        _event(
            events_path,
            {"event": "phase_finished", "phase": name, **phase_state},
        )
        print(f"phase | {name} | {phase_state['status']}")
        if phase_state["status"] == "failed" and policy == "stop":
            break

    statuses = [value.get("status") for value in state["phases"].values()]
    state["status"] = (
        "completed"
        if statuses and all(value == "completed" for value in statuses)
        else "completed_with_issues"
    )
    state["finished_at"] = _now()
    state["updated_at"] = _now()
    all_report = build_report(
        output_root,
        max_static_figures=int(cfg["report"]["max_static_figures"]),
        source_fingerprint=_source_fingerprint(),
    )
    state["all_report"] = str(all_report)
    state["paper_verdict"] = str(
        build_paper_verdict(
            output_root, source_fingerprint=_source_fingerprint()
        )
    )
    _atomic_json(state_path, state)
    print(f"suite complete | {state['status']} | {state_path}")
    if state["status"] != "completed":
        raise RuntimeError(
            f"suite finished with issues; inspect {state_path}"
        )
    return suite_dir
