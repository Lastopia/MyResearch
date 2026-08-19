"""Unified entry point for resource planning and report generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cfg import CFG
from dual_axis_transformer.reporting import build_report
from dual_axis_transformer.resources import (
    ResourcePolicy,
    WorkloadSpec,
    detect_resources,
    plan_jobs,
    write_resource_outputs,
)
from dual_axis_transformer.scheduler import ScheduledCommand, execute_schedule
from dual_axis_transformer.runner import _source_fingerprint, resolve_run_config, run_one
from dual_axis_transformer.synthetic_data import ensure_synthetic_dataset
from dual_axis_transformer.suite import describe_suite, run_suite
from dual_axis_transformer.storage import prepare_storage


def _output_root(value: str | None) -> Path:
    return Path(value or CFG["paths"]["output_root"]).resolve()


def _resource_summary(snapshot: object) -> str:
    cpu = snapshot.cpu
    physical = cpu.physical_cores or "?"
    memory = (
        f"{cpu.available_memory_gb:.1f}/{cpu.total_memory_gb:.1f} GiB"
        if cpu.available_memory_gb is not None and cpu.total_memory_gb is not None
        else "unknown"
    )
    usable = sum(gpu.torch_usable for gpu in snapshot.gpus)
    return (
        f"resources | CPU {physical}/{cpu.logical_cores} cores"
        f" | RAM {memory} | usable GPU {usable}/{len(snapshot.gpus)}"
    )


def _plan_summary(plan: object) -> str:
    if plan.device == "cpu":
        device = "CPU 0/0"
    else:
        device = f"GPU {plan.gpu_ordinal}/{plan.available_gpu_count} ({plan.device})"
    return (
        f"plan | {plan.job_name} | {device} | micro-batch {plan.micro_batch_size}"
        f" | accumulation {plan.gradient_accumulation_steps} | wave {plan.wave}"
    )


def command_resources(args: argparse.Namespace) -> int:
    settings = CFG["resources"]
    snapshot = detect_resources(required_dtype=str(settings["required_dtype"]))
    policy = ResourcePolicy.from_dict(settings)
    jobs = [
        WorkloadSpec(
            name=f"{args.name}_{index + 1}",
            global_batch_size=int(settings["global_batch_size"]),
            max_micro_batch_size=int(settings["max_micro_batch_size"]),
            estimated_model_memory_gb=float(
                settings["estimated_model_memory_gb"]
            ),
            estimated_activation_memory_per_sample_gb=float(
                settings["estimated_activation_memory_per_sample_gb"]
            ),
            profiling=args.profile,
        )
        for index in range(args.jobs)
    ]
    plans = plan_jobs(jobs, snapshot, policy)
    snapshot_path, plan_path = write_resource_outputs(
        _output_root(args.output_root), snapshot, plans
    )
    if args.verbose:
        print(json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(_resource_summary(snapshot))
        for plan in plans:
            print(_plan_summary(plan))
    print(f"resource snapshot: {snapshot_path}")
    print(f"adaptive plan: {plan_path}")
    return 0


def command_report(args: argparse.Namespace) -> int:
    report_dir = build_report(
        _output_root(args.output_root),
        stage=args.stage,
        max_static_figures=int(CFG["report"]["max_static_figures"]),
        source_fingerprint=(
            None if args.all_code_versions else _source_fingerprint()
        ),
    )
    print(f"report: {report_dir}")
    print(f"dashboard: {report_dir / 'dashboard' / 'index.html'}")
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    from dual_axis_transformer.preflight import run_preflight

    output_root = _output_root(args.output_root)
    try:
        report_path = run_preflight(
            CFG,
            size_name=args.size,
            output_root=output_root,
            prepare_data=bool(args.prepare_data),
        )
    except RuntimeError:
        report_path = output_root / "preflight" / "report.json"
        payload = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.exists()
            else {"errors": ["preflight did not produce a report"]}
        )
        print("preflight | blocked | long run was not started", file=sys.stderr)
        for error in payload.get("errors", []):
            print(f"  {error}", file=sys.stderr)
        print(f"  report: {report_path}", file=sys.stderr)
        return 2
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    usable = [
        gpu
        for gpu in payload["checks"]["resources"].get("gpus", [])
        if gpu.get("torch_usable")
    ]
    if usable:
        names = ", ".join(
            f"GPU {gpu['index']} {gpu['name']} ({gpu['total_memory_gb']:.1f} GiB)"
            for gpu in usable
        )
        print(f"preflight | pass | CUDA available | {names}")
    else:
        print("preflight | pass | CPU mode")
    for probe in payload["checks"].get("fair_synthetic_capacity_probe", []):
        peak_gib = float(probe["peak_cuda_memory_bytes"]) / 1024**3
        print(
            f"capacity | {probe['device']} | {probe['method']}"
            f" | batch {probe['micro_batch_size']}"
            f" | length {probe['sequence_length']}"
            f" | peak {peak_gib:.1f} GiB | pass"
        )
    print(f"preflight report | {report_path}")
    return 0


def command_schedule(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings = CFG["resources"]
    jobs = []
    commands = []
    for item in payload.get("jobs", []):
        name = str(item["name"])
        command = item["command"]
        if not isinstance(command, list) or not all(
            isinstance(part, str) for part in command
        ):
            raise ValueError("each command must be a JSON list of strings")
        jobs.append(
            WorkloadSpec(
                name=name,
                global_batch_size=int(
                    item.get("global_batch_size", settings["global_batch_size"])
                ),
                max_micro_batch_size=int(
                    item.get(
                        "max_micro_batch_size", settings["max_micro_batch_size"]
                    )
                ),
                estimated_model_memory_gb=float(
                    item.get(
                        "estimated_model_memory_gb",
                        settings["estimated_model_memory_gb"],
                    )
                ),
                estimated_activation_memory_per_sample_gb=float(
                    item.get(
                        "estimated_activation_memory_per_sample_gb",
                        settings["estimated_activation_memory_per_sample_gb"],
                    )
                ),
                profiling=bool(item.get("profiling", False)),
            )
        )
        commands.append(
            ScheduledCommand(
                name=name,
                command=tuple(command),
                cwd=item.get("cwd"),
            )
        )
    if not jobs:
        raise ValueError("manifest contains no jobs")
    snapshot = detect_resources(required_dtype=str(settings["required_dtype"]))
    plans = plan_jobs(jobs, snapshot, ResourcePolicy.from_dict(settings))
    output_root = _output_root(args.output_root)
    write_resource_outputs(output_root, snapshot, plans)
    if not args.execute:
        print(_resource_summary(snapshot))
        if args.verbose:
            print(json.dumps([plan.to_dict() for plan in plans], indent=2))
        else:
            for plan in plans:
                print(_plan_summary(plan))
        print("dry plan only; pass --execute to launch")
        return 0
    run_dir = execute_schedule(
        plans,
        commands,
        output_root,
        monitor=not args.no_monitor,
        monitor_interval_seconds=args.monitor_interval,
    )
    print(f"scheduler run: {run_dir}")
    return 0


def _selected_methods(args: argparse.Namespace, size: dict[str, object]) -> list[str]:
    if not args.methods:
        return [str(value) for value in size["methods"]]
    requested = [value.strip() for value in args.methods.split(",") if value.strip()]
    if len(requested) != len(set(requested)):
        raise ValueError("--methods contains duplicates")
    unknown = sorted(set(requested) - set(size["methods"]))
    if unknown:
        raise ValueError(f"methods not registered in size: {unknown}")
    return requested


def _selected_seeds(args: argparse.Namespace, size: dict[str, object]) -> list[int]:
    if not args.seeds:
        return [int(value) for value in size["seeds"]]
    requested = [
        int(value.strip()) for value in args.seeds.split(",") if value.strip()
    ]
    if len(requested) != len(set(requested)):
        raise ValueError("--seeds contains duplicates")
    unknown = sorted(set(requested) - {int(value) for value in size["seeds"]})
    if unknown:
        raise ValueError(f"seeds not registered in size: {unknown}")
    return requested


def command_run_one(args: argparse.Namespace) -> int:
    output_root = _output_root(args.output_root)
    prepare_storage(CFG, output_root)
    run_dir = run_one(
        CFG,
        size_name=args.size,
        phase_name=args.phase,
        method=args.method,
        seed=args.seed,
        output_root=output_root,
    )
    print(f"run: {run_dir}")
    return 0


def command_external_one(args: argparse.Namespace) -> int:
    output_root = _output_root(args.output_root)
    prepare_storage(CFG, output_root)
    if args.task == "clutrr":
        from dual_axis_transformer.external_tasks import run_clutrr_one

        run_dir = run_clutrr_one(
            CFG, method=args.method, seed=args.seed, output_root=output_root,
            size_name=args.size, stage=args.stage,
        )
    elif args.task in {"tinystories", "language"}:
        from dual_axis_transformer.language_model import run_language_model_one

        run_dir = run_language_model_one(
            CFG, method=args.method, seed=args.seed, output_root=output_root,
            size_name=args.size, stage=args.stage,
        )
    else:
        raise ValueError(f"unknown external task: {args.task}")
    print(f"run: {run_dir}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    size_name = args.size or str(CFG["run"]["size"])
    if size_name not in CFG["sizes"]:
        raise ValueError(f"unknown size: {size_name}")
    size = CFG["sizes"][size_name]
    methods = _selected_methods(args, size)
    seeds = _selected_seeds(args, size)
    output_root = _output_root(args.output_root)
    roots = prepare_storage(CFG, output_root)
    if size.get("suite"):
        # Validate the full matrix before CUDA probes or multi-GB downloads.
        describe_suite(
            CFG,
            size_name,
            method_override=methods if args.methods else None,
            seed_override=seeds if args.seeds else None,
        )
        if not args.dry_run:
            preflight_status = command_preflight(
                argparse.Namespace(
                    size=size_name,
                    prepare_data=True,
                    output_root=str(output_root),
                )
            )
            if preflight_status:
                return preflight_status
        try:
            run_suite(
                CFG,
                size_name=size_name,
                output_root=output_root,
                project_root=Path(__file__).resolve().parent,
                dry_run=bool(args.dry_run),
                monitor_interval=float(args.monitor_interval),
                method_override=methods if args.methods else None,
                seed_override=seeds if args.seeds else None,
            )
        except RuntimeError as error:
            if bool(size.get("require_gpu", False)) and "GPU" in str(error):
                print(
                    f"run | blocked | GPU unavailable; long run was not started | {error}",
                    file=sys.stderr,
                )
                return 2
            raise
        return 0
    bundle = ensure_synthetic_dataset(roots.data, size["data"])
    print(
        f"matrix | size {size_name} | methods {len(methods)}"
        f" | seeds {len(seeds)} | runs {len(methods) * len(seeds)}"
    )
    print(f"data | {bundle.manifest_path}")
    for method in methods:
        for seed in seeds:
            # Resolve now so an invalid method/config fails before any run starts.
            resolve_run_config(CFG, size_name, method, seed)
            print(f"  {method:<24} seed {seed}")
    if args.dry_run:
        print("dry run only; no training started")
        return 0

    if len(methods) == 1 and len(seeds) == 1 and not args.force_scheduler:
        run_one(
            CFG,
            size_name=size_name,
            method=methods[0],
            seed=seeds[0],
            output_root=output_root,
        )
    else:
        settings = CFG["resources"]
        snapshot = detect_resources(required_dtype=str(settings["required_dtype"]))
        jobs = []
        commands = []
        for method in methods:
            for seed in seeds:
                name = f"{method}_seed{seed}"
                jobs.append(
                    WorkloadSpec(
                        name=name,
                        global_batch_size=int(size["train"]["batch_size"]),
                        max_micro_batch_size=int(size["train"]["batch_size"]),
                        estimated_model_memory_gb=float(settings["estimated_model_memory_gb"]),
                        estimated_activation_memory_per_sample_gb=float(
                            settings["estimated_activation_memory_per_sample_gb"]
                        ),
                    )
                )
                commands.append(
                    ScheduledCommand(
                        name=name,
                        command=(
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "run-one",
                            "--size",
                            size_name,
                            "--method",
                            method,
                            "--seed",
                            str(seed),
                            "--output-root",
                            str(output_root),
                        ),
                        cwd=str(Path(__file__).resolve().parent),
                    )
                )
        plans = plan_jobs(jobs, snapshot, ResourcePolicy.from_dict(settings))
        write_resource_outputs(output_root, snapshot, plans)
        schedule_dir = execute_schedule(
            plans,
            commands,
            output_root,
            monitor=True,
            monitor_interval_seconds=args.monitor_interval,
        )
        statuses = json.loads(
            (schedule_dir / "status.json").read_text(encoding="utf-8")
        )
        failed = [item for item in statuses if item["status"] != "completed"]
        if failed:
            raise RuntimeError(
                f"{len(failed)} run(s) failed; inspect {schedule_dir}"
            )
    report_dir = build_report(
        output_root,
        stage=str(size["stage"]),
        max_static_figures=int(CFG["report"]["max_static_figures"]),
        source_fingerprint=_source_fingerprint(),
    )
    print(f"complete | report {report_dir / 'dashboard' / 'index.html'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    resources = subparsers.add_parser(
        "resources", help="detect hardware and write an adaptive execution plan"
    )
    resources.add_argument("--jobs", type=int, default=1)
    resources.add_argument("--name", default="concept_bus")
    resources.add_argument("--profile", action="store_true")
    resources.add_argument("--output-root")
    resources.add_argument("--verbose", action="store_true")
    resources.set_defaults(handler=command_resources)

    report = subparsers.add_parser(
        "report", help="aggregate output/runs into CSV, SVG, and HTML/JS"
    )
    report.add_argument("--stage")
    report.add_argument("--output-root")
    report.add_argument(
        "--all-code-versions",
        action="store_true",
        help="include legacy code versions (default: current source only)",
    )
    report.set_defaults(handler=command_report)

    preflight = subparsers.add_parser(
        "preflight", help="fail-fast CUDA, disk, write and real-data validation"
    )
    preflight.add_argument("--size", choices=tuple(CFG["sizes"]), default="large")
    preflight.add_argument("--prepare-data", action="store_true")
    preflight.add_argument("--output-root")
    preflight.set_defaults(handler=command_preflight)

    schedule = subparsers.add_parser(
        "schedule", help="plan a JSON job manifest and optionally execute it"
    )
    schedule.add_argument("manifest")
    schedule.add_argument("--execute", action="store_true")
    schedule.add_argument("--no-monitor", action="store_true")
    schedule.add_argument("--monitor-interval", type=float, default=5.0)
    schedule.add_argument("--output-root")
    schedule.add_argument("--verbose", action="store_true")
    schedule.set_defaults(handler=command_schedule)

    run = subparsers.add_parser(
        "run", help="generate data, execute a size matrix, and rebuild its report"
    )
    run.add_argument("--size", choices=tuple(CFG["sizes"]))
    run.add_argument("--methods", help="comma-separated subset registered by the size")
    run.add_argument("--seeds", help="comma-separated seed subset")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--force-scheduler", action="store_true")
    run.add_argument("--monitor-interval", type=float, default=5.0)
    run.add_argument("--output-root")
    run.set_defaults(handler=command_run)

    run_one_parser = subparsers.add_parser(
        "run-one", help="internal: execute one synthetic run"
    )
    run_one_parser.add_argument("--size", choices=tuple(CFG["sizes"]), required=True)
    run_one_parser.add_argument("--phase")
    run_one_parser.add_argument("--method", required=True)
    run_one_parser.add_argument("--seed", required=True, type=int)
    run_one_parser.add_argument("--output-root")
    run_one_parser.set_defaults(handler=command_run_one)

    external_one = subparsers.add_parser(
        "external-one", help="internal: execute one external or transfer run"
    )
    external_one.add_argument(
        "--task",
        choices=("clutrr", "tinystories", "language"),
        required=True,
    )
    external_one.add_argument("--method", required=True)
    external_one.add_argument("--size", choices=tuple(CFG["sizes"]), default="large")
    external_one.add_argument("--stage")
    external_one.add_argument("--seed", required=True, type=int)
    external_one.add_argument("--output-root")
    external_one.set_defaults(handler=command_external_one)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Keep explicit subcommands for internal/specialized use, while making the
    # user-facing path ``python main.py --size medium`` (or simply main.py,
    # using CFG's selected size).
    if not arguments or arguments[0].startswith("-"):
        arguments.insert(0, "run")
    args = build_parser().parse_args(arguments)
    if getattr(args, "jobs", 1) <= 0:
        raise SystemExit("--jobs must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
