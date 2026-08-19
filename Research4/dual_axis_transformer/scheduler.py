"""Execute precomputed adaptive plans without shell interpolation."""

from __future__ import annotations

import codecs
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .resources import JobPlan, detect_resources


@dataclass(frozen=True)
class ScheduledCommand:
    name: str
    command: tuple[str, ...]
    cwd: str | None = None


def _job_environment(plan: JobPlan) -> dict[str, str]:
    environment = os.environ.copy()
    # Child stdout/stderr are persisted and tailed by the parent scheduler.
    # Unbuffered mode makes each formatted training interval visible promptly.
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    if plan.gpu_index is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(plan.gpu_index)
        environment["CONCEPT_BUS_DEVICE"] = "cuda:0"
        environment["CONCEPT_BUS_GPU_INDEX"] = str(plan.gpu_index)
        environment["CONCEPT_BUS_GPU_ORDINAL"] = str(plan.gpu_ordinal or 1)
        environment["CONCEPT_BUS_GPU_COUNT"] = str(plan.available_gpu_count)
    else:
        environment["CONCEPT_BUS_DEVICE"] = "cpu"
        environment["CONCEPT_BUS_GPU_INDEX"] = ""
        environment["CONCEPT_BUS_GPU_ORDINAL"] = "0"
        environment["CONCEPT_BUS_GPU_COUNT"] = "0"
    environment["CONCEPT_BUS_MICRO_BATCH"] = str(plan.micro_batch_size)
    environment["CONCEPT_BUS_GRAD_ACCUM"] = str(
        plan.gradient_accumulation_steps
    )
    environment["CONCEPT_BUS_DATALOADER_WORKERS"] = str(
        plan.dataloader_workers
    )
    environment["CONCEPT_BUS_PREFETCH_FACTOR"] = str(plan.prefetch_factor)
    environment["CONCEPT_BUS_PIN_MEMORY"] = "1" if plan.pin_memory else "0"
    return environment


def execute_schedule(
    plans: Iterable[JobPlan],
    commands: Iterable[ScheduledCommand],
    output_root: str | Path,
    *,
    monitor: bool = True,
    monitor_interval_seconds: float = 5.0,
) -> Path:
    """Run each wave concurrently and save logs/status under output/scheduler."""

    plans = list(plans)
    command_map = {command.name: command for command in commands}
    missing = [plan.job_name for plan in plans if plan.job_name not in command_map]
    if missing:
        raise ValueError(f"missing commands for planned jobs: {missing}")
    if monitor_interval_seconds <= 0:
        raise ValueError("monitor_interval_seconds must be positive")

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    run_dir = Path(output_root) / "scheduler" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "schedule.json").write_text(
        json.dumps([plan.to_dict() for plan in plans], indent=2),
        encoding="utf-8",
    )
    history_path = run_dir / "resource_history.jsonl"
    statuses: list[dict[str, object]] = []
    console_limit = 120
    prefix_width = min(20, max((len(plan.job_name) for plan in plans), default=1))

    stream_offsets: dict[tuple[str, str], int] = {}
    stream_fragments: dict[tuple[str, str], str] = {}
    stream_decoders: dict[tuple[str, str], codecs.IncrementalDecoder] = {}

    def console_safe(value: str, target: object) -> str:
        encoding = getattr(target, "encoding", None) or "utf-8"
        return value.encode(encoding, errors="replace").decode(
            encoding, errors="replace"
        )

    def render_console(job_name: str, kind: str, value: str) -> str:
        child = value.rstrip()
        # Training rows already carry fixed-width GPU/phase/step/task fields.
        # Prefixing them duplicates the task and pushes them past 120 columns.
        if child.count(" | ") == 8:
            return child[:console_limit]
        label = (
            job_name
            if len(job_name) <= prefix_width
            else job_name[: max(1, prefix_width - 1)] + "…"
        )
        prefix = f"{label:<{prefix_width}} | {kind.upper():<6} | "
        available = max(1, console_limit - len(prefix))
        if len(child) > available:
            child = child[: max(1, available - 1)] + "…"
        return prefix + child

    def forward_new_output(active_jobs: list[tuple], *, final: bool = False) -> None:
        for plan, _, _, _, _, stdout_path, stderr_path in active_jobs:
            for kind, path, target in (
                ("stdout", stdout_path, sys.stdout),
                ("stderr", stderr_path, sys.stderr),
            ):
                key = (plan.job_name, kind)
                offset = stream_offsets.get(key, 0)
                try:
                    with path.open("rb") as handle:
                        handle.seek(offset)
                        chunk = handle.read()
                        stream_offsets[key] = handle.tell()
                except FileNotFoundError:
                    continue
                decoder = stream_decoders.setdefault(
                    key,
                    codecs.getincrementaldecoder("utf-8")(errors="replace"),
                )
                decoded = decoder.decode(chunk, final=final)
                combined = stream_fragments.get(key, "") + decoded
                lines = combined.splitlines(keepends=True)
                remainder = ""
                if lines and not lines[-1].endswith(("\n", "\r")):
                    remainder = lines.pop()
                stream_fragments[key] = remainder
                for line in lines:
                    rendered = render_console(plan.job_name, kind, line)
                    print(
                        console_safe(rendered, target),
                        file=target,
                        flush=True,
                    )
                if final and stream_fragments[key]:
                    rendered = render_console(
                        plan.job_name, kind, stream_fragments[key]
                    )
                    print(
                        console_safe(rendered, target),
                        file=target,
                        flush=True,
                    )
                    stream_fragments[key] = ""

    waves = sorted({plan.wave for plan in plans})
    for wave in waves:
        active = []
        for plan in [item for item in plans if item.wave == wave]:
            specification = command_map[plan.job_name]
            job_dir = run_dir / plan.job_name
            job_dir.mkdir(parents=True, exist_ok=False)
            stdout_handle = (job_dir / "stdout.log").open(
                "w", encoding="utf-8"
            )
            stderr_handle = (job_dir / "stderr.log").open(
                "w", encoding="utf-8"
            )
            started = datetime.now(timezone.utc).isoformat()
            process = subprocess.Popen(
                list(specification.command),
                cwd=specification.cwd,
                env=_job_environment(plan),
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                shell=False,
            )
            active.append(
                (
                    plan,
                    process,
                    stdout_handle,
                    stderr_handle,
                    started,
                    job_dir / "stdout.log",
                    job_dir / "stderr.log",
                )
            )
            device = (
                "CPU 0/0"
                if plan.gpu_index is None
                else f"GPU {plan.gpu_ordinal}/{plan.available_gpu_count}"
            )
            print(f"launch | {plan.job_name} | {device} | wave {wave}")

        while any(process.poll() is None for _, process, *_ in active):
            forward_new_output(active)
            if monitor:
                # Startup already performed the real CUDA kernel probe. Runtime
                # monitoring stays read-mostly and must not inject kernels into
                # GPUs occupied by the experiments being measured.
                snapshot = detect_resources(probe_cuda=False)
                with history_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {"wave": wave, "snapshot": snapshot.to_dict()},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            time.sleep(monitor_interval_seconds)

        forward_new_output(active, final=True)
        for (
            plan,
            process,
            stdout_handle,
            stderr_handle,
            started,
            _,
            _,
        ) in active:
            return_code = process.wait()
            stdout_handle.close()
            stderr_handle.close()
            statuses.append(
                {
                    "job_name": plan.job_name,
                    "wave": wave,
                    "device": plan.device,
                    "started_at": started,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "return_code": return_code,
                    "status": "completed" if return_code == 0 else "failed",
                }
            )
            print(
                f"complete | {plan.job_name} | "
                f"{'ok' if return_code == 0 else f'failed({return_code})'}"
            )

    (run_dir / "status.json").write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir
