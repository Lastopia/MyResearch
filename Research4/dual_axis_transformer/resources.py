"""Hardware discovery, adaptive planning, and resource recommendations.

Plans are frozen before a run.  Runtime observations may recommend a different
plan for the next launch, but never silently change effective batch semantics.
"""

from __future__ import annotations

import csv
import math
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

try:
    import psutil
except ImportError:  # pragma: no cover - exercised only in minimal installs
    psutil = None


def _gb(num_bytes: int | float) -> float:
    return round(float(num_bytes) / (1024**3), 3)


def _gpu_tier(total_memory_gb: float) -> str:
    if total_memory_gb < 16:
        return "compact"
    if total_memory_gb < 32:
        return "standard"
    if total_memory_gb < 64:
        return "large"
    return "xlarge"


@dataclass(frozen=True)
class CPUInfo:
    logical_cores: int
    physical_cores: int | None
    total_memory_gb: float | None
    available_memory_gb: float | None
    utilization_percent: float | None


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    utilization_percent: float | None
    memory_utilization_percent: float | None
    compute_capability: str | None
    bf16_supported: bool | None
    torch_usable: bool
    probe_error: str | None
    tier: str


@dataclass(frozen=True)
class ResourceSnapshot:
    captured_at: str
    hostname: str
    platform: str
    python_version: str
    torch_version: str
    cuda_version: str | None
    cpu: CPUInfo
    gpus: tuple[GPUInfo, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourcePolicy:
    min_free_memory_gb: float = 4.0
    max_initial_gpu_utilization_percent: float = 20.0
    target_memory_fraction: float = 0.85
    target_gpu_utilization_percent: float = 90.0
    allow_colocation: bool = True
    max_jobs_per_gpu: int = 2
    profile_exclusive: bool = True
    max_workers_per_job: int = 8
    prefetch_factor: int = 2

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ResourcePolicy":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in fields})


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    global_batch_size: int
    max_micro_batch_size: int
    estimated_model_memory_gb: float
    estimated_activation_memory_per_sample_gb: float
    required_micro_batch_size: int | None = None
    profiling: bool = False
    cpu_heavy: bool = False


@dataclass(frozen=True)
class JobPlan:
    job_name: str
    device: str
    gpu_index: int | None
    gpu_ordinal: int | None
    available_gpu_count: int
    gpu_tier: str | None
    wave: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    estimated_peak_memory_gb: float
    dataloader_workers: int
    prefetch_factor: int
    pin_memory: bool
    compile_recommended: bool
    exclusive: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeObservation:
    gpu_utilization_percent: float | None
    peak_memory_fraction: float | None
    cpu_utilization_percent: float | None
    available_ram_fraction: float | None
    dataloader_wait_fraction: float | None


def _nvidia_smi_rows() -> dict[int, dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,utilization.gpu,"
        "utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}

    rows: dict[int, dict[str, Any]] = {}
    reader = csv.reader(completed.stdout.splitlines())
    for row in reader:
        if len(row) != 6:
            continue
        try:
            index = int(row[0].strip())
            rows[index] = {
                "name": row[1].strip(),
                "total_memory_gb": round(float(row[2]) / 1024, 3),
                "free_memory_gb": round(float(row[3]) / 1024, 3),
                "utilization_percent": float(row[4]),
                "memory_utilization_percent": float(row[5]),
            }
        except ValueError:
            continue
    return rows


def _probe_torch_gpu(index: int, required_dtype: str) -> tuple[bool, str | None]:
    try:
        device = torch.device("cuda", index)
        with torch.cuda.device(index):
            if required_dtype in {"bf16", "bfloat16"}:
                if not torch.cuda.is_bf16_supported():
                    return False, "BF16 was required but is not supported"
                dtype = torch.bfloat16
            elif required_dtype in {"fp16", "float16"}:
                dtype = torch.float16
            else:
                dtype = torch.float32
            left = torch.randn((16, 16), device=device, dtype=dtype)
            right = torch.randn((16, 16), device=device, dtype=dtype)
            result = left @ right
            _ = float(result.float().sum().item())
            torch.cuda.synchronize(index)
            del left, right, result
        return True, None
    except Exception as error:  # CUDA driver/runtime failures vary by platform
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return False, f"{type(error).__name__}: {error}"


def detect_resources(
    *, required_dtype: str = "auto", probe_cuda: bool = True
) -> ResourceSnapshot:
    logical = os.cpu_count() or 1
    if psutil is not None:
        physical = psutil.cpu_count(logical=False)
        memory = psutil.virtual_memory()
        total_memory = _gb(memory.total)
        available_memory = _gb(memory.available)
        cpu_utilization = float(psutil.cpu_percent(interval=0.05))
    else:
        physical = None
        total_memory = available_memory = cpu_utilization = None

    smi = _nvidia_smi_rows()
    torch_gpus: dict[int, dict[str, Any]] = {}
    try:
        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    except Exception:
        cuda_available = False
        cuda_device_count = 0
    if cuda_available:
        for index in range(cuda_device_count):
            properties = torch.cuda.get_device_properties(index)
            try:
                free_bytes, _ = torch.cuda.mem_get_info(index)
                free_memory_gb = _gb(free_bytes)
            except (RuntimeError, TypeError):
                free_memory_gb = _gb(properties.total_memory)
            capability = torch.cuda.get_device_capability(index)
            usable, probe_error = (
                _probe_torch_gpu(index, required_dtype)
                if probe_cuda
                else (True, None)
            )
            with torch.cuda.device(index):
                bf16_supported = bool(torch.cuda.is_bf16_supported())
            torch_gpus[index] = {
                "name": properties.name,
                "total_memory_gb": _gb(properties.total_memory),
                "free_memory_gb": free_memory_gb,
                "compute_capability": f"{capability[0]}.{capability[1]}",
                "bf16_supported": bf16_supported,
                "torch_usable": usable,
                "probe_error": probe_error,
            }

    indices = sorted(set(smi) | set(torch_gpus))
    gpus: list[GPUInfo] = []
    for index in indices:
        merged = {**smi.get(index, {}), **torch_gpus.get(index, {})}
        total = float(merged.get("total_memory_gb", 0.0))
        gpus.append(
            GPUInfo(
                index=index,
                name=str(merged.get("name", f"GPU {index}")),
                total_memory_gb=total,
                free_memory_gb=float(merged.get("free_memory_gb", total)),
                utilization_percent=merged.get("utilization_percent"),
                memory_utilization_percent=merged.get(
                    "memory_utilization_percent"
                ),
                compute_capability=merged.get("compute_capability"),
                bf16_supported=merged.get("bf16_supported"),
                torch_usable=bool(merged.get("torch_usable", False)),
                probe_error=merged.get(
                    "probe_error",
                    None
                    if merged.get("torch_usable", False)
                    else "current PyTorch build/runtime cannot initialize CUDA",
                ),
                tier=_gpu_tier(total),
            )
        )

    return ResourceSnapshot(
        captured_at=datetime.now(timezone.utc).isoformat(),
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cpu=CPUInfo(
            logical_cores=logical,
            physical_cores=physical,
            total_memory_gb=total_memory,
            available_memory_gb=available_memory,
            utilization_percent=cpu_utilization,
        ),
        gpus=tuple(gpus),
    )


def _largest_divisor_at_most(total: int, ceiling: int) -> int:
    if total <= 0 or ceiling <= 0:
        raise ValueError("batch sizes must be positive")
    for candidate in range(min(total, ceiling), 0, -1):
        if total % candidate == 0:
            return candidate
    return 1


def _eligible_gpus(
    snapshot: ResourceSnapshot, policy: ResourcePolicy
) -> list[GPUInfo]:
    eligible = []
    for gpu in snapshot.gpus:
        if not gpu.torch_usable:
            continue
        busy = gpu.utilization_percent
        if gpu.free_memory_gb < policy.min_free_memory_gb:
            continue
        if busy is not None and busy > policy.max_initial_gpu_utilization_percent:
            continue
        eligible.append(gpu)
    return sorted(eligible, key=lambda item: item.free_memory_gb, reverse=True)


def _gpu_plan(
    job: WorkloadSpec,
    gpu: GPUInfo,
    gpu_ordinal: int,
    available_gpu_count: int,
    wave: int,
    memory_budget_gb: float,
    workers: int,
    policy: ResourcePolicy,
) -> JobPlan | None:
    remaining = memory_budget_gb - job.estimated_model_memory_gb
    if remaining <= 0:
        return None
    per_sample = max(job.estimated_activation_memory_per_sample_gb, 1e-9)
    memory_ceiling = math.floor(remaining / per_sample)
    ceiling = min(job.max_micro_batch_size, memory_ceiling)
    if ceiling < 1:
        return None
    required = job.required_micro_batch_size
    if required is not None:
        if required > ceiling or job.global_batch_size % required:
            return None
        micro_batch = required
    else:
        micro_batch = _largest_divisor_at_most(job.global_batch_size, ceiling)
    accumulation = job.global_batch_size // micro_batch
    estimated_peak = (
        job.estimated_model_memory_gb
        + micro_batch * job.estimated_activation_memory_per_sample_gb
    )
    return JobPlan(
        job_name=job.name,
        device=f"cuda:{gpu.index}",
        gpu_index=gpu.index,
        gpu_ordinal=gpu_ordinal,
        available_gpu_count=available_gpu_count,
        gpu_tier=gpu.tier,
        wave=wave,
        micro_batch_size=micro_batch,
        gradient_accumulation_steps=accumulation,
        effective_batch_size=micro_batch * accumulation,
        estimated_peak_memory_gb=round(estimated_peak, 3),
        dataloader_workers=workers,
        prefetch_factor=policy.prefetch_factor,
        pin_memory=True,
        compile_recommended=gpu.tier in {"large", "xlarge"},
        exclusive=job.profiling and policy.profile_exclusive,
    )


def plan_jobs(
    jobs: Iterable[WorkloadSpec],
    snapshot: ResourceSnapshot,
    policy: ResourcePolicy | None = None,
) -> list[JobPlan]:
    policy = policy or ResourcePolicy()
    jobs = list(jobs)
    if not jobs:
        return []
    for job in jobs:
        required = job.required_micro_batch_size
        if required is not None and (
            required < 1
            or required > job.max_micro_batch_size
            or job.global_batch_size % required
        ):
            raise ValueError(
                f"{job.name}: required micro batch {required} must be a positive "
                f"divisor of global batch {job.global_batch_size} and no larger "
                f"than max micro batch {job.max_micro_batch_size}"
            )
    gpus = _eligible_gpus(snapshot, policy)
    physical_cores = snapshot.cpu.physical_cores or snapshot.cpu.logical_cores

    if not gpus:
        workers = max(0, min(policy.max_workers_per_job, physical_cores - 1))
        def cpu_micro_batch(job: WorkloadSpec) -> int:
            return job.required_micro_batch_size or _largest_divisor_at_most(
                job.global_batch_size, job.max_micro_batch_size
            )

        return [
            JobPlan(
                job_name=job.name,
                device="cpu",
                gpu_index=None,
                gpu_ordinal=None,
                available_gpu_count=0,
                gpu_tier=None,
                wave=index,
                micro_batch_size=cpu_micro_batch(job),
                gradient_accumulation_steps=(
                    job.global_batch_size // cpu_micro_batch(job)
                ),
                effective_batch_size=job.global_batch_size,
                estimated_peak_memory_gb=job.estimated_model_memory_gb,
                dataloader_workers=workers,
                prefetch_factor=policy.prefetch_factor,
                pin_memory=False,
                compile_recommended=False,
                exclusive=True,
            )
            for index, job in enumerate(jobs)
        ]

    plans: list[JobPlan] = []
    pending = list(jobs)
    ordinal_by_index = {gpu.index: index + 1 for index, gpu in enumerate(gpus)}
    wave = 0
    while pending:
        budgets = {
            gpu.index: gpu.free_memory_gb * policy.target_memory_fraction
            for gpu in gpus
        }
        counts = {gpu.index: 0 for gpu in gpus}
        exclusive = {gpu.index: False for gpu in gpus}
        placed: list[WorkloadSpec] = []
        worker_share = max(
            1,
            min(
                policy.max_workers_per_job,
                physical_cores // max(1, min(len(pending), len(gpus))),
            ),
        )

        for job in pending:
            candidates = []
            for gpu in gpus:
                if exclusive[gpu.index]:
                    continue
                if counts[gpu.index] >= (
                    policy.max_jobs_per_gpu if policy.allow_colocation else 1
                ):
                    continue
                if job.profiling and policy.profile_exclusive and counts[gpu.index]:
                    continue
                tentative = _gpu_plan(
                    job,
                    gpu,
                    ordinal_by_index[gpu.index],
                    len(gpus),
                    wave,
                    budgets[gpu.index],
                    worker_share,
                    policy,
                )
                if tentative is not None:
                    candidates.append(
                        (
                            counts[gpu.index] == 0,
                            tentative.micro_batch_size,
                            budgets[gpu.index],
                            gpu,
                            tentative,
                        )
                    )
            if not candidates:
                continue
            _, _, _, gpu, plan = max(
                candidates, key=lambda item: (item[0], item[1], item[2])
            )
            plans.append(plan)
            budgets[gpu.index] -= plan.estimated_peak_memory_gb
            counts[gpu.index] += 1
            exclusive[gpu.index] = plan.exclusive
            placed.append(job)

        if not placed:
            names = ", ".join(job.name for job in pending)
            raise RuntimeError(
                f"no eligible GPU has enough safe memory for pending jobs: {names}"
            )
        placed_ids = {id(job) for job in placed}
        pending = [job for job in pending if id(job) not in placed_ids]
        wave += 1

    jobs_per_wave: dict[int, int] = {}
    for plan in plans:
        jobs_per_wave[plan.wave] = jobs_per_wave.get(plan.wave, 0) + 1
    adjusted = []
    for plan in plans:
        workers = max(
            1,
            min(
                policy.max_workers_per_job,
                physical_cores // jobs_per_wave[plan.wave],
            ),
        )
        adjusted.append(replace(plan, dataloader_workers=workers))
    return adjusted


def recommend_next_launch(
    observation: RuntimeObservation,
    policy: ResourcePolicy | None = None,
) -> dict[str, Any]:
    """Return recorded recommendations; never mutate a live training run."""

    policy = policy or ResourcePolicy()
    actions: list[str] = []
    worker_delta = 0
    micro_batch_scale = 1.0
    if (
        observation.dataloader_wait_fraction is not None
        and observation.dataloader_wait_fraction > 0.10
        and (observation.cpu_utilization_percent or 0.0) < 85.0
        and (observation.available_ram_fraction or 0.0) > 0.20
    ):
        worker_delta = 2
        actions.append("increase_dataloader_workers")
    if (
        observation.gpu_utilization_percent is not None
        and observation.gpu_utilization_percent < 70.0
        and observation.peak_memory_fraction is not None
        and observation.peak_memory_fraction < 0.70
    ):
        micro_batch_scale = 1.25
        actions.append("increase_micro_batch_after_recalibration")
    if (
        observation.peak_memory_fraction is not None
        and observation.peak_memory_fraction > policy.target_memory_fraction
    ):
        micro_batch_scale = 0.8
        actions.append("decrease_micro_batch_and_preserve_effective_batch")
    if (
        observation.cpu_utilization_percent is not None
        and observation.cpu_utilization_percent > 95.0
    ):
        worker_delta = min(worker_delta, -1)
        actions.append("reduce_workers_to_avoid_cpu_oversubscription")
    return {
        "apply_when": "next_launch",
        "actions": actions or ["keep_plan"],
        "worker_delta": worker_delta,
        "micro_batch_scale": micro_batch_scale,
        "effective_batch_must_remain_constant": True,
    }


def calibrate_micro_batch(
    probe: Callable[[int], tuple[float, float]],
    *,
    global_batch_size: int,
    max_micro_batch_size: int,
    memory_limit_gb: float,
) -> dict[str, Any]:
    """Probe divisors from large to small.

    The callback returns `(peak_memory_gb, step_seconds)` and may raise an OOM
    RuntimeError.  The fastest safe candidate is selected.
    """

    results = []
    for candidate in range(min(global_batch_size, max_micro_batch_size), 0, -1):
        if global_batch_size % candidate:
            continue
        try:
            peak, seconds = probe(candidate)
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                continue
            raise
        if peak <= memory_limit_gb:
            results.append(
                {
                    "micro_batch_size": candidate,
                    "gradient_accumulation_steps": global_batch_size // candidate,
                    "peak_memory_gb": peak,
                    "step_seconds": seconds,
                }
            )
    if not results:
        raise RuntimeError("no safe micro-batch candidate found")
    return min(results, key=lambda item: item["step_seconds"])


def write_resource_outputs(
    output_root: str | Path,
    snapshot: ResourceSnapshot,
    plans: Iterable[JobPlan],
) -> tuple[Path, Path]:
    import json

    resource_dir = Path(output_root) / "resources"
    resource_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = resource_dir / "snapshot.json"
    plan_path = resource_dir / "plan.json"
    snapshot_path.write_text(
        json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plan_path.write_text(
        json.dumps(
            [plan.to_dict() for plan in plans], ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    return snapshot_path, plan_path
