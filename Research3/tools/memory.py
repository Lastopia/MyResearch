from __future__ import annotations

import ctypes
import importlib.metadata
import os
import platform
try:
    import resource as _resource
except ImportError:  # pragma: no cover - Windows
    _resource = None
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch


def synchronize(device: torch.device | str | None = None) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


@contextmanager
def elapsed_timer(device: torch.device | str | None = None) -> Iterator[dict[str, float]]:
    synchronize(device)
    state: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield state
    finally:
        synchronize(device)
        state["seconds"] = time.perf_counter() - start


def reset_peak_vram(device: torch.device | str | None = None) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_vram_gb(device: torch.device | str | None = None) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**3)


def vram_usage_gb(
    device: torch.device | str | None = None,
) -> tuple[float | None, float | None]:
    if not torch.cuda.is_available():
        return None, None
    resolved = torch.device(device or "cuda")
    if resolved.type != "cuda":
        return None, None
    index = (
        torch.cuda.current_device()
        if resolved.index is None
        else int(resolved.index)
    )
    used = torch.cuda.memory_reserved(index) / (1024**3)
    total = torch.cuda.get_device_properties(index).total_memory / (1024**3)
    return used, total


def _windows_memory_gb(peak: bool) -> float | None:
    if platform.system() != "Windows":
        return None

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    handle = get_current_process()
    get_process_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ulong,
    ]
    get_process_memory.restype = ctypes.c_int
    ok = get_process_memory(
        handle, ctypes.byref(counters), counters.cb
    )
    if not ok:
        return None
    value = counters.PeakWorkingSetSize if peak else counters.WorkingSetSize
    return float(value) / (1024**3)


def process_rss_gb(peak: bool = False) -> float | None:
    if peak and platform.system() != "Windows" and _resource is not None:
        # Linux reports ru_maxrss in KiB; macOS reports bytes.  This is a
        # process high-water mark, unlike psutil's current RSS value.
        try:
            value = float(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
            if platform.system() == "Darwin":
                return value / (1024**3)
            return value / (1024**2)
        except (AttributeError, OSError, ValueError):
            pass
    try:
        import psutil

        info = psutil.Process(os.getpid()).memory_info()
        value = getattr(info, "peak_wset", info.rss) if peak else info.rss
        return float(value) / (1024**3)
    except ImportError:
        return _windows_memory_gb(peak)


@contextmanager
def phase_peak_rss_gb(
    poll_interval_seconds: float = 0.01,
) -> Iterator[dict[str, float | None]]:
    """Measure current-process RSS peak inside one explicit phase window."""
    state: dict[str, float | None] = {"peak_rss_gb": process_rss_gb()}
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(max(0.001, poll_interval_seconds)):
            current = process_rss_gb()
            if current is None:
                continue
            previous = state["peak_rss_gb"]
            state["peak_rss_gb"] = (
                current if previous is None else max(previous, current)
            )

    worker = threading.Thread(
        target=sample,
        name="phase-rss-sampler",
        daemon=True,
    )
    worker.start()
    try:
        yield state
    finally:
        current = process_rss_gb()
        if current is not None:
            previous = state["peak_rss_gb"]
            state["peak_rss_gb"] = (
                current if previous is None else max(previous, current)
            )
        stop.set()
        worker.join(timeout=1.0)


def _system_memory() -> dict[str, float | None]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {
            "ram_total_gb": float(memory.total) / (1024**3),
            "ram_used_gb": float(memory.used) / (1024**3),
            "ram_available_gb": float(memory.available) / (1024**3),
            "ram_percent": float(memory.percent),
        }
    except ImportError:
        if platform.system() == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                total = float(status.ullTotalPhys)
                available = float(status.ullAvailPhys)
                return {
                    "ram_total_gb": total / (1024**3),
                    "ram_used_gb": (total - available) / (1024**3),
                    "ram_available_gb": available / (1024**3),
                    "ram_percent": float(status.dwMemoryLoad),
                }
        return {
            "ram_total_gb": None,
            "ram_used_gb": None,
            "ram_available_gb": None,
            "ram_percent": None,
        }


def system_ram_total_gb() -> float | None:
    return _system_memory()["ram_total_gb"]


def _nvidia_smi_gpus() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
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
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    result: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        try:
            index, name, total_mb, used_mb, free_mb, utilization = fields
            result.append(
                {
                    "index": int(index),
                    "name": name,
                    "vram_total_gb": float(total_mb) / 1024.0,
                    "vram_used_gb": float(used_mb) / 1024.0,
                    "vram_free_gb": float(free_mb) / 1024.0,
                    "utilization_percent": float(utilization),
                }
            )
        except ValueError:
            continue
    return result


def _torch_gpus() -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        return []
    result: list[dict[str, object]] = []
    for index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        result.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "vram_total_gb": total_bytes / (1024**3),
                "vram_used_gb": (total_bytes - free_bytes) / (1024**3),
                "vram_free_gb": free_bytes / (1024**3),
                "utilization_percent": None,
            }
        )
    return result


def _assigned_gpu_indices(device: torch.device | str | None) -> list[int]:
    if device is None:
        return []
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return []
    return [
        torch.cuda.current_device()
        if resolved.index is None
        else int(resolved.index)
    ]


def resource_snapshot(
    device: torch.device | str | None = None,
    *,
    workspace: str | Path | None = None,
) -> dict[str, object]:
    gpus = _nvidia_smi_gpus() or _torch_gpus()
    assigned = _assigned_gpu_indices(device)
    for gpu in gpus:
        index = int(gpu["index"])
        gpu["assigned_to_task"] = index in assigned
        if torch.cuda.is_available() and index < torch.cuda.device_count():
            gpu["process_allocated_gb"] = (
                torch.cuda.memory_allocated(index) / (1024**3)
            )
            gpu["process_reserved_gb"] = (
                torch.cuda.memory_reserved(index) / (1024**3)
            )
        else:
            gpu["process_allocated_gb"] = 0.0
            gpu["process_reserved_gb"] = 0.0

    available = (
        [
            gpu
            for gpu in gpus
            if float(gpu["vram_free_gb"])
            >= max(1.0, 0.15 * float(gpu["vram_total_gb"]))
        ]
        if torch.cuda.is_available()
        else []
    )
    busy = [
        gpu
        for gpu in gpus
        if float(gpu["vram_used_gb"]) > 0.5
        or float(gpu.get("utilization_percent") or 0.0) > 5.0
    ]
    snapshot: dict[str, object] = {
        "gpu_total_count": len(gpus),
        "gpu_available_count": len(available),
        "gpu_busy_count": len(busy),
        "gpu_assigned_count": len(assigned),
        "gpu_assigned_indices": assigned,
        "gpus": gpus,
        "process_ram_gb": process_rss_gb(),
        **_system_memory(),
    }
    if workspace is not None:
        usage = shutil.disk_usage(Path(workspace))
        snapshot.update(
            {
                "disk_total_gb": usage.total / (1024**3),
                "disk_used_gb": usage.used / (1024**3),
                "disk_free_gb": usage.free / (1024**3),
            }
        )
    return snapshot


def hardware_metadata() -> dict[str, object]:
    snapshot = resource_snapshot()
    package_versions: dict[str, str | None] = {}
    for package in ("numpy", "datasets", "transformers", "psutil"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        ),
        "package_versions": package_versions,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "resources": snapshot,
    }
