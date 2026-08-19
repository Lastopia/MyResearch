import ctypes
from collections import deque
import os
import time

import torch


def _gb(x):
    return x / (1024 ** 3)


def gpu_label(index):
    return f"GPU{index + 1}"


def cuda_device(index):
    return f"cuda:{index}"


def gpu_mem(index):
    free, total = torch.cuda.mem_get_info(index)
    used_driver = total - free
    reserved = torch.cuda.memory_reserved(index)
    used = max(used_driver, reserved)
    return f"{_gb(used):.1f}/{_gb(total):.0f}G"


def gpu_total_gb(index):
    _, total = torch.cuda.mem_get_info(index)
    return _gb(total)


def gpu_free_gb(index):
    free, _ = torch.cuda.mem_get_info(index)
    return _gb(free)


def check_gpu():
    if not torch.cuda.is_available():
        print("[GPU1] : 0/0G   status = not available")
        return []
    available = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        status = "available" if free / total > 0.2 else "not available"
        print(f"[{gpu_label(i)}] : {gpu_mem(i)}   status = {status}")
        if status == "available":
            available.append(i)
    return available


def configured_gpus(cfg):
    if not torch.cuda.is_available():
        return []
    gpus = cfg.get("run", {}).get("gpus")
    if gpus:
        return [int(i) for i in gpus]
    return list(range(torch.cuda.device_count()))


def job_gpu(gpus, index, jobs_per_gpu=1):
    if not gpus:
        return 0
    slots_per_gpu = max(1, int(jobs_per_gpu))
    return gpus[(index // slots_per_gpu) % len(gpus)]


def resolve_jobs_per_gpu(cfg, gpus, stage=None):
    value = cfg.get("run", {}).get("jobs_per_gpu", 1)
    if not (isinstance(value, str) and value.lower() == "auto"):
        return max(1, int(value))
    if not gpus or not torch.cuda.is_available():
        return 1
    if cfg.get("run", {}).get("mode", "retrain") == "pretrain":
        return 1

    min_total = min(gpu_total_gb(index) for index in gpus)
    min_free = min(gpu_free_gb(index) for index in gpus)
    batch_size = int(cfg.get("train", {}).get("batch_size", 1))
    block_size = int(cfg.get("data", {}).get("block_size", cfg.get("model", {}).get("block_size", 256)))
    d_model = int(cfg.get("model", {}).get("d_model", 384))
    n_layer = int(cfg.get("model", {}).get("n_layer", 6))
    token_scale = max(1.0, batch_size * block_size * d_model * n_layer / (2 * 1024 * 384 * 6))

    if min_total >= 120:
        base = 8
    elif min_total >= 80:
        base = 6
    elif min_total >= 40:
        base = 4
    elif min_total >= 24:
        base = 2
    else:
        base = 1
    scheduler_cfg = cfg.get("run", {}).get("gpu_scheduler", {})
    base = min(base, max(1, int(scheduler_cfg.get("max_jobs_per_gpu", base))))
    if token_scale >= 8:
        base = max(1, base // 2)
    if min_free < 32:
        base = 1
    elif min_free < 64:
        base = min(base, 2)
    if stage == "sae":
        base = max(1, min(base, 3))
    return base


def parallel_slots(gpus, jobs_per_gpu):
    if not gpus:
        return 1
    return max(1, len(gpus) * max(1, int(jobs_per_gpu)))


def _auto_jobs_enabled(cfg):
    value = cfg.get("run", {}).get("jobs_per_gpu", 1)
    return isinstance(value, str) and value.lower() == "auto"


def _gpu_utilization(index):
    try:
        return float(torch.cuda.utilization(index))
    except Exception:
        return None


def _safe_gpu_memory(index):
    try:
        return gpu_free_gb(index), gpu_total_gb(index)
    except Exception:
        return None, None


def _adaptive_launch_allowed(cfg, gpu_index, state, hard_limit, now):
    if state["active"] <= 0:
        return True
    if state["active"] >= hard_limit:
        return False

    scheduler_cfg = cfg.get("run", {}).get("gpu_scheduler", {})
    settle_seconds = max(0.0, float(scheduler_cfg.get("settle_seconds", 3.0)))
    if now - state["last_launch"] < settle_seconds:
        return False

    free_gb, total_gb = _safe_gpu_memory(gpu_index)
    if free_gb is not None:
        before = state.get("free_before")
        observed = None if before is None else before - free_gb
        if observed is not None and observed >= 0.25:
            previous = state.get("job_memory_gb") or 0.0
            state["job_memory_gb"] = max(previous, observed)

        reserve_fraction = max(
            0.0, float(scheduler_cfg.get("memory_reserve_fraction", 0.10)),
        )
        reserve_gb = max(
            float(scheduler_cfg.get("min_memory_reserve_gb", 8.0)),
            total_gb * reserve_fraction,
        )
        safety_factor = max(
            1.0, float(scheduler_cfg.get("memory_safety_factor", 1.35)),
        )
        estimated_job_gb = state.get("job_memory_gb")
        if estimated_job_gb is None:
            estimated_job_gb = float(
                scheduler_cfg.get("unprofiled_job_memory_gb", 8.0),
            )
        if free_gb < reserve_gb + estimated_job_gb * safety_factor:
            return False

    utilization = _gpu_utilization(gpu_index)
    if utilization is None:
        state["utilization_samples"] = []
        fallback = max(
            1, int(scheduler_cfg.get("fallback_jobs_per_gpu", min(4, hard_limit))),
        )
        return state["active"] < min(hard_limit, fallback)
    sample_count = max(1, int(scheduler_cfg.get("utilization_samples", 5)))
    samples = state.setdefault("utilization_samples", [])
    samples.append(utilization)
    if len(samples) > sample_count:
        del samples[:-sample_count]
    if len(samples) < sample_count:
        return False
    target = min(
        100.0, max(1.0, float(scheduler_cfg.get("utilization_target", 90.0))),
    )
    return sum(samples) / len(samples) < target


def run_gpu_jobs(cfg, jobs, target, args_for_job, gpus, stage):
    """Run independent jobs with immediate refill and adaptive GPU concurrency.

    ``args_for_job`` is called in the parent as ``args_for_job(job, gpu_index)``
    and must return the positional arguments for ``target``. The scheduler only
    changes process placement; each job's training configuration and RNG setup
    remain untouched.
    """
    import multiprocessing as mp

    jobs = list(jobs)
    if not jobs:
        return []
    if not gpus:
        for job in jobs:
            target(*args_for_job(job, 0))
        return []

    hard_limit = resolve_jobs_per_gpu(cfg, gpus, stage=stage)
    if len(jobs) == 1:
        target(*args_for_job(jobs[0], gpus[0]))
        return []

    adaptive = _auto_jobs_enabled(cfg)
    scheduler_cfg = cfg.get("run", {}).get("gpu_scheduler", {})
    poll_seconds = max(0.05, float(scheduler_cfg.get("poll_seconds", 0.2)))
    pending = deque(enumerate(jobs))
    active = []
    failed = []
    gpu_state = {
        gpu_index: {
            "active": 0,
            "last_launch": 0.0,
            "free_before": None,
            "job_memory_gb": None,
            "utilization_samples": [],
        }
        for gpu_index in gpus
    }
    ctx = mp.get_context("spawn")
    next_gpu = 0

    try:
        while pending or active:
            completed = False
            survivors = []
            for item in active:
                process = item["process"]
                if process.is_alive():
                    survivors.append(item)
                    continue
                process.join()
                exitcode = process.exitcode
                process.close()
                gpu_state[item["gpu"]]["active"] -= 1
                gpu_state[item["gpu"]]["last_launch"] = 0.0
                if exitcode != 0:
                    failed.append({
                        "job_index": item["job_index"],
                        "job": repr(item["job"]),
                        "gpu": item["gpu"],
                        "exitcode": exitcode,
                    })
                completed = True
            active = survivors

            launched = False
            checked = 0
            while pending and checked < len(gpus):
                gpu_index = gpus[next_gpu % len(gpus)]
                next_gpu = (next_gpu + 1) % len(gpus)
                checked += 1
                state = gpu_state[gpu_index]
                allowed = (
                    _adaptive_launch_allowed(
                        cfg, gpu_index, state, hard_limit, time.monotonic(),
                    )
                    if adaptive else state["active"] < hard_limit
                )
                if not allowed:
                    continue

                job_index, job = pending.popleft()
                free_before, _ = _safe_gpu_memory(gpu_index)
                process = ctx.Process(
                    target=target,
                    args=tuple(args_for_job(job, gpu_index)),
                )
                process.start()
                state["active"] += 1
                state["last_launch"] = time.monotonic()
                state["free_before"] = free_before
                state["utilization_samples"] = []
                active.append({
                    "process": process,
                    "gpu": gpu_index,
                    "job_index": job_index,
                    "job": job,
                })
                launched = True
                checked = 0

            if active and not launched and not completed:
                time.sleep(poll_seconds)
    except BaseException:
        for item in active:
            process = item["process"]
            if process.is_alive():
                process.terminate()
            process.join()
            process.close()
        raise
    return failed


def allocated_cpu_count():
    for name in ["PBS_NP", "NCPUS", "SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK", "NSLOTS"]:
        value = os.environ.get(name)
        if value and value.isdigit() and int(value) > 0:
            return int(value)
    nodefile = os.environ.get("PBS_NODEFILE")
    if nodefile and os.path.exists(nodefile):
        try:
            with open(nodefile, "r", encoding="utf-8") as f:
                return max(1, sum(1 for line in f if line.strip()))
        except OSError:
            pass
    return os.cpu_count() or 1


def auto_num_threads(n_gpu, jobs_per_gpu=1):
    cpu_count = allocated_cpu_count()
    if n_gpu <= 0:
        return max(1, min(8, cpu_count))
    workers = max(1, n_gpu * max(1, int(jobs_per_gpu)))
    return max(1, min(8, cpu_count // workers))


def _mem_windows():
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
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullTotalPhys - stat.ullAvailPhys, stat.ullTotalPhys


def _mem_posix():
    pages = os.sysconf("SC_PHYS_PAGES")
    avail = os.sysconf("SC_AVPHYS_PAGES")
    size = os.sysconf("SC_PAGE_SIZE")
    total = pages * size
    used = (pages - avail) * size
    return used, total


def mem_usage():
    try:
        used, total = _mem_windows() if os.name == "nt" else _mem_posix()
        return f"{_gb(used):.1f}/{_gb(total):.0f}G"
    except Exception:
        return "NA"


def vram_usage(device):
    if not torch.cuda.is_available() or device.type != "cuda":
        return "NA"
    return gpu_mem(device.index or 0)
