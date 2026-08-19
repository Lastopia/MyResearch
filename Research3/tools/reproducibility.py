from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from tools.io import write_json
from tools.log import utc_timestamp
from tools.paths import config_path, workspace_root


def _git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[str(name).lower()] = distribution.version
    return dict(sorted(packages.items()))


def _pip_freeze() -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pip freeze failed: {result.stderr.strip()}")
    return sorted(
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    )


def _nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    versions = sorted(
        {
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        }
    )
    return ",".join(versions) if result.returncode == 0 and versions else None


def _gpu_inventory() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    return [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_bytes": torch.cuda.get_device_properties(
                index
            ).total_memory,
            "compute_capability": list(
                torch.cuda.get_device_capability(index)
            ),
        }
        for index in range(torch.cuda.device_count())
    ]


def write_reproducibility_manifest(cfg: dict[str, Any]) -> Path:
    root = workspace_root(cfg)
    status = _git_value(root, "status", "--porcelain")
    frozen_requirements = _pip_freeze()
    target = config_path(cfg).with_suffix(".environment.json")
    lock_path = config_path(cfg).with_suffix(".requirements-lock.txt")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        "\n".join(frozen_requirements) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "created_at": utc_timestamp(),
        "git": {
            "commit": _git_value(root, "rev-parse", "HEAD"),
            "branch": _git_value(root, "branch", "--show-current"),
            "dirty": bool(status),
            "status_porcelain": status or "",
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None
            ),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "determinism_policy": (
                "torch deterministic algorithms with warn_only=True; "
                "known CUDA cumsum nondeterminism is recorded as a limitation"
            ),
        },
        "nvidia_driver_version": _nvidia_driver_version(),
        "gpus": _gpu_inventory(),
        "packages": _packages(),
        "requirements_lock": str(lock_path),
        "experiment": {
            "task": cfg["run"]["task"],
            "method": cfg["run"]["method"],
            "seed": cfg["run"]["seed"],
            "dtype": cfg["run"]["dtype"],
            "device": cfg["run"]["device"],
        },
    }
    write_json(target, manifest)
    return target
