import csv
import json
import hashlib
import math
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | os.PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_dirs(paths) -> None:
    for path in paths:
        ensure_dir(path)


def save_json(obj, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stable_hash(obj) -> str:
    payload = json.dumps(namespace_to_dict(obj), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_is_current(manifest_path, config, output_paths) -> bool:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return False
    try:
        manifest = load_json(manifest_path)
    except Exception:
        return False
    if manifest.get("config_hash") != stable_hash(config):
        return False
    return all(Path(path).exists() for path in output_paths)


def valid_file(path: str | os.PathLike) -> bool:
    path = Path(path)
    return path.exists() and path.is_file() and path.stat().st_size > 0


def valid_torch_checkpoint(path: str | os.PathLike) -> bool:
    if not valid_file(path):
        return False
    try:
        torch.load(path, map_location="cpu")
    except Exception:
        return False
    return True


def save_manifest(manifest_path, stage, config, output_paths) -> None:
    save_json(
        {
            "stage": stage,
            "config_hash": stable_hash(config),
            "config": namespace_to_dict(config),
            "output_paths": [str(path) for path in output_paths],
        },
        manifest_path,
    )


def code_version_info():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
        return {"git_available": True, "commit": commit, "dirty": dirty}
    except Exception:
        return {"git_available": False, "commit": None, "dirty": None}


def runtime_environment_info():
    cuda_available = torch.cuda.is_available()
    gpus = []
    if cuda_available:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpus.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 3),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "code_version": code_version_info(),
    }


def namespace_to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {key: namespace_to_dict(value) for key, value in vars(obj).items()}
    if isinstance(obj, dict):
        return {key: namespace_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [namespace_to_dict(value) for value in obj]
    return obj


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def perplexity(loss: float) -> float:
    return float(math.exp(min(loss, 20.0)))


def mean_std(values) -> dict:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return {"mean": None, "std": None}
    mean = sum(clean) / len(clean)
    var = sum((value - mean) ** 2 for value in clean) / len(clean)
    return {"mean": mean, "std": math.sqrt(var)}
