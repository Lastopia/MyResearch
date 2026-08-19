from __future__ import annotations

import os
import random
import warnings

import torch


def _suppress_known_determinism_warnings() -> None:
    """Keep expected CUDA determinism warnings out of experiment logs."""
    warnings.filterwarnings(
        "ignore",
        message=r"Deterministic behavior was enabled with either .*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=(
            r"cumsum_cuda_kernel does not have a deterministic "
            r"implementation.*"
        ),
        category=UserWarning,
    )


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        _suppress_known_determinism_warnings()
        torch.use_deterministic_algorithms(True, warn_only=True)
