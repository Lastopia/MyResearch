from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from typing import Any

from tools.log import log_fields, stage_banner


def missing_dependencies(cfg: dict[str, Any]) -> dict[str, str]:
    required = cfg["bootstrap"]["required_packages"]
    return {
        module: requirement
        for module, requirement in required.items()
        if importlib.util.find_spec(module) is None
    }


def _physical_gpu_count() -> int:
    if shutil.which("nvidia-smi") is None:
        return 0
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return len([line for line in completed.stdout.splitlines() if line.strip()])
    except (OSError, subprocess.SubprocessError):
        return 0


def _torch_status() -> dict[str, Any]:
    code = (
        "import json,torch;"
        "print(json.dumps({"
        "'version':torch.__version__,"
        "'cuda_available':torch.cuda.is_available(),"
        "'cuda_version':torch.version.cuda"
        "}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(completed.stdout.strip())


def _ensure_cuda_torch(cfg: dict[str, Any]) -> dict[str, Any]:
    physical_gpus = _physical_gpu_count()
    status = _torch_status()
    if physical_gpus == 0 or bool(status["cuda_available"]):
        if bool(cfg["run"].get("require_cuda", False)) and not bool(
            status["cuda_available"]
        ):
            raise RuntimeError(
                "This experiment requires CUDA, but torch.cuda.is_available() is false"
            )
        return {"physical_gpus": physical_gpus, **status, "cuda_repaired": False}
    if not bool(cfg["bootstrap"]["install_cuda_torch_if_gpu_detected"]):
        if bool(cfg["run"].get("require_cuda", False)):
            raise RuntimeError(
                "This experiment requires CUDA, but the installed PyTorch build has no CUDA. "
                "Install the server's CUDA-enabled torch before running it."
            )
        log_fields(
            "bootstrap",
            cfg=cfg,
            status="warning",
            issue="physical_gpu_present_but_torch_cuda_unavailable",
        )
        return {"physical_gpus": physical_gpus, **status, "cuda_repaired": False}

    version = str(status["version"]).split("+", 1)[0]
    index_url = str(cfg["bootstrap"]["cuda_torch_index_url"])
    log_fields(
        "bootstrap",
        cfg=cfg,
        status="installing_cuda_torch",
        torch_version=version,
        index_url=index_url,
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            f"torch=={version}",
            "--index-url",
            index_url,
        ]
    )
    repaired = _torch_status()
    if not bool(repaired["cuda_available"]):
        raise RuntimeError(
            "CUDA PyTorch installation finished, but torch.cuda.is_available() "
            "is still false"
        )
    return {
        "physical_gpus": physical_gpus,
        **repaired,
        "cuda_repaired": True,
    }


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    stage_banner("BOOTSTRAP", cfg=cfg)
    missing = missing_dependencies(cfg)
    if missing and not bool(cfg["bootstrap"]["install_missing_dependencies"]):
        names = ", ".join(missing)
        raise RuntimeError(
            f"Missing dependencies: {names}. "
            "Enable bootstrap.install_missing_dependencies or install requirements.txt."
        )

    requirements = list(missing.values())
    if requirements:
        log_fields(
            "bootstrap",
            cfg=cfg,
            status="installing",
            packages=",".join(requirements),
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *requirements]
        )
        unresolved = missing_dependencies(cfg)
        if unresolved:
            raise RuntimeError(
                "Dependency installation completed but imports are still missing: "
                + ", ".join(unresolved)
            )
    cuda = _ensure_cuda_torch(cfg)
    result = {
        "status": "installed" if requirements or cuda["cuda_repaired"] else "ready",
        "missing": list(missing),
        "installed": requirements,
        "cuda": cuda,
    }
    log_fields(
        "bootstrap",
        cfg=cfg,
        status=result["status"],
        missing=len(missing),
        physical_gpus=cuda["physical_gpus"],
        torch=repr(cuda["version"]),
        torch_cuda=cuda["cuda_available"],
        cuda_runtime=cuda["cuda_version"],
    )
    stage_banner("BOOTSTRAP", "DONE", cfg=cfg)
    return result
