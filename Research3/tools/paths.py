from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROJECT_ROOT_ENV = "POSITION_BIAS_PROJECT_ROOT"


def workspace_root(cfg: dict[str, Any] | None = None) -> Path:
    """Resolve the project root without assuming the developer's local path.

    Priority is explicit config, then ``POSITION_BIAS_PROJECT_ROOT``, then the
    repository directory.  The last fallback only exists so smoke tests can be
    run from a checkout; server jobs should set one of the first two.
    """
    configured = None
    if cfg is not None:
        configured = cfg.get("paths", {}).get("project_root")
    configured = configured or os.environ.get(PROJECT_ROOT_ENV)
    if configured:
        return Path(str(configured)).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def task_name(cfg: dict[str, Any]) -> str:
    return str(cfg["run"]["task"])


def method_name(cfg: dict[str, Any]) -> str:
    return str(cfg["run"]["method"])


def seed_name(cfg: dict[str, Any]) -> str:
    return f"seed{int(cfg['run']['seed'])}"


def data_dir(cfg: dict[str, Any]) -> Path:
    data_root = cfg.get("paths", {}).get("data_root") or "data"
    return workspace_root(cfg) / str(data_root) / task_name(cfg)


def asset_dir(cfg: dict[str, Any]) -> Path:
    data_root = cfg.get("paths", {}).get("data_root") or "data"
    return (
        workspace_root(cfg)
        / str(data_root)
        / "assets"
    )


def tokenizer_dir(cfg: dict[str, Any]) -> Path:
    revision = str(cfg["data"]["tokenizer_revision"]).replace("/", "_")
    return asset_dir(cfg) / "tokenizers" / "gpt2" / revision


def huggingface_cache_dir(cfg: dict[str, Any]) -> Path:
    return asset_dir(cfg) / "huggingface_cache"


def wikitext_dir(cfg: dict[str, Any]) -> Path:
    return asset_dir(cfg) / "wikitext103"


def qasper_dir(cfg: dict[str, Any]) -> Path:
    revision = str(cfg["data"]["qasper_revision"]).replace("/", "_")
    return asset_dir(cfg) / "qasper" / revision


def checkpoint_dir(cfg: dict[str, Any]) -> Path:
    return (
        workspace_root(cfg)
        / str(cfg.get("paths", {}).get("checkpoint_root", "checkpoints"))
        / task_name(cfg)
        / method_name(cfg)
        / seed_name(cfg)
    )


def large_dir(cfg: dict[str, Any], category: str | None = None) -> Path:
    path = (
        workspace_root(cfg)
        / str(cfg.get("paths", {}).get("large_root", "large"))
        / task_name(cfg)
        / method_name(cfg)
        / seed_name(cfg)
    )
    return path / category if category else path


def output_dir(cfg: dict[str, Any]) -> Path:
    return (
        workspace_root(cfg)
        / str(cfg.get("paths", {}).get("output_root", "output"))
        / task_name(cfg)
    )


def metric_dir(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "metrics" / method_name(cfg) / seed_name(cfg)


def audit_dir(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "audits" / method_name(cfg) / seed_name(cfg)


def profile_dir(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "profiles" / method_name(cfg) / seed_name(cfg)


def log_dir(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "logs" / method_name(cfg) / seed_name(cfg)


def config_path(cfg: dict[str, Any]) -> Path:
    return (
        output_dir(cfg)
        / "configs"
        / f"{method_name(cfg)}_{seed_name(cfg)}.json"
    )


def resource_plan_path(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "resources" / "plan.json"


def resource_log_path(cfg: dict[str, Any]) -> Path:
    return log_dir(cfg) / "resources.jsonl"
