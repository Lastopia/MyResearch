from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _delimiter(cfg: dict[str, Any] | None) -> str:
    if cfg is None:
        return " | "
    return str(cfg.get("logging", {}).get("delimiter", " | "))


def _width(cfg: dict[str, Any] | None) -> int:
    if cfg is None:
        return 80
    return int(cfg.get("logging", {}).get("banner_width", 120))


def task_display_name(cfg: dict[str, Any]) -> str:
    return str(
        cfg.get("logging", {}).get("task_name")
        or cfg["run"]["task"]
    )


def gpu_display_name(cfg: dict[str, Any]) -> str | None:
    device = str(cfg.get("run", {}).get("device", "")).lower()
    if not device.startswith("cuda"):
        return None
    index = device.split(":", 1)[1] if ":" in device else "0"
    return f"GPU{index}"


def stage_banner(
    stage: str,
    state: str = "START",
    *,
    cfg: dict[str, Any] | None = None,
) -> None:
    fields: dict[str, Any] = {"stage": stage.lower(), "state": state.lower()}
    if cfg is not None:
        fields = {
            "task": task_display_name(cfg),
            "method": cfg["run"].get("method"),
            "seed": cfg["run"].get("seed"),
            **fields,
        }
    line = "=" * _width(cfg)
    print(line, flush=True)
    log_fields("stage", cfg=cfg, **fields)
    print(line, flush=True)


def log_fields(
    prefix: str,
    *,
    cfg: dict[str, Any] | None = None,
    **fields: Any,
) -> None:
    base: dict[str, Any] = {}
    if cfg is not None:
        base = {
            "task": task_display_name(cfg),
        }
        gpu = gpu_display_name(cfg)
        if gpu is not None:
            base["gpu"] = gpu
    merged = {**base, **fields}
    ordered_keys = [
        key
        for key in ("task", "method", "model", "gpu")
        if key in merged
    ]
    ordered_keys.extend(key for key in merged if key not in ordered_keys)

    def formatted(key: str) -> str:
        value = merged[key]
        if key in {"task", "method", "model"}:
            return str(value)
        if key == "gpu":
            text = str(value)
            return text if text.upper().startswith("GPU") else f"GPU{text}"
        return f"{key}={value}"

    body = _delimiter(cfg).join(formatted(key) for key in ordered_keys)
    print(f"[{prefix.upper()}]{_delimiter(cfg)}{body}", flush=True)


def log_resources(
    cfg: dict[str, Any],
    stage: str,
    *,
    device: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    from tools.io import append_jsonl
    from tools.memory import resource_snapshot
    from tools.paths import resource_log_path, workspace_root

    snapshot = resource_snapshot(device, workspace=workspace_root(cfg))
    record = {
        "timestamp": utc_timestamp(),
        "task": cfg["run"]["task"],
        "method": cfg["run"].get("method"),
        "seed": cfg["run"].get("seed"),
        "stage": stage,
        **snapshot,
        **extra,
    }
    if bool(cfg.get("logging", {}).get("resource_jsonl", True)):
        append_jsonl(resource_log_path(cfg), record)
    return record
