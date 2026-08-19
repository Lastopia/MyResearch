"""Separate generated data, checkpoints, and lightweight run outputs."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StorageRoots:
    data: Path
    checkpoints: Path
    output: Path


def storage_roots(cfg: dict[str, Any], output_root: str | Path) -> StorageRoots:
    """Resolve sibling roots while preserving custom ``--output-root`` tests/runs."""

    output = Path(output_root).resolve()
    configured_output = Path(cfg["paths"]["output_root"]).resolve()
    if output == configured_output:
        data = Path(cfg["paths"]["data_root"]).resolve()
        checkpoints = Path(cfg["paths"]["checkpoint_root"]).resolve()
    else:
        data = output.parent / "data"
        checkpoints = output.parent / "ckpt"
    return StorageRoots(data=data, checkpoints=checkpoints, output=output)


def checkpoint_dir_for_run(
    cfg: dict[str, Any], output_root: str | Path, run_dir: str | Path
) -> Path:
    roots = storage_roots(cfg, output_root)
    run = Path(run_dir).resolve()
    runs_root = roots.output / "runs"
    try:
        relative = run.relative_to(runs_root)
    except ValueError as error:
        raise ValueError(f"run directory is outside {runs_root}: {run}") from error
    return roots.checkpoints / relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_empty_tree(path: Path) -> None:
    if not path.exists():
        return
    for directory in sorted(
        (item for item in path.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        path.rmdir()
    except OSError:
        pass


def _move_tree(source: Path, destination: Path) -> int:
    """Move a tree without silently overwriting a different existing file."""

    if not source.exists() or source.resolve() == destination.resolve():
        return 0
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            source.replace(destination)
        except OSError:
            shutil.move(str(source), str(destination))
        return sum(1 for item in destination.rglob("*") if item.is_file())

    moved = 0
    for source_file in (item for item in source.rglob("*") if item.is_file()):
        relative = source_file.relative_to(source)
        destination_file = destination / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        if destination_file.exists():
            if (
                source_file.stat().st_size == destination_file.stat().st_size
                and _sha256(source_file) == _sha256(destination_file)
            ):
                source_file.unlink()
                continue
            raise RuntimeError(
                "storage migration conflict; refusing to overwrite: "
                f"{source_file} -> {destination_file}"
            )
        try:
            source_file.replace(destination_file)
        except OSError:
            shutil.move(str(source_file), str(destination_file))
        moved += 1
    _remove_empty_tree(source)
    return moved


def prepare_run_checkpoint_dir(
    cfg: dict[str, Any], output_root: str | Path, run_dir: str | Path
) -> Path:
    """Create the external checkpoint directory and migrate this run's legacy one."""

    run = Path(run_dir).resolve()
    checkpoint_dir = checkpoint_dir_for_run(cfg, output_root, run)
    _move_tree(run / "checkpoints", checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pointer = run / "checkpoint_location.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        json.dumps({"checkpoint_dir": str(checkpoint_dir)}, indent=2),
        encoding="utf-8",
    )
    return checkpoint_dir


def prepare_storage(cfg: dict[str, Any], output_root: str | Path) -> StorageRoots:
    """Create the three roots and migrate legacy ``output/data``/checkpoints."""

    roots = storage_roots(cfg, output_root)
    roots.output.mkdir(parents=True, exist_ok=True)
    moved_data = _move_tree(roots.output / "data", roots.data)
    roots.data.mkdir(parents=True, exist_ok=True)
    roots.checkpoints.mkdir(parents=True, exist_ok=True)

    # Older diagnostics sometimes nested a cache below e.g.
    # ``output/real_clutrr_check/data``. Keep output strictly result-only too.
    nested_data_dirs = [
        path
        for path in roots.output.rglob("data")
        if path.is_dir() and path != roots.output / "data"
    ]
    for legacy_data in nested_data_dirs:
        relative_parent = legacy_data.parent.relative_to(roots.output)
        destination = roots.data / "legacy_output" / relative_parent
        moved_data += _move_tree(legacy_data, destination)
        _remove_empty_tree(legacy_data.parent)

    moved_checkpoints = 0
    runs_root = roots.output / "runs"
    if runs_root.exists():
        legacy_dirs = [
            path
            for path in runs_root.rglob("checkpoints")
            if path.is_dir()
        ]
        for legacy in legacy_dirs:
            run_dir = legacy.parent
            destination = checkpoint_dir_for_run(cfg, roots.output, run_dir)
            moved_checkpoints += _move_tree(legacy, destination)
            pointer = run_dir / "checkpoint_location.json"
            pointer.write_text(
                json.dumps({"checkpoint_dir": str(destination)}, indent=2),
                encoding="utf-8",
            )

    layout_path = roots.output / "storage_layout.json"
    previous_layout: dict[str, Any] = {}
    if layout_path.exists():
        try:
            previous_layout = json.loads(layout_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_layout = {}
    migrated_legacy = bool(
        previous_layout.get("migrated_legacy_layout")
        or moved_data
        or moved_checkpoints
    )
    layout_path.write_text(
        json.dumps(
            {
                "layout_version": 2,
                "data_root": str(roots.data),
                "checkpoint_root": str(roots.checkpoints),
                "output_root": str(roots.output),
                "migrated_legacy_layout": migrated_legacy,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if moved_data or moved_checkpoints:
        print(
            "storage migration"
            f" | data files {moved_data}"
            f" | checkpoint files {moved_checkpoints}"
        )
    return roots


def legacy_layout_was_migrated(output_root: str | Path) -> bool:
    layout_path = Path(output_root).resolve() / "storage_layout.json"
    if not layout_path.exists():
        return False
    try:
        payload = json.loads(layout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("migrated_legacy_layout"))


def legacy_compatibility_pending(output_root: str | Path) -> bool:
    layout_path = Path(output_root).resolve() / "storage_layout.json"
    if not layout_path.exists():
        return False
    try:
        payload = json.loads(layout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("migrated_legacy_layout")) and not bool(
        payload.get("compatibility_consumed_by_source")
    )


def consume_legacy_compatibility(
    output_root: str | Path, source_fingerprint: str
) -> None:
    layout_path = Path(output_root).resolve() / "storage_layout.json"
    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    payload["compatibility_consumed_by_source"] = source_fingerprint
    layout_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def legacy_compatibility_active(
    output_root: str | Path, source_fingerprint: str
) -> bool:
    layout_path = Path(output_root).resolve() / "storage_layout.json"
    if not layout_path.exists():
        return False
    try:
        payload = json.loads(layout_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("migrated_legacy_layout")) and (
        payload.get("compatibility_consumed_by_source") == source_fingerprint
    )
