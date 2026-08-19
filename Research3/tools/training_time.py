from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from tools.io import write_json
from tools.log import utc_timestamp


TIME_BASIS = "optimizer_step_wall_clock_v1"


def _empty_baseline() -> dict[str, Any]:
    return {
        "time_basis": TIME_BASIS,
        "wall_clock_seconds": 0.0,
        "gpu_hours": 0.0,
        "training_sessions": 0,
        "peak_vram_gb": 0.0,
        "peak_host_ram_gb": 0.0,
    }


def _validated_baseline(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None or value.get("time_basis") != TIME_BASIS:
        raise RuntimeError(
            "Checkpoint has no valid optimizer-step time snapshot. "
            "Delete it and start this experiment from zero."
        )
    return {
        "time_basis": TIME_BASIS,
        "wall_clock_seconds": max(
            0.0,
            float(value["wall_clock_seconds"]),
        ),
        "gpu_hours": max(0.0, float(value["gpu_hours"])),
        "training_sessions": max(
            0,
            int(value["training_sessions"]),
        ),
        "peak_vram_gb": max(
            0.0,
            float(value.get("peak_vram_gb", 0.0)),
        ),
        "peak_host_ram_gb": max(
            0.0,
            float(value.get("peak_host_ram_gb", 0.0)),
        ),
    }


class TrainingTimer:
    """Count only recoverable optimizer-step wall-clock time."""

    def __init__(
        self,
        path: str | Path,
        *,
        stage: str,
        method: str,
        seed: int,
        gpu_count: int,
        resumed_from: str | None,
        resume_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        baseline = (
            _validated_baseline(resume_snapshot)
            if resumed_from is not None
            else _empty_baseline()
        )
        self.ledger: dict[str, Any] = {
            "stage": stage,
            "method": method,
            "seed": int(seed),
            "time_basis": TIME_BASIS,
            "base": baseline,
            "last_checkpoint": (
                Path(resumed_from).name
                if resumed_from is not None
                else None
            ),
            "last_checkpoint_baseline": baseline,
            "sessions": [],
        }
        self.session: dict[str, Any] = {
            "session_id": f"{utc_timestamp()}-pid{os.getpid()}",
            "pid": os.getpid(),
            "started_at": utc_timestamp(),
            "finished_at": None,
            "status": "running",
            "resumed_from": resumed_from,
            "gpu_count": int(gpu_count),
            "elapsed_seconds": 0.0,
            "step": 0,
            "tokens_seen": 0,
            "peak_vram_gb": 0.0,
            "peak_host_ram_gb": 0.0,
        }
        self.ledger["sessions"].append(self.session)
        self.update(step=0, tokens_seen=0)

    def update(
        self,
        *,
        step: int,
        tokens_seen: int,
        status: str = "running",
        add_seconds: float = 0.0,
        peak_vram_gb: float | None = None,
        peak_host_ram_gb: float | None = None,
    ) -> dict[str, Any]:
        self.session["elapsed_seconds"] = float(
            self.session["elapsed_seconds"]
        ) + max(0.0, float(add_seconds))
        self.session["step"] = int(step)
        self.session["tokens_seen"] = int(tokens_seen)
        self.session["status"] = status
        if status != "running":
            self.session["finished_at"] = utc_timestamp()
        if peak_vram_gb is not None:
            self.session["peak_vram_gb"] = max(
                float(self.session["peak_vram_gb"]),
                float(peak_vram_gb),
            )
        if peak_host_ram_gb is not None:
            self.session["peak_host_ram_gb"] = max(
                float(self.session["peak_host_ram_gb"]),
                float(peak_host_ram_gb),
            )
        snapshot = self.snapshot()
        self.ledger.update(snapshot)
        self.ledger["updated_at"] = utc_timestamp()
        write_json(self.path, self.ledger)
        return snapshot

    def commit(
        self,
        checkpoint: str | Path,
        *,
        step: int,
        tokens_seen: int,
    ) -> dict[str, Any]:
        snapshot = self.update(step=step, tokens_seen=tokens_seen)
        self.ledger["last_checkpoint"] = Path(checkpoint).name
        self.ledger["last_checkpoint_baseline"] = {
            key: snapshot[key]
            for key in (
                "time_basis",
                "wall_clock_seconds",
                "gpu_hours",
                "training_sessions",
                "peak_vram_gb",
                "peak_host_ram_gb",
            )
        }
        write_json(self.path, self.ledger)
        return snapshot

    def rollback(self) -> dict[str, Any]:
        baseline = dict(self.ledger["last_checkpoint_baseline"])
        self.ledger["base"] = baseline
        self.ledger["sessions"] = []
        self.ledger["updated_at"] = utc_timestamp()
        snapshot = self.snapshot()
        self.ledger.update(snapshot)
        write_json(self.path, self.ledger)
        return snapshot

    def snapshot(self) -> dict[str, Any]:
        baseline = self.ledger["base"]
        sessions = self.ledger["sessions"]
        wall_seconds = float(baseline["wall_clock_seconds"]) + sum(
            float(session["elapsed_seconds"]) for session in sessions
        )
        gpu_seconds = float(baseline["gpu_hours"]) * 3600.0 + sum(
            float(session["elapsed_seconds"]) * int(session["gpu_count"])
            for session in sessions
        )
        training_sessions = int(baseline["training_sessions"]) + len(
            sessions
        )
        return {
            "time_basis": TIME_BASIS,
            "wall_clock_seconds": wall_seconds,
            "gpu_hours": gpu_seconds / 3600.0,
            "session_wall_clock_seconds": (
                float(self.session["elapsed_seconds"])
                if self.session in sessions
                else 0.0
            ),
            "training_sessions": training_sessions,
            "resume_count": max(0, training_sessions - 1),
            "peak_vram_gb": max(
                [float(baseline["peak_vram_gb"])]
                + [
                    float(session["peak_vram_gb"])
                    for session in sessions
                ]
            ),
            "peak_host_ram_gb": max(
                [float(baseline["peak_host_ram_gb"])]
                + [
                    float(session["peak_host_ram_gb"])
                    for session in sessions
                ]
            ),
        }
