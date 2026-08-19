"""Fixed-column training logs and checkpoint-resumable training time."""

from __future__ import annotations

import os
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch


def _display_width(text: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        for character in text
    )


def _fit_cell(value: object | None, width: int) -> str:
    text = "" if value is None else str(value)
    if _display_width(text) > width:
        kept = []
        used = 0
        target = max(0, width - 1)
        for character in text:
            character_width = (
                2
                if unicodedata.east_asian_width(character) in {"W", "F"}
                else 1
            )
            if used + character_width > target:
                break
            kept.append(character)
            used += character_width
        text = "".join(kept) + "…"
    return text + " " * max(0, width - _display_width(text))


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.2f}s"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def checkpoint_due(
    timer: "CumulativeTrainingTimer",
    last_checkpoint_seconds: float,
    interval_minutes: float,
    *,
    final_step: bool = False,
) -> bool:
    """Return true after the requested amount of accumulated training time."""
    if interval_minutes <= 0:
        raise ValueError("checkpoint_interval_minutes must be positive")
    return final_step or (
        timer.elapsed_seconds - float(last_checkpoint_seconds)
        >= float(interval_minutes) * 60.0
    )


class CumulativeTrainingTimer:
    """Count training compute time only and survive checkpoint restoration."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._accumulated_seconds = 0.0
        self._active_since: float | None = None
        self._last_log_total = 0.0

    @property
    def running(self) -> bool:
        return self._active_since is not None

    @property
    def elapsed_seconds(self) -> float:
        active = (
            self._clock() - self._active_since
            if self._active_since is not None
            else 0.0
        )
        return self._accumulated_seconds + active

    def start(self) -> None:
        if self._active_since is None:
            self._active_since = self._clock()

    def pause(self) -> None:
        if self._active_since is not None:
            self._accumulated_seconds += self._clock() - self._active_since
            self._active_since = None

    def mark_log_interval(self) -> float:
        total = self.elapsed_seconds
        interval = total - self._last_log_total
        self._last_log_total = total
        return max(0.0, interval)

    def state_dict(self) -> dict[str, float]:
        return {
            "accumulated_seconds": self.elapsed_seconds,
            "last_log_total": self._last_log_total,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._accumulated_seconds = float(state.get("accumulated_seconds", 0.0))
        self._last_log_total = float(
            state.get("last_log_total", self._accumulated_seconds)
        )
        self._active_since = None


@dataclass(frozen=True)
class DeviceLogStatus:
    kind: str
    name: str
    ordinal: int
    available_count: int
    memory_used_gb: float | None
    memory_total_gb: float | None

    @property
    def device_text(self) -> str:
        if self.kind.upper() == "GPU":
            return f"GPU {self.ordinal}/{self.available_count} {self.name}"
        return "CPU 0/0"

    @property
    def memory_text(self) -> str:
        if self.memory_used_gb is None or self.memory_total_gb is None:
            return ""
        return f"{self.memory_used_gb:.2f}/{self.memory_total_gb:.2f} GiB"


def current_device_log_status(device: str | torch.device) -> DeviceLogStatus:
    device = torch.device(device)
    if device.type != "cuda":
        return DeviceLogStatus("CPU", "CPU", 0, 0, None, None)
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        properties = torch.cuda.get_device_properties(index)
        # Process-local allocation is the comparable model cost. Whole-card
        # occupancy is already captured separately by the resource monitor.
        used_gb = torch.cuda.memory_allocated(index) / (1024**3)
        total_gb = properties.total_memory / (1024**3)
        ordinal = int(os.environ.get("CONCEPT_BUS_GPU_ORDINAL", index + 1))
        available = int(
            os.environ.get("CONCEPT_BUS_GPU_COUNT", torch.cuda.device_count())
        )
        return DeviceLogStatus(
            "GPU", name, ordinal, available, used_gb, total_gb
        )
    except Exception as error:
        return DeviceLogStatus(
            "GPU", f"CUDA error: {type(error).__name__}", 0, 0, None, None
        )


DEFAULT_WIDTHS = {
    "time": 19,
    "device": 34,
    "phase": 8,
    "step": 17,
    "loss": 16,
    "memory": 21,
    "elapsed": 25,
    "seed": 8,
    "task": 48,
}

# Deliberately stays below 120 display columns including `` | `` separators.
# The file log uses DEFAULT_WIDTHS and therefore keeps the unabridged values.
DEFAULT_CONSOLE_WIDTHS = {
    "time": 8,
    "device": 12,
    "phase": 7,
    "step": 11,
    "loss": 10,
    "memory": 15,
    "elapsed": 15,
    "seed": 4,
    "task": 14,
}


def _compact_device(device: DeviceLogStatus) -> str:
    if device.kind.upper() != "GPU":
        return "CPU"
    name = device.name.replace("NVIDIA", "").replace("GeForce", "").strip()
    return f"G{device.ordinal}/{device.available_count} {name}"


def _compact_task(task: str | None) -> str:
    if not task:
        return ""
    stage, separator, method = task.rpartition("/")
    if not separator:
        stage, method = task, ""
    stage_lower = stage.lower()
    if "clutrr" in stage_lower:
        stage = "clutrr"
    elif "language" in stage_lower or "tinystories" in stage_lower:
        stage = "lm"
    elif "dual_tag" in stage_lower or "concept_bus" in stage_lower:
        stage = "tag"
    method = {
        "standard": "std",
        "parameter_matched": "pmatch",
        "concept_aux": "caux",
        "concept_projector": "proj",
        "concept_bus_v2": "v2",
    }.get(method, method)
    return f"{stage}/{method}" if method else stage


class FixedWidthTrainingLogger:
    """Write aligned rows; absent values occupy a blank fixed-width cell."""

    HEADERS = {
        "time": "TIME",
        "device": "DEVICE",
        "phase": "PHASE",
        "step": "STEP",
        "loss": "LOSS",
        "memory": "VRAM USED/TOTAL",
        "elapsed": "INTERVAL/TOTAL",
        "seed": "SEED",
        "task": "TASK/MODEL",
    }
    ORDER = tuple(HEADERS)

    def __init__(
        self,
        path: str | Path,
        *,
        timer: CumulativeTrainingTimer,
        device_provider: Callable[[], DeviceLogStatus],
        widths: dict[str, int] | None = None,
        console_widths: dict[str, int] | None = None,
        console_mode: str = "summary",
        flush_each_line: bool = True,
    ) -> None:
        if console_mode not in {"quiet", "summary", "interval"}:
            raise ValueError("console_mode must be quiet, summary, or interval")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.timer = timer
        self.device_provider = device_provider
        self.widths = {**DEFAULT_WIDTHS, **(widths or {})}
        self.console_widths = {
            **DEFAULT_CONSOLE_WIDTHS,
            **(console_widths or {}),
        }
        if any(self.widths[key] < len(self.HEADERS[key]) for key in self.ORDER):
            raise ValueError("column widths must fit their headers")
        if any(
            self.console_widths[key] < len(self.HEADERS[key])
            for key in self.ORDER
        ):
            raise ValueError("console column widths must fit their headers")
        self.console_mode = console_mode
        self.flush_each_line = flush_each_line
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self._handle = self.path.open("a", encoding="utf-8", newline="")
        if is_new:
            self._write_raw(self._format(self.HEADERS))
            self._write_raw(
                "-+-".join("-" * self.widths[key] for key in self.ORDER)
            )

    def _format(
        self,
        values: dict[str, object | None],
        widths: dict[str, int] | None = None,
    ) -> str:
        selected = self.widths if widths is None else widths
        return " | ".join(
            _fit_cell(values.get(key), selected[key]) for key in self.ORDER
        )

    def _write_raw(self, line: str) -> None:
        self._handle.write(line + "\n")
        if self.flush_each_line:
            self._handle.flush()

    def log(
        self,
        phase: str,
        *,
        step: int | None = None,
        total_steps: int | None = None,
        loss: float | None = None,
        seed: int | None = None,
        task: str | None = None,
    ) -> str:
        phase_upper = phase.upper()
        device = self.device_provider()
        interval = (
            self.timer.mark_log_interval() if phase_upper == "TRAIN" else None
        )
        if step is None:
            step_text = ""
        elif total_steps is None:
            step_text = str(step)
        else:
            digits = max(1, len(str(total_steps)))
            step_text = f"{step:0{digits}d}/{total_steps:0{digits}d}"
        values = {
            "time": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            "device": device.device_text,
            "phase": phase_upper,
            "step": step_text,
            "loss": None if loss is None else f"{float(loss):.8f}",
            "memory": device.memory_text,
            "elapsed": f"{_duration(interval)}/{_duration(self.timer.elapsed_seconds)}",
            "seed": seed,
            "task": task,
        }
        line = self._format(values)
        self._write_raw(line)
        should_print = self.console_mode == "interval" or (
            self.console_mode == "summary"
            and phase_upper in {"SYSTEM", "VALID", "CHECKPT", "FINAL", "ERROR"}
        )
        if should_print:
            console_values = {
                **values,
                "time": str(values["time"])[-8:],
                "device": _compact_device(device),
                "loss": None if loss is None else f"{float(loss):.6f}",
                "memory": (
                    ""
                    if device.memory_used_gb is None
                    or device.memory_total_gb is None
                    else f"{device.memory_used_gb:.1f}/{device.memory_total_gb:.1f}G"
                ),
                "task": _compact_task(task),
            }
            print(
                self._format(console_values, self.console_widths).rstrip(),
                flush=True,
            )
        return line

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "FixedWidthTrainingLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
