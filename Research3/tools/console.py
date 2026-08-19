from __future__ import annotations

import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from tools.log import gpu_display_name, task_display_name
from tools.paths import output_dir


class _TeeStream:
    def __init__(
        self,
        original: TextIO | None,
        log_file: TextIO,
        path: Path,
    ) -> None:
        self.original = original
        self.log_file = log_file
        self.console_log_path = path
        self._lock = threading.Lock()

    @property
    def encoding(self) -> str:
        return "utf-8"

    def write(self, text: str) -> int:
        with self._lock:
            if self.original is not None:
                self.original.write(text)
            self.log_file.write(text)
            if "\n" in text:
                self.flush()
        return len(text)

    def flush(self) -> None:
        if self.original is not None:
            self.original.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return bool(self.original is not None and self.original.isatty())

    def fileno(self) -> int:
        if self.original is not None:
            return self.original.fileno()
        return self.log_file.fileno()


def console_log_path(cfg: dict[str, Any]) -> Path:
    return output_dir(cfg) / "logs" / "console.log"


def install_console_log(cfg: dict[str, Any]) -> Path:
    """Tee this process's stdout and stderr into the experiment console log."""
    path = console_log_path(cfg).resolve()
    current_path = getattr(sys.stdout, "console_log_path", None)
    if current_path == path:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    log_file = path.open(
        "a",
        encoding="utf-8",
        buffering=1,
    )
    sys.stdout = _TeeStream(sys.stdout, log_file, path)
    sys.stderr = _TeeStream(sys.stderr, log_file, path)
    timestamp = datetime.now(timezone.utc).isoformat()
    gpu = gpu_display_name(cfg)
    identity = " ".join(
        value
        for value in (
            task_display_name(cfg),
            str(cfg["run"]["method"]),
            gpu,
        )
        if value is not None
    )
    print(
        "\n"
        f"[CONSOLE SESSION] timestamp={timestamp} pid={os.getpid()} "
        f"{identity} "
        f"seed={cfg['run']['seed']}",
        flush=True,
    )
    return path
