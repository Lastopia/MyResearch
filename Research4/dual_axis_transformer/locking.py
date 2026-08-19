"""Per-run process locks that prevent duplicate training after launcher restarts."""

from __future__ import annotations

import atexit
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class RunLock:
    def __init__(
        self,
        run_dir: str | Path,
        final_path: str | Path,
        *,
        poll_seconds: float = 10.0,
    ) -> None:
        self.path = Path(run_dir) / "run.lock"
        self.final_path = Path(final_path)
        self.poll_seconds = poll_seconds
        self.owned = False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                descriptor = os.open(
                    self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
            except FileExistsError:
                try:
                    metadata = json.loads(self.path.read_text(encoding="utf-8"))
                    pid = int(metadata.get("pid", -1))
                except (OSError, ValueError, json.JSONDecodeError):
                    pid = -1
                if _pid_alive(pid):
                    if self.final_path.exists():
                        return False
                    time.sleep(self.poll_seconds)
                    continue
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                continue
            payload = json.dumps(
                {
                    "pid": os.getpid(),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ).encode("utf-8")
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            self.owned = True
            atexit.register(self.release)
            return True

    def release(self) -> None:
        if not self.owned:
            return
        try:
            metadata = json.loads(self.path.read_text(encoding="utf-8"))
            if int(metadata.get("pid", -1)) == os.getpid():
                self.path.unlink()
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        self.owned = False
