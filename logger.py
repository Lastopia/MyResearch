from pathlib import Path

from para import PATH
from utils import ensure_dir


class ExperimentLogger:
    def __init__(self):
        ensure_dir(PATH.log_dir)
        self.path = Path(PATH.log_dir) / "experiment.log"

    def write(self, message: str) -> None:
        print(message, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def log_stage_start(self, stage_name: str) -> None:
        self.write(f"[start] {stage_name}")

    def log_stage_end(self, stage_name: str) -> None:
        self.write(f"[done] {stage_name}")

    def log_metric(self, name: str, value, step=None) -> None:
        prefix = f"[metric step={step}]" if step is not None else "[metric]"
        self.write(f"{prefix} {name}: {value}")

    def log_error(self, error: Exception) -> None:
        self.write(f"[error] {type(error).__name__}: {error}")
