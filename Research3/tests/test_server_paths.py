from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from cfg import get_config
from tools.paths import checkpoint_dir, data_dir, output_dir, workspace_root


class ServerPathTests(unittest.TestCase):
    def test_explicit_project_root_is_used_for_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = get_config()
            cfg["paths"]["project_root"] = directory
            cfg["run"]["task"] = "server_task"
            cfg["run"]["method"] = "alibi"
            cfg["run"]["seed"] = 7
            root = Path(directory).resolve()
            self.assertEqual(workspace_root(cfg), root)
            self.assertEqual(data_dir(cfg), root / "data" / "server_task")
            self.assertEqual(
                checkpoint_dir(cfg),
                root / "checkpoints" / "server_task" / "alibi" / "seed7",
            )
            self.assertEqual(output_dir(cfg), root / "output" / "server_task")

    def test_environment_root_is_used_when_config_is_unset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous = os.environ.get("POSITION_BIAS_PROJECT_ROOT")
            os.environ["POSITION_BIAS_PROJECT_ROOT"] = directory
            try:
                cfg = get_config()
                self.assertEqual(workspace_root(cfg), Path(directory).resolve())
            finally:
                if previous is None:
                    os.environ.pop("POSITION_BIAS_PROJECT_ROOT", None)
                else:
                    os.environ["POSITION_BIAS_PROJECT_ROOT"] = previous


if __name__ == "__main__":
    unittest.main()
