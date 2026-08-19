from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from cfg import get_config
from tools.resources import (
    _candidate_gpu_ids,
    _gpu_inventory,
    build_resource_plan,
    calibrate_micro_batch,
)


class ResourcePlanningTests(unittest.TestCase):
    def test_calibration_binary_search_honors_safe_fit(self) -> None:
        cfg = get_config()

        def trial(
            _cfg,
            method,
            batch_size,
            device,
            safety_fraction,
        ):
            return {
                "gpu": int(device.index),
                "method": method,
                "micro_batch_size": batch_size,
                "safety_fraction": safety_fraction,
                "fit": batch_size <= 16,
                "oom": batch_size > 32,
            }

        with patch("tools.resources._trial_micro_batch", side_effect=trial):
            result = calibrate_micro_batch(
                cfg,
                ["rope", "alibi", "cable", "ra_cable"],
                0,
            )
        self.assertEqual(result["method"], "ra_cable")
        self.assertEqual(result["micro_batch_size"], 16)
        self.assertEqual(result["safety_fraction"], 0.85)
        self.assertEqual(
            [item["micro_batch_size"] for item in result["trials"]],
            [8, 32, 16],
        )

    @patch("tools.resources.torch.cuda.get_device_name")
    @patch("tools.resources.torch.cuda.mem_get_info")
    @patch("tools.resources.torch.cuda.device_count")
    def test_busy_or_small_gpus_are_rejected(
        self,
        device_count,
        mem_get_info,
        get_device_name,
    ) -> None:
        gib = 1024**3
        device_count.return_value = 3
        get_device_name.side_effect = lambda index: f"GPU-{index}"
        mem_get_info.side_effect = [
            (46 * gib, 48 * gib),
            (38 * gib, 48 * gib),
            (3 * gib, 24 * gib),
        ]
        cfg = get_config()
        inventory = _gpu_inventory(cfg)
        self.assertTrue(inventory[0]["eligible"])
        self.assertIn("gpu_not_idle", inventory[1]["rejection_reasons"])
        self.assertIn(
            "insufficient_free_vram",
            inventory[2]["rejection_reasons"],
        )
        self.assertEqual(_candidate_gpu_ids(cfg, inventory), [0])

    def test_auto_plan_uses_all_safe_gpus_and_common_safe_batch(self) -> None:
        with TemporaryDirectory() as directory:
            cfg = get_config()
            cfg["paths"]["project_root"] = directory
            cfg["run"]["task"] = "resource_test"
            inventory = [
                {
                    "index": index,
                    "name": f"GPU-{index}",
                    "free_vram_gb": 46.0,
                    "total_vram_gb": 48.0,
                    "preexisting_used_vram_gb": 2.0,
                    "preexisting_used_fraction": 2 / 48,
                    "eligible": True,
                    "rejection_reasons": [],
                }
                for index in range(4)
            ]
            calibrated = {0: 32, 1: 16, 2: 32, 3: 16}

            def calibration(_cfg, _methods, gpu_id):
                return {
                    "gpu": gpu_id,
                    "method": "ra_cable",
                    "micro_batch_size": calibrated[gpu_id],
                    "candidate_micro_batches": [1, 2, 4, 8, 16, 32, 64],
                    "trials": [],
                    "safety_fraction": 0.85,
                }

            with (
                patch("tools.resources.torch.cuda.is_available", return_value=True),
                patch("tools.resources._gpu_inventory", return_value=inventory),
                patch(
                    "tools.resources._candidate_gpu_ids",
                    return_value=[0, 1, 2, 3],
                ),
                patch(
                    "tools.resources.resource_snapshot",
                    return_value={"ram_available_gb": 128.0},
                ),
                patch(
                    "tools.resources.calibrate_micro_batch",
                    side_effect=calibration,
                ),
            ):
                plan = build_resource_plan(
                    cfg,
                    methods=["rope", "alibi", "cable", "ra_cable"],
                    job_count=4,
                )

            self.assertEqual(plan["parallel_jobs"], 4)
            self.assertEqual(plan["gpu_ids"], [0, 1, 2, 3])
            self.assertEqual(plan["micro_batch_size"], 16)
            self.assertEqual(plan["gradient_accumulation_steps"], 4)
            self.assertEqual(len(plan["gpu_calibrations"]), 4)
            self.assertEqual(plan["vram_safety_fraction"], 0.85)


if __name__ == "__main__":
    unittest.main()
