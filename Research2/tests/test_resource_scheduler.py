import multiprocessing as mp
import queue
import time
import unittest
from unittest.mock import patch

from tools.resource import (
    _adaptive_launch_allowed,
    resolve_jobs_per_gpu,
    run_gpu_jobs,
)


def _timed_job(events, job_id, duration, gpu_index):
    events.put(("start", job_id, time.monotonic(), gpu_index))
    time.sleep(duration)
    events.put(("end", job_id, time.monotonic(), gpu_index))


class ResourceSchedulerTests(unittest.TestCase):
    def test_dynamic_scheduler_refills_before_slowest_job_finishes(self):
        ctx = mp.get_context("spawn")
        events = ctx.Queue()
        cfg = {
            "run": {
                "jobs_per_gpu": 2,
                "gpu_scheduler": {"poll_seconds": 0.05},
            },
        }
        # Spawned Python workers import torch before entering this function, so
        # keep the long job alive past that startup cost on slower CI hosts.
        jobs = [(0, 5.0), (1, 0.05), (2, 0.05)]
        failed = run_gpu_jobs(
            cfg,
            jobs,
            _timed_job,
            lambda job, gpu_index: (events, job[0], job[1], gpu_index),
            [0],
            stage="test",
        )
        rows = []
        for _ in range(2 * len(jobs)):
            try:
                rows.append(events.get(timeout=10.0))
            except queue.Empty as error:
                self.fail(f"missing scheduler event: {error}")
        events.close()
        events.join_thread()

        event_time = {(kind, job_id): timestamp for kind, job_id, timestamp, _ in rows}
        self.assertEqual(failed, [])
        self.assertLess(event_time[("start", 2)], event_time[("end", 0)])

    def test_adaptive_launch_obeys_utilization_and_memory_guards(self):
        cfg = {
            "run": {
                "jobs_per_gpu": "auto",
                "gpu_scheduler": {
                    "settle_seconds": 0.0,
                    "utilization_target": 90.0,
                    "utilization_samples": 1,
                    "memory_reserve_fraction": 0.10,
                    "min_memory_reserve_gb": 8.0,
                    "memory_safety_factor": 1.35,
                },
            },
        }
        state = {
            "active": 1,
            "last_launch": 0.0,
            "free_before": 100.0,
            "job_memory_gb": None,
        }
        with patch("tools.resource._safe_gpu_memory", return_value=(90.0, 140.0)):
            with patch("tools.resource._gpu_utilization", return_value=50.0):
                self.assertTrue(_adaptive_launch_allowed(cfg, 0, state, 8, 10.0))
            with patch("tools.resource._gpu_utilization", return_value=95.0):
                self.assertFalse(_adaptive_launch_allowed(cfg, 0, state, 8, 10.0))
        with patch("tools.resource._safe_gpu_memory", return_value=(20.0, 140.0)):
            with patch("tools.resource._gpu_utilization", return_value=10.0):
                self.assertFalse(_adaptive_launch_allowed(cfg, 0, state, 8, 10.0))

    def test_auto_limit_scales_with_gpu_size_and_caps_sae(self):
        cfg = {
            "run": {
                "mode": "retrain",
                "jobs_per_gpu": "auto",
                "gpu_scheduler": {"max_jobs_per_gpu": 8},
            },
            "model": {"d_model": 384, "n_layer": 6},
            "data": {"block_size": 1024},
            "train": {"batch_size": 2},
        }
        with patch("tools.resource.torch.cuda.is_available", return_value=True):
            with patch("tools.resource.gpu_total_gb", return_value=140.0):
                with patch("tools.resource.gpu_free_gb", return_value=130.0):
                    self.assertEqual(resolve_jobs_per_gpu(cfg, [0, 1], "train"), 8)
                    self.assertEqual(resolve_jobs_per_gpu(cfg, [0, 1], "sae"), 3)


if __name__ == "__main__":
    unittest.main()
