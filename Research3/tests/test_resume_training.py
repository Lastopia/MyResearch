from __future__ import annotations

import unittest
from copy import deepcopy
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from cfg import get_config
from pipeline import adapt, train
from tools.checkpoint import capture_rng_state, restore_rng_state
from tools.io import append_jsonl, read_jsonl
from tools.paths import checkpoint_dir, metric_dir
from tools.sampler import ResumableBatchSampler
from tools.training_time import TrainingTimer


class ResumeTrainingTests(unittest.TestCase):
    def test_historical_cuda_rng_maps_saved_gpu_to_current_gpu(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False):
            state = capture_rng_state()
        historical_states = [
            torch.tensor([index], dtype=torch.uint8) for index in range(4)
        ]
        state["cuda"] = historical_states

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "set_rng_state") as set_rng_state,
        ):
            restore_rng_state(
                state,
                saved_device="cuda:3",
                target_device="cuda:1",
            )

        set_rng_state.assert_called_once_with(
            historical_states[3],
            device=1,
        )

    def test_current_cuda_rng_maps_to_resumed_gpu(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False):
            state = capture_rng_state()
        current_state = torch.tensor([42], dtype=torch.uint8)
        state["cuda_device_index"] = 3
        state["cuda_device"] = current_state

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "set_rng_state") as set_rng_state,
        ):
            restore_rng_state(
                state,
                saved_device="cuda:3",
                target_device=torch.device("cuda:0"),
            )

        set_rng_state.assert_called_once_with(current_state, device=0)

    def test_failed_time_without_checkpoint_is_discarded(self) -> None:
        with TemporaryDirectory() as directory:
            timer = TrainingTimer(
                f"{directory}/training_time.json",
                stage="pretrain",
                method="ra_cable",
                seed=42,
                gpu_count=1,
                resumed_from=None,
            )
            timer.update(
                step=10,
                tokens_seen=640,
                add_seconds=10.0,
            )
            rolled_back = timer.rollback()

            self.assertEqual(rolled_back["wall_clock_seconds"], 0.0)
            self.assertEqual(rolled_back["gpu_hours"], 0.0)
            self.assertEqual(rolled_back["training_sessions"], 0)

    def test_failed_time_after_checkpoint_rolls_back_to_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            path = f"{directory}/training_time.json"
            checkpoint = f"{directory}/pretrain_step10.pt"
            timer = TrainingTimer(
                path,
                stage="pretrain",
                method="cable",
                seed=42,
                gpu_count=1,
                resumed_from=None,
            )
            timer.update(
                step=10,
                tokens_seen=640,
                add_seconds=10.0,
            )
            committed = timer.commit(
                checkpoint,
                step=10,
                tokens_seen=640,
            )
            timer.update(
                step=15,
                tokens_seen=960,
                add_seconds=5.0,
            )
            rolled_back = timer.rollback()

            self.assertAlmostEqual(
                rolled_back["wall_clock_seconds"],
                committed["wall_clock_seconds"],
                places=3,
            )
            self.assertAlmostEqual(
                rolled_back["gpu_hours"],
                committed["gpu_hours"],
                places=6,
            )
            self.assertEqual(rolled_back["training_sessions"], 1)

            resumed = TrainingTimer(
                path,
                stage="pretrain",
                method="cable",
                seed=42,
                gpu_count=1,
                resumed_from=checkpoint,
                resume_snapshot=committed,
            )
            resumed_snapshot = resumed.snapshot()
            self.assertAlmostEqual(
                resumed_snapshot["wall_clock_seconds"],
                committed["wall_clock_seconds"],
                places=2,
            )
            self.assertEqual(resumed_snapshot["training_sessions"], 2)

    def test_resumable_sampler_continues_exact_batch_sequence(self) -> None:
        dataset = list(range(20))
        complete = ResumableBatchSampler(
            dataset,
            batch_size=4,
            seed=17,
        )
        batches = list(complete)
        resumed = ResumableBatchSampler(
            dataset,
            batch_size=4,
            seed=17,
            epoch=0,
            start_batch=2,
        )
        self.assertEqual(list(resumed), batches[2:])

    def test_interrupted_training_resumes_step_and_accumulates_time(self) -> None:
        with TemporaryDirectory() as directory:
            cfg = get_config()
            cfg["paths"]["project_root"] = directory
            cfg["run"].update(
                {
                    "task": "resume_test",
                    "method": "rope",
                    "device": "cpu",
                    "dtype": "float32",
                    "require_cuda": False,
                    "force": False,
                }
            )
            cfg["data"].update(
                {
                    "source": "synthetic",
                    "vocab_size": 256,
                    "block_size": 16,
                    "train_tokens": 2_048,
                    "valid_tokens": 512,
                    "test_tokens": 512,
                    "retrieval_train_samples": 32,
                    "retrieval_eval_samples": 8,
                    "num_key_value_pairs": 2,
                    "prepare_wikitext": False,
                }
            )
            cfg["model"].update(
                {
                    "n_layer": 1,
                    "n_head": 2,
                    "n_embd": 16,
                    "ffn_dim": 32,
                    "max_seq_len": 64,
                }
            )
            cfg["train"].update(
                {
                    "token_budget": 512,
                    "micro_batch_size": 2,
                    "effective_batch_tokens": 64,
                    "log_interval": 1,
                    "eval_interval": 3,
                    "eval_batches": 1,
                    "save_interval": 2,
                }
            )
            cfg["adapt"].update(
                {
                    "max_seq_len": 16,
                    "steps": 2,
                    "batch_size": 4,
                    "micro_batch_size": 2,
                }
            )
            cfg["eval"].update(
                {
                    "lengths": [16],
                    "batch_size": 1,
                    "lm_batches": 1,
                    "retrieval_samples": 4,
                }
            )
            cfg["audit"]["lengths"] = [16]
            cfg["profile"]["lengths"] = [16]
            cfg["resources"].update(
                {
                    "data_workers": 0,
                    "resolved_pin_memory": False,
                    "monitor_interval_steps": 100,
                }
            )

            original_save = train.save_checkpoint

            def fail_after_step_four(path, **kwargs):
                original_save(path, **kwargs)
                if str(path).endswith("pretrain_step4.pt"):
                    raise RuntimeError("simulated interruption")

            with patch.object(
                train,
                "save_checkpoint",
                side_effect=fail_after_step_four,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "simulated interruption",
                ):
                    train.run(cfg)

            partial = checkpoint_dir(cfg) / "pretrain_step4.pt"
            self.assertTrue(partial.exists())
            append_jsonl(
                metric_dir(cfg) / "train.jsonl",
                {"step": 999, "step_seconds": 999.0},
            )
            append_jsonl(
                metric_dir(cfg) / "validation.jsonl",
                {"step": 999},
            )
            summary = train.run(cfg)
            self.assertTrue(summary["resumed"])
            self.assertEqual(summary["resumed_from_step"], 4)
            self.assertEqual(summary["steps"], 8)
            self.assertEqual(summary["training_sessions"], 2)
            self.assertEqual(summary["resume_count"], 1)
            self.assertGreater(
                summary["wall_clock_seconds"],
                summary["session_wall_clock_seconds"],
            )

            payload = torch.load(
                checkpoint_dir(cfg) / "pretrain_final.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(payload["step"], 8)
            self.assertTrue(
                (checkpoint_dir(cfg) / "pretrain_step6.pt").exists()
            )
            uninterrupted_cfg = deepcopy(cfg)
            uninterrupted_cfg["run"]["task"] = "uninterrupted_test"
            train.run(uninterrupted_cfg)
            uninterrupted = torch.load(
                checkpoint_dir(uninterrupted_cfg) / "pretrain_final.pt",
                map_location="cpu",
                weights_only=False,
            )
            for name, value in payload["model"].items():
                torch.testing.assert_close(
                    value,
                    uninterrupted["model"][name],
                    rtol=0,
                    atol=0,
                )

            cfg["adapt"].update(
                {
                    "steps": 4,
                    "log_interval": 1,
                    "save_interval": 2,
                }
            )
            uninterrupted_cfg["adapt"].update(cfg["adapt"])
            original_adapt_save = adapt.save_checkpoint

            def fail_after_adapt_step_two(path, **kwargs):
                original_adapt_save(path, **kwargs)
                if str(path).endswith("adapt_step2.pt"):
                    raise RuntimeError("simulated adapt interruption")

            with patch.object(
                adapt,
                "save_checkpoint",
                side_effect=fail_after_adapt_step_two,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "simulated adapt interruption",
                ):
                    adapt.run(cfg)
            append_jsonl(
                metric_dir(cfg) / "adapt.jsonl",
                {"step": 999, "step_seconds": 999.0},
            )
            adapt_summary = adapt.run(cfg)
            self.assertTrue(adapt_summary["resumed"])
            self.assertEqual(adapt_summary["resumed_from_step"], 2)
            self.assertEqual(adapt_summary["steps"], 4)
            self.assertEqual(adapt_summary["training_sessions"], 2)

            adapt.run(uninterrupted_cfg)
            resumed_adapt = torch.load(
                checkpoint_dir(cfg) / "adapt_final.pt",
                map_location="cpu",
                weights_only=False,
            )
            uninterrupted_adapt = torch.load(
                checkpoint_dir(uninterrupted_cfg) / "adapt_final.pt",
                map_location="cpu",
                weights_only=False,
            )
            for name, value in resumed_adapt["model"].items():
                torch.testing.assert_close(
                    value,
                    uninterrupted_adapt["model"][name],
                    rtol=0,
                    atol=0,
                )
            train_records = read_jsonl(metric_dir(cfg) / "train.jsonl")
            train_steps = [
                int(record["step"]) for record in train_records
            ]
            self.assertEqual(train_steps, list(range(1, 9)))
            self.assertAlmostEqual(
                summary["wall_clock_seconds"],
                sum(
                    float(record["step_seconds"])
                    for record in train_records
                ),
                places=6,
            )
            adapt_records = read_jsonl(metric_dir(cfg) / "adapt.jsonl")
            self.assertAlmostEqual(
                adapt_summary["wall_clock_seconds"],
                sum(
                    float(record["step_seconds"])
                    for record in adapt_records
                ),
                places=6,
            )
            validation_steps = [
                int(record["step"])
                for record in read_jsonl(
                    metric_dir(cfg) / "validation.jsonl"
                )
            ]
            self.assertEqual(validation_steps, [3, 6, 8])


if __name__ == "__main__":
    unittest.main()
