from __future__ import annotations

import gc
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from cfg import get_config
from pipeline import adapt, attention_audit, data, evaluate, profile, train
from tools.io import read_json, write_json
from tools.paths import metric_dir


class ExtendedPipelineTests(unittest.TestCase):
    def test_token_shingles_detect_overlap_but_not_unrelated_tokens(self) -> None:
        reference = list(range(100))
        near_duplicate = [999, *reference[10:90], 998]
        unrelated = list(range(1_000, 1_100))

        reference_shingles = data._token_shingles(
            reference,
            width=8,
            stride=2,
        )
        near_shingles = data._token_shingles(
            near_duplicate,
            width=8,
            stride=2,
        )
        unrelated_shingles = data._token_shingles(
            unrelated,
            width=8,
            stride=2,
        )

        self.assertGreater(len(reference_shingles & near_shingles), 20)
        self.assertFalse(reference_shingles & unrelated_shingles)

    def test_evaluation_disables_nested_dataloader_workers(self) -> None:
        cfg = get_config()
        cfg["resources"].update(
            {
                "data_workers": 8,
                "resolved_pin_memory": True,
                "resolved_persistent_workers": True,
            }
        )

        options = evaluate.evaluation_dataloader_kwargs(cfg)

        self.assertEqual(options["num_workers"], 0)
        self.assertTrue(options["pin_memory"])
        self.assertNotIn("persistent_workers", options)
        self.assertNotIn("prefetch_factor", options)

    def test_new_controls_run_end_to_end_on_tiny_cpu_model(self) -> None:
        with TemporaryDirectory() as directory:
            cfg = get_config()
            cfg["paths"]["project_root"] = directory
            cfg["run"].update(
                {
                    "task": "extended_pipeline_test",
                    "method": "rope",
                    "device": "cpu",
                    "dtype": "float32",
                    "require_cuda": False,
                    "force": False,
                    "bootstrap": "off",
                }
            )
            cfg["data"].update(
                {
                    "source": "synthetic",
                    "vocab_size": 256,
                    "block_size": 16,
                    "train_tokens": 512,
                    "valid_tokens": 128,
                    "test_tokens": 512,
                    "retrieval_train_samples": 16,
                    "retrieval_eval_samples": 4,
                    "num_key_value_pairs": 4,
                    "retrieval_queries_per_sample": 4,
                    "retrieval_similar_distractors": 3,
                    "prepare_wikitext": False,
                    "prepare_qasper": False,
                }
            )
            cfg["model"].update(
                {
                    "n_layer": 1,
                    "n_head": 2,
                    "n_embd": 16,
                    "ffn_dim": 32,
                    "max_seq_len": 128,
                }
            )
            cfg["train"].update(
                {
                    "token_budget": 64,
                    "micro_batch_size": 2,
                    "effective_batch_tokens": 64,
                    "log_interval": 1,
                    "eval_interval": 1,
                    "eval_batches": 1,
                    "save_interval": 1,
                }
            )
            cfg["adapt"].update(
                {
                    "max_seq_len": 16,
                    "steps": 1,
                    "batch_size": 4,
                    "micro_batch_size": 2,
                    "log_interval": 1,
                    "save_interval": 1,
                }
            )
            cfg["eval"].update(
                {
                    "lengths": [64, 128],
                    "lm_batches": 1,
                    "retrieval_samples": 2,
                    "batch_size": 1,
                    "rope_pi_enabled": True,
                    "rope_pi_train_length": 16,
                }
            )
            cfg["audit"].update(
                {
                    "lengths": [64],
                    "samples": 2,
                    "batch_size": 1,
                    "layers": [0],
                    "conditions": ["synthetic_remote_target"],
                }
            )
            cfg["profile"].update(
                {
                    "lengths": [16],
                    "train_length": 16,
                    "batch_size": 1,
                    "warmup": 1,
                    "repeat": 1,
                    "decode_tokens": 4,
                }
            )
            cfg["resources"].update(
                {
                    "data_workers": 0,
                    "resolved_pin_memory": False,
                    "monitor_interval_steps": 100,
                }
            )

            data.run(cfg)
            train.run(cfg)
            adapt.run(cfg)
            evaluation = evaluate.run(cfg)
            with patch.object(
                evaluate,
                "_evaluate_lm",
                side_effect=AssertionError("completed evaluation was rerun"),
            ):
                cached_evaluation = evaluate.run(cfg)

            pretrain_final = metric_dir(cfg) / "evaluation_pretrain.json"
            pretrain_partial = (
                metric_dir(cfg) / "evaluation_pretrain.partial.json"
            )
            interrupted = read_json(pretrain_final)
            interrupted["status"] = "partial"
            del interrupted["lengths"]["64"]["natural_language"][
                "fineweb_edu_held_out_ppl"
            ]
            write_json(pretrain_partial, interrupted)
            pretrain_final.unlink()
            with patch.object(
                evaluate,
                "_evaluate_lm",
                return_value=123.0,
            ) as resumed_lm:
                resumed_evaluation = evaluate.run(cfg)
                resumed_lm_call_count = resumed_lm.call_count
                resumed_lm.reset_mock()

            audit = attention_audit.run(cfg)
            efficiency = profile.run(cfg)

            self.assertEqual(
                set(evaluation["checkpoints"]),
                {"pretrain", "adapt"},
            )
            self.assertEqual(
                cached_evaluation["checkpoints"]["pretrain"]["status"],
                "completed",
            )
            self.assertIn(
                "length_degradation_rate",
                evaluation["checkpoints"]["pretrain"],
            )
            self.assertEqual(resumed_lm_call_count, 1)
            self.assertEqual(
                resumed_evaluation["checkpoints"]["pretrain"]["lengths"][
                    "64"
                ]["natural_language"]["fineweb_edu_held_out_ppl"],
                123.0,
            )
            at_length = evaluation["checkpoints"]["adapt"]["lengths"]["64"]
            self.assertIn(
                "multi_query_associative_recall",
                at_length["synthetic_control"],
            )
            self.assertIn(
                "target_distractor_position_swap",
                at_length["synthetic_control"],
            )
            self.assertIn("rope_pi", at_length)
            audit_summary = audit["lengths"]["64"]["conditions"][
                "synthetic_remote_target"
            ]["summary"]
            self.assertIn(
                "bias_adjacent_monotonic_violation_rate",
                audit_summary,
            )
            train_profile = efficiency["lengths"]["16"]["train_step"]
            self.assertEqual(train_profile["effective_batch_tokens"], 64)
            self.assertEqual(train_profile["gradient_accumulation_steps"], 2)
            del evaluation, cached_evaluation, resumed_evaluation
            gc.collect()


if __name__ == "__main__":
    unittest.main()
