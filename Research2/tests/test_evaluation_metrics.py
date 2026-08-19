import gc
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from pipeline.attn import normalize_attention_for_metrics
from pipeline.interpret import (
    _best_prevalence_corrected_concept,
    _best_recovery_threshold,
    _finite_median,
    _isolation_ratio_of_means,
)
from pipeline.train import (
    BlockData,
    accumulated_metric_time,
    checkpoint_compatible,
    checkpoint_elapsed_time,
    training_checkpoint_state,
)
from tools.plot import plot_metric_by_k


class AttentionMetricTests(unittest.TestCase):
    def test_unnormalized_attention_is_normalized_per_row(self):
        weights = torch.tensor([
            [[2.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        ])
        probabilities, mass, valid = normalize_attention_for_metrics(weights)
        self.assertTrue(torch.allclose(mass, torch.tensor([[4.0, 0.0]])))
        self.assertTrue(torch.equal(valid, torch.tensor([[True, False]])))
        self.assertTrue(torch.allclose(probabilities[0, 0], torch.tensor([0.5, 0.5, 0.0])))
        entropy = torch.where(
            probabilities > 0,
            -(probabilities * probabilities.log()),
            torch.zeros_like(probabilities),
        ).sum(dim=-1)
        self.assertLessEqual(float(entropy[valid].exp().max()), weights.size(-1))


class DeterministicEvaluationTests(unittest.TestCase):
    def test_deterministic_batches_repeat_without_consuming_global_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = np.arange(30, dtype=np.uint16).reshape(6, 5)
            data.tofile(root / "valid.bin")
            (root / "meta.json").write_text(json.dumps({"token_dtype": "uint16"}), encoding="utf-8")
            blocks = BlockData(root / "valid.bin", n_blocks=6, block_size=4)

            np.random.seed(123)
            expected_next = np.random.random()
            np.random.seed(123)
            first = blocks.deterministic_batch(3, torch.device("cpu"), seed=99, batch_index=2)
            observed_next = np.random.random()
            second = blocks.deterministic_batch(3, torch.device("cpu"), seed=99, batch_index=2)

            self.assertEqual(expected_next, observed_next)
            self.assertTrue(torch.equal(first[0], second[0]))
            self.assertTrue(torch.equal(first[1], second[1]))
            del blocks
            gc.collect()

    def test_training_time_recovers_old_resumed_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            rows = [
                {"step": 100, "time": 10},
                {"step": 200, "time": 20},
                {"step": 300, "time": 5},
                {"step": 400, "time": 15},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            self.assertEqual(accumulated_metric_time(path), 35)
            self.assertEqual(accumulated_metric_time(path, through_step=200), 20)
            self.assertEqual(
                checkpoint_elapsed_time({"elapsed_training_seconds": 99}, path, through_step=400),
                99,
            )


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_signature_is_authoritative_over_legacy_raw_position_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "[base_cable]seed42step100.pt"
            model = torch.nn.Linear(2, 2)
            torch.save({
                "model": model.state_dict(),
                "alias": "base_cable",
                "signature": {"resolved_position_encoding": "cable"},
                # This reproduces the old bug: cfg held the global RoPE default
                # although the alias and signature described a Cable model.
                "cfg": {"model": {"position_encoding": "rope"}},
            }, path)

            with (
                patch(
                    "pipeline.train.checkpoint_signature",
                    return_value={"resolved_position_encoding": "cable"},
                ),
                patch("pipeline.train.build_model", return_value=torch.nn.Linear(2, 2)),
            ):
                self.assertTrue(checkpoint_compatible(path, {}, "base_cable"))

    def test_new_checkpoint_stores_alias_resolved_config(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        resolved_cfg = {"model": {"position_encoding": "cable"}}
        signature = {"resolved_position_encoding": "cable"}

        with (
            patch("pipeline.train.experiment_cfg", return_value=resolved_cfg),
            patch("pipeline.train.checkpoint_signature", return_value=signature),
        ):
            state = training_checkpoint_state(
                {}, "base_cable", 42, 100, model, optimizer, scaler, 17.9,
            )

        self.assertEqual(state["cfg"], resolved_cfg)
        self.assertEqual(state["signature"], signature)
        self.assertEqual(state["alias"], "base_cable")
        self.assertEqual(state["elapsed_training_seconds"], 17)


class InterpretabilityMetricTests(unittest.TestCase):
    def test_known_concept_selection_corrects_auprc_for_prevalence(self):
        scores = torch.tensor([0.1, 0.2, 0.9, 1.0])
        labels = {
            "broad": torch.tensor([True, True, True, False]),
            "specific": torch.tensor([False, False, True, True]),
        }
        gain, ap, prevalence, name = _best_prevalence_corrected_concept(scores, labels)
        self.assertEqual(name, "specific")
        self.assertAlmostEqual(ap, 1.0)
        self.assertAlmostEqual(prevalence, 0.5)
        self.assertAlmostEqual(gain, 0.5)

    def test_ratio_of_means_is_not_dominated_by_tiny_kl_row(self):
        ratio = _isolation_ratio_of_means([1.0, 1.0], [1e-12, 1.0], floor=1e-4)
        self.assertAlmostEqual(ratio, 1.0 / np.sqrt(0.5), places=6)
        self.assertEqual(_finite_median([-100.0, 1.0, 2.0]), 1.0)

    def test_absorption_threshold_respects_tied_false_positive_budget(self):
        scores = torch.tensor([0.9, 0.8, 0.8, 0.7])
        labels = torch.tensor([True, False, False, True])
        available = torch.ones(4, dtype=torch.bool)

        strict = _best_recovery_threshold(scores, labels, available, false_positive_budget=0)
        self.assertEqual(strict["true_positive_gain"], 1)
        self.assertEqual(strict["false_positive_gain"], 0)
        self.assertAlmostEqual(strict["threshold"], 0.9, places=6)

        relaxed = _best_recovery_threshold(scores, labels, available, false_positive_budget=2)
        self.assertEqual(relaxed["true_positive_gain"], 2)
        self.assertEqual(relaxed["false_positive_gain"], 2)


class PlotTests(unittest.TestCase):
    def test_sae_plot_uses_explicit_k_axis(self):
        rows = [
            {"model": "base", "model_seed": 42, "layer": 6, "expansion": 8, "sae_seed": 42, "k": 8, "metric": 0.3},
            {"model": "base", "model_seed": 42, "layer": 6, "expansion": 8, "sae_seed": 42, "k": 32, "metric": 0.1},
        ]
        with tempfile.TemporaryDirectory() as directory:
            plot_metric_by_k(directory, rows, "metric", "curve.png")
            self.assertTrue((Path(directory) / "eval" / "curve.png").exists())


if __name__ == "__main__":
    unittest.main()
