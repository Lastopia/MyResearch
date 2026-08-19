from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from cfg import get_config, validate_config
from model.factory import build_model
from model.positional.cable import CABLE
from model.positional.ra_cable import RACABLE
from pipeline.data import (
    generate_multi_query_retrieval_batches,
    generate_nested_retrieval_batches,
    generate_position_swap_batches,
    generate_synthetic_tokens,
)
from tools.checkpoint import assert_checkpoint_compatible, save_checkpoint


class ControlInvarianceTests(unittest.TestCase):
    def test_ra_cable_lite_gates_only_the_configured_upper_layers(self) -> None:
        cfg = get_config()
        cfg["model"].update(
            {
                "n_layer": 4,
                "n_head": 2,
                "n_embd": 16,
                "ffn_dim": 32,
            }
        )
        cfg["position"]["ra_lite_layers"] = 2
        cfg["run"]["method"] = "ra_cable_lite"
        model = build_model(cfg)

        self.assertTrue(
            all(
                type(block.attention.position) is CABLE
                for block in model.blocks[:2]
            )
        )
        self.assertTrue(
            all(
                isinstance(block.attention.position, RACABLE)
                for block in model.blocks[2:]
            )
        )
        inputs = torch.randint(0, 32, (2, 8))
        output = model(inputs, inputs)
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()

        artifacts = model(inputs, return_artifacts=True)["artifacts"]
        self.assertTrue(
            all(item["relevance_gate"] is None for item in artifacts[:2])
        )
        self.assertTrue(
            all(item["relevance_gate"] is not None for item in artifacts[2:])
        )

    def test_ra_cable_lite_layer_count_is_validated(self) -> None:
        cfg = get_config()
        cfg["model"]["n_layer"] = 4
        cfg["position"]["ra_lite_layers"] = 5
        with self.assertRaisesRegex(ValueError, "ra_lite_layers"):
            validate_config(cfg)

    def test_ra_cable_lite_checkpoint_tracks_adaptive_layer_count(self) -> None:
        cfg = get_config()
        cfg["run"]["method"] = "ra_cable_lite"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pretrain_final.pt"
            save_checkpoint(
                path,
                model=torch.nn.Linear(2, 2),
                cfg=cfg,
                step=1,
                tokens_seen=8,
            )
            assert_checkpoint_compatible(path, cfg)
            changed = get_config()
            changed["run"]["method"] = "ra_cable_lite"
            changed["position"]["ra_lite_layers"] = 4
            with self.assertRaises(RuntimeError):
                assert_checkpoint_compatible(path, changed)

    def test_checkpoint_reuse_requires_matching_config(self) -> None:
        cfg = get_config()
        cfg["run"]["method"] = "rope"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pretrain_final.pt"
            model = torch.nn.Linear(2, 2)
            save_checkpoint(
                path,
                model=model,
                cfg=cfg,
                step=1,
                tokens_seen=8,
            )
            assert_checkpoint_compatible(path, cfg)
            interval_change = get_config()
            interval_change["run"]["method"] = "rope"
            interval_change["train"]["log_interval"] = 7
            interval_change["train"]["eval_interval"] = 11
            interval_change["train"]["save_interval"] = 13
            assert_checkpoint_compatible(path, interval_change)
            changed = get_config()
            changed["run"]["method"] = "rope"
            changed["model"]["n_layer"] += 1
            with self.assertRaises(RuntimeError):
                assert_checkpoint_compatible(path, changed)

    def test_shared_parameters_match_across_methods(self) -> None:
        cfg = get_config()
        cfg["model"].update(
            {
                "n_layer": 1,
                "n_head": 2,
                "n_embd": 16,
                "ffn_dim": 32,
            }
        )
        cfg["run"]["seed"] = 123
        cfg["run"]["method"] = "alibi"
        alibi = build_model(cfg)
        cfg["run"]["method"] = "ra_cable"
        ra_cable = build_model(cfg)
        alibi_state = alibi.state_dict()
        ra_state = ra_cable.state_dict()
        shared = sorted(
            key
            for key in set(alibi_state) & set(ra_state)
            if ".position." not in key
        )
        self.assertTrue(shared)
        for key in shared:
            self.assertEqual(alibi_state[key].shape, ra_state[key].shape)
            torch.testing.assert_close(alibi_state[key], ra_state[key])

    def test_data_is_independent_of_model_method_and_seed(self) -> None:
        cfg = get_config()
        first = generate_synthetic_tokens(
            128,
            int(cfg["data"]["vocab_size"]),
            int(cfg["data"]["seed"]),
        )
        cfg["run"]["method"] = "ra_cable"
        cfg["run"]["seed"] = 999
        second = generate_synthetic_tokens(
            128,
            int(cfg["data"]["vocab_size"]),
            int(cfg["data"]["seed"]),
        )
        self.assertTrue(torch.equal(first, second))

    def test_length_matched_retrieval_keeps_identity_and_covers_positions(self) -> None:
        batches = generate_nested_retrieval_batches(
            samples=6,
            lengths=[64, 128, 256],
            vocab_size=256,
            num_pairs=4,
            seed=7,
        )
        for length in (128, 256):
            self.assertTrue(
                torch.equal(
                    batches[64].labels,
                    batches[length].labels,
                )
            )
            for sample, position in enumerate(
                batches[length].relevant_positions.tolist()
            ):
                self.assertEqual(
                    int(batches[length].labels[sample]),
                    int(batches[length].input_ids[sample, position]),
                )
        fractions = batches[256].relevant_positions.float() / 255
        self.assertTrue(bool((fractions < 0.25).any()))
        self.assertTrue(bool(((fractions > 0.4) & (fractions < 0.6)).any()))
        self.assertTrue(bool((fractions > 0.75).any()))

    def test_multi_query_controls_have_shared_prefix_distractors(self) -> None:
        batch = generate_multi_query_retrieval_batches(
            samples=2,
            lengths=[128],
            vocab_size=256,
            queries_per_sample=3,
            similar_distractors=2,
            seed=17,
        )[128]
        self.assertEqual(tuple(batch.labels.shape), (2, 3))
        self.assertEqual(tuple(batch.distractor_positions.shape), (2, 3, 2))
        for sample in range(2):
            for query in range(3):
                query_suffix_position = int(
                    batch.query_positions[sample, query]
                )
                relevant_value_position = int(
                    batch.relevant_positions[sample, query]
                )
                self.assertEqual(
                    int(batch.input_ids[sample, query_suffix_position - 1]),
                    int(batch.input_ids[sample, relevant_value_position - 2]),
                )
                for distractor_value_position in batch.distractor_positions[
                    sample,
                    query,
                ].tolist():
                    self.assertEqual(
                        int(batch.input_ids[sample, query_suffix_position - 1]),
                        int(
                            batch.input_ids[
                                sample,
                                distractor_value_position - 2,
                            ]
                        ),
                    )

    def test_position_swap_exchanges_only_target_and_near_distractor(self) -> None:
        original, swapped = generate_position_swap_batches(
            samples=3,
            lengths=[64],
            vocab_size=256,
            num_pairs=4,
            seed=23,
        )[64]
        self.assertTrue(torch.equal(original.labels, swapped.labels))
        self.assertTrue(
            torch.equal(original.relevant_positions, swapped.distractor_positions)
        )
        self.assertTrue(
            torch.equal(original.distractor_positions, swapped.relevant_positions)
        )
        for sample in range(3):
            new_target = int(swapped.relevant_positions[sample])
            self.assertEqual(
                int(swapped.input_ids[sample, new_target]),
                int(swapped.labels[sample]),
            )


if __name__ == "__main__":
    unittest.main()
