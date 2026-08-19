from __future__ import annotations

import unittest

import torch

from cfg import get_config
from pipeline.evaluate import _evaluate_qasper
from pipeline.stats import _paired_comparisons
from pipeline.train import _learning_rate_lambda
from tools.metrics import (
    attention_radius,
    attention_sink_ratio_mask,
    bias_far_near_pair_violation_rate,
    bias_monotonic_violation_rate,
    context_adaptivity_score,
    false_exemption_rate,
    geometric_head_fit,
    mean_attention_distance,
    normalized_attention_distance,
    relevant_attention_advantage,
    relative_head_distance_drift,
    semantic_exemption_success_rate,
)


class MetricTests(unittest.TestCase):
    def test_inference_is_disabled_below_required_seed_count(self) -> None:
        records = {
            "cable": {42: {"metric": 1.0}},
            "ra_cable": {42: {"metric": 2.0}},
        }
        comparisons = _paired_comparisons(
            records,
            baseline_name="cable",
            method_names=["ra_cable"],
            bootstrap_samples=100,
            confidence=0.95,
            seed=0,
            minimum_inferential_seeds=8,
        )

        self.assertEqual(len(comparisons), 1)
        self.assertFalse(comparisons[0]["inferential"])
        self.assertIsNone(comparisons[0]["difference_ci_low"])
        self.assertIsNone(comparisons[0]["difference_ci_high"])
        self.assertIsNone(comparisons[0]["raw_p"])
        self.assertIsNone(comparisons[0]["holm_p"])

    def test_qasper_evaluator_scores_human_answer_continuation(self) -> None:
        class FakeModel(torch.nn.Module):
            def forward(
                self,
                input_ids: torch.Tensor,
                *,
                past_key_values: object | None = None,
                use_cache: bool = False,
                inference_scale: float | None = None,
            ) -> dict[str, object]:
                del past_key_values, inference_scale
                logits = torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    32,
                )
                logits[..., 5] = 10.0
                return {
                    "logits": logits,
                    "past_key_values": () if use_cache else None,
                }

        class FakeTokenizer:
            eos_token_id = 31

            def encode(
                self,
                text: str,
                *,
                add_special_tokens: bool,
            ) -> list[int]:
                del text, add_special_tokens
                return [7, 8]

            def decode(
                self,
                tokens: list[int],
                *,
                skip_special_tokens: bool,
            ) -> str:
                del skip_special_tokens
                return "answer" if tokens == [5] else "other"

        cfg = get_config()
        cfg["eval"]["qasper_max_answer_tokens"] = 4
        cfg["eval"]["qasper_generation_tokens"] = 1
        result = _evaluate_qasper(
            FakeModel(),
            cfg,
            examples=[
                {
                    "question": "human question",
                    "question_id": "q1",
                    "sample_sha256": "hash1",
                    "answer_texts": ["answer"],
                    "answer_token_ids": [torch.tensor([5])],
                    "document_token_ids": torch.arange(100) % 16,
                    "evidence_token_start": 10,
                    "evidence_token_end": 12,
                }
            ],
            tokenizer=FakeTokenizer(),
            length=32,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["token_f1"], 1.0)
        self.assertEqual(result["exact_match"], 1.0)
        self.assertAlmostEqual(result["evidence_utilization_gain"], 0.0)

    def test_learning_rate_warms_up_then_cosine_decays(self) -> None:
        values = [
            _learning_rate_lambda(step, 100, 10, 0.1)
            for step in range(101)
        ]
        self.assertLess(values[0], values[5])
        self.assertLess(values[5], values[9])
        self.assertAlmostEqual(values[9], 1.0)
        self.assertGreater(values[20], values[60])
        self.assertGreater(values[60], values[100])
        self.assertAlmostEqual(values[100], 0.1)

    def test_attention_distance_for_self_and_first_token_attention(self) -> None:
        identity = torch.eye(4).view(1, 1, 4, 4)
        self.assertEqual(float(mean_attention_distance(identity, 0.5)), 0.0)
        self.assertEqual(float(normalized_attention_distance(identity, 0.5)), 0.0)
        self.assertEqual(float(attention_radius(identity, 0.9, 0.5)), 0.0)

        first = torch.zeros(1, 1, 4, 4)
        first[..., 0] = 1.0
        self.assertAlmostEqual(
            float(mean_attention_distance(first, 0.5)),
            2.5,
        )
        self.assertAlmostEqual(
            float(normalized_attention_distance(first, 0.5)),
            1.0,
        )
        self.assertAlmostEqual(float(attention_radius(first, 0.9, 0.5)), 2.5)

    def test_head_geometry_and_drift(self) -> None:
        self.assertAlmostEqual(
            geometric_head_fit(torch.tensor([1.0, 2.0, 4.0, 8.0])),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            relative_head_distance_drift(
                torch.tensor([[1.0, 2.0], [2.0, 4.0], [4.0, 8.0]])
            ),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            context_adaptivity_score(
                torch.tensor([[1.0, 2.0], [2.0, 4.0]])
            ),
            0.0,
            places=6,
        )

    def test_monotonicity_and_exemption_metrics(self) -> None:
        bias = torch.zeros(1, 1, 4, 4)
        for query in range(4):
            bias[0, 0, query, : query + 1] = -torch.arange(
                query,
                -1,
                -1,
                dtype=torch.float32,
            )
        self.assertEqual(bias_monotonic_violation_rate(bias), 0.0)
        far = torch.tensor([0])
        near = torch.tensor([2])
        self.assertEqual(
            bias_far_near_pair_violation_rate(bias, far, near),
            0.0,
        )

        attention = torch.zeros(1, 1, 4, 4)
        attention[0, 0, -1, 0] = 0.8
        attention[0, 0, -1, 2] = 0.1
        relevant = torch.tensor([0])
        distractor = torch.tensor([2])
        self.assertAlmostEqual(
            relevant_attention_advantage(
                attention,
                relevant,
                distractor,
            ),
            0.7,
            places=6,
        )
        bias[0, 0, -1, 0] = 1.0
        bias[0, 0, -1, 2] = 0.0
        self.assertEqual(
            semantic_exemption_success_rate(bias, relevant, distractor),
            1.0,
        )
        gate = torch.zeros(1, 1, 4, 4)
        gate[0, 0, -1, 0] = 1.0
        gate[0, 0, -1, 1] = 1.0
        irrelevant_mask = torch.tensor([[False, True, True, False]])
        self.assertAlmostEqual(
            false_exemption_rate(
                gate,
                irrelevant_mask,
                threshold=0.5,
            ),
            0.5,
            places=6,
        )
        sink_mask = torch.tensor([[True, False, False, False]])
        self.assertEqual(
            attention_sink_ratio_mask(
                attention,
                sink_mask,
                query_fraction=0.25,
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
