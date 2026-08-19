from __future__ import annotations

import math
import unittest

import torch

from model.positional.alibi import ALiBi, alibi_slopes
from model.positional.cable import CABLE
from model.positional.dape_kerple import DAPEKerple
from model.positional.ra_cable import RACABLE, RACABLEStatic
from tools.metrics import bias_monotonic_violation_rate


def _inverse_softplus(value: float) -> float:
    return math.log(math.exp(value) - 1.0)


def _set_unit_cable(module: CABLE) -> None:
    with torch.no_grad():
        module.increment_proj.weight.zero_()
        module.increment_proj.bias.fill_(1.0)
        module.weight_proj.weight.zero_()
        module.weight_proj.bias.fill_(_inverse_softplus(1.0))


class PositionMethodTests(unittest.TestCase):
    def test_dape_kerple_matches_official_pairwise_interface(self) -> None:
        method = DAPEKerple(num_heads=2, mlp_width=4)
        content = torch.randn(1, 2, 5, 5)
        positions = torch.arange(5)
        full = method.build_bias(
            content,
            torch.empty(1, 5, 8),
            positions,
        )
        incremental, _ = method.build_incremental_bias(
            content[:, :, -1:, :],
            torch.empty(1, 1, 8),
            positions[-1:],
            positions,
            None,
        )

        self.assertEqual(full.bias.shape, content.shape)
        self.assertEqual(full.context_distance.shape, content.shape)
        torch.testing.assert_close(
            incremental.bias,
            full.bias[:, :, -1:, :],
        )
        self.assertIn("upstream_commit", full.aux)

    def test_static_ra_cable_gate_is_content_independent_and_parameter_matched(
        self,
    ) -> None:
        adaptive = RACABLE(hidden_size=8, num_heads=2)
        static = RACABLEStatic(hidden_size=8, num_heads=2)
        self.assertEqual(
            sum(parameter.numel() for parameter in adaptive.parameters()),
            sum(parameter.numel() for parameter in static.parameters()),
        )
        hidden = torch.randn(1, 4, 8)
        first = static.build_bias(
            torch.randn(1, 2, 4, 4),
            hidden,
            torch.arange(4),
        )
        second = static.build_bias(
            torch.randn(1, 2, 4, 4) * 100,
            hidden,
            torch.arange(4),
        )
        torch.testing.assert_close(
            first.relevance_gate,
            second.relevance_gate,
        )

    def test_alibi_slopes_match_power_of_two_sequence(self) -> None:
        expected = torch.tensor([0.5 ** (index + 1) for index in range(8)])
        torch.testing.assert_close(alibi_slopes(8), expected)

    def test_alibi_bias_is_monotonic(self) -> None:
        module = ALiBi(4)
        content = torch.zeros(2, 4, 8, 8)
        hidden = torch.zeros(2, 8, 16)
        output = module.build_bias(content, hidden, torch.arange(8))
        self.assertEqual(output.bias.shape, (1, 4, 8, 8))
        self.assertEqual(bias_monotonic_violation_rate(output.bias), 0.0)

    def test_cable6_matches_cumulative_relu_formula(self) -> None:
        module = CABLE(4, 2)
        _set_unit_cable(module)
        hidden = torch.zeros(1, 8, 4)
        content = torch.zeros(1, 2, 8, 8)
        output = module.build_bias(content, hidden, torch.arange(8))
        self.assertEqual(output.bias.shape, content.shape)
        self.assertAlmostEqual(
            float(output.bias[0, 0, -1, 0].detach()),
            -7.0,
            places=5,
        )
        self.assertEqual(bias_monotonic_violation_rate(output.bias), 0.0)

    def test_ra_cable_high_qk_similarity_reduces_distance_penalty(self) -> None:
        module = RACABLE(4, 2, gate_bias=0.0, sparsity_weight=0.0)
        _set_unit_cable(module)
        with torch.no_grad():
            module.log_gate_scale.fill_(_inverse_softplus(1.0))
            module.gate_bias.zero_()
        hidden = torch.zeros(2, 8, 4)
        content = torch.full((2, 2, 8, 8), -8.0)
        content[1].fill_(8.0)
        output = module.build_bias(content, hidden, torch.arange(8))
        low_similarity_penalty = output.bias[0, 0, -1, 0].abs()
        high_similarity_penalty = output.bias[1, 0, -1, 0].abs()
        self.assertLess(
            float(high_similarity_penalty.detach()),
            float(low_similarity_penalty.detach()),
        )
        self.assertGreater(
            float(output.relevance_gate[1, 0, -1, 0].detach()),
            float(output.relevance_gate[0, 0, -1, 0].detach()),
        )

    def test_ra_cable_shapes_and_regularization(self) -> None:
        hidden = torch.randn(2, 6, 8)
        content = torch.randn(2, 2, 6, 6)
        output = RACABLE(8, 2).build_bias(
            content,
            hidden,
            torch.arange(6),
        )
        self.assertEqual(output.bias.shape, content.shape)
        self.assertEqual(output.context_distance.shape, content.shape)
        self.assertEqual(output.relevance_gate.shape, content.shape)
        self.assertGreaterEqual(
            float(output.regularization_loss.detach()),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
