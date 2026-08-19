from __future__ import annotations

import unittest

import torch

from cfg import get_config
from model.factory import build_model


class AttentionOutputTests(unittest.TestCase):
    def _model(self, method: str) -> torch.nn.Module:
        cfg = get_config()
        cfg["run"]["method"] = method
        cfg["model"].update(
            {
                "n_layer": 1,
                "n_head": 2,
                "n_embd": 16,
                "ffn_dim": 32,
                "max_seq_len": 32,
            }
        )
        return build_model(cfg).eval()

    def test_artifact_switch_does_not_change_logits(self) -> None:
        model = self._model("alibi")
        inputs = torch.randint(0, 256, (2, 8))
        plain = model(inputs)["logits"]
        audited = model(inputs, return_artifacts=True)["logits"]
        torch.testing.assert_close(plain, audited)

    def test_attention_is_strictly_causal(self) -> None:
        model = self._model("alibi")
        attention = model(
            torch.randint(0, 256, (2, 8)),
            return_artifacts=True,
        )["artifacts"][0]["attention"]
        upper = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
        self.assertTrue(torch.equal(attention[..., upper], torch.zeros_like(attention[..., upper])))

    def test_content_bias_total_relationship_on_causal_entries(self) -> None:
        model = self._model("alibi")
        artifact = model(
            torch.randint(0, 256, (1, 8)),
            return_artifacts=True,
        )["artifacts"][0]
        causal = torch.tril(torch.ones(8, 8, dtype=torch.bool))
        expected = artifact["content_logits"] + artifact["position_bias"]
        torch.testing.assert_close(
            artifact["total_logits"][..., causal],
            expected[..., causal],
        )

    def test_rope_has_no_additive_position_bias(self) -> None:
        model = self._model("rope")
        artifact = model(
            torch.randint(0, 256, (1, 8)),
            return_artifacts=True,
        )["artifacts"][0]
        self.assertIsNone(artifact["position_bias"])

    def test_kv_cache_matches_full_forward_for_all_methods(self) -> None:
        methods = (
            "rope",
            "alibi",
            "cable",
            "ra_cable",
            "ra_cable_lite",
        )
        inputs = torch.randint(0, 256, (1, 7))
        for method in methods:
            with self.subTest(method=method):
                model = self._model(method)
                full = model(inputs)["logits"]
                prefix = model(inputs[:, :4], use_cache=True)
                past = prefix["past_key_values"]
                self.assertEqual(past[0]["key"].shape[-2], 4)
                for position in range(4, inputs.shape[1]):
                    incremental = model(
                        inputs[:, position : position + 1],
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = incremental["past_key_values"]
                    torch.testing.assert_close(
                        incremental["logits"][:, -1, :],
                        full[:, position, :],
                        rtol=1e-4,
                        atol=1e-5,
                    )
                self.assertEqual(past[0]["key"].shape[-2], inputs.shape[1])


if __name__ == "__main__":
    unittest.main()
