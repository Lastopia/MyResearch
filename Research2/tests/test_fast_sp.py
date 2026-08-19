import unittest

import torch
import cfg as default_cfg

from model.attn import FastSPAttn, SPAttn, StdAttn, build_attn
from model.ffn import FastSPFFN, SPFFN, StdFFN, build_ffn
from model.model import Transformer
from model.factory import active_aliases, build_model
from pipeline.benchmark import _add_relative_metrics
from pipeline.train import forward_training_loss


def tiny_model_cfg():
    return {
        "n_layer": 2,
        "n_head": 4,
        "d_model": 32,
        "d_ff": 128,
        "block_size": 16,
        "vocab_size": 101,
        "dropout": 0.0,
        "position_encoding": "rope",
    }


def fast_ffn_cfg():
    return {
        "name": "fast_sp",
        "active_width": 8,
    }


def fast_attn_cfg():
    return {
        "name": "fast_sp",
        "window_size": 8,
        "chunk_size": 8,
    }


class FastSPFFNTests(unittest.TestCase):
    def test_builder_keeps_reference_and_fast_models_separate(self):
        cfg = tiny_model_cfg()
        self.assertIsInstance(build_ffn("sp", cfg, {"k_ratio": 0.08}), SPFFN)
        self.assertIsInstance(build_ffn("fast_sp", cfg, fast_ffn_cfg()), FastSPFFN)
        self.assertIsInstance(build_attn("sp", cfg, {"k": 8}), SPAttn)
        self.assertIsInstance(build_attn("fast_sp", cfg, fast_attn_cfg()), FastSPAttn)

    def test_compact_parameter_count_matches_fused_spark_formula(self):
        cfg = tiny_model_cfg()
        fast = FastSPFFN(cfg, fast_ffn_cfg())
        self.assertEqual(
            sum(parameter.numel() for parameter in fast.parameters()),
            3 * cfg["d_model"] * fast_ffn_cfg()["active_width"],
        )

    def test_portable_forward_backward_and_density(self):
        cfg = tiny_model_cfg()
        layer = FastSPFFN(cfg, fast_ffn_cfg())
        x = torch.randn(3, 16, cfg["d_model"], requires_grad=True)
        y, stats = layer(x, collect_stats=True)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())
        self.assertAlmostEqual(float(stats["act_l0"]), 8.0)
        self.assertAlmostEqual(float(stats["act_density"]), 0.0625)
        y.square().mean().backward()
        for parameter in layer.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


class FastSPAttentionTests(unittest.TestCase):
    def test_base_cable_fused_sdpa_matches_explicit_attention(self):
        torch.manual_seed(19)
        cfg = tiny_model_cfg()
        cfg["position_encoding"] = "cable"
        layer = StdAttn(cfg).eval()
        x = torch.randn(2, cfg["block_size"], cfg["d_model"])
        with torch.no_grad():
            fused_y = layer(x)
            explicit_y, weights = layer(x, need_weights=True)
        torch.testing.assert_close(fused_y, explicit_y, atol=1e-6, rtol=1e-5)
        self.assertEqual(
            weights.shape,
            (2, cfg["n_head"], cfg["block_size"], cfg["block_size"]),
        )

    def test_base_cable_fused_sdpa_trains_position_parameters(self):
        torch.manual_seed(21)
        cfg = tiny_model_cfg()
        cfg["position_encoding"] = "cable"
        layer = StdAttn(cfg)
        x = torch.randn(2, cfg["block_size"], cfg["d_model"], requires_grad=True)
        layer(x).square().mean().backward()
        self.assertIsNotNone(layer.cable_layer.weight.grad)
        self.assertIsNotNone(layer.cable_layer_scale.weight.grad)
        self.assertTrue(torch.isfinite(layer.cable_layer.weight.grad).all())
        self.assertTrue(torch.isfinite(layer.cable_layer_scale.weight.grad).all())

    def test_forward_backward_weights_are_causal(self):
        cfg = tiny_model_cfg()
        layer = FastSPAttn(cfg, fast_attn_cfg())
        x = torch.randn(2, 16, cfg["d_model"], requires_grad=True)
        y, weights = layer(x, need_weights=True)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(weights.shape, (2, cfg["n_head"], 16, 16))
        future = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)
        self.assertEqual(
            torch.count_nonzero(weights.masked_select(future.view(1, 1, 16, 16))).item(),
            0,
        )
        y.square().mean().backward()
        for parameter in layer.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_future_inputs_do_not_change_prefix_outputs(self):
        torch.manual_seed(11)
        cfg = tiny_model_cfg()
        layer = FastSPAttn(cfg, fast_attn_cfg()).eval()
        original = torch.randn(1, 16, cfg["d_model"])
        changed = original.clone()
        changed[:, 7:] = torch.randn_like(changed[:, 7:]) * 100.0
        with torch.no_grad():
            original_y = layer(original)
            changed_y = layer(changed)
        torch.testing.assert_close(original_y[:, :7], changed_y[:, :7])

    def test_supports_a_partial_final_attention_chunk(self):
        cfg = tiny_model_cfg()
        layer = FastSPAttn(cfg, fast_attn_cfg())
        output = layer(torch.randn(1, 15, cfg["d_model"]))
        self.assertEqual(output.shape, (1, 15, cfg["d_model"]))

    def test_cable_full_window_matches_dense_cable(self):
        torch.manual_seed(23)
        cfg = tiny_model_cfg()
        cfg["position_encoding"] = "cable"
        dense = StdAttn(cfg).eval()
        fast_cfg = {
            **fast_attn_cfg(),
            "window_size": cfg["block_size"],
            "use_flex_attention": False,
        }
        fast = FastSPAttn(cfg, fast_cfg).eval()
        fast.load_state_dict(dense.state_dict())
        x = torch.randn(2, cfg["block_size"], cfg["d_model"])
        with torch.no_grad():
            dense_y, dense_weights = dense(x, need_weights=True)
            fast_y, fast_weights = fast(x, need_weights=True)
        torch.testing.assert_close(fast_y, dense_y, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(fast_weights, dense_weights, atol=1e-6, rtol=1e-5)

    def test_cable_is_causal_and_trains_position_parameters(self):
        torch.manual_seed(29)
        cfg = tiny_model_cfg()
        cfg["position_encoding"] = "cable"
        layer = FastSPAttn(cfg, fast_attn_cfg())
        original = torch.randn(1, 16, cfg["d_model"], requires_grad=True)
        changed = original.detach().clone()
        changed[:, 7:] = torch.randn_like(changed[:, 7:]) * 100.0
        original_y = layer(original)
        with torch.no_grad():
            changed_y = layer(changed)
        torch.testing.assert_close(original_y[:, :7], changed_y[:, :7])
        original_y.square().mean().backward()
        self.assertIsNotNone(layer.cable_layer.weight.grad)
        self.assertIsNotNone(layer.cable_layer_scale.weight.grad)
        self.assertTrue(torch.isfinite(layer.cable_layer.weight.grad).all())
        self.assertTrue(torch.isfinite(layer.cable_layer_scale.weight.grad).all())


class FastSPTrainingTests(unittest.TestCase):
    def test_fast_sp_both_has_paired_rope_and_cable_experiments(self):
        aliases = active_aliases(vars(default_cfg))
        self.assertEqual(
            aliases,
            [
                "base_rope", "base_cable",
                "sp_both_rope", "sp_both_cable",
                "fast_sp_both_rope", "fast_sp_both_cable",
            ],
        )
        model = build_model(vars(default_cfg), "fast_sp_both_cable")
        self.assertIsInstance(model.blocks[0].attn, FastSPAttn)
        self.assertIsInstance(model.blocks[0].ffn, FastSPFFN)
        self.assertEqual(model.blocks[0].attn.position_encoding, "cable")

    def test_benchmark_relatives_are_paired_by_position_encoding(self):
        rows = [
            {"model": "base_rope", "block_forward_backward_ms": 10.0},
            {"model": "base_cable", "block_forward_backward_ms": 20.0},
            {"model": "sp_both_rope", "block_forward_backward_ms": 30.0},
            {"model": "sp_both_cable", "block_forward_backward_ms": 40.0},
            {"model": "fast_sp_both_rope", "block_forward_backward_ms": 15.0},
            {"model": "fast_sp_both_cable", "block_forward_backward_ms": 25.0},
        ]
        _add_relative_metrics(vars(default_cfg), rows)
        self.assertEqual(rows[4]["time_over_base_percent"], 50.0)
        self.assertEqual(rows[5]["time_over_base_percent"], 25.0)
        self.assertEqual(rows[4]["speedup_vs_original_sp"], 2.0)
        self.assertEqual(rows[5]["speedup_vs_original_sp"], 1.6)

    def test_long_context_preserves_task6_tokens_per_update(self):
        tokens = (
            default_cfg.train["batch_size"]
            * default_cfg.train["gradient_accumulation_steps"]
            * default_cfg.data["block_size"]
        )
        sae_tokens = (
            default_cfg.train["batch_size"]
            * default_cfg.sae["gradient_accumulation_steps"]
            * default_cfg.data["block_size"]
        )
        self.assertEqual(tokens, 32 * 256)
        self.assertEqual(sae_tokens, 32 * 256)
        self.assertGreater(
            default_cfg.data["block_size"],
            default_cfg.models["fast_sp_both"]["attn"]["window_size"],
        )

    def test_ce_training_skips_diagnostic_reductions(self):
        cfg = tiny_model_cfg()
        variant = {
            "attn": fast_attn_cfg(),
            "ffn": fast_ffn_cfg(),
            "loss": {"name": "ce"},
        }
        model = Transformer(cfg, variant)
        x = torch.randint(0, cfg["vocab_size"], (2, 16))
        y = torch.randint(0, cfg["vocab_size"], (2, 16))
        loss = forward_training_loss(model, x, y)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(model.last_aux), {"ce_loss"})
        loss.backward()


if __name__ == "__main__":
    unittest.main()
