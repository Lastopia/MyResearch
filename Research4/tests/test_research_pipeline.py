from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest
import torch

import dual_axis_transformer.runner as runner_module
from cfg import CFG
from dual_axis_transformer.research_model import (
    ALL_METHODS,
    BUS_METHODS,
    CausalAttention,
    ResearchModelConfig,
    build_model,
    initialize_named_parameters,
    parameter_count,
)
from dual_axis_transformer.runner import (
    CumulativeTrainingTimer,
    SmoothMaxLossBalancer,
    _load_best_model,
    _load_latest,
    _optimizer,
    _save_best_model,
    _save_checkpoint,
    _scheduler,
    _semantic_hash,
    run_one,
)
from dual_axis_transformer.synthetic_data import (
    DeterministicBatcher,
    ensure_synthetic_dataset,
    load_examples,
)
from dual_axis_transformer.storage import checkpoint_dir_for_run


def tiny_model_config(method: str) -> ResearchModelConfig:
    return ResearchModelConfig(
        vocab_size=96,
        max_length=24,
        method=method,
        num_layers=1,
        d_model=32,
        d_ff=64,
        num_heads=4,
        slot_dim=32,
        num_bus_slots=2,
        bus_heads=2,
        bus_layers=1,
        concept_residual_dim=4,
        dropout=0.0,
    )


def test_scheduled_cuda_failure_never_silently_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONCEPT_BUS_DEVICE", "cuda:0")
    monkeypatch.setattr(
        runner_module,
        "_probe_device",
        lambda _device: (False, "synthetic CUDA failure"),
    )
    with pytest.raises(RuntimeError, match="failed its runtime probe"):
        runner_module.select_device()


def test_rope_attention_is_strictly_causal() -> None:
    torch.manual_seed(1)
    attention = CausalAttention(32, 4, 0.0, 10000.0, False).eval()
    first = torch.randn(2, 7, 32)
    second = first.clone()
    second[:, 5:] = torch.randn_like(second[:, 5:])
    mask = torch.ones(2, 7, dtype=torch.long)
    with torch.no_grad():
        first_output = attention(first, mask)
        second_output = attention(second, mask)
    torch.testing.assert_close(first_output[:, :5], second_output[:, :5])


def test_rope_cache_is_reused_and_excluded_from_checkpoints() -> None:
    attention = CausalAttention(32, 4, 0.0, 10000.0, False).eval()
    with torch.no_grad():
        attention(torch.randn(2, 9, 32), None)
        cache_pointer = attention.rope_cosine.data_ptr()
        attention(torch.randn(2, 5, 32), None)
    assert attention.rope_cosine.shape[-2] == 9
    assert attention.rope_cosine.data_ptr() == cache_pointer
    assert all("rope_cosine" not in key for key in attention.state_dict())
    assert all("rope_sine" not in key for key in attention.state_dict())


def test_self_only_attention_fast_path_matches_diagonal_attention() -> None:
    torch.manual_seed(2)
    attention = CausalAttention(32, 4, 0.0, 10000.0, False).eval()
    hidden = torch.randn(2, 7, 32)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0]],
        dtype=torch.long,
    )
    with torch.no_grad():
        actual = attention(hidden, mask, attention_mode="self")
        value = attention.qkv(hidden).chunk(3, dim=-1)[2]
        expected = attention.output(value * mask[..., None])
    torch.testing.assert_close(actual, expected)


def test_concept_methods_reject_nondeterministic_counterfactual_dropout() -> None:
    values = tiny_model_config("concept_bus_v2").to_dict()
    values["dropout"] = 0.1
    with pytest.raises(ValueError, match="dropout=0"):
        ResearchModelConfig(**values)


def test_v2_uses_complete_ffn_and_is_strictly_causal() -> None:
    torch.manual_seed(3)
    model = build_model(tiny_model_config("concept_bus_v2")).eval()
    bus_blocks = [block for block in model.blocks if block.uses_bus]
    assert len(bus_blocks) == 1
    assert bus_blocks[0].ffn.up.out_features == model.config.d_ff
    first = torch.randint(0, 96, (2, 10))
    second = first.clone()
    second[:, 7:] = torch.randint(0, 96, second[:, 7:].shape)
    mask = torch.ones_like(first)
    with torch.no_grad():
        first_trace = model(first, mask, return_trace=True).trace
        second_trace = model(second, mask, return_trace=True).trace
    assert first_trace is not None and second_trace is not None
    torch.testing.assert_close(
        first_trace["bus_states"][:, :7], second_trace["bus_states"][:, :7]
    )


def test_task_and_concept_losses_reach_the_extra_bus_attention() -> None:
    model = build_model(tiny_model_config("concept_bus_v2"))
    ids = torch.randint(0, 96, (2, 10))
    mask = torch.ones_like(ids)
    output = model(ids, mask)
    assert output.country_swap_country_logits is not None
    loss = (
        output.country_logits.square().mean()
        + output.concept_logits.square().mean()
        + output.country_swap_country_logits.square().mean()
    )
    loss.backward()
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if ".ffn.concept_attention." in name
    ]
    assert gradients and all(value is not None for value in gradients)
    assert sum(float(value.abs().sum()) for value in gradients if value is not None) > 0
    write_gradient = dict(model.named_parameters())[
        "blocks.0.ffn.concept_readout.write_codes"
    ].grad
    assert write_gradient is not None and float(write_gradient.abs().sum()) > 0


def test_projector_control_matches_attention_parameters_but_blocks_communication() -> None:
    control = build_model(tiny_model_config("concept_projector"))
    treatment = build_model(tiny_model_config("concept_bus_v2"))
    assert control.blocks[0].ffn.concept_attention is not None
    assert treatment.blocks[0].ffn.concept_attention is not None
    assert control.blocks[0].ffn.concept_attention.communication is False
    assert treatment.blocks[0].ffn.concept_attention.communication is True
    assert parameter_count(control) == parameter_count(treatment)
    assert control.blocks[0].ffn.private_down.out_features == 28
    assert treatment.blocks[0].ffn.projector.projection.in_features == 64


def test_v2_concept_readout_has_no_pre_attention_bypass() -> None:
    model = build_model(tiny_model_config("concept_bus_v2")).eval()
    attention = model.blocks[0].ffn.concept_attention
    assert attention is not None
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.zero_()
        inputs = torch.randint(0, 96, (2, 8))
        output = model(inputs, torch.ones_like(inputs), return_trace=True)
    assert output.trace is not None
    assert float(output.trace["projected_slots"].abs().sum()) > 0
    assert float(output.trace["bus_states"].abs().sum()) == 0


def test_zero_bus_fast_path_matches_audited_path() -> None:
    model = build_model(tiny_model_config("concept_bus_v2")).eval()
    inputs = torch.randint(0, 96, (3, 10))
    mask = torch.ones_like(inputs)
    with torch.no_grad():
        fast = model(inputs, mask, intervention="zero_bus")
        audited = model(
            inputs,
            mask,
            intervention="zero_bus",
            return_trace=True,
        )
    assert fast.trace is None
    assert audited.trace is not None
    torch.testing.assert_close(fast.country_logits, audited.country_logits)
    torch.testing.assert_close(fast.color_logits, audited.color_logits)


def test_smoothmax_balancer_prioritizes_the_weakest_normalized_objective() -> None:
    balancer = SmoothMaxLossBalancer(temperature=0.25, ema_decay=0.99)
    balancer.ema = {"task": 1.0, "concept": 1.0}
    total, diagnostics = balancer.balance(
        {
            "task": torch.tensor(1.0, requires_grad=True),
            "concept": torch.tensor(2.0, requires_grad=True),
        }
    )
    assert torch.isfinite(total)
    assert diagnostics["balance_weight_concept"] > diagnostics["balance_weight_task"]


def test_smoothmax_single_objective_is_exactly_the_original_task_loss() -> None:
    balancer = SmoothMaxLossBalancer(temperature=0.25, ema_decay=0.99)
    for value in (2.0, 0.7, 0.1):
        task = torch.tensor(value, requires_grad=True)
        total, _ = balancer.balance({"task": task})
        torch.testing.assert_close(total, task)


def test_loss_balance_ema_updates_once_per_effective_batch() -> None:
    balancer = SmoothMaxLossBalancer(temperature=0.25, ema_decay=0.5)
    balancer.ema = {"task": 1.0, "concept": 1.0}
    for task, concept in ((2.0, 4.0), (4.0, 8.0)):
        balancer.balance(
            {
                "task": torch.tensor(task),
                "concept": torch.tensor(concept),
            },
            update_ema=False,
        )
    assert balancer.ema == {"task": 1.0, "concept": 1.0}
    balancer.update_ema({"task": 3.0, "concept": 6.0})
    assert balancer.ema == {"task": 2.0, "concept": 3.5}


def test_in_forward_counterfactuals_equal_full_intervention_forwards() -> None:
    model = build_model(tiny_model_config("concept_bus_v2")).eval()
    inputs = torch.randint(0, 96, (3, 10))
    mask = torch.ones_like(inputs)
    with torch.no_grad():
        normal = model(inputs, mask)
        country = model(inputs, mask, intervention="swap_country")
        color = model(inputs, mask, intervention="swap_color")
    torch.testing.assert_close(
        normal.country_swap_country_logits, country.country_logits
    )
    torch.testing.assert_close(normal.country_swap_color_logits, country.color_logits)
    torch.testing.assert_close(normal.color_swap_country_logits, color.country_logits)
    torch.testing.assert_close(normal.color_swap_color_logits, color.color_logits)


def test_skipping_training_only_counterfactuals_preserves_predictions() -> None:
    model = build_model(tiny_model_config("concept_bus_v2")).eval()
    inputs = torch.randint(0, 96, (3, 10))
    mask = torch.ones_like(inputs)
    with torch.no_grad():
        complete = model(inputs, mask)
        economical = model(inputs, mask, compute_counterfactuals=False)
    assert economical.country_swap_country_logits is None
    assert economical.color_swap_color_logits is None
    torch.testing.assert_close(complete.country_logits, economical.country_logits)
    torch.testing.assert_close(complete.color_logits, economical.color_logits)
    torch.testing.assert_close(complete.concept_logits, economical.concept_logits)


def test_padding_does_not_change_last_real_token_predictions() -> None:
    model = build_model(tiny_model_config("concept_bus_v2")).eval()
    short = torch.randint(1, 96, (2, 7))
    padded = torch.cat((short, torch.zeros(2, 5, dtype=torch.long)), dim=1)
    short_mask = torch.ones_like(short)
    padded_mask = torch.cat((short_mask, torch.zeros(2, 5, dtype=torch.long)), dim=1)
    with torch.no_grad():
        first = model(short, short_mask)
        second = model(padded, padded_mask)
    torch.testing.assert_close(first.country_logits, second.country_logits)
    torch.testing.assert_close(first.color_logits, second.color_logits)
    torch.testing.assert_close(first.concept_logits, second.concept_logits)


def test_all_registered_methods_have_compatible_task_interface() -> None:
    input_ids = torch.randint(0, 96, (2, 12))
    attention_mask = torch.ones_like(input_ids)
    for method in sorted(ALL_METHODS):
        model = build_model(tiny_model_config(method)).eval()
        output = model(input_ids, attention_mask)
        assert output.country_logits.shape == (2, 6)
        assert output.color_logits.shape == (2, 6)
        assert (output.concept_logits is not None) == (
            method in BUS_METHODS or method == "concept_aux"
        )


def test_parameter_matching_is_within_one_percent() -> None:
    target = build_model(tiny_model_config("concept_bus_v2"))
    matched = build_model(tiny_model_config("parameter_matched"))
    difference = abs(parameter_count(target) - parameter_count(matched))
    assert difference / parameter_count(target) <= 0.01


def test_named_initialization_pairs_shared_parameters() -> None:
    standard = build_model(tiny_model_config("standard"))
    full = build_model(tiny_model_config("concept_bus_v2"))
    initialize_named_parameters(standard, 11)
    initialize_named_parameters(full, 11)
    standard_parameters = dict(standard.named_parameters())
    full_parameters = dict(full.named_parameters())
    shared = set(standard_parameters) & set(full_parameters)
    assert "embedding.weight" in shared
    for name in shared:
        if standard_parameters[name].shape == full_parameters[name].shape:
            torch.testing.assert_close(standard_parameters[name], full_parameters[name])


def test_dataset_and_batch_order_are_reproducible(tmp_path: Path) -> None:
    config = {
        "train_size": 40,
        "validation_size": 10,
        "test_size": 10,
        "max_length": 32,
        "generation_seed": 123,
        "unknown_fraction": 0.2,
    }
    bundle = ensure_synthetic_dataset(tmp_path / "data", config)
    again = ensure_synthetic_dataset(tmp_path / "data", config)
    assert bundle.manifest_path == again.manifest_path
    assert json.loads(bundle.manifest_path.read_text(encoding="utf-8"))["splits"]["train"]["size"] == 40
    examples = load_examples(bundle.split_paths["train"])
    validation = load_examples(bundle.split_paths["validation"])
    assert {item.template_family for item in examples}.isdisjoint(
        {item.template_family for item in validation}
    )
    first = DeterministicBatcher(
        examples, bundle.vocabulary, batch_size=4, max_length=32, seed=9
    )
    second = DeterministicBatcher(
        examples, bundle.vocabulary, batch_size=4, max_length=32, seed=9
    )
    torch.testing.assert_close(first.next_batch()["input_ids"], second.next_batch()["input_ids"])


def test_micro_batch_partition_is_part_of_run_identity() -> None:
    first = {"runtime": {"micro_batch_size": 64, "gradient_accumulation_steps": 2}}
    second = {"runtime": {"micro_batch_size": 32, "gradient_accumulation_steps": 4}}
    assert _semantic_hash(first, 100) != _semantic_hash(second, 100)


def test_two_step_one_click_run_writes_recoverable_artifacts(tmp_path: Path) -> None:
    cfg = copy.deepcopy(CFG)
    cfg["sizes"]["test"] = {
        "stage": "test",
        "methods": ["concept_bus_v2"],
        "seeds": [11],
        "data": {
            "train_size": 32,
            "validation_size": 8,
            "test_size": 8,
            "max_length": 24,
            "generation_seed": 4,
            "unknown_fraction": 0.0,
        },
        "model": {
            "num_layers": 1,
            "d_model": 32,
            "d_ff": 64,
            "num_heads": 4,
            "slot_dim": 32,
            "num_bus_slots": 2,
            "bus_heads": 2,
            "bus_layers": 1,
            "concept_residual_dim": 4,
            "dropout": 0.0,
        },
        "train": {
            "max_steps": 2,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "warmup_fraction": 0.5,
            "log_interval_steps": 1,
            "eval_interval_steps": 2,
            "checkpoint_interval_minutes": 120,
        },
    }
    cfg["logging"]["console_mode"] = "quiet"
    run_dir = run_one(
        cfg,
        size_name="test",
        method="concept_bus_v2",
        seed=11,
        output_root=tmp_path / "output",
    )
    checkpoint_dir = checkpoint_dir_for_run(cfg, tmp_path / "output", run_dir)
    assert (run_dir / "metrics" / "final.json").exists()
    assert (checkpoint_dir / "final.pt").exists()
    assert (checkpoint_dir / "best.pt").exists()
    assert (checkpoint_dir / "best.json").exists()
    assert (checkpoint_dir / "latest.json").exists()
    assert not list(checkpoint_dir.glob("step_*.pt"))
    assert not (run_dir / "checkpoints").exists()
    assert run_dir.is_relative_to(tmp_path / "output")
    assert checkpoint_dir.is_relative_to(tmp_path / "ckpt")
    assert any((tmp_path / "data" / "dual_tag").iterdir())
    assert "TRAIN" in (run_dir / "logs" / "train.log").read_text(encoding="utf-8")
    final = json.loads(
        (run_dir / "metrics" / "final.json").read_text(encoding="utf-8")
    )
    assert final["metrics"]["selected_checkpoint_step"] == 2
    assert math.isfinite(final["metrics"]["selected_validation_loss"])


def test_run_one_tests_the_best_validation_weights_not_the_last_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = copy.deepcopy(CFG)
    cfg["sizes"]["selection_test"] = {
        "stage": "selection_test",
        "methods": ["standard"],
        "seeds": [11],
        "data": {
            "train_size": 32,
            "validation_size": 8,
            "test_size": 8,
            "max_length": 24,
            "generation_seed": 404,
            "unknown_fraction": 0.0,
            "template_ood": False,
        },
        "model": {
            "num_layers": 1,
            "d_model": 32,
            "d_ff": 64,
            "num_heads": 4,
            "slot_dim": 32,
            "num_bus_slots": 2,
            "bus_heads": 2,
            "bus_layers": 1,
            "concept_residual_dim": 4,
            "dropout": 0.0,
        },
        "train": {
            "max_steps": 2,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "warmup_fraction": 0.5,
            "log_interval_steps": 1,
            "eval_interval_steps": 1,
            "monitor_validation_examples": 8,
            "checkpoint_interval_minutes": 120,
        },
    }
    cfg["logging"]["console_mode"] = "quiet"
    validation_calls = 0
    first_step_state: dict[str, torch.Tensor] = {}

    def controlled_evaluate(
        model: torch.nn.Module,
        examples: list[object],
        _bundle: object,
        _config: dict[str, object],
        _device: torch.device,
        *,
        return_predictions: bool = False,
        **_kwargs: object,
    ) -> object:
        nonlocal validation_calls
        metrics = {
            "loss": 0.1 if validation_calls == 0 else 1.0,
            "country_accuracy": 0.5,
            "color_accuracy": 0.5,
            "exact_accuracy": 0.25,
        }
        if return_predictions:
            assert first_step_state
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value.cpu(), first_step_state[name])
            predictions = torch.zeros(len(examples), dtype=torch.long)
            return metrics, (predictions, predictions.clone())
        validation_calls += 1
        if validation_calls == 1:
            first_step_state.update(
                {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            )
        return metrics

    monkeypatch.setattr(runner_module, "evaluate", controlled_evaluate)
    run_dir = run_one(
        cfg,
        size_name="selection_test",
        method="standard",
        seed=11,
        output_root=tmp_path / "output_selection",
    )
    final = json.loads(
        (run_dir / "metrics" / "final.json").read_text(encoding="utf-8")
    )
    assert validation_calls == 2
    assert final["metrics"]["selected_checkpoint_step"] == 1
    assert final["metrics"]["selected_validation_loss"] == 0.1


def test_checkpoint_restores_model_optimizer_rng_and_batch_order(tmp_path: Path) -> None:
    data_config = {
        "train_size": 24,
        "validation_size": 4,
        "test_size": 4,
        "max_length": 24,
        "generation_seed": 19,
        "unknown_fraction": 0.0,
        "template_ood": False,
    }
    bundle = ensure_synthetic_dataset(tmp_path / "data", data_config)
    examples = load_examples(bundle.split_paths["train"])
    batcher = DeterministicBatcher(
        examples, bundle.vocabulary, batch_size=4, max_length=24, seed=3
    )
    model = build_model(
        ResearchModelConfig(
            **{
                **tiny_model_config("concept_bus_v2").to_dict(),
                "vocab_size": len(bundle.vocabulary.tokens),
            }
        )
    )
    initialize_named_parameters(model, 11)
    train_config = {
        "train": {
            "max_steps": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "warmup_fraction": 0.5,
        },
        "optimizer": {"betas": [0.9, 0.95]},
    }
    optimizer = _optimizer(model, train_config)
    scheduler = _scheduler(optimizer, train_config)
    batch = batcher.next_batch()
    output = model(batch["input_ids"], batch["attention_mask"])
    loss = 0.5 * (
        torch.nn.functional.cross_entropy(output.country_logits, batch["country_targets"])
        + torch.nn.functional.cross_entropy(output.color_logits, batch["color_targets"])
    )
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected_parameters = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    control_batcher = DeterministicBatcher(
        examples, bundle.vocabulary, batch_size=4, max_length=24, seed=999
    )
    control_batcher.load_state_dict(batcher.state_dict())
    expected_next_batch = control_batcher.next_batch()["input_ids"]
    timer = CumulativeTrainingTimer()
    run_dir = tmp_path / "run"
    checkpoint_dir = tmp_path / "ckpt" / "test" / "seed11"
    _save_checkpoint(
        run_dir,
        checkpoint_dir=checkpoint_dir,
        name="step_00000001.pt",
        step=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        batcher=batcher,
        timer=timer,
        best_validation=1.0,
        config_hash="test-hash",
    )
    expected_random = torch.rand(4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    _ = batcher.next_batch()
    _ = torch.rand(100)
    step, best = _load_latest(
        run_dir,
        checkpoint_dir=checkpoint_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        batcher=batcher,
        timer=timer,
        config_hash="test-hash",
        device=torch.device("cpu"),
    )
    assert step == 1 and best == 1.0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_parameters[name])
    torch.testing.assert_close(batcher.next_batch()["input_ids"], expected_next_batch)
    torch.testing.assert_close(torch.rand(4), expected_random)


def test_best_validation_checkpoint_restores_selected_model(tmp_path: Path) -> None:
    model = build_model(tiny_model_config("concept_bus_v2"))
    initialize_named_parameters(model, 11)
    expected = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    checkpoint_dir = tmp_path / "ckpt" / "best"
    _save_best_model(
        checkpoint_dir=checkpoint_dir,
        model=model,
        step=500,
        validation_loss=0.25,
        config_hash="best-test",
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    step, loss = _load_best_model(
        checkpoint_dir=checkpoint_dir,
        model=model,
        config_hash="best-test",
        device=torch.device("cpu"),
    )
    assert step == 500
    assert loss == 0.25
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])
