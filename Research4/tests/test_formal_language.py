from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from cfg import CFG
from dual_axis_transformer.formal_data import (
    GPT2Tokenizer,
    TokenCorpus,
    ensure_formal_language_dataset,
)
from dual_axis_transformer.language_model import _sample_batch, language_model_config
from dual_axis_transformer.language_model import run_language_model_one


def _write_uint32(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (np.arange(count, dtype=np.uint32) % GPT2Tokenizer.vocab_size).tofile(path)


def test_gpt2_tokenizer_and_formal_model_config_are_fixed() -> None:
    assert GPT2Tokenizer.bos_id == GPT2Tokenizer.eos_id == 50256
    assert GPT2Tokenizer.vocab_size == 50257
    config = language_model_config(CFG, "concept_bus_v2", 512)
    assert (config.num_layers, config.d_model, config.d_ff) == (12, 768, 3072)
    assert config.vocab_size == 50257
    assert config.concept_residual_dim == 48
    assert config.bus_layers == 1


def test_medium_fair_recheck_keeps_the_38m_model_scale() -> None:
    config = language_model_config(CFG, "concept_bus_v2", 256, "medium")
    assert (config.num_layers, config.d_model, config.d_ff) == (12, 512, 2048)
    assert CFG["sizes"]["medium"]["language_backend"] == "none"


def test_formal_corpus_cache_is_validated_and_memory_mapped(tmp_path: Path) -> None:
    cfg = copy.deepcopy(CFG)
    settings = cfg["external"]["formal_language"]
    settings["train_tokens"] = 2048
    settings["validation_tokens"] = 1024
    cfg["external"]["allow_download"] = False
    root = tmp_path / "formal_language"
    train = root / "fineweb_edu_train.uint32"
    validation = root / "fineweb_edu_validation.uint32"
    external = root / "wikitext103_test.uint32"
    _write_uint32(train, 2048)
    _write_uint32(validation, 1024)
    _write_uint32(external, 2048)
    manifest = {
        "schema_version": 1,
        "dataset_id": settings["dataset_id"],
        "dataset_config": settings["dataset_config"],
        "dataset_revision": settings["dataset_revision"],
        "train_tokens": 2048,
        "validation_tokens": 1024,
        "external_dataset_id": settings["external_dataset_id"],
        "external_dataset_config": settings["external_dataset_config"],
        "external_dataset_revision": settings["external_dataset_revision"],
        "tokenizer": "tiktoken:gpt2",
        "vocab_size": 50257,
        "dtype": "uint32-little-endian",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bundle = ensure_formal_language_dataset(cfg, tmp_path)
    corpus = TokenCorpus(bundle.train_path)
    inputs, mask, targets = _sample_batch(
        corpus,
        batch_size=3,
        sequence_length=16,
        generator=torch.Generator().manual_seed(9),
    )
    assert inputs.shape == mask.shape == targets.shape == (3, 16)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_formal_language_path_runs_one_cpu_step(tmp_path: Path) -> None:
    cfg = copy.deepcopy(CFG)
    cfg["logging"]["console_mode"] = "quiet"
    cfg["external"]["allow_download"] = False
    settings = cfg["external"]["formal_language"]
    settings.update(
        {
            "train_tokens": 1024,
            "validation_tokens": 1024,
            "sequence_length": 16,
            "batch_size": 64,
            "validation_batches": 1,
            "external_validation_batches": 1,
            "validation_sequences": 64,
            "external_validation_sequences": 64,
            "log_interval_steps": 1,
            "eval_interval_steps": 1,
            "checkpoint_interval_minutes": 120,
        }
    )
    settings["model"] = {
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
    }
    data_root = tmp_path / "data" / "formal_language"
    _write_uint32(data_root / "fineweb_edu_train.uint32", 1024)
    _write_uint32(data_root / "fineweb_edu_validation.uint32", 1024)
    _write_uint32(data_root / "wikitext103_test.uint32", 2048)
    manifest = {
        "schema_version": 1,
        "dataset_id": settings["dataset_id"],
        "dataset_config": settings["dataset_config"],
        "dataset_revision": settings["dataset_revision"],
        "train_tokens": 1024,
        "validation_tokens": 1024,
        "external_dataset_id": settings["external_dataset_id"],
        "external_dataset_config": settings["external_dataset_config"],
        "external_dataset_revision": settings["external_dataset_revision"],
        "tokenizer": "tiktoken:gpt2",
        "vocab_size": 50257,
        "dtype": "uint32-little-endian",
    }
    (data_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cfg["paths"]["data_root"] = str(tmp_path / "data")
    cfg["paths"]["checkpoint_root"] = str(tmp_path / "ckpt")
    output = tmp_path / "output"
    run_dir = run_language_model_one(
        cfg, method="concept_bus_v2", seed=11, output_root=output
    )
    final = json.loads(
        (run_dir / "metrics" / "final.json").read_text(encoding="utf-8")
    )
    assert final["stage"] == "formal_language_model"
    assert "wikitext103_perplexity" in final["metrics"]
    assert not list(output.rglob("*.pt"))
    assert list((tmp_path / "ckpt").rglob("latest.pt"))
