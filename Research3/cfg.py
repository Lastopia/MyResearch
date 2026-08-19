from __future__ import annotations

from copy import deepcopy
from typing import Any


# This is the only experiment configuration.
# Edit values here directly when the model, data budget, or hardware changes.
CONFIG: dict[str, Any] = {
    "run": {
        "task": "task1",
        "stage": "data",
        "methods": ["rope", "alibi", "cable", "ra_cable"],
        "extension_methods": [
            "ra_cable_lite",
            "ra_cable_static",
            "dape_kerple",
        ],
        "method": "rope",
        "seed": 42,
        "seeds": [42, 43],
        "device": "auto",
        "dtype": "bfloat16",
        "world_size": 1,
        "force": False,
        "require_cuda": True,
        # verify: check only; off: skip checks; install: allow pip installation.
        "bootstrap": "verify",
    },
    "paths": {
        # Set an absolute server path, e.g. /data/position-bias.
        # None uses POSITION_BIAS_PROJECT_ROOT or the repository directory.
        "project_root": None,
        "data_root": "data",
        "checkpoint_root": "checkpoints",
        "large_root": "large",
        "output_root": "output",
    },
    "data": {
        # Recommended controlled corpus for this comparison.
        "source": "fineweb_edu",
        "seed": 2025,
        "local_tokens_path": None,
        "tokenizer_name": "openai-community/gpt2",
        "tokenizer_revision": "e7da7f221d5bf496a48136c0cd264e630fe9fcc8",
        "fineweb_dataset": "HuggingFaceFW/fineweb-edu",
        "fineweb_config": "sample-10BT",
        "fineweb_revision": "v1.0.0",
        "wikitext_dataset": "Salesforce/wikitext",
        "wikitext_config": "wikitext-103-raw-v1",
        "wikitext_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "prepare_wikitext": True,
        "qasper_dataset": "allenai/qasper",
        "qasper_data_file": "qasper/qasper-validation.parquet",
        "qasper_revision": "3065362e337ded696bbb0171b073c73e513c9410",
        "qasper_split": "validation",
        "qasper_samples": 64,
        "prepare_qasper": True,
        "leakage_shingle_width": 13,
        "leakage_shingle_stride": 4,
        "leakage_min_shared_shingles": 32,
        "leakage_overlap_threshold": 0.20,
        "streaming": True,
        "shuffle_buffer": 10_000,
        "progress_interval_documents": 10_000,
        "token_dtype": "uint16",
        "vocab_size": 50_257,
        "block_size": 1_024,
        "train_tokens": 1_000_000_000,
        "valid_tokens": 5_000_000,
        "test_tokens": 5_000_000,
        "retrieval_train_samples": 4_096,
        "retrieval_eval_samples": 512,
        "num_key_value_pairs": 8,
        "retrieval_queries_per_sample": 4,
        "retrieval_similar_distractors": 3,
    },
    "bootstrap": {
        "install_missing_dependencies": False,
        "install_cuda_torch_if_gpu_detected": False,
        "cuda_torch_index_url": "https://download.pytorch.org/whl/cu130",
        "required_packages": {
            "torch": "torch>=2.2",
            "numpy": "numpy>=1.26",
            "psutil": "psutil>=5.9",
            "datasets": "datasets>=2.18",
            "transformers": "transformers>=4.40",
        },
    },
    "resources": {
        # Detect eligible GPUs automatically and run one independent method
        # job per selected GPU; this experiment is not DDP.
        "parallel_jobs": "auto",
        "auto_plan": True,
        "calibrate_micro_batch": True,
        "vram_safety_fraction": 0.85,
        "require_idle_gpus": True,
        "max_preexisting_vram_fraction": 0.10,
        "min_free_vram_gb": 4.0,
        "ram_safety_fraction": 0.80,
        "disk_safety_fraction": 0.90,
        "min_micro_batch_size": 1,
        "max_micro_batch_size": 64,
        "max_data_workers": 8,
        "prefetch_factor": 2,
        "pin_memory": True,
        "persistent_workers": True,
        "monitor_interval_steps": 20,
        "data_workers": 0,
        "resolved_pin_memory": False,
        "resolved_persistent_workers": False,
        "resolved_prefetch_factor": 2,
        "resolved_parallel_jobs": 2,
    },
    "logging": {
        "delimiter": " | ",
        "banner_width": 120,
        "resource_jsonl": True,
    },
    # GPT-2 Small class: approximately 124M parameters.
    "model": {
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "ffn_dim": 3_072,
        "dropout": 0.0,
        "max_seq_len": 8_192,
        "bias": True,
    },
    "position": {
        "rope_base": 10_000.0,
        # RA-CABLE: gate = sigmoid(scale * QK + bias).
        "ra_gate_bias": -3.0,
        "ra_sparsity_weight": 1e-3,
        # RA-CABLE-Lite keeps the adaptive gate only in the final N layers;
        # earlier layers use the cheaper CABLE bias.
        "ra_lite_layers": 6,
        "dape_mlp_width": 32,
        "dape_kerple_epsilon": 1e-2,
    },
    "train": {
        "token_budget": 1_000_000_000,
        "micro_batch_size": 8,
        "effective_batch_tokens": 65_536,
        "learning_rate": 6e-4,
        "min_learning_rate": 6e-5,
        # Linear warmup, followed by cosine decay to min_learning_rate.
        "warmup_fraction": 0.01,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "log_interval": 125,
        "eval_interval": 500,
        "eval_batches": 20,
        "save_interval": 2_500,
    },
    "adapt": {
        "enabled": True,
        "max_seq_len": 1_024,
        "steps": 500,
        "batch_size": 16,
        "micro_batch_size": 8,
        "learning_rate": 1e-4,
        "log_interval": 100,
        "save_interval": 100,
    },
    "eval": {
        # Fast exploratory evaluation preserving all primary endpoint lengths.
        "lengths": [1_024, 4_096, 8_192],
        "lm_batches": 10,
        "retrieval_samples": 128,
        "batch_size": 1,
        "checkpoints": ["pretrain", "adapt"],
        "distance_bins": [0, 256, 512, 1_024, 2_048, 4_096, 8_192],
        "target_position_bins": [0.0, 0.25, 0.5, 0.75, 1.0],
        "rope_pi_enabled": False,
        "rope_pi_train_length": 1_024,
        "qasper_samples": 16,
        "qasper_max_answer_tokens": 32,
        "qasper_generation_tokens": 16,
    },
    "audit": {
        # Fast exploratory audit retaining the 4K mechanistic endpoint.
        "lengths": [1_024, 4_096],
        "samples": 4,
        "batch_size": 1,
        "layers": [0, 5, 11],
        "query_fraction": 0.25,
        "gate_threshold": 0.5,
        "save_full_artifacts": False,
        "artifact_sample_limit": 1,
        "checkpoint": "adapt",
        "conditions": ["synthetic_remote_target"],
        # GPT-2: <|endoftext|>, newline. Synthetic controls additionally
        # report their BOS, separator and query tokens by semantic name.
        "natural_sink_token_ids": {
            "bos_or_eos": [50_256],
            "newline": [198],
        },
    },
    "profile": {
        # Fast exploratory efficiency profile.
        "lengths": [1_024, 4_096],
        "batch_size": 1,
        "train_length": 1_024,
        "warmup": 5,
        "repeat": 20,
        "decode_tokens": 64,
        "use_kv_cache": True,
        "attention_kernel": "eager_torch_matmul_softmax",
        "checkpoint": "adapt",
    },
    "stats": {
        "bootstrap_samples": 1_000,
        "confidence": 0.95,
        # Eight pairs allow a two-sided exact sign-flip test to survive Holm
        # correction across the three confirmatory endpoints.
        "minimum_inferential_seeds": 8,
        "primary_endpoints": {
            "metrics.checkpoints.pretrain.lengths.1024.natural_language.fineweb_edu_held_out_ppl": "minimize",
            "metrics.checkpoints.pretrain.lengths.4096.real_long_document_qa.qasper.token_f1": "maximize",
            "metrics.checkpoints.adapt.lengths.8192.synthetic_control.single_query.accuracy": "maximize",
        },
        "secondary_endpoints": {
            "metrics.checkpoints.pretrain.lengths.4096.real_long_document_qa.qasper.evidence_utilization_gain": "maximize",
            "metrics.checkpoints.adapt.lengths.4096.synthetic_control.single_query.accuracy": "maximize",
            "audits.lengths.4096.conditions.synthetic_remote_target.summary.context_adaptivity_score": "maximize",
            "audits.lengths.4096.conditions.synthetic_remote_target.summary.false_exemption_rate": "minimize",
            "profiles.lengths.1024.train_step.median_seconds": "minimize",
            "profiles.full_training.gpu_hours": "minimize",
        },
    },
}


def get_config() -> dict[str, Any]:
    return deepcopy(CONFIG)


def set_by_path(cfg: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cursor = cfg
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            raise KeyError(f"Unknown config path: {path}")
        cursor = cursor[key]
    if keys[-1] not in cursor:
        raise KeyError(f"Unknown config path: {path}")
    cursor[keys[-1]] = value


def validate_config(cfg: dict[str, Any]) -> None:
    model = cfg["model"]
    data = cfg["data"]
    run = cfg["run"]
    methods = list(run["methods"])
    if methods != ["rope", "alibi", "cable", "ra_cable"]:
        raise ValueError(
            "run.methods must contain exactly: rope, alibi, cable, ra_cable"
        )
    allowed_methods = methods + list(run["extension_methods"])
    if str(run["method"]) not in allowed_methods:
        raise ValueError(
            "run.method must be in run.methods or run.extension_methods"
        )
    seeds = [int(seed) for seed in run["seeds"]]
    if not seeds:
        raise ValueError("run.seeds must not be empty")
    if len(seeds) != len(set(seeds)):
        raise ValueError("run.seeds must not contain duplicates")
    if int(run["world_size"]) != 1:
        raise ValueError(
            "world_size must remain 1; this experiment assigns one full model "
            "to each GPU"
        )
    if int(model["n_embd"]) <= 0 or int(model["n_head"]) <= 0:
        raise ValueError("model.n_embd and model.n_head must be positive")
    if int(model["n_embd"]) % int(model["n_head"]):
        raise ValueError("model.n_embd must be divisible by model.n_head")
    if int(model["max_seq_len"]) < int(data["block_size"]):
        raise ValueError("model.max_seq_len must be >= data.block_size")
    lite_layers = int(cfg["position"]["ra_lite_layers"])
    if not 1 <= lite_layers <= int(model["n_layer"]):
        raise ValueError(
            "position.ra_lite_layers must be between 1 and model.n_layer"
        )
    micro_tokens = int(cfg["train"]["micro_batch_size"]) * int(data["block_size"])
    if int(cfg["train"]["effective_batch_tokens"]) % max(1, micro_tokens):
        raise ValueError(
            "train.effective_batch_tokens must be divisible by "
            "train.micro_batch_size * data.block_size"
        )
    if int(cfg["train"]["token_budget"]) <= 0:
        raise ValueError("train.token_budget must be positive")
    requested_parallel = cfg["resources"]["parallel_jobs"]
    if (
        str(requested_parallel).lower() != "auto"
        and int(requested_parallel) <= 0
    ):
        raise ValueError("resources.parallel_jobs must be auto or positive")
    for key in ("vram_safety_fraction", "ram_safety_fraction"):
        value = float(cfg["resources"][key])
        if not 0 < value <= 1:
            raise ValueError(f"resources.{key} must be in (0, 1]")
    existing_fraction = float(
        cfg["resources"]["max_preexisting_vram_fraction"]
    )
    if not 0 <= existing_fraction < 1:
        raise ValueError(
            "resources.max_preexisting_vram_fraction must be in [0, 1)"
        )
    if float(cfg["resources"]["min_free_vram_gb"]) <= 0:
        raise ValueError("resources.min_free_vram_gb must be positive")
    minimum_micro_batch = int(cfg["resources"]["min_micro_batch_size"])
    maximum_micro_batch = int(cfg["resources"]["max_micro_batch_size"])
    if minimum_micro_batch <= 0 or maximum_micro_batch < minimum_micro_batch:
        raise ValueError(
            "resource micro-batch bounds must be positive and ordered"
        )
    for key in ("log_interval", "eval_interval", "eval_batches", "save_interval"):
        if int(cfg["train"][key]) <= 0:
            raise ValueError(f"train.{key} must be positive")
    for key in ("log_interval", "save_interval"):
        if int(cfg["adapt"][key]) <= 0:
            raise ValueError(f"adapt.{key} must be positive")
    peak_lr = float(cfg["train"]["learning_rate"])
    min_lr = float(cfg["train"]["min_learning_rate"])
    if not 0 < min_lr <= peak_lr:
        raise ValueError(
            "train.min_learning_rate must be positive and no larger than "
            "train.learning_rate"
        )
    warmup = float(cfg["train"]["warmup_fraction"])
    if not 0 < warmup < 1:
        raise ValueError("train.warmup_fraction must be between 0 and 1")
    if int(data["vocab_size"]) < 32:
        raise ValueError("data.vocab_size must be at least 32")
    if int(data["retrieval_queries_per_sample"]) <= 1:
        raise ValueError("data.retrieval_queries_per_sample must be greater than one")
    if int(data["retrieval_similar_distractors"]) <= 0:
        raise ValueError("data.retrieval_similar_distractors must be positive")
    if int(cfg["eval"]["rope_pi_train_length"]) <= 0:
        raise ValueError("eval.rope_pi_train_length must be positive")
    if int(data["qasper_samples"]) <= 0:
        raise ValueError("data.qasper_samples must be positive")
    for key in (
        "qasper_samples",
        "qasper_max_answer_tokens",
        "qasper_generation_tokens",
    ):
        if int(cfg["eval"][key]) <= 0:
            raise ValueError(f"eval.{key} must be positive")
    for key in ("warmup", "repeat", "decode_tokens", "train_length"):
        if int(cfg["profile"][key]) <= 0:
            raise ValueError(f"profile.{key} must be positive")
    if int(cfg["stats"]["minimum_inferential_seeds"]) < 8:
        raise ValueError("stats.minimum_inferential_seeds must be at least 8")
    if int(data["leakage_shingle_width"]) < 2:
        raise ValueError("data.leakage_shingle_width must be at least 2")
    if int(data["leakage_shingle_stride"]) < 1:
        raise ValueError("data.leakage_shingle_stride must be positive")
    if int(data["leakage_min_shared_shingles"]) < 1:
        raise ValueError(
            "data.leakage_min_shared_shingles must be positive"
        )
    if not 0.0 < float(data["leakage_overlap_threshold"]) <= 1.0:
        raise ValueError(
            "data.leakage_overlap_threshold must be in (0, 1]"
        )
    eval_checkpoints = list(cfg["eval"]["checkpoints"])
    if eval_checkpoints != ["pretrain", "adapt"]:
        raise ValueError(
            "eval.checkpoints must be exactly ['pretrain', 'adapt']"
        )
    primary_endpoints = dict(cfg["stats"]["primary_endpoints"])
    if not primary_endpoints:
        raise ValueError("stats.primary_endpoints must not be empty")
    if set(primary_endpoints.values()) - {"minimize", "maximize"}:
        raise ValueError(
            "stats.primary_endpoints directions must be minimize or maximize"
        )
    secondary_endpoints = dict(cfg["stats"]["secondary_endpoints"])
    if set(secondary_endpoints.values()) - {"minimize", "maximize"}:
        raise ValueError(
            "stats.secondary_endpoints directions must be minimize or maximize"
        )
    if set(primary_endpoints) & set(secondary_endpoints):
        raise ValueError("primary and secondary endpoints must be disjoint")
    lengths = (
        list(cfg["eval"]["lengths"])
        + list(cfg["audit"]["lengths"])
        + list(cfg["profile"]["lengths"])
    )
    if max(int(length) for length in lengths) > int(model["max_seq_len"]):
        raise ValueError(
            "all eval/audit/profile lengths must be <= model.max_seq_len"
        )
    if str(run["bootstrap"]).lower() not in {"off", "verify", "install"}:
        raise ValueError("run.bootstrap must be one of: off, verify, install")
