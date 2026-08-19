"""The single human-edited configuration for every experiment.

Choose only ``small``, ``medium`` or ``large`` below. Every run saves the
fully resolved mapping next to its outputs.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

RESEARCH_METHODS = [
    "standard",
    "parameter_matched",
    "concept_aux",
    "concept_projector",
    "concept_bus_v2",
]

CFG = {
    "paths": {
        "project_root": str(PROJECT_ROOT),
        "data_root": str(PROJECT_ROOT / "data"),
        "checkpoint_root": str(PROJECT_ROOT / "ckpt"),
        "output_root": str(PROJECT_ROOT / "output"),
    },
    "run": {
        # Double-click run.bat after changing only this value if desired.
        "size": "small",  # small | medium | large
        "resume": True,
        "skip_completed": True,
        "attempt": 1,
        "max_phase_attempts": 2,
    },
    "sizes": {
        "small": {
            "stage": "smoke",
            "language_backend": "none",
            "methods": ["concept_bus_v2"],
            "seeds": [11],
            "data": {
                "train_size": 128,
                "validation_size": 32,
                "test_size": 32,
                "max_length": 48,
                "generation_seed": 20260806,
                "unknown_fraction": 0.10,
                "template_ood": False,
            },
            "model": {
                "num_layers": 1,
                "d_model": 64,
                "d_ff": 128,
                "num_heads": 4,
                "slot_dim": 32,
                "num_bus_slots": 2,
                "bus_heads": 2,
                "bus_layers": 1,
                "concept_residual_dim": 4,
                "dropout": 0.0,
            },
            "train": {
                "max_steps": 5,
                "batch_size": 8,
                "learning_rate": 1e-3,
                "weight_decay": 0.01,
                "warmup_fraction": 0.05,
                "log_interval_steps": 1,
                "eval_interval_steps": 5,
                "monitor_validation_examples": 32,
                "checkpoint_interval_minutes": 120,
            },
        },
        "medium": {
            # One-seed, three-model fair recheck.  All three runs use exactly
            # the same effective and micro batch, data order, and validation set.
            "stage": "fair_dual_tag",
            "language_backend": "none",
            "require_gpu": True,
            "methods": ["standard", "concept_projector", "concept_bus_v2"],
            "seeds": [11, 22, 33],
            "data": {
                "train_size": 200000,
                "validation_size": 20000,
                "test_size": 20000,
                "max_length": 256,
                "generation_seed": 20260806,
                "unknown_fraction": 0.20,
                "template_ood": True,
            },
            "model": {
                "num_layers": 12,
                "d_model": 512,
                "d_ff": 2048,
                "num_heads": 8,
                "slot_dim": 64,
                "num_bus_slots": 2,
                "bus_heads": 4,
                "bus_layers": 1,
                "concept_residual_dim": 32,
                "dropout": 0.0,
            },
            "train": {
                "max_steps": 6000,
                "batch_size": 128,
                "learning_rate": 2e-4,
                "weight_decay": 0.01,
                "warmup_fraction": 0.05,
                "log_interval_steps": 100,
                "eval_interval_steps": 500,
                # Use the complete validation split for fair best-checkpoint
                # selection.  The test split remains untouched until the end.
                "monitor_validation_examples": 20000,
                "checkpoint_interval_minutes": 120,
            },
            "suite": [
                {
                    "name": "fair_concept_subspace",
                    "runner": "synthetic",
                    "stage": "fair_dual_tag",
                    "methods": [
                        "standard",
                        "concept_projector",
                        "concept_bus_v2",
                    ],
                    # SmoothMax is nonlinear across samples, so gradient
                    # accumulation with different micro batches is not an
                    # equivalent experiment.  Require the same 64 x 2 split.
                    "required_micro_batch_size": 64,
                    "resources": {
                        "allow_colocation": False,
                        "max_jobs_per_gpu": 1,
                    },
                },
            ],
        },
        "large": {
            "stage": "dual_tag",
            "language_backend": "formal",
            "require_gpu": True,
            "methods": list(RESEARCH_METHODS),
            # Formal paper matrix. Keep all three seeds for reported means/CI.
            "seeds": [11, 22, 33],
            "data": {
                "train_size": 200000,
                "validation_size": 20000,
                "test_size": 20000,
                "max_length": 256,
                "generation_seed": 20260806,
                "unknown_fraction": 0.20,
                "template_ood": True,
            },
            "model": {
                "num_layers": 12,
                "d_model": 512,
                "d_ff": 2048,
                "num_heads": 8,
                "slot_dim": 64,
                "num_bus_slots": 2,
                "bus_heads": 4,
                "bus_layers": 1,
                "concept_residual_dim": 32,
                "dropout": 0.0,
            },
            "train": {
                "max_steps": 6000,
                "batch_size": 128,
                "learning_rate": 2e-4,
                "weight_decay": 0.01,
                "warmup_fraction": 0.05,
                "log_interval_steps": 100,
                "eval_interval_steps": 500,
                "monitor_validation_examples": 2000,
                "checkpoint_interval_minutes": 120,
            },
            # Minimal confirmatory paper matrix: 27 runs, all three seeds.
            "suite": [
                {
                    "name": "dual_tag_confirm",
                    "runner": "synthetic",
                    "methods": [
                        "standard",
                        "parameter_matched",
                        "concept_projector",
                        "concept_bus_v2",
                    ],
                },
                {
                    "name": "clutrr_external",
                    "runner": "clutrr",
                    "methods": ["standard", "concept_bus_v2"],
                },
                {
                    "name": "formal_language_model",
                    "runner": "language_model",
                    "methods": ["standard", "parameter_matched", "concept_bus_v2"],
                    "resources": {
                        "max_jobs_per_gpu": 1,
                        "estimated_model_memory_gb": 12.0,
                        "estimated_activation_memory_per_sample_gb": 1.8,
                    },
                },
            ],
        },
    },
    "model_defaults": {
        "architecture": "decoder_only",
        "norm": "rmsnorm",
        "norm_eps": 1e-5,
        "position_encoding": "rope",
        "rope_theta": 10000.0,
        "activation": "gelu",
        "bias": False,
        "tie_embeddings": True,
        "keep_residual_attention": True,
    },
    "loss": {
        # EMA normalization removes unit mismatch; smooth-max concentrates
        # gradient on the currently weakest task/concept/causal objective.
        "balance_temperature": 0.25,
        "balance_ema_decay": 0.99,
        "orthogonality_weight": 0.01,
    },
    "optimizer": {
        "betas": [0.9, 0.95],
        "gradient_clip": 1.0,
    },
    "resources": {
        "auto_detect": True,
        "required_dtype": "auto",
        "min_free_memory_gb": 4.0,
        "max_initial_gpu_utilization_percent": 20.0,
        "target_memory_fraction": 0.85,
        "target_gpu_utilization_percent": 90.0,
        "allow_colocation": True,
        # Spread across GPUs first, then colocate small jobs while the memory
        # budget permits. Dedicated hardware_profile jobs remain exclusive.
        "max_jobs_per_gpu": 2,
        "profile_exclusive": True,
        "max_workers_per_job": 8,
        "prefetch_factor": 2,
        # Used only by the pre-run planner; run-one uses the resolved size.
        "global_batch_size": 128,
        "max_micro_batch_size": 128,
        "estimated_model_memory_gb": 4.0,
        "estimated_activation_memory_per_sample_gb": 0.12,
    },
    "preflight": {
        "minimum_free_disk_gb": 50.0,
    },
    # Compatibility view used by the existing logging tests and simple tools.
    "train": {
        "log_interval_steps": 50,
        "eval_interval_steps": 250,
        "checkpoint_interval_minutes": 120,
    },
    "logging": {
        # Every configured TRAIN interval plus validation/checkpoint/final is
        # written to train.log and forwarded by the scheduler to the terminal.
        "console_mode": "interval",
        "file_name": "train.log",
        "flush_each_line": True,
        "column_widths": {
            "time": 19,
            "device": 34,
            "phase": 8,
            "step": 17,
            "loss": 16,
            "memory": 21,
            "elapsed": 25,
            "seed": 8,
            "task": 48,
        },
        # Terminal rows remain below 120 columns; train.log keeps full values.
        "console_column_widths": {
            "time": 8,
            "device": 12,
            "phase": 7,
            "step": 11,
            "loss": 10,
            "memory": 15,
            "elapsed": 15,
            "seed": 4,
            "task": 14,
        },
    },
    "audit": {
        # Full test metrics use every item; expensive causal passes use this
        # deterministic prefix so unattended suites remain bounded.
        "max_examples": 2000,
        "trace_examples": 64,
    },
    "external": {
        "allow_download": True,
        "download_timeout_seconds": 120,
        "failure_policy": "continue",
        "clutrr": {
            "archive_url": "https://drive.usercontent.google.com/download?id=1SEq_e1IVCDDzsBIBhoUQ5pOVH5kxRoZF&export=download&confirm=t",
            # Official paper split. The Hugging Face loader uses these CSV
            # mirrors; they are a fallback when Google Drive serves a nested
            # or otherwise incompatible publication bundle.
            "dataset_id": "data_089907f8",
            "fallback_split": "gen_train23_test2to10",
            "fallback_csv_urls": {
                "train": "https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/e5b496941e91abb7c319d2618a3ce96752bc4ab7/gen_train23_test2to10/train.csv",
                "validation": "https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/e5b496941e91abb7c319d2618a3ce96752bc4ab7/gen_train23_test2to10/validation.csv",
                "test": "https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/e5b496941e91abb7c319d2618a3ce96752bc4ab7/gen_train23_test2to10/test.csv",
            },
            # data_089907f8: 10,094 train CSV rows; reserve 2,000 for
            # validation and use the remaining 8,094 for training.
            "train_rows": 8094,
            "validation_rows": 2000,
            "test_rows_per_length": 2000,
            "train_lengths": [2, 3],
            "test_lengths": [2, 3, 4, 5, 6],
            "max_length": 256,
            "max_steps": 3000,
            "batch_size": 128,
        },
        "tinystories": {
            "train_url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt?download=true",
            "validation_url": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt?download=true",
            "train_bytes": 300000000,
            "validation_bytes": 20000000,
            "train_tokens": 100000000,
            "sequence_length": 256,
            "batch_size": 64,
            "validation_batches": 100,
            # Fixed sample count makes validation cost independent of the
            # adaptive micro-batch and is far cheaper than 100 full batches.
            "validation_sequences": 1024,
            "log_interval_steps": 50,
            "eval_interval_steps": 250,
            "checkpoint_interval_minutes": 120,
        },
        "formal_language": {
            "enabled": True,
            "dataset_id": "HuggingFaceFW/fineweb-edu",
            "dataset_config": "sample-10BT",
            "dataset_revision": "593b3a867298afb8ce42625a270ef20ddcad28f9",
            "external_dataset_id": "Salesforce/wikitext",
            "external_dataset_config": "wikitext-103-raw-v1",
            "external_dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "external_split": "test",
            "train_tokens": 1000000000,
            "validation_tokens": 10000000,
            "sequence_length": 512,
            "batch_size": 64,
            "validation_batches": 100,
            "external_validation_batches": 100,
            "validation_sequences": 1024,
            "external_validation_sequences": 1024,
            "learning_rate": 0.0003,
            "weight_decay": 0.1,
            "warmup_fraction": 0.01,
            "log_interval_steps": 100,
            "eval_interval_steps": 1000,
            "checkpoint_interval_minutes": 120,
            "tokenizer_batch_size": 128,
            "preparation_log_tokens": 10000000,
            # GPT-2-small scale: ~124M parameters with tied embeddings.
            "model": {
                "num_layers": 12,
                "d_model": 768,
                "d_ff": 3072,
                "num_heads": 12,
                "slot_dim": 64,
                "num_bus_slots": 2,
                "bus_heads": 4,
                "bus_layers": 1,
                "concept_residual_dim": 48,
                "dropout": 0.0,
            },
        },
    },
    "report": {
        "max_static_figures": 12,
        "write_svg": True,
        "write_interactive_dashboard": True,
    },
}
