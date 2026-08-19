from __future__ import annotations

import copy
import csv
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

import torch
import pytest

import dual_axis_transformer.preflight as preflight_module
import dual_axis_transformer.suite as suite_module
from cfg import CFG
from dual_axis_transformer.external_model import (
    ByteTokenizer,
    CausalLanguageModel,
    SequenceClassifierTransformer,
)
from dual_axis_transformer.external_tasks import (
    RelationExample,
    _load_examples as _load_clutrr_examples,
    _normalization_config,
    _relation_length,
    _split_from_path,
    _write_examples,
    ensure_clutrr_dataset,
    run_clutrr_one,
)
from dual_axis_transformer.data_download import sha256_file
from dual_axis_transformer.language_model import run_language_model_one
from dual_axis_transformer.metrics import (
    binary_average_precision,
    binary_auroc,
    expected_calibration_error,
    multilabel_metrics,
)
from dual_axis_transformer.preflight import run_preflight
from dual_axis_transformer.reporting import paired_comparisons
from dual_axis_transformer.research_model import ResearchModelConfig
from dual_axis_transformer.runner import (
    _SCIENTIFIC_SOURCE_FILES,
    _normalize_cfg_for_fingerprint,
    resolve_run_config,
    run_one,
)
from dual_axis_transformer.suite import _phase_signature, describe_suite, run_suite


def _tiny_cfg() -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["external"]["formal_language"]["enabled"] = False
    cfg["logging"]["console_mode"] = "quiet"
    cfg["sizes"]["large"]["model"] = {
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
    cfg["sizes"]["large"]["train"].update(
        {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "warmup_fraction": 0.5,
            "log_interval_steps": 1,
            "eval_interval_steps": 1,
            "checkpoint_interval_minutes": 120,
        }
    )
    return cfg


def _official_clutrr_csv(rows: list[tuple[str, str, str]]) -> str:
    """Create the schema used by the published/Hugging Face CLUTRR files."""

    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "index",
            "id",
            "story",
            "query",
            "target",
            "target_text",
            "clean_story",
            "proof_state",
            "f_comb",
            "task_name",
            "story_edges",
            "edge_types",
            "query_edge",
            "genders",
            "task_split",
        ]
    )
    for index, (task_name, split, target) in enumerate(rows):
        story = f"Ann is Bob's {target}."
        writer.writerow(
            [
                index,
                f"id-{split}-{index}",
                story,
                "('Ann', 'Bob')",
                index % 2,
                target,
                story,
                "{}",
                target,
                task_name,
                "[]",
                "[]",
                "(0, 1)",
                "Ann:female,Bob:male",
                split,
            ]
        )
    return output.getvalue()


def _write_cached_clutrr_manifest(
    cfg: dict, directory: Path, labels: list[str]
) -> None:
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "labels": labels,
                "normalization_config": _normalization_config(
                    cfg["external"]["clutrr"]
                ),
                "splits": {
                    name: {"sha256": sha256_file(directory / f"{name}.jsonl")}
                    for name in ("train", "validation", "test")
                },
            }
        ),
        encoding="utf-8",
    )


def test_metrics_cover_calibration_and_multilabel_quality() -> None:
    targets = torch.tensor([0, 0, 1, 1])
    perfect = torch.tensor([0.0, 0.1, 0.9, 1.0])
    assert binary_auroc(targets, perfect) == 1.0
    assert binary_average_precision(targets, perfect) == 1.0
    assert expected_calibration_error(targets, targets.float()) == 0.0
    matrix = torch.stack((targets, ~targets.bool()), dim=-1).float()
    metrics = multilabel_metrics(matrix, matrix)
    assert metrics["concept_macro_f1"] == 1.0
    assert metrics["dead_concept_fraction"] == 0.0


def test_paired_report_uses_only_common_seeds() -> None:
    records = []
    for method, values in {"concept_bus_v2": [0.8, 0.9], "standard": [0.6, 0.7]}.items():
        for seed, value in zip((11, 22), values):
            records.append(
                {"stage": "x", "method": method, "seed": seed, "metrics": {"accuracy": value}}
            )
    rows = paired_comparisons(records)
    assert len(rows) == 1
    assert rows[0]["n"] == 2
    assert abs(rows[0]["mean_difference"] - 0.2) < 1e-7
    assert rows[0]["randomization_p"] == 0.5


def test_large_suite_dry_run_has_all_paper_phases(tmp_path: Path) -> None:
    descriptions = describe_suite(CFG, "large")
    assert [item["runner"] for item in descriptions] == [
        "synthetic",
        "clutrr",
        "language_model",
    ]
    suite = run_suite(
        CFG,
        size_name="large",
        output_root=tmp_path / "output",
        project_root=Path(__file__).parents[1],
        dry_run=True,
        monitor_interval=0.01,
    )
    plan = json.loads((suite / "dry_run_plan.json").read_text(encoding="utf-8"))
    assert sum(item["runs"] for item in plan["phases"]) == 27


def test_three_size_presets_have_distinct_scopes() -> None:
    small = CFG["sizes"]["small"]
    medium = CFG["sizes"]["medium"]
    large = CFG["sizes"]["large"]
    assert small["stage"] == "smoke"
    assert small["train"]["max_steps"] == 5
    assert "suite" not in small
    medium_plan = describe_suite(CFG, "medium")
    assert sum(item["runs"] for item in medium_plan) == 9
    assert medium["language_backend"] == "none"
    assert medium_plan[0]["methods"] == [
        "standard",
        "concept_projector",
        "concept_bus_v2",
    ]
    assert medium["suite"][0]["required_micro_batch_size"] == 64
    assert medium["train"]["monitor_validation_examples"] == 20000
    assert medium["model"]["d_model"] == 512
    assert medium["seeds"] == [11, 22, 33]
    assert sum(item["runs"] for item in describe_suite(CFG, "large")) == 27
    assert large["language_backend"] == "formal"
    assert large["seeds"] == [11, 22, 33]


def test_medium_three_models_and_seeds_resolve_identical_fairness_controls() -> None:
    phase = "fair_concept_subspace"
    configs = [
        resolve_run_config(CFG, "medium", method, seed, phase)
        for seed in (11, 22, 33)
        for method in ("standard", "concept_projector", "concept_bus_v2")
    ]
    reference = configs[0]
    for config in configs:
        assert config["seed"] in (11, 22, 33)
        assert config["data"] == reference["data"]
        assert config["model"] == reference["model"]
        assert config["train"] == reference["train"]
        assert config["optimizer"] == reference["optimizer"]
    assert reference["train"]["batch_size"] == 128
    assert reference["train"]["monitor_validation_examples"] == 20000


def test_suite_overrides_filter_every_phase_without_marking_a_full_matrix(
    tmp_path: Path,
) -> None:
    descriptions = describe_suite(
        CFG,
        "large",
        method_override=["concept_bus_v2"],
        seed_override=[11],
    )
    assert [item["name"] for item in descriptions] == [
        "dual_tag_confirm",
        "clutrr_external",
        "formal_language_model",
    ]
    assert all(item["methods"] == ["concept_bus_v2"] for item in descriptions)
    assert all(item["seeds"] == [11] for item in descriptions)
    assert sum(item["runs"] for item in descriptions) == 3
    projector_only = describe_suite(
        CFG, "large", method_override=["concept_projector"], seed_override=[11]
    )
    assert [item["name"] for item in projector_only] == ["dual_tag_confirm"]
    suite = run_suite(
        CFG,
        size_name="large",
        output_root=tmp_path / "output",
        project_root=Path(__file__).parents[1],
        dry_run=True,
        monitor_interval=0.01,
        method_override=["concept_bus_v2"],
        seed_override=[11],
    )
    plan = json.loads((suite / "dry_run_plan.json").read_text(encoding="utf-8"))
    assert sum(item["runs"] for item in plan["phases"]) == 3


def test_suite_rejects_unknown_or_duplicate_jobs_before_launch() -> None:
    cfg = copy.deepcopy(CFG)
    cfg["sizes"]["large"]["suite"][0]["methods"] = ["standard", "typo"]
    with pytest.raises(ValueError, match="unregistered methods"):
        describe_suite(cfg, "large")
    cfg = copy.deepcopy(CFG)
    cfg["sizes"]["large"]["seeds"] = [11, 11]
    with pytest.raises(ValueError, match="duplicate seeds"):
        describe_suite(cfg, "large")


def test_suite_returns_failure_when_a_requested_final_artifact_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = copy.deepcopy(CFG)
    cfg["run"]["max_phase_attempts"] = 1
    cfg["sizes"]["artifact_test"] = {
        **copy.deepcopy(cfg["sizes"]["small"]),
        "stage": "artifact_test",
        "suite": [
            {
                "name": "artifact_test",
                "runner": "synthetic",
                "methods": ["concept_bus_v2"],
            }
        ],
    }
    monkeypatch.setattr(
        suite_module,
        "_synthetic_phase",
        lambda *_args, **_kwargs: {"status": "completed", "runs": 1},
    )
    with pytest.raises(RuntimeError, match="suite finished with issues"):
        run_suite(
            cfg,
            size_name="artifact_test",
            output_root=tmp_path / "missing_final_output",
            project_root=Path(__file__).parents[1],
            dry_run=False,
            monitor_interval=0.01,
        )
    state = json.loads(
        (
            tmp_path
            / "missing_final_output"
            / "suites"
            / "artifact_test"
            / "state.json"
        ).read_text(encoding="utf-8")
    )
    assert state["status"] == "completed_with_issues"
    assert state["phases"]["artifact_test"]["status"] == "completed_with_failures"


def test_one_large_seed_array_controls_every_scientific_phase() -> None:
    cfg = copy.deepcopy(CFG)
    cfg["sizes"]["large"]["seeds"] = [11, 22]
    descriptions = describe_suite(cfg, "large")
    by_name = {item["name"]: item for item in descriptions}
    assert by_name["dual_tag_confirm"]["seeds"] == [11, 22]
    for name in (
        "clutrr_external",
        "formal_language_model",
    ):
        assert by_name[name]["seeds"] == [11, 22]


def test_main_accepts_direct_size_shorthand(monkeypatch: object) -> None:
    import main as entrypoint

    observed = {}

    def fake_run(args: object) -> int:
        observed["size"] = args.size
        observed["command"] = args.command
        return 0

    monkeypatch.setattr(entrypoint, "command_run", fake_run)
    assert entrypoint.main(["--size", "medium", "--dry-run"]) == 0
    assert observed == {"size": "medium", "command": "run"}


def test_adding_seeds_changes_phase_plan_but_not_code_identity() -> None:
    before = '{"seeds": [11], "d_model": 128}'
    after = '{"seeds": [11, 22, 33], "d_model": 128}'
    assert _normalize_cfg_for_fingerprint(before) == _normalize_cfg_for_fingerprint(after)
    cfg = copy.deepcopy(CFG)
    phase = next(
        item
        for item in cfg["sizes"]["large"]["suite"]
        if item["name"] == "formal_language_model"
    )
    first = _phase_signature(cfg, "large", phase)
    cfg["sizes"]["large"]["seeds"] = [11, 22]
    assert _phase_signature(cfg, "large", phase) != first


def test_scientific_fingerprint_excludes_reporting_only_code() -> None:
    assert "research_model.py" in _SCIENTIFIC_SOURCE_FILES
    assert "runner.py" in _SCIENTIFIC_SOURCE_FILES
    assert "reporting.py" not in _SCIENTIFIC_SOURCE_FILES
    assert "verdict.py" not in _SCIENTIFIC_SOURCE_FILES
    assert "scheduler.py" not in _SCIENTIFIC_SOURCE_FILES


def test_byte_models_share_causal_backbone() -> None:
    tokenizer = ByteTokenizer()
    assert tokenizer.encode("A", 8) == [1, 68, 2]
    config = ResearchModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_length=8,
        method="concept_bus_v2",
        num_layers=1,
        d_model=32,
        d_ff=64,
        num_heads=4,
        slot_dim=32,
        num_bus_slots=2,
        bus_heads=2,
        bus_layers=1,
        concept_residual_dim=4,
    )
    classifier = SequenceClassifierTransformer(config, 3)
    lm = CausalLanguageModel(config)
    inputs = torch.randint(3, tokenizer.vocab_size, (2, 8))
    mask = torch.ones_like(inputs)
    assert classifier(inputs, mask).logits.shape == (2, 3)
    first = lm(inputs, mask).logits
    changed = inputs.clone()
    changed[:, 5:] = torch.randint(3, tokenizer.vocab_size, changed[:, 5:].shape)
    second = lm(changed, mask).logits
    torch.testing.assert_close(first[:, :5], second[:, :5])


def test_external_models_reject_unknown_method_instead_of_running_standard() -> None:
    values = ResearchModelConfig(
        vocab_size=32,
        max_length=8,
        method="standard",
        num_layers=1,
        d_model=32,
        d_ff=64,
        num_heads=4,
        slot_dim=32,
        bus_heads=2,
        bus_layers=1,
        concept_residual_dim=4,
    ).to_dict()
    values["method"] = "misspelled_method"
    invalid = ResearchModelConfig(**values)
    with pytest.raises(ValueError, match="unknown method"):
        CausalLanguageModel(invalid)
    with pytest.raises(ValueError, match="unknown method"):
        SequenceClassifierTransformer(invalid, 3)


def test_external_runners_complete_from_offline_cache(tmp_path: Path) -> None:
    cfg = _tiny_cfg()
    output = tmp_path / "output"

    clutrr = tmp_path / "data" / "clutrr" / "normalized"
    clutrr.mkdir(parents=True)
    examples = [
        RelationExample("Ann is Bob's mother. Query: Ann to Bob?", index % 2, ("mother", "father")[index % 2], 2 + index % 2)
        for index in range(12)
    ]
    for name, rows in {
        "train": examples[:8],
        "validation": examples[8:10],
        "test": examples[8:],
    }.items():
        _write_examples(clutrr / f"{name}.jsonl", rows)
    _write_cached_clutrr_manifest(cfg, clutrr, ["mother", "father"])
    cfg["external"]["clutrr"].update(
        {"max_length": 48, "max_steps": 1, "batch_size": 2}
    )
    clutrr_run = run_clutrr_one(
        cfg, method="standard", seed=11, output_root=output
    )
    assert (clutrr_run / "metrics" / "final.json").exists()

    stories = tmp_path / "data" / "tinystories"
    stories.mkdir(parents=True)
    payload = ("Once upon a time there was a small cat. " * 100).encode()
    (stories / "train_prefix.txt").write_bytes(payload)
    (stories / "validation_prefix.txt").write_bytes(payload)
    (stories / "manifest.json").write_text("{}", encoding="utf-8")
    cfg["external"]["tinystories"].update(
        {
            "train_tokens": 32,
            "sequence_length": 8,
            "batch_size": 2,
            "validation_batches": 1,
            "validation_sequences": 2,
            "log_interval_steps": 1,
            "eval_interval_steps": 1,
            "checkpoint_interval_minutes": 120,
        }
    )
    lm_run = run_language_model_one(
        cfg, method="standard", seed=11, output_root=output
    )
    assert (lm_run / "metrics" / "final.json").exists()


def test_clutrr_length_parser_handles_official_style_names() -> None:
    assert _relation_length(Path("task_1.4_test.csv")) == 4
    assert (
        _relation_length(
            PurePosixPath(r"data_emnlp_final\data_089907f8\1.4_test.csv")
        )
        == 4
    )
    assert (
        _split_from_path(
            PurePosixPath(r"data_emnlp_final\data_089907f8\1.4_test.csv")
        )
        == "test"
    )


def test_clutrr_parser_handles_nested_archive_and_row_task_names(
    tmp_path: Path,
) -> None:
    cfg = _tiny_cfg()
    cfg["external"]["clutrr"].update(
        {"train_rows": 4, "validation_rows": 2, "test_rows_per_length": 1}
    )
    root = tmp_path / "data" / "clutrr"
    root.mkdir(parents=True)
    train_rows = [
        ("task_1.2", "train", "mother"),
        ("task_1.3", "train", "father"),
        ("task_1.2", "train", "mother"),
        ("task_1.3", "train", "father"),
    ]
    validation_rows = [
        ("task_1.2", "validation", "mother"),
        ("task_1.3", "validation", "father"),
    ]
    test_rows = [
        (f"task_1.{length}", "test", "mother" if length % 2 else "father")
        for length in range(2, 7)
    ]
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as handle:
        base = "published/data_089907f8"
        handle.writestr(f"{base}/train.csv", _official_clutrr_csv(train_rows))
        handle.writestr(
            f"{base}/validation.csv", _official_clutrr_csv(validation_rows)
        )
        handle.writestr(f"{base}/test.csv", _official_clutrr_csv(test_rows))
    with zipfile.ZipFile(root / "data_publish.zip", "w", zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("official_bundle.zip", inner.getvalue())

    bundle = ensure_clutrr_dataset(cfg, tmp_path / "data")
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_mode"] == "official_archive"
    assert manifest["dataset_id"] == "data_089907f8"
    assert {item.length for item in _load_clutrr_examples(bundle.test_path)} == {
        2,
        3,
        4,
        5,
        6,
    }
    assert {item.relation for item in _load_clutrr_examples(bundle.train_path)} == {
        "mother",
        "father",
    }


def test_clutrr_parser_falls_back_to_huggingface_loader_csvs(
    tmp_path: Path,
) -> None:
    cfg = _tiny_cfg()
    cfg["external"]["clutrr"].update(
        {"train_rows": 4, "validation_rows": 2, "test_rows_per_length": 1}
    )
    output = tmp_path / "output"
    root = tmp_path / "data" / "clutrr"
    root.mkdir(parents=True)
    with zipfile.ZipFile(root / "data_publish.zip", "w") as handle:
        handle.writestr("README.txt", "publication bundle without flat CSV files")

    source = tmp_path / "mirror_source"
    source.mkdir()
    rows = {
        "train": [
            ("task_1.2", "train", "mother"),
            ("task_1.3", "train", "father"),
        ],
        "validation": [("task_1.2", "validation", "mother")],
        "test": [
            (f"task_1.{length}", "test", "mother" if length % 2 else "father")
            for length in range(2, 7)
        ],
    }
    for split, split_rows in rows.items():
        path = source / f"{split}.csv"
        path.write_text(_official_clutrr_csv(split_rows), encoding="utf-8")
    cfg["external"]["clutrr"]["fallback_csv_urls"] = {
        split: (source / f"{split}.csv").resolve().as_uri()
        for split in rows
    }

    bundle = ensure_clutrr_dataset(cfg, tmp_path / "data")
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_mode"] == "huggingface_loader_mirror"
    assert len(manifest["fallback_sources"]) == 3
    assert {item.length for item in _load_clutrr_examples(bundle.test_path)} == {
        2,
        3,
        4,
        5,
        6,
    }


def test_preflight_validates_cached_real_data_and_training_gradient(
    tmp_path: Path,
) -> None:
    cfg = _tiny_cfg()
    cfg["preflight"]["minimum_free_disk_gb"] = 0.0
    output = tmp_path / "output"
    stories = tmp_path / "data" / "tinystories"
    stories.mkdir(parents=True)
    payload = ("A tiny story about France and a blue boat. " * 100).encode()
    (stories / "train_prefix.txt").write_bytes(payload)
    (stories / "validation_prefix.txt").write_bytes(payload)
    (stories / "manifest.json").write_text("{}", encoding="utf-8")
    clutrr = tmp_path / "data" / "clutrr" / "normalized"
    clutrr.mkdir(parents=True)
    train = [RelationExample("A is B's parent.", 0, "parent", 2)]
    test = [
        RelationExample(f"A is B's parent at length {length}.", 0, "parent", length)
        for length in range(2, 7)
    ]
    _write_examples(clutrr / "train.jsonl", train)
    _write_examples(clutrr / "validation.jsonl", train)
    _write_examples(clutrr / "test.jsonl", test)
    _write_cached_clutrr_manifest(cfg, clutrr, ["parent"])
    report = run_preflight(
        cfg, size_name="small", output_root=output, prepare_data=True
    )
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["status"] == "pass"
    assert result["checks"]["training_probe"]["concept_bus_v2_gradient_norm"] > 0
    assert all(
        value > 0
        for value in result["checks"]["training_probe"]["gradient_norms"].values()
    )
    assert result["checks"]["training_probe"]["future_leakage_max_abs"] <= 1e-5
    assert result["checks"]["clutrr_data"]["test_lengths"] == [2, 3, 4, 5, 6]


def test_medium_preflight_prepares_only_the_declared_synthetic_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = copy.deepcopy(CFG)
    cfg["sizes"]["medium"]["require_gpu"] = False
    cfg["sizes"]["medium"]["data"].update(
        {
            "train_size": 32,
            "validation_size": 8,
            "test_size": 8,
            "max_length": 24,
        }
    )
    cfg["preflight"]["minimum_free_disk_gb"] = 0.0
    monkeypatch.setattr(
        preflight_module,
        "_training_probe",
        lambda *_args, **_kwargs: {"device": "cpu", "loss": 1.0},
    )
    monkeypatch.setattr(
        preflight_module,
        "ensure_clutrr_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("CLUTRR must not be prepared for medium")
        ),
    )
    monkeypatch.setattr(
        preflight_module,
        "ensure_active_language_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("language data must not be prepared for medium")
        ),
    )
    report = preflight_module.run_preflight(
        cfg,
        size_name="medium",
        output_root=tmp_path / "output_medium_preflight",
        prepare_data=True,
    )
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["status"] == "pass"
    assert "synthetic_data" in result["checks"]
    assert "clutrr_data" not in result["checks"]
    assert "language_data" not in result["checks"]


def test_fair_capacity_probe_executes_real_forward_backward_and_adam_step() -> None:
    cfg = copy.deepcopy(CFG)
    cfg["sizes"]["medium"]["data"]["max_length"] = 24
    cfg["sizes"]["medium"]["model"].update(
        {
            "num_layers": 1,
            "d_model": 32,
            "d_ff": 64,
            "num_heads": 4,
            "slot_dim": 32,
            "bus_heads": 2,
            "concept_residual_dim": 4,
        }
    )
    cfg["sizes"]["medium"]["suite"][0]["required_micro_batch_size"] = 2
    result = preflight_module._fair_synthetic_capacity_probe(
        torch.device("cpu"),
        cfg,
        "medium",
        "fair_concept_subspace",
    )
    assert result["method"] == "concept_bus_v2"
    assert result["micro_batch_size"] == 2
    assert result["sequence_length"] == 24
    assert result["gradient_norm"] > 0
