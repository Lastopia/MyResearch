from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tools.io import read_json, write_json
from tools.log import stage_banner
from tools.paths import output_dir


def _read_if_exists(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def _evaluation_views(
    evaluation: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    return [
        (str(kind), payload)
        for kind, payload in evaluation.get("checkpoints", {}).items()
    ]


def _with_checkpoint(
    rows: list[dict[str, Any]],
    checkpoint_kind: str,
) -> list[dict[str, Any]]:
    return [
        {"checkpoint_kind": checkpoint_kind, **row}
        for row in rows
    ]


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def _source_fields(
    source: dict[str, Any],
    *,
    source_name: str,
) -> dict[str, Any]:
    return {
        "data_source": source_name,
        "dataset": source.get("dataset"),
        "dataset_config": source.get("config"),
        "dataset_revision": source.get("revision"),
        "split": source.get("local_split") or source.get("upstream_split"),
        "sample_count": source.get("sample_count")
        or source.get("samples_per_length"),
        "program_generated": source.get("program_generated"),
        "generator": source.get("generator"),
        "generator_version": source.get("generator_version"),
        "seed": source.get("seed"),
    }


def _natural_rows(
    evaluation: dict[str, Any],
    *,
    method: str,
    seed: str,
) -> list[dict[str, Any]]:
    sources = evaluation.get("data_sources", {}).get("natural_language", {})
    rows: list[dict[str, Any]] = []
    for length, payload in evaluation.get("lengths", {}).items():
        natural = payload.get("natural_language", {})
        source_mapping = {
            "fineweb_edu_held_out_ppl": "fineweb_edu_held_out",
            "wikitext103_ppl": "wikitext103",
        }
        for metric, source_name in source_mapping.items():
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "position_mode": "native",
                    "length": int(length),
                    "metric": "perplexity",
                    "value": natural.get(metric),
                    **_source_fields(
                        sources.get(source_name, {}),
                        source_name=source_name,
                    ),
                }
            )
        rope_pi = payload.get("rope_pi")
        if rope_pi:
            for metric, source_name in source_mapping.items():
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "position_mode": "rope_pi",
                        "inference_scale": rope_pi.get("inference_scale"),
                        "length": int(length),
                        "metric": "perplexity",
                        "value": rope_pi.get("natural_language", {}).get(metric),
                        **_source_fields(
                            sources.get(source_name, {}),
                            source_name=source_name,
                        ),
                    }
                )
    return rows


def _single_query_row(
    metrics: dict[str, Any],
    *,
    method: str,
    seed: str,
    length: int,
    position_mode: str,
    source: dict[str, Any],
    ldr: float | None,
    inference_scale: float | None = None,
) -> dict[str, Any]:
    return {
        "method": method,
        "seed": seed,
        "position_mode": position_mode,
        "inference_scale": inference_scale,
        "length": length,
        "condition": "single_query",
        "accuracy": metrics.get("accuracy"),
        "near_accuracy": metrics.get("near_accuracy"),
        "far_accuracy": metrics.get("far_accuracy"),
        "rcug": metrics.get("rcug"),
        "length_degradation_rate": ldr,
        "input_sha256": metrics.get("input_sha256"),
        **_source_fields(source, source_name="single_query"),
    }


def _synthetic_rows(
    evaluation: dict[str, Any],
    *,
    method: str,
    seed: str,
) -> list[dict[str, Any]]:
    sources = evaluation.get("data_sources", {}).get("synthetic_control", {})
    rows: list[dict[str, Any]] = []
    for length_text, payload in evaluation.get("lengths", {}).items():
        length = int(length_text)
        controls = payload.get("synthetic_control", {})
        single = controls.get(
            "single_query",
            {
                key: payload.get(key)
                for key in (
                    "accuracy",
                    "near_accuracy",
                    "far_accuracy",
                    "rcug",
                )
            },
        )
        rows.append(
            _single_query_row(
                single,
                method=method,
                seed=seed,
                length=length,
                position_mode="native",
                source=sources.get("single_query", {}),
                ldr=evaluation.get("length_degradation_rate"),
            )
        )

        multi = controls.get("multi_query_associative_recall")
        if multi:
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "position_mode": "native",
                    "length": length,
                    "condition": "multi_query_associative_recall",
                    "accuracy": multi.get("association_accuracy"),
                    "sample_exact_match": multi.get("sample_exact_match"),
                    "near_accuracy": multi.get("near_accuracy"),
                    "far_accuracy": multi.get("far_accuracy"),
                    "queries_per_sample": multi.get("queries_per_sample"),
                    "similar_distractors_per_query": multi.get(
                        "similar_distractors_per_query"
                    ),
                    "input_sha256": multi.get("input_sha256"),
                    **_source_fields(
                        sources.get("multi_query_associative_recall", {}),
                        source_name="multi_query_associative_recall",
                    ),
                }
            )

        position_swap = controls.get("target_distractor_position_swap")
        if position_swap:
            for condition, metrics in (
                ("position_swap_original", position_swap.get("original", {})),
                ("position_swap_swapped", position_swap.get("swapped", {})),
            ):
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "position_mode": "native",
                        "length": length,
                        "condition": condition,
                        "accuracy": metrics.get("accuracy"),
                        "near_accuracy": metrics.get("near_accuracy"),
                        "far_accuracy": metrics.get("far_accuracy"),
                        "rcug": metrics.get("rcug"),
                        "accuracy_delta_swapped_minus_original": position_swap.get(
                            "accuracy_delta_swapped_minus_original"
                        ),
                        "input_sha256": position_swap.get(
                            "original_input_sha256"
                            if condition.endswith("original")
                            else "swapped_input_sha256"
                        ),
                        **_source_fields(
                            sources.get(
                                "target_distractor_position_swap",
                                {},
                            ),
                            source_name="target_distractor_position_swap",
                        ),
                    }
                )

        rope_pi = payload.get("rope_pi")
        if rope_pi:
            rope_pi_controls = rope_pi.get("synthetic_control", {})
            rope_pi_single = rope_pi_controls.get("single_query")
            if rope_pi_single:
                rows.append(
                    _single_query_row(
                        rope_pi_single,
                        method=method,
                        seed=seed,
                        length=length,
                        position_mode="rope_pi",
                        inference_scale=rope_pi.get("inference_scale"),
                        source=sources.get("single_query", {}),
                        ldr=evaluation.get(
                            "rope_pi_length_degradation_rate"
                        ),
                    )
                )
            rope_pi_multi = rope_pi_controls.get(
                "multi_query_associative_recall"
            )
            if rope_pi_multi:
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "position_mode": "rope_pi",
                        "inference_scale": rope_pi.get("inference_scale"),
                        "length": length,
                        "condition": "multi_query_associative_recall",
                        "accuracy": rope_pi_multi.get(
                            "association_accuracy"
                        ),
                        "sample_exact_match": rope_pi_multi.get(
                            "sample_exact_match"
                        ),
                        "near_accuracy": rope_pi_multi.get("near_accuracy"),
                        "far_accuracy": rope_pi_multi.get("far_accuracy"),
                        **_source_fields(
                            sources.get(
                                "multi_query_associative_recall",
                                {},
                            ),
                            source_name="multi_query_associative_recall",
                        ),
                    }
                )
    return rows


def _real_long_document_rows(
    evaluation: dict[str, Any],
    *,
    method: str,
    seed: str,
) -> list[dict[str, Any]]:
    source = (
        evaluation.get("data_sources", {})
        .get("real_long_document_qa", {})
        .get("qasper", {})
    )
    rows: list[dict[str, Any]] = []
    for length_text, payload in evaluation.get("lengths", {}).items():
        variants = [
            (
                "native",
                None,
                payload.get("real_long_document_qa", {}).get("qasper"),
            )
        ]
        rope_pi = payload.get("rope_pi")
        if rope_pi:
            variants.append(
                (
                    "rope_pi",
                    rope_pi.get("inference_scale"),
                    rope_pi.get("real_long_document_qa", {}).get("qasper"),
                )
            )
        for position_mode, inference_scale, metrics in variants:
            if not metrics:
                continue
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "position_mode": position_mode,
                    "inference_scale": inference_scale,
                    "length": int(length_text),
                    "dataset": source.get("dataset"),
                    "data_file": source.get("data_file"),
                    "dataset_revision": source.get("revision"),
                    "split": source.get("split"),
                    "sample_count": metrics.get("samples"),
                    "program_generated": False,
                    "program_modified": True,
                    "question_ids": json.dumps(
                        metrics.get("question_ids", []),
                        ensure_ascii=False,
                    ),
                    "sample_sha256": json.dumps(
                        metrics.get("sample_sha256", []),
                        ensure_ascii=False,
                    ),
                    "leakage_check_complete": source.get(
                        "leakage_check_complete"
                    ),
                    "fineweb_documents_excluded_for_evaluation_overlap": (
                        source.get(
                            "fineweb_documents_excluded_for_evaluation_overlap",
                            0,
                        )
                    ),
                    "answer_nll": metrics.get("answer_nll"),
                    "answer_perplexity": metrics.get("answer_perplexity"),
                    "token_f1": metrics.get("token_f1"),
                    "exact_match": metrics.get("exact_match"),
                    "evidence_utilization_gain": metrics.get(
                        "evidence_utilization_gain"
                    ),
                    "mean_evidence_distance_tokens": metrics.get(
                        "mean_evidence_distance_tokens"
                    ),
                }
            )
    return rows


def _audit_rows(
    audit: dict[str, Any],
    *,
    method: str,
    seed: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = audit.get("data_sources", {})
    cross = audit.get("cross_length_by_condition", {})
    for length_text, payload in audit.get("lengths", {}).items():
        conditions = payload.get(
            "conditions",
            {
                "synthetic_remote_target": {
                    "summary": payload.get("summary", {}),
                    "samples": None,
                    "input_sha256": None,
                }
            },
        )
        for condition, condition_result in conditions.items():
            summary = condition_result.get("summary", {})
            source = sources.get(condition, {})
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "length": int(length_text),
                    "condition": condition,
                    "sample_count": condition_result.get("samples"),
                    "input_sha256": condition_result.get("input_sha256"),
                    "program_generated": source.get("program_generated"),
                    "dataset": source.get("dataset"),
                    "dataset_config": source.get("config"),
                    "dataset_revision": source.get("revision"),
                    "split": source.get("local_split"),
                    "generator": source.get("generator"),
                    "generator_version": source.get("generator_version"),
                    "mad": summary.get("mad_mean"),
                    "nad": summary.get("nad_mean"),
                    "r90": summary.get("r90_mean"),
                    "ghf": summary.get("geometric_head_fit"),
                    "rhdd": cross.get(condition, {})
                    .get("summary", {})
                    .get("relative_head_distance_drift"),
                    "cas": summary.get("context_adaptivity_score"),
                    "bmvr_far_near_pair": summary.get(
                        "bias_monotonic_violation_rate"
                    ),
                    "bmvr_adjacent_diagnostic": summary.get(
                        "bias_adjacent_monotonic_violation_rate"
                    ),
                    "raa": summary.get("relevant_attention_advantage"),
                    "sesr": summary.get(
                        "semantic_exemption_success_rate"
                    ),
                    "fer_all_explicit_far_irrelevant": summary.get(
                        "false_exemption_rate"
                    ),
                    "attention_sink_special_combined": summary.get(
                        "attention_sink_ratio"
                    ),
                    "attention_sink_bos_or_eos": summary.get(
                        "attention_sink_bos_or_eos_ratio"
                    ),
                    "attention_sink_newline": summary.get(
                        "attention_sink_newline_ratio"
                    ),
                    "attention_sink_separator": summary.get(
                        "attention_sink_separator_ratio"
                    ),
                    "attention_sink_query": summary.get(
                        "attention_sink_query_ratio"
                    ),
                }
            )
    return rows


def _efficiency_rows(
    profile: dict[str, Any],
    *,
    method: str,
    seed: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hardware = profile.get("hardware", {})
    package_versions = hardware.get("package_versions", {})
    full_training = profile.get("full_training", {})
    for length_text, payload in profile.get("lengths", {}).items():
        train = payload.get("train_step", {})
        prefill = payload.get("prefill", {})
        decode = payload.get("decode", {})
        rows.append(
            {
                "method": method,
                "seed": seed,
                "length": int(length_text),
                "status": payload.get("status"),
                "input_source": "torch.randint uniform token IDs",
                "program_generated": True,
                "warmup_trials": profile.get(
                    "measurement_protocol", {}
                ).get("warmup_trials"),
                "measurement_trials": profile.get(
                    "measurement_protocol", {}
                ).get("measurement_trials"),
                "decode_tokens": profile.get(
                    "measurement_protocol", {}
                ).get("decode_tokens"),
                "dtype": profile.get("dtype"),
                "attention_kernel": profile.get("attention_kernel"),
                "decode_mode": profile.get("decode_mode"),
                "gpu_names": ",".join(hardware.get("gpu_names", [])),
                "torch_version": hardware.get("torch"),
                "cuda_version": hardware.get("cuda_version"),
                "cudnn_version": hardware.get("cudnn_version"),
                "transformers_version": package_versions.get("transformers"),
                "datasets_version": package_versions.get("datasets"),
                "numpy_version": package_versions.get("numpy"),
                "train_effective_batch_tokens": train.get(
                    "effective_batch_tokens"
                ),
                "train_micro_batch_size": train.get("micro_batch_size"),
                "train_gradient_accumulation_steps": train.get(
                    "gradient_accumulation_steps"
                ),
                "train_median_seconds": train.get("median_seconds"),
                "train_p95_seconds": train.get("p95_seconds"),
                "train_tokens_per_second": train.get("tokens_per_second"),
                "train_peak_vram_gb": train.get("peak_vram_gb"),
                "train_peak_host_ram_gb": train.get("peak_host_ram_gb"),
                "prefill_median_seconds": prefill.get("median_seconds"),
                "prefill_p95_seconds": prefill.get("p95_seconds"),
                "prefill_tokens_per_second": prefill.get(
                    "tokens_per_second"
                ),
                "prefill_peak_vram_gb": prefill.get("peak_vram_gb"),
                "prefill_peak_host_ram_gb": prefill.get(
                    "peak_host_ram_gb"
                ),
                "decode_median_seconds": decode.get("median_seconds"),
                "decode_p95_seconds": decode.get("p95_seconds"),
                "decode_ms_per_token": decode.get(
                    "milliseconds_per_token"
                ),
                "decode_tokens_per_second": decode.get("tokens_per_second"),
                "decode_peak_vram_gb": decode.get("peak_vram_gb"),
                "decode_peak_host_ram_gb": decode.get(
                    "peak_host_ram_gb"
                ),
                "pretrain_wall_seconds": full_training.get(
                    "pretrain_wall_clock_seconds"
                ),
                "adapt_wall_seconds": full_training.get(
                    "adapt_wall_clock_seconds"
                ),
                "total_wall_seconds": full_training.get(
                    "wall_clock_seconds"
                ),
                "gpu_hours": full_training.get("gpu_hours"),
                "total_parameters": profile.get("total_parameters"),
                "position_parameters": profile.get("position_parameters"),
                "position_parameter_ratio": profile.get(
                    "position_parameter_ratio"
                ),
            }
        )
    return rows


def _markdown_table(
    lines: list[str],
    *,
    headers: list[str],
    rows: list[list[Any]],
) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    stage_banner("REPORT", cfg=cfg)
    root = output_dir(cfg)
    natural_rows: list[dict[str, Any]] = []
    real_long_document_rows: list[dict[str, Any]] = []
    synthetic_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    efficiency_rows: list[dict[str, Any]] = []
    source_manifest: dict[str, Any] = {}

    method_seed_pairs: set[tuple[str, str]] = set()
    for category in ("metrics", "audits", "profiles"):
        category_root = root / category
        if not category_root.exists():
            continue
        for method_dir in category_root.iterdir():
            if not method_dir.is_dir():
                continue
            for seed_dir in method_dir.glob("seed*"):
                method_seed_pairs.add(
                    (method_dir.name, seed_dir.name.removeprefix("seed"))
                )

    for method, seed in sorted(method_seed_pairs):
        seed_name = f"seed{seed}"
        evaluation = _read_if_exists(
            root / "metrics" / method / seed_name / "evaluation.json"
        )
        audit = _read_if_exists(
            root / "audits" / method / seed_name / "attention_audit.json"
        )
        profile = _read_if_exists(
            root / "profiles" / method / seed_name / "efficiency.json"
        )
        if evaluation:
            for checkpoint_kind, evaluation_view in _evaluation_views(
                evaluation
            ):
                natural_rows.extend(
                    _with_checkpoint(
                        _natural_rows(
                            evaluation_view,
                            method=method,
                            seed=seed,
                        ),
                        checkpoint_kind,
                    )
                )
                real_long_document_rows.extend(
                    _with_checkpoint(
                        _real_long_document_rows(
                            evaluation_view,
                            method=method,
                            seed=seed,
                        ),
                        checkpoint_kind,
                    )
                )
                synthetic_rows.extend(
                    _with_checkpoint(
                        _synthetic_rows(
                            evaluation_view,
                            method=method,
                            seed=seed,
                        ),
                        checkpoint_kind,
                    )
                )
                source_manifest[
                    f"{method}/seed{seed}/evaluation/{checkpoint_kind}"
                ] = evaluation_view.get("data_sources", {})
        if audit:
            audit_rows.extend(_audit_rows(audit, method=method, seed=seed))
            source_manifest[f"{method}/seed{seed}/audit"] = audit.get(
                "data_sources", {}
            )
        if profile:
            efficiency_rows.extend(
                _efficiency_rows(profile, method=method, seed=seed)
            )

    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    table_paths = {
        "natural_language": _write_csv(
            table_dir / "natural_language.csv",
            natural_rows,
        ),
        "real_long_document_qa": _write_csv(
            table_dir / "real_long_document_qa.csv",
            real_long_document_rows,
        ),
        "synthetic_control": _write_csv(
            table_dir / "synthetic_control.csv",
            synthetic_rows,
        ),
        "attention_audit": _write_csv(
            table_dir / "attention_audit.csv",
            audit_rows,
        ),
        "efficiency": _write_csv(
            table_dir / "efficiency.csv",
            efficiency_rows,
        ),
    }
    manifest_rows = [
        {
            "table": name,
            "row_count": len(rows),
            "path": path,
            "contains_program_generated_inputs": (
                name in {"synthetic_control", "attention_audit", "efficiency"}
            ),
        }
        for name, path, rows in (
            ("natural_language", table_paths["natural_language"], natural_rows),
            (
                "real_long_document_qa",
                table_paths["real_long_document_qa"],
                real_long_document_rows,
            ),
            (
                "synthetic_control",
                table_paths["synthetic_control"],
                synthetic_rows,
            ),
            ("attention_audit", table_paths["attention_audit"], audit_rows),
            ("efficiency", table_paths["efficiency"], efficiency_rows),
        )
    ]
    summary_path = _write_csv(table_dir / "summary.csv", manifest_rows)
    write_json(table_dir / "data_sources.json", source_manifest)

    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"
    lines = [
        "# 实验结果摘要",
        "",
        "自然语料 PPL、真实长文档问答、合成控制任务、attention 机制审计"
        "和效率结果严格分表。"
        "所有程序生成或程序修改的输入均在 CSV 的 `program_generated` "
        "列中明确标记；合成控制结果不能表述为真实检索、真实问答或真实用户场景表现。",
        "",
        f"当前配置包含 {len(cfg['run']['seeds'])} 个配对 seed；统计推断至少需要 "
        f"{int(cfg['stats']['minimum_inferential_seeds'])} 个完整配对 seed。"
        "未达到门槛时，结果仅作为探索性描述，不报告统计检验。",
        "",
        "## 真实自然语料 PPL",
        "",
    ]
    _markdown_table(
        lines,
        headers=["方法", "seed", "checkpoint", "模式", "长度", "数据集", "PPL"],
        rows=[
            [
                row["method"],
                row["seed"],
                row["checkpoint_kind"],
                row["position_mode"],
                row["length"],
                row["data_source"],
                row["value"],
            ]
            for row in natural_rows
        ],
    )
    lines.extend(["", "## 真实长文档问答（QASPER）", ""])
    _markdown_table(
        lines,
        headers=[
            "方法",
            "seed",
            "checkpoint",
            "模式",
            "长度",
            "样本",
            "Answer NLL",
            "Token F1",
            "EM",
            "证据利用增益",
            "证据距离",
            "泄漏检查完整",
        ],
        rows=[
            [
                row["method"],
                row["seed"],
                row["checkpoint_kind"],
                row["position_mode"],
                row["length"],
                row["sample_count"],
                row["answer_nll"],
                row["token_f1"],
                row["exact_match"],
                row["evidence_utilization_gain"],
                row["mean_evidence_distance_tokens"],
                row["leakage_check_complete"],
            ]
            for row in real_long_document_rows
        ],
    )
    lines.extend(["", "## 合成控制任务", ""])
    _markdown_table(
        lines,
        headers=[
            "方法",
            "seed",
            "checkpoint",
            "模式",
            "长度",
            "条件",
            "Accuracy",
            "Near",
            "Far",
            "RCUG",
            "Exact",
            "LDR",
        ],
        rows=[
            [
                row["method"],
                row["seed"],
                row["checkpoint_kind"],
                row["position_mode"],
                row["length"],
                row["condition"],
                row.get("accuracy"),
                row.get("near_accuracy"),
                row.get("far_accuracy"),
                row.get("rcug"),
                row.get("sample_exact_match"),
                row.get("length_degradation_rate"),
            ]
            for row in synthetic_rows
        ],
    )
    lines.extend(["", "## Attention 机制审计", ""])
    _markdown_table(
        lines,
        headers=[
            "方法",
            "seed",
            "长度",
            "条件",
            "MAD",
            "NAD",
            "GHF",
            "RHDD",
            "CAS",
            "BMVR",
            "RAA",
            "SESR",
            "FER",
            "Sink",
        ],
        rows=[
            [
                row["method"],
                row["seed"],
                row["length"],
                row["condition"],
                row.get("mad"),
                row.get("nad"),
                row.get("ghf"),
                row.get("rhdd"),
                row.get("cas"),
                row.get("bmvr_far_near_pair"),
                row.get("raa"),
                row.get("sesr"),
                row.get("fer_all_explicit_far_irrelevant"),
                row.get("attention_sink_special_combined"),
            ]
            for row in audit_rows
        ],
    )
    lines.extend(["", "## 效率与资源", ""])
    _markdown_table(
        lines,
        headers=[
            "方法",
            "seed",
            "长度",
            "训练(s)",
            "Prefill(s)",
            "Decode(ms/token)",
            "Train VRAM",
            "Prefill VRAM",
            "Decode VRAM",
            "Train RAM",
            "Prefill RAM",
            "Decode RAM",
            "GPU-hours",
        ],
        rows=[
            [
                row["method"],
                row["seed"],
                row["length"],
                row.get("train_median_seconds"),
                row.get("prefill_median_seconds"),
                row.get("decode_ms_per_token"),
                row.get("train_peak_vram_gb"),
                row.get("prefill_peak_vram_gb"),
                row.get("decode_peak_vram_gb"),
                row.get("train_peak_host_ram_gb"),
                row.get("prefill_peak_host_ram_gb"),
                row.get("decode_peak_host_ram_gb"),
                row.get("gpu_hours"),
            ]
            for row in efficiency_rows
        ],
    )
    lines.extend(
        [
            "",
            "## 统计与限制",
            "",
            "统计检验位于 `tables/stats.json`，其中同时包含相对 RoPE 的"
            "比较和 RA-CABLE 相对 CABLE 的核心配对比较。逐层逐头结果位于"
            " `audits/`，完整来源清单位于 `tables/data_sources.json`。",
            "",
            "只有合成任务改善而真实长文档任务没有改善时，只能将结果报告为"
            "机制线索，不得据此形成“长程信息利用得到改善”的最终结论。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stage_banner("REPORT", "DONE", cfg=cfg)
    return {
        "tables": table_paths,
        "summary": summary_path,
        "data_sources": str(table_dir / "data_sources.json"),
        "report": str(report_path),
        "row_counts": {
            "natural_language": len(natural_rows),
            "real_long_document_qa": len(real_long_document_rows),
            "synthetic_control": len(synthetic_rows),
            "attention_audit": len(audit_rows),
            "efficiency": len(efficiency_rows),
        },
    }
