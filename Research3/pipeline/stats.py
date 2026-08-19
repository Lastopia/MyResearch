from __future__ import annotations

import itertools
import math
import random
import statistics
from pathlib import Path
from typing import Any

from tools.io import read_json, write_json
from tools.log import stage_banner
from tools.paths import output_dir


def _flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten_numeric(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            output.update(_flatten_numeric(child, child_prefix))
    elif (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        output[prefix] = float(value)
    return output


def _bootstrap_ci(
    values: list[float],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    estimates = sorted(
        statistics.mean(rng.choice(values) for _ in values)
        for _ in range(max(1, samples))
    )
    tail = (1.0 - confidence) / 2.0
    low = estimates[min(len(estimates) - 1, int(tail * len(estimates)))]
    high = estimates[
        min(len(estimates) - 1, int((1.0 - tail) * len(estimates)))
    ]
    return low, high


def _sign_flip_pvalue(differences: list[float]) -> float:
    if not differences:
        return 1.0
    observed = abs(statistics.mean(differences))
    if len(differences) <= 12:
        estimates = [
            abs(
                statistics.mean(
                    sign * value
                    for sign, value in zip(signs, differences, strict=True)
                )
            )
            for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
        ]
    else:
        rng = random.Random(0)
        estimates = [
            abs(
                statistics.mean(
                    rng.choice((-1.0, 1.0)) * value for value in differences
                )
            )
            for _ in range(4096)
        ]
    return (sum(value >= observed for value in estimates) + 1) / (
        len(estimates) + 1
    )


def _holm_adjust(raw: list[float]) -> list[float]:
    order = sorted(range(len(raw)), key=raw.__getitem__)
    adjusted = [1.0] * len(raw)
    running = 0.0
    total = len(raw)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * raw[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _load_seed_records(root: Path) -> dict[str, dict[int, dict[str, float]]]:
    records: dict[str, dict[int, dict[str, float]]] = {}
    sources = [
        ("metrics", "evaluation.json"),
        ("audits", "attention_audit.json"),
        ("profiles", "efficiency.json"),
    ]
    for category, filename in sources:
        category_root = root / category
        if not category_root.exists():
            continue
        for method_dir in category_root.iterdir():
            if not method_dir.is_dir():
                continue
            for seed_dir in method_dir.glob("seed*"):
                path = seed_dir / filename
                if not path.exists():
                    continue
                try:
                    seed = int(seed_dir.name.removeprefix("seed"))
                except ValueError:
                    continue
                flattened = _flatten_numeric(read_json(path), category)
                records.setdefault(method_dir.name, {}).setdefault(seed, {}).update(
                    flattened
                )
    return records


def _paired_comparisons(
    records: dict[str, dict[int, dict[str, float]]],
    *,
    baseline_name: str,
    method_names: list[str],
    bootstrap_samples: int,
    confidence: float,
    seed: int,
    minimum_inferential_seeds: int,
    metric_directions: dict[str, str] | None = None,
    inferential_enabled: bool = True,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    baseline = records.get(baseline_name, {})
    for method in method_names:
        seed_records = records.get(method, {})
        common_seeds = sorted(set(baseline) & set(seed_records))
        common_keys = sorted(
            set.intersection(
                *[
                    set(baseline[item_seed]) & set(seed_records[item_seed])
                    for item_seed in common_seeds
                ]
            )
            if common_seeds
            else set()
        )
        if metric_directions is not None:
            common_keys = [
                key for key in common_keys if key in metric_directions
            ]
        for key in common_keys:
            differences = [
                seed_records[item_seed][key] - baseline[item_seed][key]
                for item_seed in common_seeds
            ]
            std = (
                statistics.stdev(differences)
                if len(differences) > 1
                else 0.0
            )
            difference_low, difference_high = _bootstrap_ci(
                differences,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=seed,
            )
            inferential = (
                inferential_enabled
                and len(differences) >= minimum_inferential_seeds
            )
            direction = (
                metric_directions.get(key)
                if metric_directions is not None
                else None
            )
            mean_difference = statistics.mean(differences)
            comparisons.append(
                {
                    "baseline": baseline_name,
                    "method": method,
                    "metric": key,
                    "direction": direction,
                    "n": len(differences),
                    "mean_difference": mean_difference,
                    "favorable_difference": (
                        mean_difference
                        * (1.0 if direction == "maximize" else -1.0)
                        if direction is not None
                        else None
                    ),
                    "difference_ci_low": difference_low,
                    "difference_ci_high": difference_high,
                    "paired_effect_size": (
                        mean_difference / std
                        if std > 0
                        else None
                    ),
                    "inferential": inferential,
                    "raw_p": (
                        _sign_flip_pvalue(differences)
                        if inferential
                        else None
                    ),
                    "holm_p": None,
                }
            )
    eligible = [
        (index, float(item["raw_p"]))
        for index, item in enumerate(comparisons)
        if item["raw_p"] is not None
    ]
    adjusted = _holm_adjust([value for _, value in eligible])
    for (index, _), value in zip(eligible, adjusted, strict=True):
        comparisons[index]["holm_p"] = value
    return comparisons


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y, strict=True)
    )
    denominator = math.sqrt(
        sum((value - x_mean) ** 2 for value in x)
        * sum((value - y_mean) ** 2 for value in y)
    )
    return numerator / denominator if denominator > 0 else None


def _synthetic_accuracy(payload: dict[str, Any]) -> float | None:
    value = (
        payload.get("synthetic_control", {})
        .get("single_query", {})
        .get("accuracy")
    )
    return float(value) if value is not None else None


def _evaluation_checkpoint(
    payload: dict[str, Any],
    checkpoint_kind: str,
) -> dict[str, Any]:
    return payload.get("checkpoints", {}).get(checkpoint_kind, {})


def _audit_summary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    return (
        payload.get("conditions", {})
        .get("synthetic_remote_target", {})
        .get("summary")
        or {}
    )


def _mechanism_retrieval_pairing(
    root: Path,
    *,
    minimum_inferential_seeds: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cable_root = root / "metrics" / "cable"
    ra_root = root / "metrics" / "ra_cable"
    if not cable_root.exists() or not ra_root.exists():
        return {
            "comparison": "ra_cable_minus_cable",
            "rows": [],
            "summary": {
                "paired_seed_count": 0,
                "retrieval_cas_pearson": None,
                "statistical_interpretation_allowed": False,
            },
        }
    cable_seeds = {path.name for path in cable_root.glob("seed*")}
    ra_seeds = {path.name for path in ra_root.glob("seed*")}
    common_seeds = sorted(cable_seeds & ra_seeds)
    for seed_name in common_seeds:
        cable_eval_path = cable_root / seed_name / "evaluation.json"
        ra_eval_path = ra_root / seed_name / "evaluation.json"
        cable_audit_path = (
            root / "audits" / "cable" / seed_name / "attention_audit.json"
        )
        ra_audit_path = (
            root
            / "audits"
            / "ra_cable"
            / seed_name
            / "attention_audit.json"
        )
        if not all(
            path.exists()
            for path in (
                cable_eval_path,
                ra_eval_path,
                cable_audit_path,
                ra_audit_path,
            )
        ):
            continue
        cable_eval = _evaluation_checkpoint(
            read_json(cable_eval_path),
            "adapt",
        )
        ra_eval = _evaluation_checkpoint(
            read_json(ra_eval_path),
            "adapt",
        )
        cable_audit = read_json(cable_audit_path)
        ra_audit = read_json(ra_audit_path)
        lengths = sorted(
            set(cable_eval.get("lengths", {}))
            & set(ra_eval.get("lengths", {}))
            & set(cable_audit.get("lengths", {}))
            & set(ra_audit.get("lengths", {})),
            key=int,
        )
        cable_rhdd = cable_audit.get("cross_length_summary", {}).get(
            "relative_head_distance_drift"
        )
        ra_rhdd = ra_audit.get("cross_length_summary", {}).get(
            "relative_head_distance_drift"
        )
        for length in lengths:
            cable_accuracy = _synthetic_accuracy(
                cable_eval["lengths"][length]
            )
            ra_accuracy = _synthetic_accuracy(ra_eval["lengths"][length])
            cable_summary = _audit_summary(
                cable_audit["lengths"][length]
            )
            ra_summary = _audit_summary(ra_audit["lengths"][length])
            if cable_accuracy is None or ra_accuracy is None:
                continue
            row: dict[str, Any] = {
                "seed": int(seed_name.removeprefix("seed")),
                "length": int(length),
                "retrieval_accuracy_difference": (
                    ra_accuracy - cable_accuracy
                ),
                "rhdd_difference": (
                    float(ra_rhdd) - float(cable_rhdd)
                    if ra_rhdd is not None and cable_rhdd is not None
                    else None
                ),
            }
            for output_name, metric_name in (
                ("cas_difference", "context_adaptivity_score"),
                ("raa_difference", "relevant_attention_advantage"),
                ("sesr_difference", "semantic_exemption_success_rate"),
                ("fer_difference", "false_exemption_rate"),
                ("attention_sink_difference", "attention_sink_ratio"),
            ):
                cable_value = cable_summary.get(metric_name)
                ra_value = ra_summary.get(metric_name)
                row[output_name] = (
                    float(ra_value) - float(cable_value)
                    if ra_value is not None and cable_value is not None
                    else None
                )
            rows.append(row)
    correlation_rows = [
        row
        for row in rows
        if row["cas_difference"] is not None
    ]
    paired_seed_count = len({row["seed"] for row in rows})
    return {
        "comparison": "ra_cable_minus_cable",
        "rows": rows,
        "summary": {
            "paired_seed_count": paired_seed_count,
            "paired_length_rows": len(rows),
            "retrieval_cas_pearson": _pearson(
                [
                    float(row["retrieval_accuracy_difference"])
                    for row in correlation_rows
                ],
                [
                    float(row["cas_difference"])
                    for row in correlation_rows
                ],
            ),
            "statistical_interpretation_allowed": (
                paired_seed_count >= minimum_inferential_seeds
            ),
            "note": (
                "With one seed these paired rows are descriptive only; they "
                "do not establish that an attention change caused a retrieval "
                "change."
            ),
        },
    }


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    stage_banner("STATS", cfg=cfg)
    root = output_dir(cfg)
    records = _load_seed_records(root)
    bootstrap_samples = int(cfg["stats"]["bootstrap_samples"])
    confidence = float(cfg["stats"]["confidence"])
    minimum_inferential_seeds = int(
        cfg["stats"]["minimum_inferential_seeds"]
    )
    primary_endpoints = {
        str(key): str(value)
        for key, value in cfg["stats"]["primary_endpoints"].items()
    }
    secondary_endpoints = {
        str(key): str(value)
        for key, value in cfg["stats"]["secondary_endpoints"].items()
    }
    aggregates: dict[str, Any] = {}
    for method, seed_records in sorted(records.items()):
        keys = sorted(
            set().union(*(record.keys() for record in seed_records.values()))
        )
        method_result: dict[str, Any] = {}
        for key in keys:
            values = [
                record[key] for record in seed_records.values() if key in record
            ]
            if not values:
                continue
            low, high = _bootstrap_ci(
                values,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=int(cfg["data"]["seed"]),
            )
            method_result[key] = {
                "n": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "ci_low": low,
                "ci_high": high,
            }
        aggregates[method] = method_result

    comparisons = _paired_comparisons(
        records,
        baseline_name="rope",
        method_names=[
            method for method in sorted(records) if method != "rope"
        ],
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=int(cfg["data"]["seed"]),
        minimum_inferential_seeds=minimum_inferential_seeds,
        metric_directions=primary_endpoints,
        inferential_enabled=False,
    )
    ra_cable_vs_cable = _paired_comparisons(
        records,
        baseline_name="cable",
        method_names=["ra_cable"],
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=int(cfg["data"]["seed"]),
        minimum_inferential_seeds=minimum_inferential_seeds,
        metric_directions=primary_endpoints,
    )
    ra_cable_vs_static = _paired_comparisons(
        records,
        baseline_name="ra_cable_static",
        method_names=["ra_cable"],
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=int(cfg["data"]["seed"]),
        minimum_inferential_seeds=minimum_inferential_seeds,
        metric_directions=primary_endpoints,
    )
    ra_cable_vs_dape = _paired_comparisons(
        records,
        baseline_name="dape_kerple",
        method_names=["ra_cable"],
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
        seed=int(cfg["data"]["seed"]),
        minimum_inferential_seeds=minimum_inferential_seeds,
        metric_directions=primary_endpoints,
    )

    result = {
        "confidence": confidence,
        "bootstrap_samples": bootstrap_samples,
        "minimum_inferential_seeds": minimum_inferential_seeds,
        "primary_endpoints": primary_endpoints,
        "secondary_endpoints": secondary_endpoints,
        "aggregates": aggregates,
        "inferential_families": {
            "ra_cable_vs_cable": "three confirmatory endpoints",
            "ra_cable_vs_static": (
                "separate three-endpoint ablation family"
            ),
            "ra_cable_vs_dape": (
                "separate three-endpoint strong-baseline family"
            ),
            "method_vs_rope": "descriptive only",
        },
        "paired_comparisons_vs_rope": comparisons,
        "paired_comparisons_ra_cable_vs_cable": ra_cable_vs_cable,
        "paired_comparisons_ra_cable_vs_static": ra_cable_vs_static,
        "paired_comparisons_ra_cable_vs_dape": ra_cable_vs_dape,
        "mechanism_retrieval_pairing_ra_cable_vs_cable": (
            _mechanism_retrieval_pairing(
                root,
                minimum_inferential_seeds=minimum_inferential_seeds,
            )
        ),
    }
    write_json(root / "tables" / "stats.json", result)
    stage_banner("STATS", "DONE", cfg=cfg)
    return result
