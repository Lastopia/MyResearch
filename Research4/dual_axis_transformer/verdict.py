"""Pre-registered, machine-readable verdicts for the unattended suite."""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .reporting import discover_final_records


def _values(
    records: list[dict[str, Any]], stage: str, method: str, metric: str
) -> list[float]:
    return [
        float(record["metrics"][metric])
        for record in records
        if record["stage"] == stage
        and record["method"] == method
        and metric in record["metrics"]
    ]


def _gate(
    records: list[dict[str, Any]],
    *,
    name: str,
    stage: str,
    method: str,
    metric: str,
    minimum_runs: int,
    threshold: float,
    comparison: str,
) -> dict[str, Any]:
    values = _values(records, stage, method, metric)
    if len(values) < minimum_runs:
        return {
            "name": name,
            "status": "not_available",
            "stage": stage,
            "method": method,
            "metric": metric,
            "n": len(values),
            "required_n": minimum_runs,
            "threshold": threshold,
            "comparison": comparison,
        }
    mean = statistics.fmean(values)
    passed = mean >= threshold if comparison == ">=" else mean <= threshold
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "stage": stage,
        "method": method,
        "metric": metric,
        "n": len(values),
        "mean": mean,
        "threshold": threshold,
        "comparison": comparison,
        "values": values,
    }


def _difference_gate(
    records: list[dict[str, Any]],
    *,
    name: str,
    stage: str,
    treatment: str,
    reference: str,
    metric: str,
    minimum_runs: int,
    threshold: float,
    require_positive_each_seed: bool = False,
) -> dict[str, Any]:
    treatment_values = {
        int(record["seed"]): float(record["metrics"][metric])
        for record in records
        if record["stage"] == stage
        and record["method"] == treatment
        and metric in record["metrics"]
    }
    reference_values = {
        int(record["seed"]): float(record["metrics"][metric])
        for record in records
        if record["stage"] == stage
        and record["method"] == reference
        and metric in record["metrics"]
    }
    seeds = sorted(set(treatment_values) & set(reference_values))
    count = len(seeds)
    if count < minimum_runs:
        return {
            "name": name,
            "status": "not_available",
            "n": count,
            "required_n": minimum_runs,
            "threshold": threshold,
            "comparison": ">=",
        }
    differences = [
        treatment_values[seed] - reference_values[seed] for seed in seeds
    ]
    difference = statistics.fmean(differences)
    direction_consistent = all(value > 0.0 for value in differences)
    passed = difference >= threshold and (
        direction_consistent or not require_positive_each_seed
    )
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "stage": stage,
        "treatment": treatment,
        "reference": reference,
        "metric": metric,
        "n": count,
        "seeds": seeds,
        "differences": differences,
        "require_positive_each_seed": require_positive_each_seed,
        "direction_consistent": direction_consistent,
        "mean_difference": difference,
        "threshold": threshold,
        "comparison": ">=",
    }


def build_paper_verdict(
    output_root: str | Path, *, source_fingerprint: str | None = None
) -> Path:
    output_root = Path(output_root)
    records = discover_final_records(
        output_root, source_fingerprint=source_fingerprint
    )
    specifications = [
        ("dual_tag_concept_f1", "dual_tag", "concept_bus_v2", "test_concept_macro_f1", 3, 0.80, ">="),
        ("dual_tag_exact", "dual_tag", "concept_bus_v2", "test_exact_accuracy", 3, 0.70, ">="),
        ("unknown_auroc", "dual_tag", "concept_bus_v2", "test_unknown_auroc", 3, 0.75, ">="),
        ("bus_necessity", "dual_tag", "concept_bus_v2", "zero_bus_exact_drop", 3, 0.05, ">="),
        ("country_counterfactual", "dual_tag", "concept_bus_v2", "country_counterfactual_success", 3, 0.70, ">="),
        ("color_counterfactual", "dual_tag", "concept_bus_v2", "color_counterfactual_success", 3, 0.70, ">="),
        ("country_cf_side_effect", "dual_tag", "concept_bus_v2", "country_counterfactual_color_side_effect", 3, 0.05, "<="),
        ("color_cf_side_effect", "dual_tag", "concept_bus_v2", "color_counterfactual_country_side_effect", 3, 0.05, "<="),
        ("projector_orthogonality", "dual_tag", "concept_bus_v2", "projector_orthogonality", 3, 0.10, "<="),
        ("clutrr_length_4", "clutrr", "concept_bus_v2", "test_accuracy_length_4", 3, 0.70, ">="),
        ("clutrr_length_5", "clutrr", "concept_bus_v2", "test_accuracy_length_5", 3, 0.60, ">="),
        ("clutrr_length_6", "clutrr", "concept_bus_v2", "test_accuracy_length_6", 3, 0.50, ">="),
    ]
    gates = [
        _gate(
            records,
            name=name,
            stage=stage,
            method=method,
            metric=metric,
            minimum_runs=minimum,
            threshold=threshold,
            comparison=comparison,
        )
        for name, stage, method, metric, minimum, threshold, comparison in specifications
    ]
    gates.extend(
        [
            _difference_gate(
                records,
                name="dual_tag_performance_gap",
                stage="dual_tag",
                treatment="concept_bus_v2",
                reference="parameter_matched",
                metric="test_exact_accuracy",
                minimum_runs=3,
                threshold=-0.02,
            ),
            _difference_gate(
                records,
                name="attention_increment_over_projector",
                stage="dual_tag",
                treatment="concept_bus_v2",
                reference="concept_projector",
                metric="test_concept_macro_f1",
                minimum_runs=3,
                threshold=0.01,
                require_positive_each_seed=True,
            ),
        ]
    )
    bus_ppl = _values(
        records, "formal_language_model", "concept_bus_v2", "validation_perplexity"
    )
    mac_ppl = _values(
        records, "formal_language_model", "parameter_matched", "validation_perplexity"
    )
    if len(bus_ppl) >= 3 and len(mac_ppl) >= 3:
        ratio = statistics.fmean(bus_ppl) / statistics.fmean(mac_ppl)
        gates.append(
            {
                "name": "formal_lm_ppl_ratio",
                "status": "pass" if ratio <= 1.02 else "fail",
                "n": min(len(bus_ppl), len(mac_ppl)),
                "mean_ratio_v2_over_parameter_matched": ratio,
                "threshold": 1.02,
                "comparison": "<=",
            }
        )
    else:
        gates.append(
            {
                "name": "formal_lm_ppl_ratio",
                "status": "not_available",
                "n": min(len(bus_ppl), len(mac_ppl)),
                "required_n": 3,
            }
        )
    for name, metric, threshold in (
        ("formal_lm_training_time_ratio", "training_seconds", 1.25),
        ("formal_lm_peak_memory_ratio", "peak_cuda_memory_bytes", 1.25),
    ):
        bus_values = _values(
            records, "formal_language_model", "concept_bus_v2", metric
        )
        baseline_values = _values(
            records, "formal_language_model", "parameter_matched", metric
        )
        if len(bus_values) >= 3 and len(baseline_values) >= 3:
            ratio = statistics.fmean(bus_values) / max(
                1e-12, statistics.fmean(baseline_values)
            )
            gates.append(
                {
                    "name": name,
                    "status": "pass" if ratio <= threshold else "fail",
                    "n": min(len(bus_values), len(baseline_values)),
                    "mean_ratio_v2_over_parameter_matched": ratio,
                    "threshold": threshold,
                    "comparison": "<=",
                }
            )
        else:
            gates.append(
                {
                    "name": name,
                    "status": "not_available",
                    "n": min(len(bus_values), len(baseline_values)),
                    "required_n": 3,
                }
            )
    available = [gate for gate in gates if gate["status"] != "not_available"]
    overall = (
        "not_available"
        if not available
        else "pass"
        if len(available) == len(gates) and all(gate["status"] == "pass" for gate in gates)
        else "fail_or_incomplete"
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": overall,
        "warning": "A pass means the pre-registered thresholds were met; it is not a claim of novelty or independent replication.",
        "gates": gates,
    }
    path = output_root / "reports" / "paper_verdict.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
