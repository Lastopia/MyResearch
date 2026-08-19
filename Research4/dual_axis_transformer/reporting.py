"""Centralized CSV, SVG, and dependency-free HTML/JavaScript reporting."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape


def _flatten_numeric(
    values: dict[str, Any], prefix: str = ""
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            flattened[name] = float(value)
        elif isinstance(value, dict):
            flattened.update(_flatten_numeric(value, name))
    return flattened


def discover_final_records(
    output_root: str | Path,
    *,
    stage: str | None = None,
    source_fingerprint: str | None = None,
) -> list[dict[str, Any]]:
    output_root = Path(output_root)
    # A seed is a replicate; rerunning the same seed under another attempt or
    # code version must never inflate n. Keep only the newest compatible final.
    records_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted((output_root / "runs").glob("**/metrics/final.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        relative = path.relative_to(output_root / "runs")
        parts = relative.parts
        inferred_stage = parts[0] if len(parts) > 0 else "unknown"
        method = parts[1] if len(parts) > 1 else "unknown"
        seed_text = parts[2] if len(parts) > 2 else "seed-1"
        match = re.search(r"-?\d+", seed_text)
        seed = int(match.group()) if match else -1
        record_stage = str(payload.get("stage", inferred_stage))
        if stage is not None and record_stage != stage:
            continue
        record_source = payload.get("source_fingerprint")
        if record_source is None:
            resolved_path = path.parent.parent / "resolved_config.json"
            try:
                resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
                record_source = resolved.get(
                    "source_fingerprint", resolved.get("source")
                )
            except (OSError, json.JSONDecodeError):
                record_source = None
        if (
            source_fingerprint is not None
            and record_source != source_fingerprint
        ):
            continue
        metrics_source = payload.get("metrics", payload)
        metrics = _flatten_numeric(metrics_source)
        record_method = str(payload.get("method", method))
        record_seed = int(payload.get("seed", seed))
        record = {
            "stage": record_stage,
            "method": record_method,
            "seed": record_seed,
            "metrics": metrics,
            "path": str(path),
            "source_fingerprint": record_source,
            "config_hash": payload.get("config_hash"),
            "_attempt": next(
                (
                    int(match.group(1))
                    for part in reversed(parts)
                    if (match := re.fullmatch(r"attempt(\d+)", part))
                ),
                0,
            ),
            "_mtime_ns": path.stat().st_mtime_ns,
        }
        key = (record_stage, record_method, record_seed)
        previous = records_by_key.get(key)
        if previous is None or (
            record["_attempt"],
            record["_mtime_ns"],
            record["path"],
        ) > (
            previous["_attempt"],
            previous["_mtime_ns"],
            previous["path"],
        ):
            records_by_key[key] = record
    records = sorted(
        records_by_key.values(),
        key=lambda item: (item["stage"], item["method"], item["seed"]),
    )
    for record in records:
        record.pop("_attempt", None)
        record.pop("_mtime_ns", None)
    return records


def aggregate_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for record in records:
        for metric, value in record["metrics"].items():
            grouped[(record["stage"], record["method"], metric)].append(
                (record["seed"], float(value))
            )
    summaries = []
    for (stage, method, metric), observations in sorted(grouped.items()):
        values = [value for _, value in observations]
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        critical = _t_critical_975(len(values) - 1)
        half_width = (
            critical * std / math.sqrt(len(values))
            if len(values) > 1
            else float("nan")
        )
        summaries.append(
            {
                "stage": stage,
                "method": method,
                "metric": metric,
                "n": len(values),
                "mean": mean,
                "std": std,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
                "min": min(values),
                "max": max(values),
                "points": [
                    {"seed": seed, "value": value}
                    for seed, value in observations
                ],
            }
        )
    return summaries


def _t_critical_975(degrees: int) -> float:
    """Two-sided 95% Student-t critical value without a scipy dependency."""

    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        24: 2.064,
        30: 2.042,
        40: 2.021,
        60: 2.000,
        120: 1.980,
    }
    if degrees <= 0:
        return float("nan")
    if degrees in table:
        return table[degrees]
    larger = sorted(value for value in table if value >= degrees)
    return table[larger[0]] if larger else 1.960


def _exact_sign_flip_pvalue(differences: list[float]) -> float:
    """Exact paired randomization p-value for small seed counts."""

    count = len(differences)
    if count == 0:
        return float("nan")
    observed = abs(statistics.fmean(differences))
    if count > 20:
        # Paper suites use 3--5 seeds. Refuse a silent exponential blow-up if
        # users later increase it far beyond the registered design.
        return float("nan")
    extreme = 0
    total = 1 << count
    for mask in range(total):
        permuted = [
            value if mask & (1 << index) else -value
            for index, value in enumerate(differences)
        ]
        if abs(statistics.fmean(permuted)) >= observed - 1e-12:
            extreme += 1
    return extreme / total


def paired_comparisons(
    records: Iterable[dict[str, Any]], *, reference: str = "concept_bus_v2"
) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    methods_by_stage: dict[str, set[str]] = defaultdict(set)
    for record in records:
        stage = record["stage"]
        method = record["method"]
        methods_by_stage[stage].add(method)
        for metric, value in record["metrics"].items():
            indexed[(stage, method, metric)][record["seed"]] = float(value)
    rows: list[dict[str, Any]] = []
    for stage, methods in sorted(methods_by_stage.items()):
        stage_reference = reference
        if stage_reference not in methods:
            continue
        metrics = sorted(
            metric
            for (item_stage, method, metric) in indexed
            if item_stage == stage and method == stage_reference
        )
        for comparator in sorted(methods - {stage_reference}):
            for metric in metrics:
                left = indexed.get((stage, stage_reference, metric), {})
                right = indexed.get((stage, comparator, metric), {})
                seeds = sorted(set(left) & set(right))
                if not seeds:
                    continue
                differences = [left[seed] - right[seed] for seed in seeds]
                mean = statistics.fmean(differences)
                std = statistics.stdev(differences) if len(differences) > 1 else 0.0
                critical = _t_critical_975(len(differences) - 1)
                half_width = (
                    critical * std / math.sqrt(len(differences))
                    if len(differences) > 1
                    else float("nan")
                )
                rows.append(
                    {
                        "stage": stage,
                        "reference": stage_reference,
                        "comparator": comparator,
                        "metric": metric,
                        "n": len(seeds),
                        "seeds": seeds,
                        "mean_difference": mean,
                        "std_difference": std,
                        "ci95_low": mean - half_width,
                        "ci95_high": mean + half_width,
                        "randomization_p": _exact_sign_flip_pvalue(differences),
                    }
                )
    finite = [row for row in rows if math.isfinite(row["randomization_p"])]
    ordered = sorted(finite, key=lambda row: row["randomization_p"])
    running = 0.0
    total = len(ordered)
    for index, row in enumerate(ordered):
        adjusted = min(1.0, (total - index) * row["randomization_p"])
        running = max(running, adjusted)
        row["holm_p"] = running
    for row in rows:
        row.setdefault("holm_p", float("nan"))
    return rows


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    return safe[:120] or "metric"


def _write_metric_svg(
    path: Path, metric: str, summaries: list[dict[str, Any]]
) -> None:
    width = 960
    height = 420
    margin_left, margin_right, margin_top, margin_bottom = 90, 30, 52, 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    if not summaries:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="220" '
            'viewBox="0 0 960 220"><rect width="100%" height="100%" fill="#fff"/>'
            '<text x="480" y="110" text-anchor="middle" font-family="sans-serif" '
            'font-size="18">No completed run metrics yet</text></svg>',
            encoding="utf-8",
        )
        return

    low = min(0.0, min(item["min"] for item in summaries))
    high = max(0.0, max(item["max"] for item in summaries))
    if math.isclose(low, high):
        high = low + 1.0

    def y_pos(value: float) -> float:
        return margin_top + (high - value) / (high - low) * plot_height

    zero_y = y_pos(0.0)
    slot_width = plot_width / len(summaries)
    bar_width = min(96.0, slot_width * 0.55)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{escape(metric)} comparison</title>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{margin_left}" y="28" font-family="sans-serif" '
        f'font-size="20" font-weight="600">{escape(metric)}</text>',
        f'<line x1="{margin_left}" y1="{zero_y:.2f}" x2="{width-margin_right}" '
        f'y2="{zero_y:.2f}" stroke="#777" stroke-width="1"/>',
    ]
    for index, item in enumerate(summaries):
        center = margin_left + (index + 0.5) * slot_width
        mean_y = y_pos(item["mean"])
        rect_y = min(mean_y, zero_y)
        rect_height = max(1.0, abs(zero_y - mean_y))
        elements.append(
            f'<rect x="{center-bar_width/2:.2f}" y="{rect_y:.2f}" '
            f'width="{bar_width:.2f}" height="{rect_height:.2f}" '
            'fill="#4c78a8" opacity="0.78"/>'
        )
        std_top = y_pos(item["mean"] + item["std"])
        std_bottom = y_pos(item["mean"] - item["std"])
        elements.extend(
            [
                f'<line x1="{center:.2f}" y1="{std_top:.2f}" x2="{center:.2f}" '
                f'y2="{std_bottom:.2f}" stroke="#222" stroke-width="2"/>',
                f'<line x1="{center-8:.2f}" y1="{std_top:.2f}" x2="{center+8:.2f}" '
                f'y2="{std_top:.2f}" stroke="#222" stroke-width="2"/>',
                f'<line x1="{center-8:.2f}" y1="{std_bottom:.2f}" '
                f'x2="{center+8:.2f}" y2="{std_bottom:.2f}" '
                'stroke="#222" stroke-width="2"/>',
            ]
        )
        for point_index, point in enumerate(item["points"]):
            jitter = (point_index - (len(item["points"]) - 1) / 2) * 7
            elements.append(
                f'<circle cx="{center+jitter:.2f}" cy="{y_pos(point["value"]):.2f}" '
                'r="4" fill="#f58518" stroke="#fff" stroke-width="1"/>'
            )
        elements.extend(
            [
                f'<text x="{center:.2f}" y="{max(margin_top+14, mean_y-10):.2f}" '
                'text-anchor="middle" font-family="sans-serif" font-size="13">'
                f'{item["mean"]:.4g}</text>',
                f'<text x="{center:.2f}" y="{height-margin_bottom+28}" '
                'text-anchor="middle" font-family="sans-serif" font-size="13">'
                f'{escape(item["method"])}</text>',
            ]
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _write_long_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["stage", "method", "seed", "metric", "value", "path"]
        )
        writer.writeheader()
        for record in records:
            for metric, value in sorted(record["metrics"].items()):
                writer.writerow(
                    {
                        "stage": record["stage"],
                        "method": record["method"],
                        "seed": record["seed"],
                        "metric": metric,
                        "value": value,
                        "path": record["path"],
                    }
                )


def _write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "stage",
            "method",
            "metric",
            "n",
            "mean",
            "std",
            "ci95_low",
            "ci95_high",
            "min",
            "max",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            writer.writerow({key: item[key] for key in fieldnames})


def _write_paired_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "stage",
        "reference",
        "comparator",
        "metric",
        "n",
        "seeds",
        "mean_difference",
        "std_difference",
        "ci95_low",
        "ci95_high",
        "randomization_p",
        "holm_p",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["seeds"] = ",".join(str(value) for value in row["seeds"])
            writer.writerow({key: payload[key] for key in fieldnames})


def build_report(
    output_root: str | Path,
    *,
    stage: str | None = None,
    max_static_figures: int = 12,
    source_fingerprint: str | None = None,
) -> Path:
    output_root = Path(output_root)
    records = discover_final_records(
        output_root,
        stage=stage,
        source_fingerprint=source_fingerprint,
    )
    summaries = aggregate_records(records)
    comparisons = paired_comparisons(records)
    report_name = stage or "all"
    report_dir = output_root / "reports" / report_name
    figure_dir = report_dir / "figures"
    dashboard_dir = report_dir / "dashboard"
    figure_dir.mkdir(parents=True, exist_ok=True)
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    _write_long_csv(report_dir / "per_seed.csv", records)
    _write_summary_csv(report_dir / "summary.csv", summaries)
    _write_paired_csv(report_dir / "paired_comparisons.csv", comparisons)
    metrics = sorted({item["metric"] for item in summaries})
    if not metrics:
        _write_metric_svg(figure_dir / "no-results.svg", "No results", [])
    for metric in metrics[:max_static_figures]:
        selected = [item for item in summaries if item["metric"] == metric]
        _write_metric_svg(figure_dir / f"{_safe_name(metric)}.svg", metric, selected)

    resource_path = output_root / "resources" / "snapshot.json"
    try:
        resources = json.loads(resource_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        resources = None
    report_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": report_name,
        "records": records,
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "resources": resources,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (dashboard_dir / "data.js").write_text(
        "window.REPORT_DATA = "
        + json.dumps(report_data, ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )

    assets = Path(__file__).with_name("report_assets")
    for name in ("index.html", "report.js", "styles.css"):
        shutil.copyfile(assets / name, dashboard_dir / name)
    return report_dir
