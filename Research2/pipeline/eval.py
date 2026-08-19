import json
import math

from model.factory import active_aliases, build_model, default_position_encoding, experiment_alias, is_pretrain_mode, resolve_experiment, validate_aliases
from pipeline.train import run_seeds
from tools.io import ensure_dir, output_dir, read_json, save_config, write_json
from tools.log import stage_title
from tools.metrics import benjamini_hochberg, hierarchical_bootstrap_mean, paired_sign_flip_test, paired_t_test
from tools.plot import plot_metric_bars, plot_metric_by_k


def latest_jsonl(path):
    if not path.exists():
        return {}
    last = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    return last


def _read_models(path, key="models"):
    if not path.exists():
        return []
    data = read_json(path)
    return data.get(key, []) if isinstance(data, dict) else []


def _sae_key(row):
    return (
        row.get("model"), row.get("model_seed"), row.get("layer"), row.get("expansion"),
        row.get("k"), row.get("sae_seed"),
    )


def _stability_key(row):
    return row.get("model"), row.get("model_seed"), row.get("layer"), row.get("expansion"), row.get("k")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _copy_metrics(target, source, prefix=""):
    identity = {"model", "seed", "model_seed", "layer", "expansion", "k", "sae_seed", "sae_id", "gpu", "time", "pairs"}
    for key, value in source.items():
        if key not in identity:
            target[f"{prefix}{key}"] = value


def _safe_delta(row, baseline, metric, suffix=""):
    if _is_number(row.get(metric)) and _is_number(baseline.get(metric)):
        row[f"delta_{metric}{suffix}"] = float(row[metric]) - float(baseline[metric])


def _baseline_alias(cfg, aliases):
    requested = cfg.get("eval", {}).get("baseline", "base")
    if is_pretrain_mode(cfg) or requested in aliases:
        return requested
    expanded = experiment_alias(cfg, requested, default_position_encoding(cfg))
    return expanded if expanded in aliases else requested


def _baseline_alias_for(cfg, alias, aliases):
    requested = cfg.get("eval", {}).get("baseline", "base")
    if is_pretrain_mode(cfg):
        return requested
    experiment = resolve_experiment(cfg, alias)
    candidate = experiment_alias(cfg, requested, experiment["position_encoding"])
    if candidate in aliases:
        return candidate
    return _baseline_alias(cfg, aliases)


def model_level_rows(cfg, aliases, seeds, metrics_dir):
    attn_rows = _read_models(metrics_dir / "attn_summary.json")
    ffn_rows = _read_models(metrics_dir / "ffn_summary.json")
    attn_map = {(row.get("model"), row.get("seed", row.get("model_seed"))): row for row in attn_rows}
    ffn_map = {(row.get("model"), row.get("seed", row.get("model_seed"))): row for row in ffn_rows}
    rows = []
    parameter_counts = {}
    if not is_pretrain_mode(cfg):
        for alias in aliases:
            parameter_counts[alias] = sum(parameter.numel() for parameter in build_model(cfg, alias).parameters())
    for seed in seeds:
        for alias in aliases:
            row = {"model": alias, "model_seed": seed, "parameter_count": parameter_counts.get(alias)}
            _copy_metrics(row, latest_jsonl(metrics_dir / f"[{alias}]seed{seed}train.jsonl"), "train_")
            _copy_metrics(row, latest_jsonl(metrics_dir / f"[{alias}]seed{seed}valid.jsonl"), "valid_")
            _copy_metrics(row, attn_map.get((alias, seed), {}), "attn_")
            _copy_metrics(row, ffn_map.get((alias, seed), {}), "ffn_")
            rows.append(row)
    return rows


def joined_sae_rows(metrics_dir):
    sae_data = read_json(metrics_dir / "sae_summary.json") if (metrics_dir / "sae_summary.json").exists() else {}
    sae_rows = sae_data.get("models", [])
    interpret_data = read_json(metrics_dir / "interpret_summary.json") if (metrics_dir / "interpret_summary.json").exists() else {}
    interpret_map = {_sae_key(row): row for row in interpret_data.get("models", [])}
    stability_map = {_stability_key(row): row for row in interpret_data.get("stability", [])}
    rows = []
    for sae_row in sae_rows:
        row = dict(sae_row)
        _copy_metrics(row, interpret_map.get(_sae_key(row), {}))
        _copy_metrics(row, stability_map.get(_stability_key(row), {}))
        rows.append(row)
    return rows, sae_data.get("frontiers", [])


COMPARISON_METRICS = [
    "normalized_mse",
    "explained_variance",
    "loss_recovered",
    "kl_recovered",
    "actual_l0",
    "fraction_alive",
    "rare_feature_rate",
    "feature_usage_entropy",
    "activation_entropy",
    "decoder_duplication_proxy",
    "feature_ablation_loss_delta_mean",
    "concept_macro_auprc",
    "concept_macro_auroc",
    "concept_macro_f1",
    "concept_auprc_above_permutation",
    "trained_sae_auprc_above_untrained_sae",
    "trained_sae_auprc_above_random_transformer_untrained_sae",
    "fade_clarity_annotation_proxy_mean",
    "fade_responsiveness_mean",
    "fade_purity_mean",
    "fade_faithfulness_next_token_proxy_mean",
    "no_explanation_rate",
    "strict_no_explanation_rate",
    "low_context_coherence_rate",
    "hard_negative_separation_failure_rate",
    "feature_context_coherence_mean",
    "known_concept_best_auprc_mean",
    "known_concept_best_auprc_gain_mean",
    "hard_negative_coherence_margin_mean",
    "human_semantic_coherence_mean",
    "human_hard_negative_separability_mean",
    "human_monosemanticity_mean",
    "absorption_rate_mean",
    "absorption_false_positive_rate_increase_mean",
    "causal_necessity_control_adjusted",
    "causal_sufficiency_control_adjusted",
    "causal_necessity_isolation",
    "causal_sufficiency_isolation",
    "causal_necessity_isolation_condition_median",
    "causal_sufficiency_isolation_condition_median",
    "stability_pw_mcc",
    "stability_decoder_hungarian_cosine",
]


HIGHER_IS_BETTER = {
    "explained_variance", "loss_recovered", "kl_recovered", "fraction_alive", "feature_usage_entropy",
    "feature_ablation_loss_delta_mean", "concept_macro_auprc", "concept_macro_auroc", "concept_macro_f1",
    "concept_auprc_above_permutation", "feature_context_coherence_mean", "known_concept_best_auprc_mean",
    "known_concept_best_auprc_gain_mean",
    "trained_sae_auprc_above_untrained_sae", "trained_sae_auprc_above_random_transformer_untrained_sae",
    "fade_clarity_annotation_proxy_mean", "fade_responsiveness_mean", "fade_purity_mean", "fade_faithfulness_next_token_proxy_mean",
    "hard_negative_coherence_margin_mean", "causal_necessity_control_adjusted", "causal_sufficiency_control_adjusted",
    "human_semantic_coherence_mean", "human_hard_negative_separability_mean", "human_monosemanticity_mean",
    "causal_necessity_isolation", "causal_sufficiency_isolation", "stability_pw_mcc",
    "causal_necessity_isolation_condition_median", "causal_sufficiency_isolation_condition_median",
    "stability_decoder_hungarian_cosine",
}


LOWER_IS_BETTER = {
    "normalized_mse", "actual_l0", "rare_feature_rate", "decoder_duplication_proxy",
    "no_explanation_rate", "strict_no_explanation_rate", "low_context_coherence_rate",
    "hard_negative_separation_failure_rate", "absorption_rate_mean",
    "absorption_false_positive_rate_increase_mean",
}


def add_matched_comparisons(rows, baseline_by_model):
    baseline_aliases = set(baseline_by_model.values())
    baseline_rows_by_alias = {
        alias: [row for row in rows if row.get("model") == alias]
        for alias in baseline_aliases
    }
    exact_maps = {
        alias: {_sae_key(row)[1:]: row for row in baseline_rows}
        for alias, baseline_rows in baseline_rows_by_alias.items()
    }
    for row in rows:
        baseline_alias = baseline_by_model.get(row.get("model"))
        baseline_rows = baseline_rows_by_alias.get(baseline_alias, [])
        exact = exact_maps.get(baseline_alias, {}).get(_sae_key(row)[1:])
        if exact:
            for metric in COMPARISON_METRICS:
                _safe_delta(row, exact, metric)
            row["baseline_exact_k"] = exact.get("k")

        pool = [
            candidate for candidate in baseline_rows
            if candidate.get("model_seed") == row.get("model_seed")
            and candidate.get("layer") == row.get("layer")
            and candidate.get("expansion") == row.get("expansion")
            and candidate.get("sae_seed") == row.get("sae_seed")
        ]
        if pool and _is_number(row.get("actual_l0")):
            l0_match = min(pool, key=lambda candidate: abs(float(candidate.get("actual_l0", float("inf"))) - float(row["actual_l0"])))
            row["baseline_l0_matched_k"] = l0_match.get("k")
            row["baseline_l0_match_error"] = abs(float(l0_match["actual_l0"]) - float(row["actual_l0"]))
            for metric in COMPARISON_METRICS:
                _safe_delta(row, l0_match, metric, "_at_l0_match")
        if pool and _is_number(row.get("loss_recovered")):
            fidelity_match = min(pool, key=lambda candidate: abs(float(candidate.get("loss_recovered", -float("inf"))) - float(row["loss_recovered"])))
            row["baseline_fidelity_matched_k"] = fidelity_match.get("k")
            row["baseline_fidelity_match_error"] = abs(float(fidelity_match["loss_recovered"]) - float(row["loss_recovered"]))
            for metric in COMPARISON_METRICS:
                _safe_delta(row, fidelity_match, metric, "_at_fidelity_match")

        for metric in COMPARISON_METRICS:
            delta_key = f"delta_{metric}"
            if _is_number(row.get(delta_key)):
                direction = 1.0 if metric in HIGHER_IS_BETTER else -1.0 if metric in LOWER_IS_BETTER else 1.0
                row[f"gain_{metric}"] = direction * float(row[delta_key])


def select_primary_operating_points(cfg, rows, baseline_by_model):
    layer = int(cfg.get("eval", {}).get("primary_layer", cfg["model"].get("n_layer", -1)))
    expansion = int(cfg.get("eval", {}).get("primary_expansion", cfg["sae"].get("expansion", 8)))
    target = float(cfg.get("eval", {}).get("loss_recovered_target", 0.95))
    candidates = [row for row in rows if int(row.get("layer", -999)) == layer and int(row.get("expansion", -999)) == expansion]
    groups = {}
    for row in candidates:
        groups.setdefault((row.get("model"), row.get("model_seed"), row.get("sae_seed")), []).append(row)
    selected = []
    for key, group in groups.items():
        eligible = [row for row in group if _is_number(row.get("loss_recovered")) and row["loss_recovered"] >= target]
        if eligible:
            chosen = min(eligible, key=lambda row: (float(row.get("actual_l0", float("inf"))), int(row.get("k", 10 ** 9))))
            target_met = True
        else:
            available = [row for row in group if _is_number(row.get("loss_recovered"))]
            if not available:
                continue
            chosen = max(available, key=lambda row: row["loss_recovered"])
            target_met = False
        primary = dict(chosen)
        primary["fidelity_target"] = target
        primary["fidelity_target_met"] = target_met
        primary["primary_operating_point"] = True
        selected.append(primary)

    selected_map = {
        (row["model"], row["model_seed"], row["sae_seed"]): row
        for row in selected
    }
    for row in selected:
        baseline_alias = baseline_by_model.get(row["model"])
        baseline = selected_map.get((baseline_alias, row["model_seed"], row["sae_seed"]))
        if not baseline or not row.get("fidelity_target_met") or not baseline.get("fidelity_target_met"):
            continue
        row["baseline_primary_k"] = baseline.get("k")
        for metric in COMPARISON_METRICS:
            _safe_delta(row, baseline, metric, "_primary")
            key = f"delta_{metric}_primary"
            if _is_number(row.get(key)):
                direction = 1.0 if metric in HIGHER_IS_BETTER else -1.0 if metric in LOWER_IS_BETTER else 1.0
                row[f"gain_{metric}_primary"] = direction * float(row[key])
    return selected


def aggregate_primary(cfg, primary_rows, aliases, baseline_by_model):
    stats_cfg = cfg["interpretability"]["statistics"]
    repeats = int(stats_cfg.get("bootstrap_repeats", 2000))
    confidence = float(stats_cfg.get("confidence", 0.95))
    primary_metrics = list(stats_cfg.get("primary_metrics", []))
    metric_map = {
        "k_at_95_loss_recovered": "k",
        "concept_macro_auprc": "concept_macro_auprc",
        "causal_necessity_isolation": "causal_necessity_isolation",
        "stability_pw_mcc": "stability_pw_mcc",
    }
    all_metrics = list(dict.fromkeys(primary_metrics + [
        "normalized_mse", "loss_recovered", "actual_l0", "concept_macro_auprc", "no_explanation_rate",
        "strict_no_explanation_rate", "low_context_coherence_rate", "hard_negative_separation_failure_rate",
        "known_concept_best_auprc_gain_mean",
        "fade_responsiveness_mean", "fade_purity_mean", "fade_faithfulness_next_token_proxy_mean",
        "absorption_rate_mean", "absorption_false_positive_rate_increase_mean",
        "causal_necessity_isolation", "causal_sufficiency_isolation",
        "causal_necessity_isolation_condition_median", "causal_sufficiency_isolation_condition_median",
        "stability_pw_mcc",
    ]))
    aggregates = []
    tests = []
    for alias in aliases:
        all_alias_rows = [row for row in primary_rows if row["model"] == alias]
        alias_rows = [row for row in all_alias_rows if row.get("fidelity_target_met")]
        aggregate = {"model": alias, "primary_rows": len(alias_rows), "attempted_primary_rows": len(all_alias_rows)}
        for requested in all_metrics:
            metric = metric_map.get(requested, requested)
            if not any(_is_number(row.get(metric)) for row in alias_rows):
                continue
            ci = hierarchical_bootstrap_mean(
                alias_rows, metric, "model_seed", "sae_seed", repeats, confidence,
                seed=int(cfg["interpretability"].get("eval_seed", 314159)) + len(aggregates) * 101,
            )
            aggregate[f"{requested}_mean"] = ci["mean"]
            aggregate[f"{requested}_ci_low"] = ci["ci_low"]
            aggregate[f"{requested}_ci_high"] = ci["ci_high"]
        aggregate["fidelity_target_success_rate"] = sum(bool(row.get("fidelity_target_met")) for row in all_alias_rows) / max(1, len(all_alias_rows))
        aggregates.append(aggregate)

        if alias == baseline_by_model.get(alias):
            continue
        for requested in primary_metrics:
            metric = metric_map.get(requested, requested)
            delta_key = "delta_actual_l0_primary" if requested == "k_at_95_loss_recovered" else f"delta_{metric}_primary"
            by_seed = {}
            for row in alias_rows:
                if _is_number(row.get(delta_key)):
                    by_seed.setdefault(row["model_seed"], []).append(float(row[delta_key]))
            paired_differences = [sum(values) / len(values) for values in by_seed.values()]
            difference_ci = hierarchical_bootstrap_mean(
                alias_rows, delta_key, "model_seed", "sae_seed", repeats, confidence,
                seed=int(cfg["interpretability"].get("eval_seed", 314159)) + len(tests) * 211,
            )
            if len(paired_differences) > 1:
                difference_mean = sum(paired_differences) / len(paired_differences)
                difference_variance = sum((value - difference_mean) ** 2 for value in paired_differences) / (len(paired_differences) - 1)
                cohen_dz = difference_mean / max(1e-12, math.sqrt(difference_variance))
            else:
                cohen_dz = None
            tests.append({
                "model": alias,
                "metric": requested,
                "difference_key": delta_key,
                "paired_model_seeds": len(paired_differences),
                "mean_difference": None if not paired_differences else sum(paired_differences) / len(paired_differences),
                "difference_ci_low": difference_ci["ci_low"],
                "difference_ci_high": difference_ci["ci_high"],
                "cohen_dz": cohen_dz,
                "p_value": paired_sign_flip_test(paired_differences, seed=1729),
                "paired_t_p_value": paired_t_test(paired_differences),
            })
    corrections = benjamini_hochberg([row["p_value"] for row in tests], float(stats_cfg.get("fdr_alpha", 0.05)))
    for row, correction in zip(tests, corrections):
        row.update(correction)
    return aggregates, tests


def run(cfg):
    stage_title("eval")
    out = output_dir(cfg)
    metrics_dir = out / "metrics"
    ensure_dir(metrics_dir)
    aliases = active_aliases(cfg)
    validate_aliases(cfg, aliases)
    seeds = run_seeds(cfg)
    baseline_by_model = {alias: _baseline_alias_for(cfg, alias, aliases) for alias in aliases}

    model_rows = model_level_rows(cfg, aliases, seeds, metrics_dir)
    model_map = {(row["model"], row["model_seed"]): row for row in model_rows}
    for row in model_rows:
        baseline = model_map.get((baseline_by_model[row["model"]], row["model_seed"]), {})
        _safe_delta(row, baseline, "valid_valid_ce_loss")

    sae_rows, frontiers = joined_sae_rows(metrics_dir)
    add_matched_comparisons(sae_rows, baseline_by_model)
    primary_rows = select_primary_operating_points(cfg, sae_rows, baseline_by_model)
    aggregates, tests = aggregate_primary(cfg, primary_rows, aliases, baseline_by_model)

    result = {
        "baseline_by_model": baseline_by_model,
        "comparison_design": {
            "exact_spec": "same layer, expansion, k, model seed, and SAE seed",
            "l0_matched": "nearest baseline actual L0 within layer, expansion, model seed, and SAE seed",
            "fidelity_matched": "nearest baseline loss-recovered within layer, expansion, model seed, and SAE seed",
            "primary": "minimum-L0 SAE reaching the configured loss-recovered target at the preregistered layer and expansion",
        },
        "model_runs": model_rows,
        "sae_runs": sae_rows,
        "sae_frontiers": frontiers,
        "primary_operating_points": primary_rows,
        "mean_by_model": aggregates,
        "paired_primary_tests": tests,
    }
    write_json(metrics_dir / "final_summary.json", result)
    with (metrics_dir / "final_summary.jsonl").open("w", encoding="utf-8") as handle:
        for row in primary_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for metric in [
        "concept_macro_auprc_mean", "causal_necessity_isolation_mean", "stability_pw_mcc_mean",
        "no_explanation_rate_mean", "actual_l0_mean", "normalized_mse_mean",
    ]:
        plot_metric_bars(out, aggregates, metric, f"summary_interpret_{metric}.png")
    for metric in [
        "normalized_mse", "explained_variance", "reconstruction_cosine", "loss_recovered",
        "kl_recovered", "actual_l0", "fraction_alive", "rare_feature_rate",
        "feature_usage_entropy", "activation_entropy", "decoder_duplication_proxy",
        "feature_ablation_loss_delta_mean",
    ]:
        plot_metric_by_k(out, sae_rows, metric, f"sae_{metric}.png")
    save_config(cfg)
    print(f"[eval] done | primary_points={len(primary_rows)} | metrics={metrics_dir / 'final_summary.json'}", flush=True)
