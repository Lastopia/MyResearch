import json
import hashlib
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model.factory import active_aliases, build_model, hidden_size, is_pretrain_mode, validate_aliases
from model.sae import TopKSAE
from pipeline.concepts import build_concept_masks, concept_log_odds, normalized_entropy
from pipeline.sae import _continue_logits, load_base, resolved_hook_layer
from pipeline.sae_grid import load_sae, sae_dir, sae_specs
from pipeline.train import BlockData, run_seeds
from tools.io import block_dir, ensure_dir, output_dir, save_config, write_json
from tools.log import event_line, stage_title
from tools.metrics import average_precision, best_f1_threshold, binary_metrics, fit_logistic_probe, krippendorff_alpha_interval, roc_auc, select_top_features, standardized_mean_gap
from tools.resource import configured_gpus, cuda_device, gpu_label, run_gpu_jobs


INTERPRETABILITY_METRICS_VERSION = 2


def interpret_dir(cfg, alias, model_seed, spec):
    return output_dir(cfg) / "interpretability" / f"{alias}_mseed{model_seed}_{spec['sae_id']}"


def interpret_summary_path(cfg, alias, model_seed, spec):
    return output_dir(cfg) / "metrics" / f"[{alias}]mseed{model_seed}_{spec['sae_id']}_interpret_summary.json"


def _safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        # The RuntimeError branch covers older checkpoints containing objects
        # rejected by the restricted weights-only loader.
        return torch.load(path, map_location=map_location)


@torch.no_grad()
def collect_activation_cache(base, data, cfg, spec, device, dtype):
    interpret_cfg = cfg["interpretability"]
    max_tokens = int(interpret_cfg.get("max_tokens", 32768))
    block_size = int(cfg["data"]["block_size"])
    requested_blocks = max(3, math.ceil(max_tokens / block_size))
    requested_blocks = min(requested_blocks, data.n_blocks)
    batch_size = min(int(cfg["train"]["batch_size"]), requested_blocks)
    seed = int(interpret_cfg.get("eval_seed", 314159))
    hook_layer = resolved_hook_layer(base, cfg, spec)
    hidden_rows = []
    input_rows = []
    target_rows = []
    for batch_index, start in enumerate(range(0, requested_blocks, batch_size)):
        current = min(batch_size, requested_blocks - start)
        x, y = data.deterministic_batch(current, device, seed, batch_index)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            _, hidden, _ = base(x, return_hidden=True, hook_layer=hook_layer)
        if hidden is None:
            raise RuntimeError(f"missing hidden state at layer {hook_layer}")
        hidden_rows.append(hidden.detach().float().cpu())
        input_rows.append(x.detach().cpu())
        target_rows.append(y.detach().cpu())
    hidden = torch.cat(hidden_rows, dim=0)
    inputs = torch.cat(input_rows, dim=0)
    targets = torch.cat(target_rows, dim=0)
    return {
        "hidden": hidden,
        "inputs": inputs,
        "targets": targets,
        "hook_layer": hook_layer,
        "blocks": int(hidden.size(0)),
        "tokens": int(hidden.size(0) * hidden.size(1)),
    }


@torch.no_grad()
def encode_cache(sae, hidden, encode_batch_size, device):
    flat = hidden.reshape(-1, hidden.size(-1))
    rows = []
    for start in range(0, flat.size(0), int(encode_batch_size)):
        batch = flat[start:start + int(encode_batch_size)].to(device)
        rows.append(sae.encode(batch).to(torch.float16).cpu())
    return torch.cat(rows, dim=0)


def block_split_indices(cache, cfg):
    n_blocks = int(cache["blocks"])
    sequence = int(cache["hidden"].size(1))
    train_fraction = float(cfg["interpretability"].get("train_fraction", 0.6))
    validation_fraction = float(cfg["interpretability"].get("validation_fraction", 0.2))
    train_blocks = max(1, int(n_blocks * train_fraction))
    validation_blocks = max(1, int(n_blocks * validation_fraction))
    if train_blocks + validation_blocks >= n_blocks:
        validation_blocks = 1
        train_blocks = max(1, n_blocks - 2)
    train_end = train_blocks * sequence
    validation_end = (train_blocks + validation_blocks) * sequence
    total = n_blocks * sequence
    return {
        "train": torch.arange(0, train_end),
        "validation": torch.arange(train_end, validation_end),
        "test": torch.arange(validation_end, total),
    }


def _finite_mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not values else float(sum(values) / len(values))


def _finite_median(values):
    values = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


def _isolation_ratio_of_means(effects, side_effects, floor):
    effect = _finite_mean(effects)
    side_effect = _finite_mean(side_effects)
    if effect is None or side_effect is None:
        return None
    return effect / math.sqrt(max(float(floor), side_effect))


def _probe_row(concept, representation, budget, metrics):
    return {
        "concept": concept,
        "representation": representation,
        "feature_budget": int(budget),
        **{key: value for key, value in metrics.items() if isinstance(value, (int, float))},
    }


def evaluate_concepts(cfg, latent, hidden, targets, masks, splits):
    concept_cfg = cfg["interpretability"]["concepts"]
    counts = [int(value) for value in concept_cfg.get("probe_feature_counts", [1, 5, 10])]
    max_budget = min(max(counts), latent.size(1))
    regularization = concept_cfg.get("probe_regularization", [1e-3])
    steps = int(concept_cfg.get("probe_steps", 250))
    lr = float(concept_cfg.get("probe_lr", 0.05))
    patience = int(concept_cfg.get("probe_validation_patience", 25))
    min_train = int(concept_cfg.get("min_positive_train", 64))
    min_test = int(concept_cfg.get("min_positive_test", 64))
    target_flat = targets.reshape(-1).long()
    hidden = hidden.reshape(-1, hidden.size(-1)).float()
    train_i, validation_i, test_i = splits["train"], splits["validation"], splits["test"]

    projection_dim = min(int(concept_cfg.get("random_projection_dim", 256)), hidden.size(1))
    generator = torch.Generator().manual_seed(int(cfg["interpretability"].get("eval_seed", 314159)))
    projection = torch.randn(hidden.size(1), projection_dim, generator=generator) / math.sqrt(hidden.size(1))
    projected = hidden @ projection
    permutation = torch.randperm(latent.size(0), generator=generator)
    permuted = latent[permutation]

    probe_rows = []
    feature_rows = []
    rankings = {}
    eligible = []
    for concept, vocab_mask in masks.items():
        labels = vocab_mask[target_flat]
        positive_counts = {name: int(labels[index].sum()) for name, index in splits.items()}
        if positive_counts["train"] < min_train or positive_counts["test"] < min_test:
            continue
        if int((~labels[train_i]).sum()) < min_train or int((~labels[test_i]).sum()) < min_test:
            continue
        if not bool(labels[validation_i].any()) or not bool((~labels[validation_i]).any()):
            continue
        eligible.append(concept)
        train_y, validation_y, test_y = labels[train_i], labels[validation_i], labels[test_i]

        latent_indices, latent_gaps = select_top_features(latent[train_i], train_y, max_budget)
        hidden_indices, _ = select_top_features(hidden[train_i], train_y, min(max(counts), hidden.size(1)))
        random_indices, _ = select_top_features(projected[train_i], train_y, min(max(counts), projected.size(1)))
        permuted_indices, _ = select_top_features(permuted[train_i], train_y, max_budget)
        rankings[concept] = {
            "features": [int(value) for value in latent_indices],
            "gaps": [float(value) for value in latent_gaps],
            "positive_mean": [float(latent[train_i][:, value][train_y].float().mean()) for value in latent_indices],
            "positive_counts": positive_counts,
        }

        for rank, feature in enumerate(latent_indices[:max_budget]):
            validation_scores = latent[validation_i, feature]
            threshold = best_f1_threshold(validation_scores, validation_y)
            metrics = binary_metrics(latent[test_i, feature], test_y, threshold)
            feature_rows.append({
                "concept": concept,
                "feature": int(feature),
                "rank": rank + 1,
                "direction": 1 if latent_gaps[rank] >= 0 else -1,
                "standardized_gap": float(latent_gaps[rank]),
                **metrics,
            })

        representations = {
            "sae": (latent, latent_indices),
            "residual": (hidden, hidden_indices),
            "random_projection": (projected, random_indices),
            "permuted_latents": (permuted, permuted_indices),
        }
        for budget in counts:
            for representation, (matrix, ordered) in representations.items():
                width = min(int(budget), ordered.numel())
                chosen = ordered[:width]
                metrics, _ = fit_logistic_probe(
                    matrix[train_i][:, chosen], train_y,
                    matrix[validation_i][:, chosen], validation_y,
                    matrix[test_i][:, chosen], test_y,
                    regularization, steps, lr, patience,
                )
                probe_rows.append(_probe_row(concept, representation, width, metrics))

        # Dense residual-stream ceiling: it distinguishes a concept absent from
        # the model from one present in the model but not cleanly represented by
        # SAE latents.
        dense_metrics, _ = fit_logistic_probe(
            hidden[train_i], train_y, hidden[validation_i], validation_y, hidden[test_i], test_y,
            regularization, steps, lr, patience,
        )
        probe_rows.append(_probe_row(concept, "residual_dense_ceiling", hidden.size(1), dense_metrics))

        max_concepts = concept_cfg.get("max_concepts")
        if max_concepts is not None and len(eligible) >= int(max_concepts):
            break

    primary_budget = int(concept_cfg.get("primary_probe_budget", 5))
    primary_rows = [row for row in probe_rows if row["representation"] == "sae" and row["feature_budget"] == primary_budget]
    if not primary_rows and counts:
        chosen_budget = min(counts, key=lambda value: abs(value - primary_budget))
        primary_rows = [row for row in probe_rows if row["representation"] == "sae" and row["feature_budget"] == chosen_budget]
    summary = {
        "eligible_concepts": len(eligible),
        "concept_macro_auprc": _finite_mean([row.get("auprc") for row in primary_rows]),
        "concept_macro_auroc": _finite_mean([row.get("auroc") for row in primary_rows]),
        "concept_macro_f1": _finite_mean([row.get("f1") for row in primary_rows]),
        "concept_primary_budget": primary_budget,
    }
    for representation in ("sae", "residual", "random_projection", "permuted_latents", "residual_dense_ceiling"):
        rows = [row for row in probe_rows if row["representation"] == representation]
        if representation != "residual_dense_ceiling":
            rows = [row for row in rows if row["feature_budget"] == primary_budget]
        summary[f"{representation}_macro_auprc"] = _finite_mean([row.get("auprc") for row in rows])
    if summary.get("concept_macro_auprc") is not None and summary.get("permuted_latents_macro_auprc") is not None:
        summary["concept_auprc_above_permutation"] = summary["concept_macro_auprc"] - summary["permuted_latents_macro_auprc"]
    return summary, probe_rows, feature_rows, rankings, eligible


def evaluate_null_latents(cfg, latent, targets, masks, splits, eligible, label):
    concept_cfg = cfg["interpretability"]["concepts"]
    null_cfg = cfg["interpretability"].get("null_controls", {})
    budget = min(int(null_cfg.get("probe_budget", 5)), latent.size(1))
    target_flat = targets.reshape(-1).long()
    train_i, validation_i, test_i = splits["train"], splits["validation"], splits["test"]
    rows = []
    for concept in eligible:
        labels = masks[concept][target_flat]
        chosen, _ = select_top_features(latent[train_i].float(), labels[train_i], budget)
        metrics, _ = fit_logistic_probe(
            latent[train_i][:, chosen].float(), labels[train_i],
            latent[validation_i][:, chosen].float(), labels[validation_i],
            latent[test_i][:, chosen].float(), labels[test_i],
            concept_cfg.get("probe_regularization", [1e-3]),
            int(concept_cfg.get("probe_steps", 250)),
            float(concept_cfg.get("probe_lr", 0.05)),
            int(concept_cfg.get("probe_validation_patience", 25)),
        )
        rows.append({"concept": concept, "representation": label, "feature_budget": budget, **metrics})
    return {
        f"{label}_macro_auprc": _finite_mean([row.get("auprc") for row in rows]),
        f"{label}_macro_auroc": _finite_mean([row.get("auroc") for row in rows]),
    }, rows


@torch.no_grad()
def random_transformer_latents(cfg, alias, model_seed, spec, cache, device, dtype):
    if is_pretrain_mode(cfg):
        return None
    seed = int(cfg["interpretability"].get("null_controls", {}).get("null_seed", 8675309)) + int(model_seed)
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    torch.manual_seed(seed)
    random_base = build_model(cfg, alias).to(device).eval()
    hook_layer = resolved_hook_layer(random_base, cfg, spec)
    hidden_rows = []
    batch_size = int(cfg["train"]["batch_size"])
    for start in range(0, cache["inputs"].size(0), batch_size):
        x = cache["inputs"][start:start + batch_size].to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            _, hidden, _ = random_base(x, return_hidden=True, hook_layer=hook_layer)
        hidden_rows.append(hidden.float().cpu())
    d_in = hidden_size(random_base, cfg)
    d_sae = d_in * int(spec["expansion"])
    random_sae = TopKSAE(d_in, d_sae, int(spec["k"]), tied_init=True, normalize_decoder=True).to(device).eval()
    latent = encode_cache(random_sae, torch.cat(hidden_rows, dim=0), cfg["interpretability"].get("encode_batch_size", 4096), device)
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)
    return latent


def _activation_stratified_indices(scores, sample_count, top_fraction, percentile_ranges, seed):
    scores = torch.as_tensor(scores, dtype=torch.float32)
    sample_count = min(int(sample_count), scores.numel())
    if sample_count <= 0:
        return torch.empty(0, dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    selected = []
    top_count = min(sample_count, max(1, int(sample_count * float(top_fraction))))
    selected.extend(int(value) for value in torch.topk(scores, top_count).indices)
    remaining_count = sample_count - len(selected)
    per_range = max(1, math.ceil(remaining_count / max(1, len(percentile_ranges))))
    for low, high in percentile_ranges:
        low_value = torch.quantile(scores, float(low))
        high_value = torch.quantile(scores, float(high))
        candidates = torch.where((scores >= low_value) & (scores <= high_value))[0]
        if candidates.numel():
            order = torch.randperm(candidates.numel(), generator=generator)
            added = 0
            for value in candidates[order]:
                value = int(value)
                if value not in selected:
                    selected.append(value)
                    added += 1
                    if len(selected) >= sample_count or added >= per_range:
                        break
        if len(selected) >= sample_count:
            break
    if len(selected) < sample_count:
        pool = torch.tensor([index for index in range(scores.numel()) if index not in set(selected)], dtype=torch.long)
        if pool.numel():
            order = torch.randperm(pool.numel(), generator=generator)
            selected.extend(int(value) for value in pool[order[:sample_count - len(selected)]])
    return torch.tensor(selected[:sample_count], dtype=torch.long)


def evaluate_fade_activation(cfg, latent, targets, masks, splits, rankings, eligible):
    """Annotation-backed FADE activation metrics.

    This uses the paper's absolute-Gini and AP definitions. Because the local
    pipeline has audited token annotations rather than LLM-generated open-text
    descriptions, Clarity is explicitly named a proxy; it must not be reported
    as the paper's synthetic-generation Clarity score.
    """
    fade_cfg = cfg["interpretability"].get("fade", {})
    if not fade_cfg.get("enabled", True):
        return {}, []
    target_flat = targets.reshape(-1).long()
    test_i = splits["test"]
    sample_count = int(fade_cfg.get("natural_samples_per_feature", 100))
    top_fraction = float(fade_cfg.get("top_activation_fraction", 0.25))
    ranges = fade_cfg.get("percentile_ranges", [[0.0, 0.5], [0.5, 0.75], [0.75, 0.95], [0.95, 1.0]])
    features_per_concept = int(fade_cfg.get("features_per_concept", 1))
    rows = []
    for concept_index, concept in enumerate(eligible):
        labels = masks[concept][target_flat][test_i]
        ranked = rankings.get(concept, {})
        features = [
            int(feature) for feature, gap in zip(ranked.get("features", []), ranked.get("gaps", [])) if gap > 0
        ][:features_per_concept]
        for feature in features:
            scores = latent[test_i, feature].float()
            full_auc = roc_auc(scores, labels)
            sampled = _activation_stratified_indices(
                scores, sample_count, top_fraction, ranges,
                int(cfg["interpretability"].get("eval_seed", 314159)) + concept_index * 1009 + feature,
            )
            if not sampled.numel() or not bool(labels[sampled].any()) or not bool((~labels[sampled]).any()):
                sampled = torch.arange(labels.numel())
            sampled_auc = roc_auc(scores[sampled], labels[sampled])
            rows.append({
                "concept": concept,
                "feature": feature,
                "description_source": "audited_token_concept",
                "fade_clarity_annotation_proxy": abs(2.0 * full_auc - 1.0),
                "fade_responsiveness": abs(2.0 * sampled_auc - 1.0),
                "fade_purity": average_precision(scores[sampled], labels[sampled]),
                "natural_samples": int(sampled.numel()),
            })
    return {
        "fade_features": len(rows),
        "fade_clarity_annotation_proxy_mean": _finite_mean([row["fade_clarity_annotation_proxy"] for row in rows]),
        "fade_responsiveness_mean": _finite_mean([row["fade_responsiveness"] for row in rows]),
        "fade_purity_mean": _finite_mean([row["fade_purity"] for row in rows]),
        "fade_full_clarity_available": False,
    }, rows


def attach_fade_faithfulness(fade_summary, fade_rows, causal_rows):
    causal_by_concept = {}
    for row in causal_rows:
        if int(row.get("feature_budget", 0)) == 1:
            causal_by_concept.setdefault(row["concept"], []).append(row)
    faithfulness_values = []
    for row in fade_rows:
        conditions = causal_by_concept.get(row["concept"], [])
        if not conditions:
            row["fade_faithfulness_next_token_proxy"] = None
            continue
        zero_probability = _finite_mean([condition.get("sufficiency_zero_concept_probability") for condition in conditions])
        maximum_probability = max(condition["sufficiency_patched_concept_probability"] for condition in conditions)
        if zero_probability is None:
            faithfulness = None
        else:
            faithfulness = max(maximum_probability - zero_probability, 0.0) / max(1e-12, 1.0 - zero_probability)
        row["fade_faithfulness_next_token_proxy"] = faithfulness
        if faithfulness is not None:
            faithfulness_values.append(faithfulness)
    fade_summary["fade_faithfulness_next_token_proxy_mean"] = _finite_mean(faithfulness_values)
    return fade_summary, fade_rows


def _sample_features_by_frequency(frequency, count, bins, seed):
    frequency = torch.as_tensor(frequency, dtype=torch.float32)
    generator = torch.Generator().manual_seed(int(seed))
    selected = []
    per_bin = max(1, math.ceil(int(count) / max(1, len(bins) - 1)))
    for low, high in zip(bins[:-1], bins[1:]):
        if high == bins[-1]:
            candidates = torch.where((frequency >= low) & (frequency <= high))[0]
        else:
            candidates = torch.where((frequency >= low) & (frequency < high))[0]
        if candidates.numel():
            order = torch.randperm(candidates.numel(), generator=generator)
            selected.extend(int(value) for value in candidates[order[:per_bin]])
    if len(selected) < int(count):
        remaining = torch.tensor([index for index in range(frequency.numel()) if index not in set(selected)], dtype=torch.long)
        if remaining.numel():
            order = torch.randperm(remaining.numel(), generator=generator)
            selected.extend(int(value) for value in remaining[order[:int(count) - len(selected)]])
    return torch.tensor(selected[:int(count)], dtype=torch.long)


def _context_coherence(hidden_rows):
    if hidden_rows.size(0) <= 1:
        return 1.0
    vectors = F.normalize(hidden_rows.float(), dim=-1)
    centroid = F.normalize(vectors.mean(dim=0, keepdim=True), dim=-1)
    return float((vectors @ centroid.T).mean())


def _best_prevalence_corrected_concept(scores, concept_labels):
    candidates = []
    for name, labels in concept_labels.items():
        if bool(labels.any()) and bool((~labels).any()):
            ap = average_precision(scores, labels)
            if math.isfinite(ap):
                prevalence = float(labels.float().mean())
                candidates.append((ap - prevalence, ap, prevalence, name))
    return max(candidates, default=(None, None, None, None), key=lambda item: item[0])


def evaluate_no_explanation(cfg, sae, latent, cache, frequency, masks, decode_tokens, target_dir):
    noexp_cfg = cfg["interpretability"]["no_explanation"]
    if not noexp_cfg.get("enabled", True):
        return {}, []
    count = min(int(noexp_cfg.get("sample_features", 256)), latent.size(1))
    examples = int(noexp_cfg.get("examples_per_feature", 20))
    hard_count = int(noexp_cfg.get("hard_negatives_per_feature", 20))
    bins = noexp_cfg.get("activation_frequency_bins", [0.0, 1e-4, 1e-3, 1.0])
    seed = int(cfg["interpretability"].get("eval_seed", 314159))
    chosen = _sample_features_by_frequency(frequency, count, bins, seed)
    hidden = cache["hidden"].reshape(-1, cache["hidden"].size(-1)).float()
    inputs = cache["inputs"]
    targets = cache["targets"].reshape(-1).long()
    sequence = cache["inputs"].size(1)
    context_width = int(noexp_cfg.get("context_tokens", 16))
    ap_gain_threshold = float(noexp_cfg.get("known_concept_auprc_gain_threshold", 0.05))
    coherence_threshold = float(noexp_cfg.get("context_coherence_threshold", 0.55))
    hard_margin_threshold = float(noexp_cfg.get("hard_negative_margin_threshold", 0.05))
    rows = []
    context_rows = []
    decoder = sae.decoder.weight.detach().float().cpu()
    encoder = sae.encoder.weight.detach().float().cpu()
    encoder_bias = sae.encoder.bias.detach().float().cpu()
    decoder_bias = sae.decoder.bias.detach().float().cpu()
    centered_hidden = hidden - decoder_bias

    concept_labels = {name: mask[targets] for name, mask in masks.items()}
    for feature in chosen:
        feature = int(feature)
        scores = latent[:, feature].float()
        active = torch.where(scores > 0)[0]
        if not active.numel():
            continue
        top_n = min(examples, active.numel())
        top_indices = active[torch.topk(scores[active], top_n).indices]
        preactivation = F.relu(centered_hidden @ encoder[feature] + encoder_bias[feature])
        suppressed = torch.where((preactivation > 0) & (scores <= 0))[0]
        hard_indices = torch.empty(0, dtype=torch.long)
        if suppressed.numel():
            hard_n = min(hard_count, suppressed.numel())
            hard_indices = suppressed[torch.topk(preactivation[suppressed], hard_n).indices]

        token_ids, token_counts = torch.unique(targets[top_indices], return_counts=True)
        token_entropy = normalized_entropy(token_counts)
        coherence = _context_coherence(hidden[top_indices])
        hard_coherence = _context_coherence(hidden[hard_indices]) if hard_indices.numel() else None
        best_gain, best_ap, best_prevalence, best_name = _best_prevalence_corrected_concept(
            scores, concept_labels,
        )
        top_token_purity = float(token_counts.max() / token_counts.sum())
        suppressed_ratio = float(suppressed.numel() / max(1, int((preactivation > 0).sum())))
        row = {
            "feature": feature,
            "frequency": float(frequency[feature]),
            "mean_top_activation": float(scores[top_indices].mean()),
            "top_token_entropy": token_entropy,
            "top_token_purity": top_token_purity,
            "context_coherence": coherence,
            "hard_negative_context_coherence": hard_coherence,
            "hard_negative_count": int(hard_indices.numel()),
            "topk_suppression_ratio": suppressed_ratio,
            "best_known_concept": best_name,
            "best_known_concept_auprc": best_ap,
            "best_known_concept_prevalence": best_prevalence,
            "best_known_concept_auprc_gain": best_gain,
            "decoder_norm": float(decoder[:, feature].norm()),
        }
        if hard_coherence is not None:
            row["hard_negative_coherence_margin"] = coherence - hard_coherence
        row["no_known_concept"] = best_gain is None or best_gain < ap_gain_threshold
        row["low_context_coherence"] = coherence < coherence_threshold
        margin = row.get("hard_negative_coherence_margin")
        row["hard_negative_separation_failure"] = None if margin is None else margin < hard_margin_threshold
        row["strict_no_explanation"] = (
            row["no_known_concept"]
            and row["low_context_coherence"]
            and row["hard_negative_separation_failure"] is not False
        )
        rows.append(row)

        if noexp_cfg.get("export_contexts", True):
            for kind, indices in (("top", top_indices), ("hard_negative", hard_indices)):
                for flat_index in indices:
                    flat_index = int(flat_index)
                    block_index, position = divmod(flat_index, sequence)
                    left = max(0, position - context_width)
                    right = min(sequence, position + context_width + 1)
                    context_rows.append({
                        "feature": feature,
                        "kind": kind,
                        "activation": float(scores[flat_index]),
                        "preactivation": float(preactivation[flat_index]),
                        "target_token": int(targets[flat_index]),
                        "context": decode_tokens(cache["inputs"][block_index, left:right].tolist()),
                    })

    no_known_concept = [row for row in rows if row["no_known_concept"]]
    strict_unexplained = [row for row in rows if row["strict_no_explanation"]]
    low_coherence = [row for row in rows if row["low_context_coherence"]]
    hard_negative_evaluable = [row for row in rows if row["hard_negative_separation_failure"] is not None]
    hard_negative_failures = [row for row in hard_negative_evaluable if row["hard_negative_separation_failure"]]
    summary = {
        "sampled_features": len(rows),
        # This primary automated rate is deliberately scoped to the declared
        # concept vocabulary. Open-vocabulary and human audits remain separate.
        "no_explanation_scope": "no_known_concept_above_prevalence_corrected_auprc_gain",
        "no_explanation_rate": len(no_known_concept) / max(1, len(rows)),
        "strict_no_explanation_rate": len(strict_unexplained) / max(1, len(rows)),
        "low_context_coherence_rate": len(low_coherence) / max(1, len(rows)),
        "hard_negative_separation_failure_rate": len(hard_negative_failures) / max(1, len(hard_negative_evaluable)),
        "known_concept_auprc_gain_threshold": ap_gain_threshold,
        "context_coherence_threshold": coherence_threshold,
        "hard_negative_margin_threshold": hard_margin_threshold,
        "feature_context_coherence_mean": _finite_mean([row["context_coherence"] for row in rows]),
        "feature_token_entropy_mean": _finite_mean([row["top_token_entropy"] for row in rows]),
        "known_concept_best_auprc_mean": _finite_mean([row.get("best_known_concept_auprc") for row in rows]),
        "known_concept_best_auprc_gain_mean": _finite_mean([row.get("best_known_concept_auprc_gain") for row in rows]),
        "hard_negative_coherence_margin_mean": _finite_mean([row.get("hard_negative_coherence_margin") for row in rows]),
        "topk_suppression_ratio_mean": _finite_mean([row["topk_suppression_ratio"] for row in rows]),
    }
    if context_rows:
        path = target_dir / "feature_contexts.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in context_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        human_cfg = cfg["interpretability"].get("human_audit", {})
        if human_cfg.get("enabled", True) and human_cfg.get("export_blinded_packets", True):
            grouped_contexts = {}
            for context_row in context_rows:
                grouped_contexts.setdefault(context_row["feature"], {}).setdefault(context_row["kind"], []).append(context_row["context"])
            audit_features = sorted(grouped_contexts)[:int(human_cfg.get("features_per_run", 100))]
            packet = []
            key = []
            for feature in audit_features:
                item_id = hashlib.sha256(f"{target_dir.resolve()}::{feature}".encode("utf-8")).hexdigest()[:16]
                examples_limit = int(human_cfg.get("examples_per_feature", 10))
                negatives_limit = int(human_cfg.get("hard_negatives_per_feature", 10))
                packet.append({
                    "item_id": item_id,
                    "high_activation_examples": grouped_contexts[feature].get("top", [])[:examples_limit],
                    "hard_negative_examples": grouped_contexts[feature].get("hard_negative", [])[:negatives_limit],
                    "rating_scale": human_cfg.get("rating_scale", [1, 2, 3, 4, 5]),
                    "questions": ["semantic_coherence", "hard_negative_separability", "monosemanticity"],
                })
                key.append({"item_id": item_id, "feature": feature})
            write_json(target_dir / "human_audit_packet.json", packet)
            write_json(target_dir / "human_audit_key.json", key)
            summary["human_audit_items_exported"] = len(packet)
    return summary, rows


def evaluate_human_annotations(cfg, target_dir):
    human_cfg = cfg["interpretability"].get("human_audit", {})
    annotation_file = human_cfg.get("annotation_file")
    key_path = target_dir / "human_audit_key.json"
    if not annotation_file or not Path(annotation_file).exists() or not key_path.exists():
        return {"human_annotations_available": False}
    text = Path(annotation_file).read_text(encoding="utf-8").strip()
    if not text:
        return {"human_annotations_available": False}
    annotations = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    valid_items = {row["item_id"] for row in json.loads(key_path.read_text(encoding="utf-8"))}
    annotations = [row for row in annotations if row.get("item_id") in valid_items]
    summary = {"human_annotations_available": bool(annotations), "human_annotation_rows": len(annotations)}
    for question in ("semantic_coherence", "hard_negative_separability", "monosemanticity"):
        grouped = {}
        for row in annotations:
            if isinstance(row.get(question), (int, float)):
                grouped.setdefault(row["item_id"], []).append(float(row[question]))
        values = [value for ratings in grouped.values() for value in ratings]
        summary[f"human_{question}_mean"] = _finite_mean(values)
        if human_cfg.get("report_inter_rater_agreement", True):
            summary[f"human_{question}_krippendorff_alpha"] = krippendorff_alpha_interval(list(grouped.values()))
    return summary


def _best_recovery_threshold(scores, labels, available, false_positive_budget):
    """Maximize newly recovered positives under an absolute FP budget.

    Threshold selection is performed only on validation data. Equal-score
    groups are kept intact so ties cannot silently violate the budget.
    """
    scores = torch.as_tensor(scores, dtype=torch.float64).flatten().cpu()
    labels = torch.as_tensor(labels, dtype=torch.bool).flatten().cpu()
    available = torch.as_tensor(available, dtype=torch.bool).flatten().cpu()
    indices = torch.where(available)[0]
    if not indices.numel():
        return None
    order = torch.argsort(scores[indices], descending=True, stable=True)
    sorted_indices = indices[order]
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]
    cumulative_tp = sorted_labels.to(torch.int64).cumsum(0)
    cumulative_fp = (~sorted_labels).to(torch.int64).cumsum(0)
    group_end = torch.ones(sorted_scores.numel(), dtype=torch.bool)
    if sorted_scores.numel() > 1:
        group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    endpoints = torch.where(group_end)[0]
    allowed = cumulative_fp[endpoints] <= max(0, int(false_positive_budget))
    useful = cumulative_tp[endpoints] > 0
    candidates = endpoints[allowed & useful]
    if not candidates.numel():
        return None
    tp = cumulative_tp[candidates]
    fp = cumulative_fp[candidates]
    # Prefer more recovery; break ties with fewer false positives and then the
    # stricter threshold (the first remaining candidate).
    best_tp = tp.max()
    best_candidates = candidates[tp == best_tp]
    best_fp = cumulative_fp[best_candidates].min()
    best_index = int(best_candidates[cumulative_fp[best_candidates] == best_fp][0])
    threshold = float(sorted_scores[best_index])
    prediction = available & (scores >= threshold)
    return {
        "threshold": threshold,
        "prediction": prediction,
        "true_positive_gain": int((prediction & labels).sum()),
        "false_positive_gain": int((prediction & ~labels).sum()),
    }


def evaluate_absorption(cfg, latent, targets, masks, splits, rankings):
    absorption_cfg = cfg["interpretability"]["absorption"]
    if not absorption_cfg.get("enabled", True):
        return {}, []
    target_flat = targets.reshape(-1).long()
    train_i, validation_i, test_i = splits["train"], splits["validation"], splits["test"]
    candidate_count = int(absorption_cfg.get("candidate_features", 20))
    max_selected = int(absorption_cfg.get("max_selected_candidates", 5))
    max_fpr_increase = float(absorption_cfg.get("max_validation_fpr_increase", 0.02))
    rows = []
    for concept in absorption_cfg.get("parent_concepts", []):
        if concept not in masks or concept not in rankings:
            continue
        labels = masks[concept][target_flat]
        parent_features = rankings[concept]["features"]
        parent_gaps = rankings[concept]["gaps"]
        positive = [(feature, gap) for feature, gap in zip(parent_features, parent_gaps) if gap > 0]
        if not positive:
            continue
        parent = int(positive[0][0])
        threshold = best_f1_threshold(latent[validation_i, parent], labels[validation_i])
        train_parent_pred = latent[train_i, parent] >= threshold
        validation_parent_pred = latent[validation_i, parent] >= threshold
        test_parent_pred = latent[test_i, parent] >= threshold
        train_absorbed_label = labels[train_i] & ~train_parent_pred
        validation_absorbed_label = labels[validation_i] & ~validation_parent_pred
        if int(train_absorbed_label.sum()) < 4 or int(validation_absorbed_label.sum()) < 2:
            continue
        candidate_indices, _ = select_top_features(latent[train_i], train_absorbed_label, candidate_count + 1)
        candidate_indices = [int(value) for value in candidate_indices if int(value) != parent][:candidate_count]
        if not candidate_indices:
            continue
        validation_labels = labels[validation_i]
        validation_combined = validation_parent_pred.clone()
        validation_negatives = int((~validation_labels).sum())
        parent_validation_fp = int((validation_parent_pred & ~validation_labels).sum())
        additional_fp_budget = int(math.floor(max_fpr_increase * validation_negatives))
        selected_candidates = []
        remaining = list(candidate_indices)
        for _ in range(min(max_selected, len(remaining))):
            current_fp = int((validation_combined & ~validation_labels).sum())
            remaining_fp_budget = parent_validation_fp + additional_fp_budget - current_fp
            available = ~validation_combined
            best = None
            for candidate in remaining:
                proposal = _best_recovery_threshold(
                    latent[validation_i, candidate], validation_labels, available, remaining_fp_budget,
                )
                if proposal is None:
                    continue
                key = (proposal["true_positive_gain"], -proposal["false_positive_gain"])
                if best is None or key > best[0]:
                    best = (key, candidate, proposal)
            if best is None:
                break
            _, candidate, proposal = best
            validation_combined |= proposal["prediction"]
            selected_candidates.append({
                "feature": int(candidate),
                "threshold": proposal["threshold"],
                "validation_true_positive_gain": proposal["true_positive_gain"],
                "validation_false_positive_gain": proposal["false_positive_gain"],
            })
            remaining.remove(candidate)

        candidate_pred = torch.zeros(test_i.numel(), dtype=torch.bool)
        candidate_details = []
        for detail in selected_candidates:
            candidate = detail["feature"]
            candidate_threshold = detail["threshold"]
            pred = latent[test_i, candidate] >= candidate_threshold
            candidate_pred |= pred
            candidate_details.append(detail)
        test_labels = labels[test_i]
        false_negative = test_labels & ~test_parent_pred
        recovered = false_negative & candidate_pred
        combined = test_parent_pred | candidate_pred
        parent_recall = float((test_parent_pred & test_labels).sum() / test_labels.sum().clamp_min(1))
        combined_recall = float((combined & test_labels).sum() / test_labels.sum().clamp_min(1))
        false_positive_rate = float((combined & ~test_labels).sum() / (~test_labels).sum().clamp_min(1))
        parent_false_positive_rate = float((test_parent_pred & ~test_labels).sum() / (~test_labels).sum().clamp_min(1))
        validation_parent_fpr = parent_validation_fp / max(1, validation_negatives)
        validation_combined_fpr = float(
            (validation_combined & ~validation_labels).sum() / (~validation_labels).sum().clamp_min(1)
        )
        rows.append({
            "concept": concept,
            "parent_feature": parent,
            "candidate_features": candidate_details,
            "candidate_pool_size": len(candidate_indices),
            "selected_candidate_count": len(candidate_details),
            "parent_recall": parent_recall,
            "combined_recall": combined_recall,
            "recall_gain": combined_recall - parent_recall,
            "false_negative_count": int(false_negative.sum()),
            "absorbed_fraction": float(recovered.sum() / false_negative.sum().clamp_min(1)),
            "parent_false_positive_rate": parent_false_positive_rate,
            "combined_false_positive_rate": false_positive_rate,
            "false_positive_rate_increase": false_positive_rate - parent_false_positive_rate,
            "validation_parent_false_positive_rate": validation_parent_fpr,
            "validation_combined_false_positive_rate": validation_combined_fpr,
            "validation_false_positive_rate_budget": max_fpr_increase,
        })
    return {
        "absorption_parent_count": len(rows),
        "absorption_rate_mean": _finite_mean([row["absorbed_fraction"] for row in rows]),
        "absorption_recall_gain_mean": _finite_mean([row["recall_gain"] for row in rows]),
        "absorption_parent_false_positive_rate_mean": _finite_mean([row["parent_false_positive_rate"] for row in rows]),
        "absorption_false_positive_rate_mean": _finite_mean([row["combined_false_positive_rate"] for row in rows]),
        "absorption_false_positive_rate_increase_mean": _finite_mean([row["false_positive_rate_increase"] for row in rows]),
        "absorption_selected_candidates_mean": _finite_mean([row["selected_candidate_count"] for row in rows]),
    }, rows


def _frequency_matched_random_features(frequency, selected, count, controls, seed):
    frequency = torch.as_tensor(frequency, dtype=torch.float32).clamp_min(1e-12)
    log_frequency = frequency.log10()
    excluded = {int(value) for value in selected}
    available = torch.tensor([index for index in range(frequency.numel()) if index not in excluded], dtype=torch.long)
    if not available.numel():
        return []
    generator = torch.Generator().manual_seed(int(seed))
    sets = []
    target = log_frequency[torch.as_tensor(selected, dtype=torch.long)]
    for _ in range(int(controls)):
        picked = []
        pool = available.clone()
        for value in target[:int(count)]:
            distance = (log_frequency[pool] - value).abs()
            nearest_count = min(32, pool.numel())
            nearest = torch.topk(distance, nearest_count, largest=False).indices
            choice = int(nearest[torch.randint(nearest_count, (1,), generator=generator)])
            picked.append(int(pool[choice]))
            pool = torch.cat((pool[:choice], pool[choice + 1:]))
            if not pool.numel():
                break
        if picked:
            sets.append(picked)
    return sets


def _selected_kl(original_logits, patched_logits, indices):
    if indices.numel() == 0:
        return float("nan")
    original = original_logits.reshape(-1, original_logits.size(-1))[indices].float()
    patched = patched_logits.reshape(-1, patched_logits.size(-1))[indices].float()
    probabilities = original.softmax(dim=-1)
    return float(F.kl_div(patched.log_softmax(dim=-1), probabilities, reduction="batchmean"))


@torch.no_grad()
def evaluate_causal(cfg, base, sae, cache, latent, masks, rankings, eligible, frequency, device, dtype):
    causal_cfg = cfg["interpretability"]["causal"]
    if not causal_cfg.get("enabled", True):
        return {}, []
    max_blocks = min(int(causal_cfg.get("max_batches", 8)), cache["blocks"])
    hidden = cache["hidden"][:max_blocks].to(device)
    targets = cache["targets"][:max_blocks].reshape(-1).long().to(device)
    z = latent[:max_blocks * cache["hidden"].size(1)].float().to(device)
    flat_hidden = hidden.reshape(-1, hidden.size(-1)).float()
    hook_layer = int(cache["hook_layer"])
    try:
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            original_logits = _continue_logits(base, hidden, hook_layer)
    except (NotImplementedError, RuntimeError):
        return {"causal_supported": False}, []

    budgets = [int(value) for value in causal_cfg.get("feature_budgets", [1, 5, 10])]
    base_strengths = [float(value) for value in causal_cfg.get("strengths", [1.0])]
    fade_cfg = cfg["interpretability"].get("fade", {})
    fade_strengths = [float(value) for value in fade_cfg.get("modification_factors", [])] if fade_cfg.get("enabled", True) else []
    max_concepts = int(causal_cfg.get("max_concepts", 24))
    max_examples = int(causal_cfg.get("max_examples_per_concept", 512))
    controls = int(causal_cfg.get("random_controls", 5))
    isolation_kl_floor = float(causal_cfg.get("isolation_kl_floor", 1e-4))
    decoder = sae.decoder.weight.detach().float()
    rows = []

    def intervention_metrics(patched_flat, vocab_mask, indices, direction):
        patched_hidden = patched_flat.to(hidden.dtype).view_as(hidden)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            patched_logits = _continue_logits(base, patched_hidden, hook_layer)
        original_odds = concept_log_odds(original_logits, vocab_mask.to(device)).reshape(-1)[indices]
        patched_odds = concept_log_odds(patched_logits, vocab_mask.to(device)).reshape(-1)[indices]
        if direction == "necessity":
            effect = original_odds - patched_odds
        else:
            effect = patched_odds - original_odds
        kl = _selected_kl(original_logits, patched_logits, indices)
        mask = vocab_mask.to(device)
        original_probability = original_logits.reshape(-1, original_logits.size(-1))[indices].float().softmax(dim=-1)[:, mask].sum(dim=-1).mean()
        patched_probability = patched_logits.reshape(-1, patched_logits.size(-1))[indices].float().softmax(dim=-1)[:, mask].sum(dim=-1).mean()
        return float(effect.mean()), kl, float(original_probability), float(patched_probability)

    for concept_index, concept in enumerate(eligible[:max_concepts]):
        ranking = rankings.get(concept, {})
        positive_features = [
            int(feature) for feature, gap in zip(ranking.get("features", []), ranking.get("gaps", [])) if gap > 0
        ]
        if not positive_features:
            continue
        vocab_mask = masks[concept]
        labels = vocab_mask.to(device)[targets]
        positive_indices = torch.where(labels)[0][:max_examples]
        negative_indices = torch.where(~labels)[0][:max_examples]
        if not positive_indices.numel() or not negative_indices.numel():
            continue
        for budget in budgets:
            selected = positive_features[:min(budget, len(positive_features))]
            if not selected:
                continue
            selected_tensor = torch.tensor(selected, device=device)
            positive_mean = z[positive_indices][:, selected_tensor].mean(dim=0)
            sufficiency_zero_flat = flat_hidden.clone()
            sufficiency_zero_flat[negative_indices] -= (
                z[negative_indices][:, selected_tensor] @ decoder[:, selected_tensor].T
            )
            _, sufficiency_zero_kl, _, sufficiency_zero_probability = intervention_metrics(
                sufficiency_zero_flat, vocab_mask, negative_indices, "sufficiency"
            )
            random_sets = _frequency_matched_random_features(
                frequency, selected, len(selected), controls,
                int(cfg["interpretability"].get("eval_seed", 314159)) + concept_index * 1009 + budget,
            ) if cfg["interpretability"].get("null_controls", {}).get("random_feature_ablation", True) else []

            strengths = sorted(set(base_strengths + (fade_strengths if len(selected) == 1 else [])))
            for strength in strengths:
                necessity_flat = flat_hidden.clone()
                contribution = z[positive_indices][:, selected_tensor] @ decoder[:, selected_tensor].T
                necessity_flat[positive_indices] -= strength * contribution
                necessity_effect, necessity_kl, necessity_base_probability, necessity_patched_probability = intervention_metrics(
                    necessity_flat, vocab_mask, positive_indices, "necessity"
                )

                sufficiency_flat = sufficiency_zero_flat.clone()
                steering = (positive_mean @ decoder[:, selected_tensor].T).view(1, -1)
                sufficiency_flat[negative_indices] += strength * steering
                sufficiency_effect, sufficiency_kl, sufficiency_base_probability, sufficiency_patched_probability = intervention_metrics(
                    sufficiency_flat, vocab_mask, negative_indices, "sufficiency"
                )

                random_necessity = []
                random_sufficiency = []
                random_necessity_kl = []
                random_sufficiency_kl = []
                for random_features in random_sets:
                    random_tensor = torch.tensor(random_features, device=device)
                    random_positive_mean = z[positive_indices][:, random_tensor].mean(dim=0)
                    random_nec_flat = flat_hidden.clone()
                    random_nec_flat[positive_indices] -= strength * (
                        z[positive_indices][:, random_tensor] @ decoder[:, random_tensor].T
                    )
                    effect, kl, _, _ = intervention_metrics(random_nec_flat, vocab_mask, positive_indices, "necessity")
                    random_necessity.append(effect)
                    random_necessity_kl.append(kl)

                    random_suf_flat = flat_hidden.clone()
                    random_steering = (random_positive_mean @ decoder[:, random_tensor].T).view(1, -1)
                    random_suf_flat[negative_indices] += strength * random_steering
                    effect, kl, _, _ = intervention_metrics(random_suf_flat, vocab_mask, negative_indices, "sufficiency")
                    random_sufficiency.append(effect)
                    random_sufficiency_kl.append(kl)

                random_nec_mean = _finite_mean(random_necessity) or 0.0
                random_suf_mean = _finite_mean(random_sufficiency) or 0.0
                necessity_specific = necessity_effect - random_nec_mean
                sufficiency_specific = sufficiency_effect - random_suf_mean
                rows.append({
                    "concept": concept,
                    "feature_budget": len(selected),
                    "strength": strength,
                    "features": selected,
                    "positive_examples": int(positive_indices.numel()),
                    "negative_examples": int(negative_indices.numel()),
                    "necessity_delta_log_odds": necessity_effect,
                    "necessity_base_concept_probability": necessity_base_probability,
                    "necessity_patched_concept_probability": necessity_patched_probability,
                    "necessity_kl": necessity_kl,
                    "necessity_random_delta_log_odds": random_nec_mean,
                    "necessity_random_kl": _finite_mean(random_necessity_kl),
                    "necessity_control_adjusted": necessity_specific,
                    "necessity_isolation": necessity_specific / math.sqrt(max(isolation_kl_floor, necessity_kl)),
                    "sufficiency_delta_log_odds": sufficiency_effect,
                    "sufficiency_base_concept_probability": sufficiency_base_probability,
                    "sufficiency_zero_concept_probability": sufficiency_zero_probability,
                    "sufficiency_zero_kl": sufficiency_zero_kl,
                    "sufficiency_patched_concept_probability": sufficiency_patched_probability,
                    "sufficiency_kl": sufficiency_kl,
                    "sufficiency_random_delta_log_odds": random_suf_mean,
                    "sufficiency_random_kl": _finite_mean(random_sufficiency_kl),
                    "sufficiency_control_adjusted": sufficiency_specific,
                    "sufficiency_isolation": sufficiency_specific / math.sqrt(max(isolation_kl_floor, sufficiency_kl)),
                })
    primary_candidates = [row for row in rows if row["strength"] == 1.0]
    # A concept with fewer positive features can produce identical selected
    # sets for several requested budgets. Count that intervention only once.
    primary = []
    seen = set()
    for row in primary_candidates:
        key = (row["concept"], tuple(row["features"]), row["strength"])
        if key not in seen:
            seen.add(key)
            primary.append(row)
    if not primary:
        primary = rows
    necessity_effects = [row["necessity_control_adjusted"] for row in primary]
    sufficiency_effects = [row["sufficiency_control_adjusted"] for row in primary]
    necessity_kls = [row["necessity_kl"] for row in primary]
    sufficiency_kls = [row["sufficiency_kl"] for row in primary]
    return {
        "causal_supported": True,
        "causal_conditions": len(rows),
        "causal_primary_conditions": len(primary),
        "causal_isolation_aggregation": "ratio_of_means_over_deduplicated_strength_1_conditions",
        "causal_isolation_kl_floor": isolation_kl_floor,
        "causal_necessity_control_adjusted": _finite_mean(necessity_effects),
        "causal_sufficiency_control_adjusted": _finite_mean(sufficiency_effects),
        "causal_necessity_isolation": _isolation_ratio_of_means(
            necessity_effects, necessity_kls, isolation_kl_floor,
        ),
        "causal_sufficiency_isolation": _isolation_ratio_of_means(
            sufficiency_effects, sufficiency_kls, isolation_kl_floor,
        ),
        "causal_necessity_isolation_condition_mean": _finite_mean([row["necessity_isolation"] for row in primary]),
        "causal_sufficiency_isolation_condition_mean": _finite_mean([row["sufficiency_isolation"] for row in primary]),
        "causal_necessity_isolation_condition_median": _finite_median([row["necessity_isolation"] for row in primary]),
        "causal_sufficiency_isolation_condition_median": _finite_median([row["sufficiency_isolation"] for row in primary]),
        "causal_necessity_kl_below_floor_rate": sum(value < isolation_kl_floor for value in necessity_kls) / max(1, len(necessity_kls)),
        "causal_sufficiency_kl_below_floor_rate": sum(value < isolation_kl_floor for value in sufficiency_kls) / max(1, len(sufficiency_kls)),
        "causal_necessity_kl": _finite_mean(necessity_kls),
        "causal_sufficiency_kl": _finite_mean(sufficiency_kls),
    }, rows


def evaluate_one(cfg, alias, model_seed, spec, gpu_index):
    if cfg["run"].get("num_threads"):
        torch.set_num_threads(int(cfg["run"]["num_threads"]))
    device = torch.device(cuda_device(gpu_index)) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = torch.float16 if cfg["run"].get("dtype") == "float16" else torch.bfloat16
    target_dir = interpret_dir(cfg, alias, model_seed, spec)
    ensure_dir(target_dir)
    summary_path = interpret_summary_path(cfg, alias, model_seed, spec)
    if not cfg["interpretability"].get("recompute", False) and summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("interpretability_metrics_version") == INTERPRETABILITY_METRICS_VERSION:
            event_line("interpret", gpu_label(gpu_index), alias, model_seed, "metrics exist", spec["sae_id"])
            return existing
        event_line("interpret", gpu_label(gpu_index), alias, model_seed, "metrics stale", spec["sae_id"])

    base, _ = load_base(cfg, alias, model_seed, device)
    valid_data = BlockData(
        block_dir(cfg, alias) / "valid.bin", cfg["data"]["valid_blocks"], cfg["data"]["block_size"]
    )
    cache = collect_activation_cache(base, valid_data, cfg, spec, device, dtype)
    sae, _ = load_sae(cfg, alias, model_seed, spec, hidden_size(base, cfg), device)
    latent = encode_cache(sae, cache["hidden"], cfg["interpretability"].get("encode_batch_size", 4096), device)
    stats_path = sae_dir(cfg, alias, model_seed, spec) / "feature_stats.pt"
    if stats_path.exists():
        feature_stats = _safe_torch_load(stats_path)
        frequency = feature_stats["frequency"].float()
    else:
        frequency = (latent > 0).float().mean(dim=0)
    vocab_size = int(getattr(base, "vocab_size", cfg["model"]["vocab_size"]))
    masks, concept_metadata, _, decode_tokens = build_concept_masks(cfg, alias, vocab_size)
    splits = block_split_indices(cache, cfg)

    concept_summary, probe_rows, feature_rows, rankings, eligible = evaluate_concepts(
        cfg, latent, cache["hidden"], cache["targets"], masks, splits
    ) if cfg["interpretability"]["concepts"].get("enabled", True) else ({}, [], [], {}, [])
    null_summary = {}
    null_rows = []
    null_cfg = cfg["interpretability"].get("null_controls", {})
    if null_cfg.get("untrained_sae", True) and eligible:
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        torch.manual_seed(int(null_cfg.get("null_seed", 8675309)) + int(spec["sae_seed"]))
        untrained_sae = TopKSAE(
            cache["hidden"].size(-1), cache["hidden"].size(-1) * int(spec["expansion"]), int(spec["k"]),
            tied_init=True, normalize_decoder=True,
        ).to(device).eval()
        untrained_latent = encode_cache(
            untrained_sae, cache["hidden"], cfg["interpretability"].get("encode_batch_size", 4096), device
        )
        metrics, rows = evaluate_null_latents(
            cfg, untrained_latent, cache["targets"], masks, splits, eligible, "untrained_sae"
        )
        null_summary.update(metrics)
        null_rows.extend(rows)
        del untrained_sae, untrained_latent
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)
    if null_cfg.get("random_transformer", True) and eligible:
        random_latent = random_transformer_latents(cfg, alias, model_seed, spec, cache, device, dtype)
        if random_latent is not None:
            metrics, rows = evaluate_null_latents(
                cfg, random_latent, cache["targets"], masks, splits, eligible, "random_transformer_untrained_sae"
            )
            null_summary.update(metrics)
            null_rows.extend(rows)
            del random_latent
    if concept_summary.get("concept_macro_auprc") is not None:
        for key in ("untrained_sae_macro_auprc", "random_transformer_untrained_sae_macro_auprc"):
            if null_summary.get(key) is not None:
                null_summary[f"trained_sae_auprc_above_{key.removesuffix('_macro_auprc')}"] = concept_summary["concept_macro_auprc"] - null_summary[key]
    fade_summary, fade_rows = evaluate_fade_activation(
        cfg, latent, cache["targets"], masks, splits, rankings, eligible
    )
    noexp_summary, noexp_rows = evaluate_no_explanation(
        cfg, sae, latent, cache, frequency, masks, decode_tokens, target_dir
    )
    human_summary = evaluate_human_annotations(cfg, target_dir)
    absorption_summary, absorption_rows = evaluate_absorption(
        cfg, latent, cache["targets"], masks, splits, rankings
    )
    causal_summary, causal_rows = evaluate_causal(
        cfg, base, sae, cache, latent, masks, rankings, eligible, frequency, device, dtype
    )
    fade_summary, fade_rows = attach_fade_faithfulness(fade_summary, fade_rows, causal_rows)
    summary = {
        "model": alias,
        "model_seed": model_seed,
        "interpretability_metrics_version": INTERPRETABILITY_METRICS_VERSION,
        **spec,
        "evaluation_tokens": cache["tokens"],
        "evaluation_blocks": cache["blocks"],
        **concept_summary,
        **null_summary,
        **fade_summary,
        **noexp_summary,
        **human_summary,
        **absorption_summary,
        **causal_summary,
    }
    write_json(target_dir / "concept_metadata.json", concept_metadata)
    write_json(target_dir / "concept_probes.json", probe_rows)
    write_json(target_dir / "concept_features.json", feature_rows)
    write_json(target_dir / "feature_rankings.json", rankings)
    write_json(target_dir / "null_controls.json", null_rows)
    write_json(target_dir / "fade.json", fade_rows)
    write_json(target_dir / "no_explanation.json", noexp_rows)
    write_json(target_dir / "absorption.json", absorption_rows)
    write_json(target_dir / "causal.json", causal_rows)
    write_json(target_dir / "metrics.json", summary)
    write_json(summary_path, summary)
    return summary


def _selected_active_features(cfg, alias, model_seed, spec, limit):
    stats_path = sae_dir(cfg, alias, model_seed, spec) / "feature_stats.pt"
    if not stats_path.exists():
        return None
    stats = _safe_torch_load(stats_path)
    frequency = stats["frequency"].float()
    active = torch.where(frequency > 0)[0]
    if active.numel() > int(limit):
        active = active[torch.topk(frequency[active], int(limit)).indices]
    return active


def _pairwise_maximum_correlation(first, second, chunk_size=256):
    first = first.float()
    second = second.float()
    first = first - first.mean(dim=0, keepdim=True)
    second = second - second.mean(dim=0, keepdim=True)
    first = first / first.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
    second = second / second.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
    denominator = max(1, first.size(0))
    row_max = []
    column_max = torch.zeros(second.size(1), device=second.device)
    for start in range(0, first.size(1), int(chunk_size)):
        correlations = (first[:, start:start + int(chunk_size)].T @ second / denominator).abs()
        row_max.append(correlations.max(dim=1).values.cpu())
        column_max = torch.maximum(column_max, correlations.max(dim=0).values)
    forward = torch.cat(row_max).mean().item() if row_max else float("nan")
    backward = column_max.mean().item() if column_max.numel() else float("nan")
    return 0.5 * (forward + backward), forward, backward


def _hungarian_decoder_metrics(first_decoder, second_decoder, thresholds):
    first = F.normalize(first_decoder.float(), dim=-1)
    second = F.normalize(second_decoder.float(), dim=-1)
    similarities = (first @ second.T).abs().cpu()
    try:
        from scipy.optimize import linear_sum_assignment

        first_indices, second_indices = linear_sum_assignment(-similarities.numpy())
        first_indices = torch.from_numpy(first_indices).long()
        second_indices = torch.from_numpy(second_indices).long()
    except (ImportError, ValueError):
        # Deterministic greedy fallback keeps the evaluation usable in minimal
        # environments; scipy remains in requirements for the exact result.
        candidate = similarities.flatten().argsort(descending=True)
        used_first = set()
        used_second = set()
        pairs = []
        for flat_index in candidate:
            row = int(flat_index // similarities.size(1))
            col = int(flat_index % similarities.size(1))
            if row not in used_first and col not in used_second:
                pairs.append((row, col))
                used_first.add(row)
                used_second.add(col)
                if len(pairs) == min(similarities.shape):
                    break
        first_indices = torch.tensor([row for row, _ in pairs], dtype=torch.long)
        second_indices = torch.tensor([col for _, col in pairs], dtype=torch.long)
    matched = similarities[first_indices, second_indices]
    metrics = {
        "decoder_hungarian_cosine_mean": float(matched.mean()) if matched.numel() else None,
        "decoder_hungarian_cosine_median": float(matched.median()) if matched.numel() else None,
        "matched_features": int(matched.numel()),
    }
    for threshold in thresholds:
        metrics[f"decoder_share_at_{str(threshold).replace('.', '_')}"] = float((matched >= float(threshold)).float().mean()) if matched.numel() else None
    return metrics, first_indices, second_indices


def stability_path(cfg, alias, model_seed, group_spec):
    group_id = f"layer{group_spec['layer']}_e{group_spec['expansion']}x_k{group_spec['k']}"
    return output_dir(cfg) / "metrics" / f"[{alias}]mseed{model_seed}_{group_id}_stability.json"


@torch.no_grad()
def evaluate_stability_group(cfg, alias, model_seed, group_specs, gpu_index):
    stability_cfg = cfg["interpretability"]["stability"]
    representative = group_specs[0]
    path = stability_path(cfg, alias, model_seed, representative)
    if not cfg["interpretability"].get("recompute", False) and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    device = torch.device(cuda_device(gpu_index)) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = torch.float16 if cfg["run"].get("dtype") == "float16" else torch.bfloat16
    base, _ = load_base(cfg, alias, model_seed, device)
    valid_data = BlockData(
        block_dir(cfg, alias) / "valid.bin", cfg["data"]["valid_blocks"], cfg["data"]["block_size"]
    )
    local_cfg = dict(cfg)
    local_cfg["interpretability"] = dict(cfg["interpretability"])
    local_cfg["interpretability"]["max_tokens"] = min(
        int(stability_cfg.get("activation_tokens", 8192)), int(cfg["interpretability"].get("max_tokens", 32768))
    )
    cache = collect_activation_cache(base, valid_data, local_cfg, representative, device, dtype)
    hidden = cache["hidden"]
    token_limit = min(int(stability_cfg.get("activation_tokens", 8192)), cache["tokens"])
    limit = int(stability_cfg.get("max_features", 8192))
    thresholds = stability_cfg.get("similarity_thresholds", [0.7, 0.8, 0.9])
    loaded = {}
    for spec in group_specs:
        sae, _ = load_sae(cfg, alias, model_seed, spec, hidden_size(base, cfg), device)
        selected = _selected_active_features(cfg, alias, model_seed, spec, limit)
        if selected is None:
            all_latent = encode_cache(sae, hidden, cfg["interpretability"].get("encode_batch_size", 4096), device)
            frequency = (all_latent > 0).float().mean(dim=0)
            selected = torch.where(frequency > 0)[0]
            if selected.numel() > limit:
                selected = selected[torch.topk(frequency[selected], limit).indices]
        flat = hidden.reshape(-1, hidden.size(-1))[:token_limit]
        activation_rows = []
        for start in range(0, flat.size(0), int(cfg["interpretability"].get("encode_batch_size", 4096))):
            activation_rows.append(sae.encode(flat[start:start + int(cfg["interpretability"].get("encode_batch_size", 4096))].to(device))[:, selected.to(device)].float().cpu())
        activations = torch.cat(activation_rows, dim=0)
        decoder = sae.decoder.weight.detach()[:, selected.to(device)].T.float().cpu()
        loaded[int(spec["sae_seed"])] = {"activation": activations, "decoder": decoder, "selected": selected}
        del sae

    pair_rows = []
    seeds = sorted(loaded)
    for first_seed, second_seed in combinations(seeds, 2):
        first = loaded[first_seed]
        second = loaded[second_seed]
        decoder_metrics, first_match, second_match = _hungarian_decoder_metrics(
            first["decoder"], second["decoder"], thresholds
        )
        first_activation = first["activation"].to(device)
        second_activation = second["activation"].to(device)
        pw_mcc, forward_mcc, backward_mcc = _pairwise_maximum_correlation(
            first_activation, second_activation, int(stability_cfg.get("correlation_chunk_size", 256))
        )
        matched_first = first_activation[:, first_match.to(device)]
        matched_second = second_activation[:, second_match.to(device)]
        matched_first = matched_first - matched_first.mean(dim=0, keepdim=True)
        matched_second = matched_second - matched_second.mean(dim=0, keepdim=True)
        numerator = (matched_first * matched_second).mean(dim=0)
        denominator = matched_first.square().mean(dim=0).sqrt() * matched_second.square().mean(dim=0).sqrt()
        matched_correlation = (numerator / denominator.clamp_min(1e-6)).abs().mean().item()
        pair_rows.append({
            "first_sae_seed": first_seed,
            "second_sae_seed": second_seed,
            "first_features": int(first["activation"].size(1)),
            "second_features": int(second["activation"].size(1)),
            "activation_tokens": int(token_limit),
            "stability_pw_mcc": pw_mcc,
            "pw_mcc_forward": forward_mcc,
            "pw_mcc_backward": backward_mcc,
            "decoder_matched_activation_correlation": matched_correlation,
            **decoder_metrics,
        })
    summary = {
        "model": alias,
        "model_seed": model_seed,
        "layer": representative["layer"],
        "expansion": representative["expansion"],
        "k": representative["k"],
        "sae_seeds": seeds,
        "seed_pairs": len(pair_rows),
        "stability_pw_mcc": _finite_mean([row["stability_pw_mcc"] for row in pair_rows]),
        "stability_decoder_hungarian_cosine": _finite_mean([row["decoder_hungarian_cosine_mean"] for row in pair_rows]),
        "stability_matched_activation_correlation": _finite_mean([row["decoder_matched_activation_correlation"] for row in pair_rows]),
        "pairs": pair_rows,
    }
    for threshold in thresholds:
        key = f"decoder_share_at_{str(threshold).replace('.', '_')}"
        summary[f"stability_{key}"] = _finite_mean([row.get(key) for row in pair_rows])
    write_json(path, summary)
    return summary


def _stability_groups(specs):
    groups = {}
    for spec in specs:
        key = (spec["layer"], spec["expansion"], spec["k"])
        groups.setdefault(key, []).append(spec)
    return [values for values in groups.values() if len(values) >= 2]


def selected_interpret_specs(cfg):
    specs = sae_specs(cfg)
    selection = cfg["interpretability"].get("sae_selection", {"mode": "all"})
    mode = selection.get("mode", "all")
    if mode == "all":
        return specs
    if mode == "primary_grid":
        layers = {int(cfg["eval"].get("primary_layer", cfg["model"].get("n_layer", -1)))}
        expansions = {int(cfg["eval"].get("primary_expansion", cfg["sae"].get("expansion", 8)))}
    elif mode == "custom":
        layers = {int(value) for value in selection.get("layers", [])}
        expansions = {int(value) for value in selection.get("expansions", [])}
    else:
        raise ValueError("interpretability.sae_selection.mode must be all, primary_grid, or custom")
    k_values = {int(value) for value in selection.get("k_values", [])}
    sae_seed_values = {int(value) for value in selection.get("sae_seeds", [])}
    return [
        spec for spec in specs
        if (not layers or spec["layer"] in layers)
        and (not expansions or spec["expansion"] in expansions)
        and (not k_values or spec["k"] in k_values)
        and (not sae_seed_values or spec["sae_seed"] in sae_seed_values)
    ]


def experiment_counts(cfg, aliases, model_seeds, specs):
    model_runs = len(aliases) * len(model_seeds)
    all_sae_specs = sae_specs(cfg)
    sae_runs = model_runs * len(all_sae_specs)
    interpret_runs = model_runs * len(specs)
    stability_groups = model_runs * len(_stability_groups(specs))
    sae_tokens_per_run = (
        int(cfg["sae"]["max_steps"])
        * int(cfg["train"]["batch_size"])
        * int(cfg["sae"].get("gradient_accumulation_steps", 1))
        * int(cfg["data"]["block_size"])
    )
    train_tokens_per_run = (
        int(cfg["train"]["max_steps"])
        * int(cfg["train"]["batch_size"])
        * int(cfg["train"].get("gradient_accumulation_steps", 1))
        * int(cfg["data"]["block_size"])
    )
    concept_count_upper = 14 + (26 if cfg["interpretability"]["concepts"].get("include_first_letter", True) else 0)
    concept_count_upper += len(cfg["interpretability"]["concepts"].get("keyword_concepts", {}))
    budget_values = [int(value) for value in cfg["interpretability"]["causal"].get("feature_budgets", [])]
    budgets = len(budget_values)
    strength_values = [float(value) for value in cfg["interpretability"]["causal"].get("strengths", [])]
    base_strength_count = len(set(strength_values))
    fade_extra_count = 0
    if cfg["interpretability"].get("fade", {}).get("enabled", True):
        fade_values = [float(value) for value in cfg["interpretability"]["fade"].get("modification_factors", [])]
        fade_extra_count = len(set(fade_values) - set(strength_values)) if 1 in budget_values else 0
    max_concepts = min(concept_count_upper, int(cfg["interpretability"]["causal"].get("max_concepts", 24)))
    controls = int(cfg["interpretability"]["causal"].get("random_controls", 5))
    causal_conditions = max_concepts * (budgets * base_strength_count + fade_extra_count)
    causal_forwards_per_sae = 1 + causal_conditions * (2 + 2 * controls) + max_concepts * budgets
    causal_tokens_per_forward = int(cfg["interpretability"]["causal"].get("max_batches", 8)) * int(cfg["data"]["block_size"])
    sae_seed_count = len(set(spec["sae_seed"] for spec in specs))
    storage = {}
    if not is_pretrain_mode(cfg):
        parameter_counts = {alias: sum(parameter.numel() for parameter in build_model(cfg, alias).parameters()) for alias in aliases}
        checkpoints_per_run = int(cfg["train"]["max_steps"]) // int(cfg["train"]["save_interval"]) + 2
        model_checkpoint_bytes = sum(parameter_counts.values()) * len(model_seeds) * checkpoints_per_run * 12
        d_in = int(cfg["model"]["d_model"])
        sae_parameter_total = 0
        for spec in all_sae_specs:
            d_sae = d_in * int(spec["expansion"])
            sae_parameter_total += 2 * d_in * d_sae + d_sae + d_in
        sae_checkpoint_bytes = sae_parameter_total * model_runs * 4
        storage = {
            "model_parameter_count_min": min(parameter_counts.values()),
            "model_parameter_count_max": max(parameter_counts.values()),
            "model_checkpoints_per_run_upper_bound": checkpoints_per_run,
            "estimated_model_checkpoint_storage_gib": model_checkpoint_bytes / (1024 ** 3),
            "estimated_sae_checkpoint_storage_gib": sae_checkpoint_bytes / (1024 ** 3),
        }
    return {
        "model_variants": len(aliases),
        "model_seeds": len(model_seeds),
        "base_model_training_runs": model_runs,
        "sae_specs_per_checkpoint": len(all_sae_specs),
        "interpret_specs_per_checkpoint": len(specs),
        "sae_training_runs": sae_runs,
        "interpretability_runs": interpret_runs,
        "stability_groups": stability_groups,
        "stability_seed_pair_comparisons": stability_groups * (math.comb(sae_seed_count, 2) if sae_seed_count >= 2 else 0),
        "train_tokens_per_model_run": train_tokens_per_run,
        "total_base_training_tokens": model_runs * train_tokens_per_run,
        "sae_activation_tokens_per_run": sae_tokens_per_run,
        "total_sae_training_activation_tokens": sae_runs * sae_tokens_per_run,
        "causal_conditions_per_sae_upper_bound": causal_conditions,
        "causal_forward_passes_per_sae_upper_bound": causal_forwards_per_sae,
        "total_causal_forward_passes_upper_bound": interpret_runs * causal_forwards_per_sae,
        "total_causal_token_forwards_upper_bound": interpret_runs * causal_forwards_per_sae * causal_tokens_per_forward,
        **storage,
    }


def run(cfg):
    stage_title("interpret")
    if not cfg.get("interpretability", {}).get("enabled", True):
        print("[interpret] disabled")
        return
    aliases = active_aliases(cfg)
    validate_aliases(cfg, aliases)
    model_seeds = run_seeds(cfg)
    specs = selected_interpret_specs(cfg)
    jobs = [(alias, model_seed, spec) for model_seed in model_seeds for alias in aliases for spec in specs]
    gpus = configured_gpus(cfg)
    n_gpu = len(gpus)
    failed = run_gpu_jobs(
        cfg,
        jobs,
        evaluate_one,
        lambda job, gpu_index: (cfg, job[0], job[1], job[2], gpu_index),
        gpus,
        stage="interpret",
    )
    if failed:
        raise RuntimeError(f"interpret subprocess failed: {failed}")
    rows = [json.loads(interpret_summary_path(cfg, alias, model_seed, spec).read_text(encoding="utf-8")) for alias, model_seed, spec in jobs]

    stability_rows = []
    if cfg["interpretability"]["stability"].get("enabled", True):
        stability_jobs = [
            (alias, model_seed, group)
            for model_seed in model_seeds for alias in aliases for group in _stability_groups(specs)
        ]
        max_parallel = min(max(1, int(cfg["interpretability"]["stability"].get("max_parallel_jobs", 1))), max(1, n_gpu))
        stability_scheduler_cfg = dict(cfg)
        stability_scheduler_cfg["run"] = dict(cfg["run"])
        stability_scheduler_cfg["run"]["jobs_per_gpu"] = 1
        failed = run_gpu_jobs(
            stability_scheduler_cfg,
            stability_jobs,
            evaluate_stability_group,
            lambda job, gpu_index: (cfg, job[0], job[1], job[2], gpu_index),
            gpus[:max_parallel],
            stage="stability",
        )
        if failed:
            raise RuntimeError(f"stability subprocess failed: {failed}")
        stability_rows = [
            json.loads(stability_path(cfg, alias, model_seed, group[0]).read_text(encoding="utf-8"))
            for alias, model_seed, group in stability_jobs
        ]

    metrics_dir = output_dir(cfg) / "metrics"
    write_json(metrics_dir / "interpret_summary.json", {"models": rows, "stability": stability_rows})
    write_json(metrics_dir / "compute_plan.json", experiment_counts(cfg, aliases, model_seeds, specs))
    save_config(cfg)
    print(f"[interpret] done | runs={len(rows)} | stability_groups={len(stability_rows)}", flush=True)
