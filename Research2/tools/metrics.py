import math

import numpy as np
import torch
import torch.nn.functional as F


def _binary_tensors(scores, labels):
    scores = torch.as_tensor(scores, dtype=torch.float64).flatten().cpu()
    labels = torch.as_tensor(labels, dtype=torch.bool).flatten().cpu()
    if scores.numel() != labels.numel():
        raise ValueError("scores and labels must have equal length")
    return scores, labels


def average_precision(scores, labels):
    scores, labels = _binary_tensors(scores, labels)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = labels[order].to(torch.float64)
    precision = ranked.cumsum(0) / torch.arange(1, ranked.numel() + 1, dtype=torch.float64)
    return float(precision[ranked.bool()].mean())


def roc_auc(scores, labels):
    scores, labels = _binary_tensors(scores, labels)
    positives = int(labels.sum())
    negatives = labels.numel() - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    ranks = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    start = 0
    while start < sorted_scores.numel():
        end = start + 1
        while end < sorted_scores.numel() and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = ranks[start:end].mean()
        start = end
    original_ranks = torch.empty_like(ranks)
    original_ranks[order] = ranks
    positive_rank_sum = original_ranks[labels].sum()
    auc = (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return float(auc)


def best_f1_threshold(scores, labels):
    scores, labels = _binary_tensors(scores, labels)
    positives = int(labels.sum())
    if positives == 0:
        return float("inf")
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order].to(torch.float64)
    tp = sorted_labels.cumsum(0)
    fp = torch.arange(1, labels.numel() + 1, dtype=torch.float64) - tp
    fn = positives - tp
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1e-12)
    index = int(torch.argmax(f1))
    return float(sorted_scores[index])


def binary_metrics(scores, labels, threshold=None):
    scores, labels = _binary_tensors(scores, labels)
    if threshold is None:
        threshold = best_f1_threshold(scores, labels)
    pred = scores >= threshold
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    tn = int((~pred & ~labels).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "auprc": average_precision(scores, labels),
        "auroc": roc_auc(scores, labels),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "threshold": float(threshold),
        "positives": int(labels.sum()),
        "negatives": int((~labels).sum()),
    }


def standardized_mean_gap(features, labels):
    features = torch.as_tensor(features, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.bool, device=features.device)
    if not bool(labels.any()) or not bool((~labels).any()):
        return features.new_zeros(features.size(-1))
    positive = features[labels]
    negative = features[~labels]
    gap = positive.mean(dim=0) - negative.mean(dim=0)
    pooled = (0.5 * (positive.var(dim=0, unbiased=False) + negative.var(dim=0, unbiased=False))).sqrt().clamp_min(1e-6)
    return gap / pooled


def select_top_features(features, labels, count):
    gap = standardized_mean_gap(features, labels)
    count = min(max(1, int(count)), gap.numel())
    indices = torch.topk(gap.abs(), count).indices
    return indices, gap[indices]


def _standardize(train, validation, test):
    mean = train.mean(dim=0, keepdim=True)
    scale = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    return (train - mean) / scale, (validation - mean) / scale, (test - mean) / scale


def fit_logistic_probe(train_x, train_y, validation_x, validation_y, test_x, test_y, regularization, steps, lr, patience=25):
    train_x = torch.as_tensor(train_x, dtype=torch.float32)
    validation_x = torch.as_tensor(validation_x, dtype=torch.float32)
    test_x = torch.as_tensor(test_x, dtype=torch.float32)
    train_y = torch.as_tensor(train_y, dtype=torch.float32).flatten()
    validation_y = torch.as_tensor(validation_y, dtype=torch.bool).flatten()
    test_y = torch.as_tensor(test_y, dtype=torch.bool).flatten()
    train_x, validation_x, test_x = _standardize(train_x, validation_x, test_x)

    best = None
    positive_weight = ((~train_y.bool()).sum() / train_y.sum().clamp_min(1)).clamp(0.1, 10.0)
    for penalty in regularization:
        weight = torch.zeros(train_x.size(-1), dtype=torch.float32, requires_grad=True)
        bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([weight, bias], lr=float(lr))
        best_state = None
        best_ap = -float("inf")
        stale = 0
        for step in range(int(steps)):
            logits = train_x @ weight + bias
            loss = F.binary_cross_entropy_with_logits(logits, train_y, pos_weight=positive_weight)
            loss = loss + float(penalty) * weight.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 5 == 0 or step + 1 == int(steps):
                with torch.no_grad():
                    val_scores = validation_x @ weight + bias
                    val_ap = average_precision(val_scores, validation_y)
                if val_ap > best_ap + 1e-6:
                    best_ap = val_ap
                    best_state = (weight.detach().clone(), bias.detach().clone())
                    stale = 0
                else:
                    stale += 5
                if stale >= int(patience):
                    break
        if best_state is None:
            best_state = (weight.detach(), bias.detach())
        candidate = (best_ap, float(penalty), best_state)
        if best is None or candidate[0] > best[0]:
            best = candidate

    _, penalty, (weight, bias) = best
    with torch.no_grad():
        validation_scores = validation_x @ weight + bias
        threshold = best_f1_threshold(validation_scores, validation_y)
        test_scores = test_x @ weight + bias
    metrics = binary_metrics(test_scores, test_y, threshold=threshold)
    metrics["regularization"] = penalty
    metrics["weight_l0"] = int((weight.abs() > 1e-6).sum())
    return metrics, {"weight": weight, "bias": bias, "threshold": threshold}


def mean_confidence_interval(values, confidence=0.95):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"mean": None, "std": None, "ci_low": None, "ci_high": None, "n": 0}
    tensor = torch.tensor(values, dtype=torch.float64)
    mean = tensor.mean().item()
    std = tensor.std(unbiased=True).item() if tensor.numel() > 1 else 0.0
    z = 1.959963984540054 if abs(float(confidence) - 0.95) < 1e-6 else 1.959963984540054
    half = z * std / math.sqrt(max(1, tensor.numel()))
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half, "n": int(tensor.numel())}


def bootstrap_mean_ci(values, repeats=2000, confidence=0.95, seed=0):
    values = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(values, size=(int(repeats), values.size), replace=True).mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
        "n": int(values.size),
    }


def hierarchical_bootstrap_mean(rows, value_key, cluster_key, subcluster_key=None, repeats=2000, confidence=0.95, seed=0):
    """Bootstrap a mean without treating repeated SAE seeds as independent."""
    grouped = {}
    for row in rows:
        value = row.get(value_key)
        if value is None or not math.isfinite(float(value)):
            continue
        grouped.setdefault(row.get(cluster_key), []).append(row)
    clusters = list(grouped)
    if not clusters:
        return {"mean": None, "ci_low": None, "ci_high": None, "clusters": 0, "observations": 0}
    cluster_means = [np.mean([float(row[value_key]) for row in grouped[cluster]]) for cluster in clusters]
    rng = np.random.default_rng(int(seed))
    draws = []
    for _ in range(int(repeats)):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        values = []
        for cluster in sampled_clusters:
            cluster_rows = grouped[cluster]
            if subcluster_key is None:
                sampled = rng.choice(len(cluster_rows), size=len(cluster_rows), replace=True)
                values.extend(float(cluster_rows[index][value_key]) for index in sampled)
            else:
                subgroups = {}
                for row in cluster_rows:
                    subgroups.setdefault(row.get(subcluster_key), []).append(float(row[value_key]))
                subkeys = list(subgroups)
                sampled_subkeys = rng.choice(subkeys, size=len(subkeys), replace=True)
                values.extend(float(rng.choice(subgroups[subkey])) for subkey in sampled_subkeys)
        draws.append(float(np.mean(values)))
    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mean": float(np.mean(cluster_means)),
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha)),
        "clusters": len(clusters),
        "observations": sum(len(value) for value in grouped.values()),
    }


def paired_sign_flip_test(values, repeats=10000, seed=0):
    """Two-sided paired randomization test on already-computed differences."""
    values = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=np.float64)
    if values.size == 0:
        return None
    observed = abs(values.mean())
    if values.size <= 20:
        patterns = np.arange(2 ** values.size, dtype=np.uint64)[:, None]
        bits = ((patterns >> np.arange(values.size, dtype=np.uint64)) & 1).astype(np.float64)
        signs = bits * 2.0 - 1.0
        null = np.abs((signs * values).mean(axis=1))
        return float(np.sum(null >= observed - 1e-15) / null.size)
    else:
        rng = np.random.default_rng(int(seed))
        signs = rng.choice(np.array([-1.0, 1.0]), size=(int(repeats), values.size), replace=True)
        null = np.abs((signs * values).mean(axis=1))
        return float((1 + np.sum(null >= observed - 1e-15)) / (1 + null.size))


def paired_t_test(values):
    values = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=np.float64)
    if values.size < 2 or np.allclose(values.std(ddof=1), 0.0):
        return None
    try:
        from scipy.stats import ttest_1samp

        return float(ttest_1samp(values, popmean=0.0, alternative="two-sided").pvalue)
    except ImportError:
        return None


def benjamini_hochberg(p_values, alpha=0.05):
    """Return BH q-values and rejection flags, preserving None entries."""
    valid = [(index, float(value)) for index, value in enumerate(p_values) if value is not None and math.isfinite(float(value))]
    output = [{"q_value": None, "reject_fdr": False} for _ in p_values]
    if not valid:
        return output
    ordered = sorted(valid, key=lambda item: item[1])
    total = len(ordered)
    adjusted = [0.0] * total
    running = 1.0
    for rank in range(total - 1, -1, -1):
        value = ordered[rank][1] * total / (rank + 1)
        running = min(running, value)
        adjusted[rank] = min(1.0, running)
    for (index, _), q_value in zip(ordered, adjusted):
        output[index] = {"q_value": q_value, "reject_fdr": q_value <= float(alpha)}
    return output


def krippendorff_alpha_interval(item_ratings):
    """Krippendorff's alpha with squared-distance disagreement."""
    grouped = [[float(value) for value in ratings if value is not None] for ratings in item_ratings]
    grouped = [ratings for ratings in grouped if len(ratings) >= 2]
    if not grouped:
        return None
    observed_terms = []
    pooled = []
    for ratings in grouped:
        pooled.extend(ratings)
        for first in range(len(ratings)):
            for second in range(first + 1, len(ratings)):
                observed_terms.append((ratings[first] - ratings[second]) ** 2)
    if len(pooled) < 2:
        return None
    expected_terms = []
    for first in range(len(pooled)):
        for second in range(first + 1, len(pooled)):
            expected_terms.append((pooled[first] - pooled[second]) ** 2)
    observed = float(np.mean(observed_terms))
    expected = float(np.mean(expected_terms))
    return 1.0 if expected <= 1e-12 else 1.0 - observed / expected
