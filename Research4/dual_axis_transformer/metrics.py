"""Dependency-free metrics used by confirmatory and audit runs.

The functions intentionally operate on CPU tensors and return plain floats so
the complete evidence needed by the report is serializable in ``final.json``.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def binary_auroc(targets: Tensor, scores: Tensor) -> float:
    targets = targets.bool().flatten().cpu()
    scores = scores.float().flatten().cpu()
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty(len(scores), dtype=torch.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(ranks[targets].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def binary_average_precision(targets: Tensor, scores: Tensor) -> float:
    truth = targets.bool().flatten().cpu()
    values = scores.float().flatten().cpu()
    positives = int(truth.sum())
    if positives == 0:
        return float("nan")
    order = torch.argsort(values, descending=True)
    ordered_truth = truth[order].float()
    precision = ordered_truth.cumsum(0) / torch.arange(
        1, len(order) + 1, dtype=torch.float32
    )
    return float((precision * ordered_truth).sum() / positives)


def expected_calibration_error(
    targets: Tensor, scores: Tensor, *, bins: int = 15
) -> float:
    truth = targets.float().flatten().cpu()
    values = scores.float().flatten().clamp(0, 1).cpu()
    total = max(1, len(values))
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = (values >= low) & (
            values <= high if index == bins - 1 else values < high
        )
        count = int(selected.sum())
        if count:
            confidence = float(values[selected].mean())
            accuracy = float(truth[selected].mean())
            error += count / total * abs(confidence - accuracy)
    return error


def false_positive_rate_at_tpr(
    targets: Tensor, scores: Tensor, *, target_tpr: float = 0.95
) -> float:
    truth = targets.bool().flatten().cpu()
    values = scores.float().flatten().cpu()
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(values, descending=True)
    ordered = truth[order]
    true_positive = ordered.cumsum(0)
    false_positive = (~ordered).cumsum(0)
    reached = torch.nonzero(true_positive.float() / positives >= target_tpr)
    if not len(reached):
        return 1.0
    return float(false_positive[int(reached[0])]) / negatives


def gini(values: Tensor) -> float:
    values = values.float().flatten().clamp_min(0).cpu()
    if not len(values) or float(values.sum()) == 0.0:
        return 0.0
    ordered, _ = values.sort()
    indices = torch.arange(1, len(values) + 1, dtype=torch.float32)
    return float(
        (2 * (indices * ordered).sum() / (len(values) * ordered.sum()))
        - (len(values) + 1) / len(values)
    )


def multilabel_metrics(
    targets: Tensor,
    probabilities: Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    truth = targets.bool().cpu()
    scores = probabilities.float().cpu()
    predicted = scores >= threshold
    per_precision: list[float] = []
    per_recall: list[float] = []
    per_f1: list[float] = []
    per_ap: list[float] = []
    association_counts: list[int] = []
    for column in range(truth.shape[1]):
        target = truth[:, column]
        guess = predicted[:, column]
        true_positive = int((target & guess).sum())
        false_positive = int((~target & guess).sum())
        false_negative = int((target & ~guess).sum())
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        per_precision.append(precision)
        per_recall.append(recall)
        per_f1.append(_safe_divide(2 * precision * recall, precision + recall))
        ap = binary_average_precision(target, scores[:, column])
        if math.isfinite(ap):
            per_ap.append(ap)

        activated = guess
        associated = 0
        if int(activated.sum()):
            for label in range(truth.shape[1]):
                conditional = float(truth[activated, label].float().mean())
                marginal = float(truth[:, label].float().mean())
                if conditional >= max(0.5, marginal + 0.20):
                    associated += 1
        association_counts.append(associated)

    true_positive = int((truth & predicted).sum())
    false_positive = int((~truth & predicted).sum())
    false_negative = int((truth & ~predicted).sum())
    micro_precision = _safe_divide(true_positive, true_positive + false_positive)
    micro_recall = _safe_divide(true_positive, true_positive + false_negative)
    usage = predicted.float().mean(dim=0)
    dead = float((usage == 0).float().mean())
    active = predicted.float().sum(dim=-1)
    return {
        "concept_macro_precision": sum(per_precision) / len(per_precision),
        "concept_macro_recall": sum(per_recall) / len(per_recall),
        "concept_macro_f1": sum(per_f1) / len(per_f1),
        "concept_micro_f1": _safe_divide(
            2 * micro_precision * micro_recall, micro_precision + micro_recall
        ),
        "concept_map": sum(per_ap) / len(per_ap) if per_ap else float("nan"),
        "active_concepts_per_example": float(active.mean()),
        "activation_sparsity": float(1.0 - predicted.float().mean()),
        "dead_concept_fraction": dead,
        "concept_usage_gini": gini(usage),
        "polysemantic_labels_per_node": sum(association_counts)
        / len(association_counts),
    }


def unknown_metrics(targets: Tensor, scores: Tensor) -> dict[str, float]:
    return {
        "unknown_auroc": binary_auroc(targets, scores),
        "unknown_auprc": binary_average_precision(targets, scores),
        "unknown_ece": expected_calibration_error(targets, scores),
        "unknown_fpr95": false_positive_rate_at_tpr(targets, scores),
    }


def cosine_alignment(first: Tensor, second: Tensor) -> float:
    first = torch.nn.functional.normalize(first.float(), dim=-1)
    second = torch.nn.functional.normalize(second.float(), dim=-1)
    return float((first * second).sum(dim=-1).mean())


def mean_jaccard(first: Iterable[set[int]], second: Iterable[set[int]]) -> float:
    scores = []
    for left, right in zip(first, second):
        union = left | right
        scores.append(1.0 if not union else len(left & right) / len(union))
    return sum(scores) / len(scores) if scores else float("nan")
