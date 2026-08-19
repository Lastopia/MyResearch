from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def perplexity(total_nll: float, token_count: int) -> float:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    return math.exp(total_nll / token_count)


def _query_start(length: int, query_fraction: float) -> int:
    if not 0.0 < query_fraction <= 1.0:
        raise ValueError("query_fraction must be in (0, 1]")
    return max(0, length - max(1, math.ceil(length * query_fraction)))


def mean_attention_distance(
    attention: torch.Tensor, query_fraction: float = 0.25
) -> torch.Tensor:
    """Return mean absolute causal attention distance for each head."""
    if attention.ndim != 4:
        raise ValueError("attention must have shape [batch, head, query, key]")
    _, _, query_len, key_len = attention.shape
    start = _query_start(query_len, query_fraction)
    q_pos = torch.arange(query_len, device=attention.device).view(query_len, 1)
    k_pos = torch.arange(key_len, device=attention.device).view(1, key_len)
    distance = (q_pos - k_pos).clamp_min(0).to(attention.dtype)
    selected = attention[:, :, start:, :]
    selected_distance = distance[start:, :]
    return (selected * selected_distance).sum(dim=-1).mean(dim=(0, 2))


def normalized_attention_distance(
    attention: torch.Tensor, query_fraction: float = 0.25
) -> torch.Tensor:
    if attention.ndim != 4:
        raise ValueError("attention must have shape [batch, head, query, key]")
    _, _, query_len, key_len = attention.shape
    start = _query_start(query_len, query_fraction)
    q_pos = torch.arange(query_len, device=attention.device).view(query_len, 1)
    k_pos = torch.arange(key_len, device=attention.device).view(1, key_len)
    distance = (q_pos - k_pos).clamp_min(0).to(attention.dtype)
    denom = q_pos.clamp_min(1).to(attention.dtype)
    normalized = distance / denom
    return (attention[:, :, start:, :] * normalized[start:, :]).sum(dim=-1).mean(
        dim=(0, 2)
    )


def attention_radius(
    attention: torch.Tensor,
    mass: float = 0.9,
    query_fraction: float = 0.25,
) -> torch.Tensor:
    """Return the average radius containing ``mass`` attention for each head."""
    if not 0.0 < mass <= 1.0:
        raise ValueError("mass must be in (0, 1]")
    batch, heads, query_len, _ = attention.shape
    start = _query_start(query_len, query_fraction)
    radii = torch.zeros(batch, heads, query_len - start, device=attention.device)
    for offset, query_index in enumerate(range(start, query_len)):
        causal_row = attention[:, :, query_index, : query_index + 1].flip(-1)
        cumulative = causal_row.cumsum(dim=-1)
        reached = cumulative >= mass
        first = reached.to(torch.int64).argmax(dim=-1)
        never = ~reached.any(dim=-1)
        first = torch.where(
            never,
            torch.full_like(first, query_index),
            first,
        )
        radii[:, :, offset] = first.to(radii.dtype)
    return radii.mean(dim=(0, 2))


def geometric_head_fit(mean_distances: torch.Tensor, eps: float = 1e-8) -> float:
    distances = mean_distances.detach().float().flatten()
    if distances.numel() < 2:
        return float("nan")
    x = torch.arange(distances.numel(), dtype=torch.float32, device=distances.device)
    y = torch.log(distances.clamp_min(eps))
    x_centered = x - x.mean()
    denominator = (x_centered.square()).sum()
    if denominator <= eps:
        return float("nan")
    slope = (x_centered * (y - y.mean())).sum() / denominator
    prediction = y.mean() + slope * x_centered
    ss_total = ((y - y.mean()).square()).sum()
    if ss_total <= eps:
        return 1.0
    r_squared = 1.0 - ((y - prediction).square()).sum() / ss_total
    return float(r_squared.clamp(min=0.0, max=1.0).item())


def relative_head_distance_drift(
    distances_by_length: torch.Tensor, eps: float = 1e-8
) -> float:
    """Mean coefficient of variation of head distance shares over lengths."""
    if distances_by_length.ndim != 2:
        raise ValueError("distances_by_length must have shape [length, head]")
    shares = distances_by_length / distances_by_length.sum(dim=-1, keepdim=True).clamp_min(
        eps
    )
    mean = shares.mean(dim=0)
    std = shares.std(dim=0, unbiased=False)
    return float((std / mean.clamp_min(eps)).mean().item())


def context_adaptivity_score(head_distances: torch.Tensor, eps: float = 1e-8) -> float:
    """Mean Jensen-Shannon divergence between per-sample head shares and their mean."""
    if head_distances.ndim != 2:
        raise ValueError("head_distances must have shape [sample, head]")
    probs = head_distances / head_distances.sum(dim=-1, keepdim=True).clamp_min(eps)
    reference = probs.mean(dim=0, keepdim=True)
    midpoint = 0.5 * (probs + reference)
    kl_left = (probs * (probs.clamp_min(eps).log() - midpoint.clamp_min(eps).log())).sum(
        dim=-1
    )
    kl_right = (
        reference
        * (reference.clamp_min(eps).log() - midpoint.clamp_min(eps).log())
    ).sum(dim=-1)
    return float((0.5 * (kl_left + kl_right)).mean().item())


def bias_monotonic_violation_rate(
    bias: torch.Tensor, query_fraction: float = 0.25, tolerance: float = 1e-7
) -> float:
    """Fraction of adjacent causal key pairs where a farther key has larger bias."""
    if bias.ndim != 4:
        raise ValueError("bias must have shape [batch, head, query, key]")
    _, _, query_len, _ = bias.shape
    start = _query_start(query_len, query_fraction)
    violations: list[torch.Tensor] = []
    for query_index in range(max(1, start), query_len):
        row = bias[:, :, query_index, : query_index + 1]
        violations.append(row[..., :-1] > row[..., 1:] + tolerance)
    if not violations:
        return 0.0
    flat = torch.cat([item.reshape(-1) for item in violations])
    return float(flat.float().mean().item())


def bias_far_near_pair_violation_rate(
    bias: torch.Tensor,
    far_positions: torch.Tensor,
    near_positions: torch.Tensor,
    tolerance: float = 1e-7,
) -> float:
    """Compare explicit far/near keys for the same final query.

    A violation occurs when the farther key receives a larger additive bias
    than its paired nearer key. This is the document-level pair definition of
    BMVR; the adjacent-key helper above remains available as a diagnostic.
    """
    far = _gather_final_query(bias, far_positions)
    near = _gather_final_query(bias, near_positions)
    return float((far > near + tolerance).float().mean().item())


def _gather_final_query(
    matrix: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    if matrix.ndim != 4:
        raise ValueError("matrix must have shape [batch, head, query, key]")
    if positions.ndim != 1 or positions.shape[0] != matrix.shape[0]:
        raise ValueError("positions must have shape [batch]")
    batch_index = torch.arange(matrix.shape[0], device=matrix.device)
    return matrix[batch_index, :, -1, positions]


def relevant_attention_advantage(
    attention: torch.Tensor,
    relevant_positions: torch.Tensor,
    distractor_positions: torch.Tensor,
) -> float:
    relevant = _gather_final_query(attention, relevant_positions)
    distractor = _gather_final_query(attention, distractor_positions)
    return float((relevant - distractor).mean().item())


def semantic_exemption_success_rate(
    bias: torch.Tensor,
    relevant_positions: torch.Tensor,
    distractor_positions: torch.Tensor,
) -> float:
    relevant = _gather_final_query(bias, relevant_positions)
    distractor = _gather_final_query(bias, distractor_positions)
    return float((relevant > distractor).float().mean().item())


def false_exemption_rate(
    gate: torch.Tensor,
    irrelevant_mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    if gate.ndim != 4:
        raise ValueError("gate must have shape [batch, head, query, key]")
    rows = gate[:, :, -1, :-1]
    if (
        irrelevant_mask.ndim != 2
        or irrelevant_mask.shape[0] != rows.shape[0]
        or irrelevant_mask.shape[1] < rows.shape[-1]
    ):
        raise ValueError(
            "irrelevant_mask must have shape [batch, key] and cover "
            "all causal keys"
        )
    mask = irrelevant_mask[:, : rows.shape[-1]].to(
        device=rows.device,
        dtype=torch.bool,
    )
    selected = rows[mask[:, None, :].expand_as(rows)]
    if selected.numel() == 0:
        return 0.0
    return float((selected > threshold).float().mean().item())


def attention_sink_ratio_mask(
    attention: torch.Tensor,
    sink_mask: torch.Tensor,
    *,
    query_fraction: float = 0.25,
    threshold: float = 0.5,
) -> float | None:
    if attention.ndim != 4:
        raise ValueError("attention must have shape [batch, head, query, key]")
    if sink_mask.ndim != 2 or sink_mask.shape != (
        attention.shape[0],
        attention.shape[-1],
    ):
        raise ValueError("sink_mask must have shape [batch, key]")
    mask = sink_mask.to(device=attention.device, dtype=attention.dtype)
    if not bool(mask.any()):
        return None
    selected = (attention * mask[:, None, None, :]).sum(dim=-1)
    start = _query_start(attention.shape[-2], query_fraction)
    return float((selected[:, :, start:] > threshold).float().mean().item())


def retrieval_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return float((predictions == labels).float().mean().item())


def target_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels, reduction="none")
