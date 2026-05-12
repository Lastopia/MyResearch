import torch


def compute_attention_entropy(attn_weights: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    probs = attn_weights.clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def compute_attention_distance(attn_weights: torch.Tensor) -> torch.Tensor:
    seq_len = attn_weights.size(-1)
    pos = torch.arange(seq_len, device=attn_weights.device)
    distance = (pos[None, :] - pos[:, None]).abs().float()
    return (attn_weights * distance).sum(dim=-1)


def compute_local_attention_mass(attn_weights: torch.Tensor, window: int) -> torch.Tensor:
    seq_len = attn_weights.size(-1)
    pos = torch.arange(seq_len, device=attn_weights.device)
    local = (pos[None, :] - pos[:, None]).abs() <= window
    return (attn_weights * local.float()).sum(dim=-1)


def compute_far_attention_mass(attn_weights: torch.Tensor, min_distance: int) -> torch.Tensor:
    seq_len = attn_weights.size(-1)
    pos = torch.arange(seq_len, device=attn_weights.device)
    far = (pos[None, :] - pos[:, None]).abs() > min_distance
    return (attn_weights * far.float()).sum(dim=-1)


def prepare_causal_logits_for_svd(logits: torch.Tensor) -> torch.Tensor:
    seq_len = logits.size(-1)
    causal = torch.ones(seq_len, seq_len, device=logits.device, dtype=torch.bool).tril()
    finite = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits)).float()
    finite = finite.masked_fill(~causal, 0.0)
    row_counts = causal.sum(dim=-1).clamp_min(1).to(finite.device)
    row_means = finite.sum(dim=-1, keepdim=True) / row_counts.view(1, 1, seq_len, 1)
    return (finite - row_means).masked_fill(~causal, 0.0)


def compute_singular_values(logits: torch.Tensor, max_items: int = 16) -> torch.Tensor:
    clean = prepare_causal_logits_for_svd(logits)
    flat = clean.reshape(-1, clean.size(-2), clean.size(-1))[:max_items]
    return torch.linalg.svdvals(flat.float())


def compute_head_singular_values(logits: torch.Tensor) -> torch.Tensor:
    clean = prepare_causal_logits_for_svd(logits)
    return torch.linalg.svdvals(clean.float())


def compute_spectral_concentration(svals: torch.Tensor, top_k: int = 8, eps: float = 1e-9) -> torch.Tensor:
    energy = svals.square()
    return energy[..., :top_k].sum(dim=-1) / energy.sum(dim=-1).clamp_min(eps)


def compute_toeplitz_deviation(logits: torch.Tensor) -> torch.Tensor:
    seq_len = logits.size(-1)
    causal = torch.ones(seq_len, seq_len, device=logits.device, dtype=torch.bool).tril()
    clean = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits)).float()
    clean = clean.masked_fill(~causal, 0.0)
    flat = clean.reshape(-1, clean.size(-2), clean.size(-1))
    seq_len = flat.size(-1)
    deviations = []
    for offset in range(-(seq_len - 1), 1):
        diagonal = torch.diagonal(flat, offset=offset, dim1=-2, dim2=-1)
        if diagonal.size(-1) > 1:
            deviations.append(diagonal.var(dim=-1, unbiased=False))
    return torch.stack(deviations, dim=-1).mean(dim=-1)


def compute_l0(feature_acts: torch.Tensor) -> torch.Tensor:
    return (feature_acts != 0).float().sum(dim=-1).mean()


def compute_dead_feature_rate(feature_acts: torch.Tensor) -> torch.Tensor:
    active = (feature_acts != 0).any(dim=0)
    return 1.0 - active.float().mean()
