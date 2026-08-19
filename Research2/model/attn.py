import math
import os
import tempfile
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:  # Older PyTorch keeps the portable SDPA fallback usable.
    create_block_mask = None
    flex_attention = None


_COMPILED_FLEX_ATTENTION = None
_FLEX_RUNTIME_DISABLED = False


def _compiled_flex_attention():
    global _COMPILED_FLEX_ATTENTION
    if _COMPILED_FLEX_ATTENTION is None:
        # Triton-Windows cannot decode a non-ASCII cache path in some setups.
        # The system temp directory is normally ASCII and is outside the repo,
        # so compiled artifacts do not pollute experiment outputs.
        os.environ.setdefault(
            "TRITON_CACHE_DIR", os.path.join(tempfile.gettempdir(), "fast_sp_triton"),
        )
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR",
            os.path.join(tempfile.gettempdir(), "fast_sp_torchinductor"),
        )
        _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, dynamic=False)
    return _COMPILED_FLEX_ATTENTION


def _normal_quantile(prob, device):
    prob = torch.as_tensor(prob, device=device, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
    if hasattr(torch.special, "ndtri"):
        return torch.special.ndtri(prob)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
    )
    return normal.icdf(prob)


def _spark_k(attn_cfg, seq_len):
    if "k" in attn_cfg:
        return int(attn_cfg["k"])
    return int(attn_cfg.get("max_attended_tokens", min(256, seq_len)))


def _spark_r(attn_cfg, head_dim):
    if "r" in attn_cfg:
        return int(attn_cfg["r"])
    r = int(round(head_dim * attn_cfg.get("r_ratio", 0.5)))
    return max(1, min(head_dim - 1, r))


def statistical_topk_neg_inf(score, causal, k):
    seq_len = score.size(-2)
    valid = causal[None, None, :, :]
    count = causal.sum(dim=-1, keepdim=True).to(score.device, dtype=torch.float32)
    count = count.view(1, 1, seq_len, 1)
    keep = torch.minimum(torch.full_like(count, float(max(1, int(k)))), count)
    keep_all = keep >= count

    sf = score.float()
    masked = sf.masked_fill(~valid, 0.0)
    mean = masked.sum(dim=-1, keepdim=True) / count
    centered = (sf - mean).masked_fill(~valid, 0.0)
    denom = (count - 1).clamp_min(1.0)
    std = (centered.square().sum(dim=-1, keepdim=True) / denom).sqrt().clamp_min(1e-6)
    q = _normal_quantile(1.0 - keep / count, score.device).to(sf.dtype)
    theta = mean + std * q
    sparse = valid & (keep_all | (sf >= theta))
    empty = sparse.sum(dim=-1, keepdim=True) == 0
    if empty.any():
        fallback_idx = sf.masked_fill(~valid, float("-inf")).argmax(dim=-1, keepdim=True)
        fallback = torch.zeros_like(sparse).scatter_(-1, fallback_idx, True)
        sparse = sparse | (empty & fallback)
    return score.masked_fill(~sparse, float("-inf")), sparse


def alibi_slopes(n_head):
    x = (2 ** 8) ** (1 / n_head)
    return torch.tensor([1 / x ** (i + 1) for i in range(n_head)], dtype=torch.float32)


def causal_mask(seq_len, device):
    return torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()


def alibi_bias(seq_len, slopes, device, dtype):
    pos = torch.arange(seq_len, device=device)
    distance = (pos[:, None] - pos[None, :]).clamp_min(0).to(dtype)
    return -slopes.to(device=device, dtype=dtype).view(1, -1, 1, 1) * distance.view(1, 1, seq_len, seq_len)


def rope_cache(seq_len, head_dim, device):
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    idx = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (idx / head_dim))
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out = torch.empty_like(x)
    out[..., ::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class StdAttn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg["n_head"]
        self.d_model = cfg["d_model"]
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model must be divisible by n_head, got d_model={self.d_model}, n_head={self.n_head}")
        self.head_dim = self.d_model // self.n_head
        self.position_encoding = cfg.get("position_encoding", "rope")
        if self.position_encoding == "rope" and self.head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got d_model/n_head={self.head_dim}")
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = cfg.get("dropout", 0.0)
        self.cable_fused_sdpa = bool(cfg.get("cable_fused_sdpa", True))
        self.cable_augmented_dims = int(cfg.get("cable_augmented_dims", 8))
        if self.cable_augmented_dims < 1:
            raise ValueError("cable_augmented_dims must be positive")
        if self.position_encoding == "alibi":
            self.register_buffer("slopes", alibi_slopes(self.n_head), persistent=False)
        elif self.position_encoding == "cable":
            self.cable_layer = nn.Linear(self.d_model, self.n_head, bias=True)
            self.cable_layer_scale = nn.Linear(self.d_model, self.n_head, bias=True)

    def _apply_position_to_qk(self, q, k, x):
        if self.position_encoding == "rope":
            cos, sin = rope_cache(q.size(-2), self.head_dim, x.device)
            return apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.position_encoding in ("none", "alibi", "cable"):
            return q, k
        raise ValueError(f"unknown position_encoding: {self.position_encoding}")

    def _position_bias(self, x, score):
        if self.position_encoding == "alibi":
            return alibi_bias(score.size(-1), self.slopes, x.device, score.dtype)
        if self.position_encoding == "cable":
            prefix, scale = self._cable_terms(x)
            return scale.unsqueeze(-1) * (prefix.unsqueeze(3) - prefix.unsqueeze(2))
        return None

    def _cable_terms(self, x):
        signal = self.cable_layer(x).float()
        prefix = torch.cumsum(-F.relu(signal), dim=1).permute(0, 2, 1)
        scale = F.softplus(self.cable_layer_scale(x).float()).permute(0, 2, 1)
        return prefix, scale

    def _cable_augmented_qkv(self, q, k, v, x):
        prefix, scale = self._cable_terms(x)
        # scale_i * prefix_i is constant across keys and cancels in softmax;
        # the remaining -scale_i * prefix_j term is one augmented Q/K product.
        q_first = scale.to(dtype=q.dtype).unsqueeze(-1) * math.sqrt(self.head_dim)
        k_first = -prefix.to(dtype=k.dtype).unsqueeze(-1)
        pad_shape = q_first.shape[:-1] + (self.cable_augmented_dims - 1,)
        q_extra = torch.cat((q_first, q_first.new_zeros(pad_shape)), dim=-1)
        k_extra = torch.cat((k_first, k_first.new_zeros(pad_shape)), dim=-1)
        v_extra = v.new_zeros(v.shape[:-1] + (self.cable_augmented_dims,))
        return (
            torch.cat((q, q_extra), dim=-1),
            torch.cat((k, k_extra), dim=-1),
            torch.cat((v, v_extra), dim=-1),
        )

    def _score(self, q, k, x):
        score = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        bias = self._position_bias(x, score)
        if bias is not None:
            score = score + bias
        return score

    def forward(self, x, need_weights=False):
        bsz, seq_len, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self._apply_position_to_qk(q, k, x)
        if need_weights or self.position_encoding == "alibi" or (
            self.position_encoding == "cable" and not self.cable_fused_sdpa
        ):
            score = self._score(q, k, x)
            mask = causal_mask(seq_len, x.device)
            score = score.masked_fill(~mask[None, None, :, :], float("-inf"))
            weights = torch.softmax(score, dim=-1)
            weights = F.dropout(weights, p=self.dropout, training=self.training)
            y = weights @ v
        elif self.position_encoding == "cable":
            q_sdpa, k_sdpa, v_sdpa = self._cable_augmented_qkv(q, k, v, x)
            y = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                scale=1.0 / math.sqrt(self.head_dim),
            )[..., :self.head_dim]
            weights = None
        else:
            weights = None
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        y = self.proj(y)
        return (y, weights) if need_weights else y


def build_attn(name, cfg, attn_cfg=None):
    attn_cfg = attn_cfg or {}
    if name == "std":
        return StdAttn(cfg)
    if name == "alibi":
        return ALiBiAttn(cfg)
    if name == "cable":
        return CableAttn(cfg)
    if name in ("sp", "spark"):
        return SPAttn(cfg, attn_cfg)
    if name in ("fast_sp", "fast_spark"):
        return FastSPAttn(cfg, attn_cfg)
    if name in ("fast_sp_routed", "fast_spark_routed"):
        return RoutedFastSPAttn(cfg, attn_cfg)
    if name == "topk":
        return TopKAttn(cfg, attn_cfg)
    if name == "adaptive_threshold":
        return AdaptiveThresholdAttn(cfg, attn_cfg)
    raise ValueError(f"unknown attn: {name}")


class TopKAttn(StdAttn):
    def __init__(self, cfg, attn_cfg):
        super().__init__(cfg)
        self.k = attn_cfg["k"]

    def forward(self, x, need_weights=False):
        bsz, seq_len, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self._apply_position_to_qk(q, k, x)

        score = self._score(q, k, x)
        causal = causal_mask(seq_len, x.device)
        score = score.masked_fill(~causal[None, None, :, :], float("-inf"))

        keep = min(self.k, seq_len)
        topv, topi = torch.topk(score, keep, dim=-1)
        sparse_score = torch.full_like(score, float("-inf"))
        sparse_score.scatter_(-1, topi, topv)
        weights = torch.softmax(sparse_score, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        y = self.proj(y)
        return (y, weights) if need_weights else y


class ALiBiAttn(StdAttn):
    def __init__(self, cfg):
        cfg = dict(cfg)
        cfg["position_encoding"] = "alibi"
        super().__init__(cfg)


class CableAttn(StdAttn):
    def __init__(self, cfg):
        cfg = dict(cfg)
        cfg["position_encoding"] = "cable"
        super().__init__(cfg)


class AdaptiveThresholdAttn(StdAttn):
    def __init__(self, cfg, attn_cfg):
        super().__init__(cfg)
        self.relative_threshold = attn_cfg.get("relative_threshold", 0.05)
        self.length_scale = attn_cfg.get("length_scale", 1.0)
        self.keep_self = attn_cfg.get("keep_self", True)
        self.renorm = attn_cfg.get("renorm", True)

    def forward(self, x, need_weights=False):
        bsz, seq_len, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self._apply_position_to_qk(q, k, x)

        score = self._score(q, k, x)
        causal = causal_mask(seq_len, x.device)
        score = score.masked_fill(~causal[None, None, :, :], float("-inf"))
        weights = torch.softmax(score, dim=-1)

        valid_len = causal.sum(dim=-1).to(weights.dtype)[None, None, :, None]
        length_threshold = self.length_scale / valid_len
        peak_threshold = self.relative_threshold * weights.max(dim=-1, keepdim=True).values
        adaptive_threshold = torch.maximum(length_threshold, peak_threshold)
        sparse_mask = weights >= adaptive_threshold
        if self.keep_self:
            eye = torch.eye(seq_len, device=x.device, dtype=torch.bool)
            sparse_mask = sparse_mask | eye[None, None, :, :]
        weights = weights.masked_fill(~sparse_mask, 0.0)
        if self.renorm:
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
            weights = weights / denom
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        y = self.proj(y)
        return (y, weights) if need_weights else y


class SPAttn(StdAttn):
    def __init__(self, cfg, attn_cfg):
        super().__init__(cfg)
        self.r = _spark_r(attn_cfg, self.head_dim)
        self.k = attn_cfg.get("k", attn_cfg.get("max_attended_tokens", 256))
        self.renorm = attn_cfg.get("renorm", False)

    def forward(self, x, need_weights=False):
        bsz, seq_len, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self._apply_position_to_qk(q, k, x)

        q1, q2 = q[..., :self.r], q[..., self.r:]
        k1, k2 = k[..., :self.r], k[..., self.r:]
        pred_score = q1 @ k1.transpose(-2, -1) / math.sqrt(self.r)
        bias = self._position_bias(x, pred_score)
        if bias is not None:
            pred_score = pred_score + bias
        value_score = q2 @ k2.transpose(-2, -1) / math.sqrt(self.head_dim - self.r)
        causal = causal_mask(seq_len, x.device)
        sparse_pred_score, sparse_mask = statistical_topk_neg_inf(pred_score, causal, min(int(self.k), seq_len))
        pred_weights = torch.softmax(sparse_pred_score, dim=-1)
        value_gate = F.softplus(value_score).masked_fill(~sparse_mask, 0.0)
        weights = pred_weights * value_gate
        if self.renorm:
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
            weights = weights / denom
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        y = self.proj(y)
        return (y, weights) if need_weights else y


class RoutedFastSPAttn(StdAttn):
    """Block-sparse Spark attention that avoids the dense value-score path.

    A cheap block-level predictor first selects a fixed number of causal key
    blocks for every query token. Full Spark predictor/value scores and the
    weighted value sum are then evaluated only inside those selected blocks.
    This makes the expensive attention work proportional to ``seq_len * k``
    rather than ``seq_len ** 2`` once the context is longer than ``k``.

    The reference :class:`SPAttn` remains unchanged for direct comparisons.
    """

    def __init__(self, cfg, attn_cfg):
        super().__init__(cfg)
        self.r = _spark_r(attn_cfg, self.head_dim)
        self.k = int(attn_cfg.get("k", attn_cfg.get("max_attended_tokens", 256)))
        self.block_size = int(attn_cfg.get("block_size", 32))
        self.active_blocks = int(attn_cfg.get(
            "active_blocks", math.ceil(self.k / self.block_size),
        ))
        self.renorm = bool(attn_cfg.get("renorm", False))
        self.keep_current_block = bool(attn_cfg.get("keep_current_block", True))

    @staticmethod
    def _gather_blocks(blocks, block_ids):
        # blocks: [B, H, K_BLOCKS, BLOCK, ...]
        # ids:    [B, H, QUERIES, ACTIVE_BLOCKS]
        bsz, n_head = blocks.shape[:2]
        batch = torch.arange(bsz, device=blocks.device).view(bsz, 1, 1, 1)
        heads = torch.arange(n_head, device=blocks.device).view(1, n_head, 1, 1)
        return blocks[batch, heads, block_ids]

    def _cable_terms(self, x):
        prefix = torch.cumsum(-F.relu(self.cable_layer(x)), dim=1).permute(0, 2, 1)
        scale = F.softplus(self.cable_layer_scale(x)).permute(0, 2, 1)
        return prefix, scale

    def forward(self, x, need_weights=False):
        bsz, seq_len, _ = x.shape
        if seq_len % self.block_size != 0:
            raise ValueError(
                f"FastSPAttn requires seq_len divisible by block_size, got "
                f"seq_len={seq_len}, block_size={self.block_size}"
            )
        block_count = seq_len // self.block_size
        active_blocks = min(self.active_blocks, block_count)

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self._apply_position_to_qk(q, k, x)
        q1, q2 = q[..., :self.r], q[..., self.r:]
        k1, k2 = k[..., :self.r], k[..., self.r:]

        k1_blocks = k1.view(bsz, self.n_head, block_count, self.block_size, self.r)
        k2_blocks = k2.view(
            bsz, self.n_head, block_count, self.block_size, self.head_dim - self.r,
        )
        v_blocks = v.view(bsz, self.n_head, block_count, self.block_size, self.head_dim)

        # Key-block representatives make routing O(seq_len * num_blocks)
        # instead of O(seq_len^2). Each query routes independently, preserving
        # strict autoregressive causality while full token scores are computed
        # only after routing.
        k1_representative = k1_blocks.mean(dim=3)
        selector = q1 @ k1_representative.transpose(-2, -1)
        selector = selector / math.sqrt(self.r)
        query_blocks = torch.arange(seq_len, device=x.device) // self.block_size
        key_blocks = torch.arange(block_count, device=x.device)
        causal_blocks = key_blocks.view(1, -1) <= query_blocks.view(-1, 1)
        selector = selector.masked_fill(~causal_blocks[None, None, :, :], float("-inf"))

        cable_prefix = cable_scale = None
        if self.position_encoding == "cable":
            cable_prefix, cable_scale = self._cable_terms(x)
            prefix_blocks = cable_prefix.view(
                bsz, self.n_head, block_count, self.block_size,
            ).mean(dim=-1)
            selector = selector + cable_scale.unsqueeze(-1) * (
                cable_prefix.unsqueeze(-1) - prefix_blocks.unsqueeze(-2)
            )
        elif self.position_encoding not in ("rope", "none"):
            raise ValueError(f"FastSPAttn does not support position_encoding={self.position_encoding}")

        if self.keep_current_block:
            # The current key block contains tokens later than some queries.
            # Give it a constant winning score so its routing decision cannot
            # depend on those future keys; the exact token mask below removes
            # future positions from the actual attention computation.
            current = query_blocks.view(1, 1, seq_len, 1).expand(
                bsz, self.n_head, seq_len, 1,
            )
            selector = selector.scatter(-1, current, float("inf"))
        selected_blocks = torch.topk(selector, active_blocks, dim=-1).indices

        selected_k1 = self._gather_blocks(k1_blocks, selected_blocks)
        selected_k2 = self._gather_blocks(k2_blocks, selected_blocks)
        selected_v = self._gather_blocks(v_blocks, selected_blocks)
        selected_k1 = selected_k1.flatten(3, 4)
        selected_k2 = selected_k2.flatten(3, 4)
        selected_v = selected_v.flatten(3, 4)

        token_offsets = torch.arange(self.block_size, device=x.device)
        key_indices = selected_blocks.unsqueeze(-1) * self.block_size + token_offsets
        key_indices = key_indices.flatten(-2, -1)
        query_indices = torch.arange(seq_len, device=x.device)
        valid = key_indices <= query_indices.view(1, 1, seq_len, 1)

        pred_score = (q1.unsqueeze(-2) @ selected_k1.transpose(-2, -1)).squeeze(-2)
        pred_score = pred_score / math.sqrt(self.r)
        if cable_prefix is not None:
            prefix_blocks = cable_prefix.view(
                bsz, self.n_head, block_count, self.block_size,
            )
            selected_prefix = self._gather_blocks(
                prefix_blocks.unsqueeze(-1), selected_blocks,
            ).squeeze(-1).flatten(3, 4)
            pred_score = pred_score + cable_scale.unsqueeze(-1) * (
                cable_prefix.unsqueeze(-1) - selected_prefix
            )
        pred_score = pred_score.masked_fill(~valid, float("-inf"))
        pred_weights = torch.softmax(pred_score, dim=-1)

        value_score = (q2.unsqueeze(-2) @ selected_k2.transpose(-2, -1)).squeeze(-2)
        value_score = value_score / math.sqrt(self.head_dim - self.r)
        value_gate = F.softplus(value_score).masked_fill(~valid, 0.0)
        weights = pred_weights * value_gate
        if self.renorm:
            denom = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
            weights = weights / denom
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        y = (weights.unsqueeze(-2) @ selected_v).squeeze(-2)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        y = self.proj(y)

        if not need_weights:
            return y
        full_weights = weights.new_zeros(bsz, self.n_head, seq_len, seq_len)
        scatter_indices = key_indices
        full_weights.scatter_add_(-1, scatter_indices, weights)
        return y, full_weights


class FastSPAttn(StdAttn):
    """Fused causal-window attention for the hardware-oriented Fast SP model.

    The exact routed Spark implementation above avoids arithmetic but its
    irregular gathers cost more wall-clock time than the dense reference on
    common GPUs. Fast SP uses a regular causal window and a compiled
    FlexAttention block mask instead. Every query attends to at most
    ``window_size`` tokens and the Triton kernel skips empty blocks. In the
    FlexAttention path, CABLE is folded into extra Q/K coordinates so gradients
    flow through tensor inputs instead of captured score_mod buffers. A chunked
    implementation remains as the portable fallback. The original SPAttn
    remains available for mathematical comparison.

    This is an explicitly hardware-oriented approximation: it keeps the full
    attention parameter capacity but replaces Spark's second value-score gate
    with fused scaled-dot-product attention inside the sparse support.
    """

    def __init__(self, cfg, attn_cfg):
        super().__init__(cfg)
        self.window_size = int(attn_cfg.get(
            "window_size", attn_cfg.get("k", attn_cfg.get("max_attended_tokens", 256)),
        ))
        self.chunk_size = int(attn_cfg.get("chunk_size", 512))
        self.flex_block_size = int(attn_cfg.get("flex_block_size", 128))
        self.cable_augmented_dims = int(attn_cfg.get("cable_augmented_dims", 8))
        self.use_flex_attention = bool(attn_cfg.get("use_flex_attention", True))
        self._flex_mask_cache = {}
        if self.window_size < 1 or self.chunk_size < 1:
            raise ValueError("FastSPAttn window_size and chunk_size must be positive")
        if self.cable_augmented_dims < 1:
            raise ValueError("FastSPAttn cable_augmented_dims must be positive")
        if self.position_encoding not in ("rope", "none", "cable"):
            raise ValueError(
                "FastSPAttn supports position_encoding=rope, cable, or none"
            )

    def _cable_terms(self, x):
        prefix = torch.cumsum(-F.relu(self.cable_layer(x)), dim=1).permute(0, 2, 1)
        scale = F.softplus(self.cable_layer_scale(x)).permute(0, 2, 1)
        return prefix, scale

    def _cable_augmented_qk(self, q, k, cable_prefix, cable_scale):
        # CABLE bias is scale_q * (prefix_q - prefix_k). The prefix_q term is a
        # per-query softmax constant, so the effective bias is -scale_q*prefix_k.
        q_first = cable_scale.to(dtype=q.dtype).unsqueeze(-1) * math.sqrt(self.head_dim)
        k_first = -cable_prefix.to(dtype=k.dtype).unsqueeze(-1)
        if self.cable_augmented_dims == 1:
            q_extra, k_extra = q_first, k_first
        else:
            pad_shape = q_first.shape[:-1] + (self.cable_augmented_dims - 1,)
            q_extra = torch.cat((q_first, q_first.new_zeros(pad_shape)), dim=-1)
            k_extra = torch.cat((k_first, k_first.new_zeros(pad_shape)), dim=-1)
        return torch.cat((q, q_extra), dim=-1), torch.cat((k, k_extra), dim=-1)

    def _chunk_bounds(self, seq_len):
        for query_start in range(0, seq_len, self.chunk_size):
            query_end = min(seq_len, query_start + self.chunk_size)
            key_start = max(0, query_start - self.window_size + 1)
            yield query_start, query_end, key_start

    def _window_mask(self, query_start, query_end, key_start, device):
        query_index = torch.arange(query_start, query_end, device=device).unsqueeze(-1)
        key_index = torch.arange(key_start, query_end, device=device).unsqueeze(0)
        return (key_index <= query_index) & (key_index > query_index - self.window_size)

    def _flex_mask(self, seq_len, device):
        key = (str(device), int(seq_len), self.window_size, self.flex_block_size)
        if key not in self._flex_mask_cache:
            window_size = self.window_size

            def causal_window(batch, head, query_index, key_index):
                del batch, head
                return (
                    (query_index >= key_index)
                    & (query_index - key_index < window_size)
                )

            self._flex_mask_cache[key] = create_block_mask(
                causal_window,
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
                BLOCK_SIZE=self.flex_block_size,
            )
        return self._flex_mask_cache[key]

    def _sdpa_fallback(self, q, k, v, x, cable_prefix=None, cable_scale=None):
        output_chunks = []
        for query_start, query_end, key_start in self._chunk_bounds(x.size(1)):
            mask = self._window_mask(query_start, query_end, key_start, x.device)
            query = q[:, :, query_start:query_end]
            key = k[:, :, key_start:query_end]
            value = v[:, :, key_start:query_end]
            if cable_prefix is None:
                output = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
            else:
                score = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
                query_prefix = cable_prefix[:, :, query_start:query_end]
                key_prefix = cable_prefix[:, :, key_start:query_end]
                query_scale = cable_scale[:, :, query_start:query_end]
                score = score + query_scale.unsqueeze(-1) * (
                    query_prefix.unsqueeze(-1) - key_prefix.unsqueeze(-2)
                )
                score = score.masked_fill(
                    ~mask.view(1, 1, *mask.shape), float("-inf"),
                )
                weights = torch.softmax(score, dim=-1)
                weights = F.dropout(weights, p=self.dropout, training=self.training)
                output = weights @ value
            output_chunks.append(output)
        return torch.cat(output_chunks, dim=2)

    def forward(self, x, need_weights=False):
        bsz, seq_len, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self._apply_position_to_qk(q, k, x)

        cable_prefix = cable_scale = None
        if self.position_encoding == "cable":
            cable_prefix, cable_scale = self._cable_terms(x)

        full_weights = None
        if not need_weights:
            global _FLEX_RUNTIME_DISABLED
            can_flex = (
                self.use_flex_attention
                and not _FLEX_RUNTIME_DISABLED
                and flex_attention is not None
                and create_block_mask is not None
                and x.is_cuda
                and self.dropout == 0.0
            )
            if can_flex:
                try:
                    q_flex, k_flex = q, k
                    if cable_prefix is not None:
                        q_flex, k_flex = self._cable_augmented_qk(
                            q, k, cable_prefix, cable_scale,
                        )
                    flex_kwargs = {
                        "block_mask": self._flex_mask(seq_len, x.device),
                        "scale": 1.0 / math.sqrt(self.head_dim),
                    }
                    y = _compiled_flex_attention()(q_flex, k_flex, v, **flex_kwargs)
                except Exception as error:
                    _FLEX_RUNTIME_DISABLED = True
                    warnings.warn(
                        f"FastSPAttn FlexAttention unavailable; using SDPA fallback: {error}",
                        RuntimeWarning,
                    )
                    y = self._sdpa_fallback(
                        q, k, v, x, cable_prefix=cable_prefix, cable_scale=cable_scale,
                    )
            else:
                y = self._sdpa_fallback(
                    q, k, v, x, cable_prefix=cable_prefix, cable_scale=cable_scale,
                )
        else:
            output_chunks = []
            full_weights = q.new_zeros(bsz, self.n_head, seq_len, seq_len)
            for query_start, query_end, key_start in self._chunk_bounds(seq_len):
                mask = self._window_mask(query_start, query_end, key_start, x.device)
                query = q[:, :, query_start:query_end]
                key = k[:, :, key_start:query_end]
                value = v[:, :, key_start:query_end]
                score = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
                if cable_prefix is not None:
                    query_prefix = cable_prefix[:, :, query_start:query_end]
                    key_prefix = cable_prefix[:, :, key_start:query_end]
                    query_scale = cable_scale[:, :, query_start:query_end]
                    score = score + query_scale.unsqueeze(-1) * (
                        query_prefix.unsqueeze(-1) - key_prefix.unsqueeze(-2)
                    )
                score = score.masked_fill(~mask.view(1, 1, *mask.shape), float("-inf"))
                weights = torch.softmax(score, dim=-1)
                weights = F.dropout(weights, p=self.dropout, training=self.training)
                output_chunks.append(weights @ value)
                full_weights[:, :, query_start:query_end, key_start:query_end] = weights
            y = torch.cat(output_chunks, dim=2)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        y = self.proj(y)
        return (y, full_weights) if need_weights else y
