import math
import os
import tempfile
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


_COMPILED_SEQUENCE_EXPERT = None
_SEQUENCE_EXPERT_COMPILE_DISABLED = False


def _sequence_expert_math(
    x,
    router_weight,
    pred_weight,
    value_weight,
    down_weight,
    r,
    route_block_size,
    router_jitter,
    training,
):
    bsz, seq_len, d_model = x.shape
    route_count = seq_len // route_block_size
    routed_x = x.reshape(
        bsz, route_count, route_block_size, d_model,
    ).reshape(-1, route_block_size, d_model)
    router_logits = F.linear(routed_x[:, 0, :r], router_weight)
    if training and router_jitter:
        router_logits = router_logits + torch.empty_like(router_logits).uniform_(
            -router_jitter, router_jitter,
        )
    router_probability = torch.softmax(router_logits.float(), dim=-1)
    expert_ids = torch.argmax(router_logits, dim=-1)
    selected_probability = router_probability.gather(1, expert_ids.unsqueeze(-1))
    route_scale = selected_probability / selected_probability.detach().clamp_min(1e-12)

    selected_pred = pred_weight.index_select(0, expert_ids)
    selected_value = value_weight.index_select(0, expert_ids)
    selected_down = down_weight.index_select(0, expert_ids)
    pred = torch.bmm(routed_x[..., :r], selected_pred.transpose(1, 2))
    value = torch.bmm(routed_x[..., r:], selected_value.transpose(1, 2))
    hidden = F.gelu(pred) * value
    output = torch.bmm(hidden, selected_down.transpose(1, 2))
    output = output * route_scale.to(output.dtype).unsqueeze(-1)
    return (
        output.reshape(bsz, seq_len, d_model),
        hidden,
        pred,
        expert_ids,
        router_probability,
    )


def _compiled_sequence_expert():
    global _COMPILED_SEQUENCE_EXPERT
    if _COMPILED_SEQUENCE_EXPERT is None:
        os.environ.setdefault(
            "TRITON_CACHE_DIR", os.path.join(tempfile.gettempdir(), "fast_sp_triton"),
        )
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR",
            os.path.join(tempfile.gettempdir(), "fast_sp_torchinductor"),
        )
        _COMPILED_SEQUENCE_EXPERT = torch.compile(
            _sequence_expert_math, dynamic=False,
        )
    return _COMPILED_SEQUENCE_EXPERT


def _normal_quantile(prob, device):
    prob = torch.as_tensor(prob, device=device, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
    if hasattr(torch.special, "ndtri"):
        return torch.special.ndtri(prob)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
    )
    return normal.icdf(prob)


def statistical_topk_soft(x, k):
    d = x.size(-1)
    keep = min(max(1, int(k)), d)
    if keep >= d:
        return x
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    std = xf.std(dim=-1, keepdim=True, unbiased=True).clamp_min(1e-6)
    q = _normal_quantile(1.0 - keep / d, x.device).to(xf.dtype)
    theta = mean + std * q
    return F.relu(xf - theta).to(x.dtype)


def _spark_k(ffn_cfg, d_ff):
    if "k" in ffn_cfg:
        return int(ffn_cfg["k"])
    return max(1, int(round(d_ff * ffn_cfg.get("k_ratio", 0.08))))


def _spark_r(ffn_cfg, d_model):
    if "r" in ffn_cfg:
        return int(ffn_cfg["r"])
    return max(1, min(d_model - 1, int(round(d_model * ffn_cfg.get("r_ratio", 0.5)))))


class StdFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.up = nn.Linear(cfg["d_model"], cfg["d_ff"], bias=False)
        self.down = nn.Linear(cfg["d_ff"], cfg["d_model"], bias=False)

    def forward(self, x, collect_stats=False):
        h = F.gelu(self.up(x))
        y = self.down(h)
        if not collect_stats:
            return y
        stats = {
            "l1_act": h.abs().mean(),
            "act_l0": (h != 0).float().sum(dim=-1).mean(),
            "act_density": (h != 0).float().mean(),
        }
        return y, stats


class GroupedFFN(nn.Module):
    def __init__(self, cfg, ffn_cfg):
        super().__init__()
        self.d_model = cfg["d_model"]
        self.d_ff = cfg["d_ff"]
        self.groups = ffn_cfg.get("groups", 4)
        self.alpha = ffn_cfg.get("mix_alpha", 0.25)
        if self.d_model % self.groups != 0 or self.d_ff % self.groups != 0:
            raise ValueError("d_model and d_ff must be divisible by ffn.groups")
        self.d_model_g = self.d_model // self.groups
        self.d_ff_g = self.d_ff // self.groups
        self.up = nn.ModuleList([nn.Linear(self.d_model_g, self.d_ff_g, bias=False) for _ in range(self.groups)])
        self.down = nn.ModuleList([nn.Linear(self.d_ff_g, self.d_model_g, bias=False) for _ in range(self.groups)])
        mix_ratio = ffn_cfg.get("mix_ratio", 0.125)
        mix_d_ff = ffn_cfg.get("mix_d_ff", max(1, int(self.d_ff * mix_ratio)))
        self.mix_up = nn.Linear(self.d_model, mix_d_ff, bias=False)
        self.mix_down = nn.Linear(mix_d_ff, self.d_model, bias=False)

    def forward(self, x, collect_stats=False):
        xs = x.split(self.d_model_g, dim=-1)
        ys = []
        hs = []
        for group_x, up, down in zip(xs, self.up, self.down):
            h = F.gelu(up(group_x))
            hs.append(h)
            ys.append(down(h))
        block_y = torch.cat(ys, dim=-1)
        mix_h = F.gelu(self.mix_up(x))
        y = block_y + self.alpha * self.mix_down(mix_h)
        if not collect_stats:
            return y
        h_all = torch.cat(hs, dim=-1)
        group_means = torch.stack([h.abs().mean() for h in hs])
        stats = {
            "l1_act": h_all.abs().mean(),
            "act_l0": (h_all != 0).float().sum(dim=-1).mean(),
            "act_density": (h_all != 0).float().mean(),
            "group_utilization_std": group_means.std(unbiased=False),
        }
        return y, stats


class SparseFFN(nn.Module):
    def __init__(self, cfg, ffn_cfg):
        super().__init__()
        self.up = nn.Linear(cfg["d_model"], cfg["d_ff"], bias=False)
        self.down = nn.Linear(cfg["d_ff"], cfg["d_model"], bias=False)
        self.mode = ffn_cfg.get("mode", "topk")
        self.k = ffn_cfg.get("k", max(1, cfg["d_ff"] // 4))
        self.threshold = ffn_cfg.get("threshold", 0.0)

    def forward(self, x, collect_stats=False):
        h = F.gelu(self.up(x))
        if self.mode == "topk":
            keep = min(self.k, h.size(-1))
            _, idx = torch.topk(h, keep, dim=-1)
            mask = torch.zeros_like(h, dtype=torch.bool).scatter_(-1, idx, True)
            h_sparse = h.masked_fill(~mask, 0.0)
        elif self.mode == "threshold":
            h_sparse = h.masked_fill(h <= self.threshold, 0.0)
        else:
            raise ValueError(f"unknown sparse ffn mode: {self.mode}")
        y = self.down(h_sparse)
        if not collect_stats:
            return y
        active = (h_sparse != 0).float()
        stats = {
            "l1_act": h_sparse.abs().mean(),
            "act_l0": active.sum(dim=-1).mean(),
            "act_density": active.mean(),
            "pre_sparse_l1": h.abs().mean(),
        }
        return y, stats


class FixedMaskFFN(nn.Module):
    """Parameter-matched non-adaptive sparsity control.

    The same randomly selected hidden units are retained for every token. This
    separates benefits of conditional TopK routing from merely reducing the
    number of active FFN units.
    """
    def __init__(self, cfg, ffn_cfg):
        super().__init__()
        self.up = nn.Linear(cfg["d_model"], cfg["d_ff"], bias=False)
        self.down = nn.Linear(cfg["d_ff"], cfg["d_model"], bias=False)
        keep = min(max(1, int(ffn_cfg.get("k", cfg["d_ff"] // 4))), cfg["d_ff"])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(ffn_cfg.get("mask_seed", 1729)))
        chosen = torch.randperm(cfg["d_ff"], generator=generator)[:keep]
        mask = torch.zeros(cfg["d_ff"], dtype=torch.bool)
        mask[chosen] = True
        self.register_buffer("mask", mask, persistent=True)
        self.scale_output = bool(ffn_cfg.get("scale_output", False))

    def forward(self, x, collect_stats=False):
        h = F.gelu(self.up(x))
        h_sparse = h.masked_fill(~self.mask, 0.0)
        if self.scale_output:
            h_sparse = h_sparse * (h.size(-1) / int(self.mask.sum()))
        y = self.down(h_sparse)
        if not collect_stats:
            return y
        active = (h_sparse != 0).float()
        return y, {
            "l1_act": h_sparse.abs().mean(),
            "act_l0": active.sum(dim=-1).mean(),
            "act_density": active.mean(),
            "pre_sparse_l1": h.abs().mean(),
        }


class StructuredSparseFFN(nn.Module):
    def __init__(self, cfg, ffn_cfg):
        super().__init__()
        self.d_model = cfg["d_model"]
        self.d_ff = cfg["d_ff"]
        self.groups = ffn_cfg.get("groups", 8)
        self.active_groups = ffn_cfg.get("active_groups", 1)
        self.route = ffn_cfg.get("route", "first")
        self.scale_output = ffn_cfg.get("scale_output", True)
        if self.d_ff % self.groups != 0:
            raise ValueError("d_ff must be divisible by ffn.groups")
        if self.active_groups < 1 or self.active_groups > self.groups:
            raise ValueError("ffn.active_groups must be in [1, groups]")
        self.d_ff_g = self.d_ff // self.groups
        self.up = nn.ModuleList([nn.Linear(self.d_model, self.d_ff_g, bias=False) for _ in range(self.groups)])
        self.down = nn.ModuleList([nn.Linear(self.d_ff_g, self.d_model, bias=False) for _ in range(self.groups)])

    def _active_for_position(self, position):
        if self.route == "first":
            return list(range(self.active_groups))
        if self.route == "cycle":
            start = position % self.groups
            return [(start + i) % self.groups for i in range(self.active_groups)]
        raise ValueError(f"unknown structured sparse route: {self.route}")

    def forward(self, x, collect_stats=False):
        bsz, seq_len, _ = x.shape
        y = None
        h_abs_sum = x.new_tensor(0.0)
        group_counts = x.new_zeros(self.groups)
        total_active = 0

        if self.route == "first":
            active = self._active_for_position(0)
            for group in active:
                h = F.gelu(self.up[group](x))
                group_y = self.down[group](h)
                if y is None:
                    y = group_y.new_zeros(x.shape)
                y = y + group_y
                if collect_stats:
                    h_abs_sum = h_abs_sum + h.abs().sum()
                    group_counts[group] = group_counts[group] + bsz * seq_len
                    total_active += h.numel()
        elif self.route == "cycle":
            positions = torch.arange(seq_len, device=x.device)
            for group in range(self.groups):
                active_pos = ((group - positions).remainder(self.groups) < self.active_groups).nonzero(as_tuple=False).squeeze(-1)
                if active_pos.numel() == 0:
                    continue
                x_group = x.index_select(1, active_pos).reshape(-1, x.size(-1))
                h = F.gelu(self.up[group](x_group))
                group_y = self.down[group](h).view(bsz, active_pos.numel(), -1)
                if y is None:
                    y = group_y.new_zeros(x.shape)
                y.index_add_(1, active_pos, group_y)
                if collect_stats:
                    h_abs_sum = h_abs_sum + h.abs().sum()
                    group_counts[group] = group_counts[group] + bsz * active_pos.numel()
                    total_active += h.numel()
        else:
            raise ValueError(f"unknown structured sparse route: {self.route}")

        if y is None:
            y = torch.zeros_like(x)
        if self.scale_output:
            y = y * (self.groups / self.active_groups)
        if not collect_stats:
            return y

        usage = group_counts / max(1, bsz * seq_len * self.active_groups)
        stats = {
            "l1_act": h_abs_sum / max(1, total_active),
            "act_l0": x.new_tensor(float(self.active_groups * self.d_ff_g)),
            "act_density": x.new_tensor(float(self.active_groups / self.groups)),
            "group_utilization_std": usage.std(unbiased=False),
        }
        return y, stats


class SPFFN(nn.Module):
    def __init__(self, cfg, ffn_cfg):
        super().__init__()
        self.d_model = cfg["d_model"]
        self.d_ff = cfg["d_ff"]
        self.r = _spark_r(ffn_cfg, self.d_model)
        self.k = _spark_k(ffn_cfg, self.d_ff)
        self.pred = nn.Linear(self.r, self.d_ff, bias=False)
        self.value = nn.Linear(self.d_model - self.r, self.d_ff, bias=False)
        self.down = nn.Linear(self.d_ff, self.d_model, bias=False)

    def forward(self, x, collect_stats=False):
        x_pred = x[..., :self.r]
        x_value = x[..., self.r:]
        pred_score = self.pred(x_pred)
        gate = F.gelu(statistical_topk_soft(pred_score, self.k))
        value = self.value(x_value)
        h = gate * value
        y = self.down(h)
        if not collect_stats:
            return y
        active = (gate > 0).float()
        stats = {
            "l1_act": h.abs().mean(),
            "act_l0": active.sum(dim=-1).mean(),
            "act_density": active.mean(),
            "pre_sparse_l1": pred_score.abs().mean(),
        }
        return y, stats


class RoutedFastSPFFN(nn.Module):
    """Hardware-oriented Spark FFN with learned dynamic block routing.

    Unlike :class:`SPFFN`, this implementation does not calculate the value
    and down projections for all ``d_ff`` neurons.  The predictor first routes
    each token to a small number of contiguous neuron blocks, after which the
    two expensive projections are evaluated only for those blocks.  On recent
    CUDA devices the routed projections use PyTorch's grouped GEMM operator;
    the portable fallback keeps the same math for CPU tests and debugging.

    The original SPFFN is deliberately left unchanged so experiments can
    compare the mathematical reference implementation with this fast path.
    """

    def __init__(self, cfg, ffn_cfg):
        super().__init__()
        self.d_model = int(cfg["d_model"])
        self.d_ff = int(cfg["d_ff"])
        self.r = _spark_r(ffn_cfg, self.d_model)
        self.block_size = int(ffn_cfg.get("block_size", 32))
        if self.d_ff % self.block_size != 0:
            raise ValueError("d_ff must be divisible by fast_sp block_size")
        self.groups = self.d_ff // self.block_size
        default_active = max(1, int(round(self.groups * ffn_cfg.get("k_ratio", 0.08))))
        self.active_blocks = int(ffn_cfg.get("active_blocks", default_active))
        if not 1 <= self.active_blocks <= self.groups:
            raise ValueError("fast_sp active_blocks must be in [1, groups]")
        self.use_grouped_mm = bool(ffn_cfg.get("use_grouped_mm", True))

        # Parameter count matches SPFFN and the standard two-layer FFN: the
        # predictor/value matrices partition the input dimension.
        self.pred = nn.Linear(self.r, self.d_ff, bias=False)
        self.value = nn.Linear(self.d_model - self.r, self.d_ff, bias=False)
        self.down = nn.Linear(self.d_ff, self.d_model, bias=False)

    def _can_grouped_mm(self, tensor):
        if not self.use_grouped_mm or not tensor.is_cuda or not hasattr(F, "grouped_mm"):
            return False
        major, _ = torch.cuda.get_device_capability(tensor.device)
        return major >= 8

    def _portable_grouped_projection(self, x_value, gate, token_ids, block_ids):
        output = gate.new_zeros(token_ids.numel(), self.d_model)
        routed_hidden = gate.new_zeros(gate.shape)
        value_weight = self.value.weight.view(self.groups, self.block_size, -1)
        down_weight = self.down.weight.view(self.d_model, self.groups, self.block_size)
        for group in range(self.groups):
            positions = torch.where(block_ids == group)[0]
            if positions.numel() == 0:
                continue
            selected_x = x_value.index_select(0, token_ids.index_select(0, positions))
            value = F.linear(selected_x, value_weight[group])
            hidden = gate.index_select(0, positions) * value
            contribution = F.linear(hidden, down_weight[:, group, :])
            routed_hidden.index_copy_(0, positions, hidden)
            output.index_copy_(0, positions, contribution)
        return output, routed_hidden

    def forward(self, x, collect_stats=False):
        shape = x.shape
        flat = x.reshape(-1, self.d_model)
        x_pred = flat[:, :self.r]
        x_value = flat[:, self.r:]
        pred_score = self.pred(x_pred)
        pred_blocks = pred_score.view(flat.size(0), self.groups, self.block_size)

        # Exact top-k is cheap over 48 blocks (for d_ff=1536, block_size=32)
        # and gives a fixed amount of work to the grouped GEMMs.
        block_score = pred_blocks.mean(dim=-1)
        active_blocks = torch.topk(block_score, self.active_blocks, dim=-1).indices
        token_ids = torch.arange(flat.size(0), device=x.device).view(-1, 1)
        token_ids = token_ids.expand(-1, self.active_blocks).reshape(-1)
        block_ids = active_blocks.reshape(-1)

        # Group assignments by block so a single grouped-GEMM call can process
        # all routed tokens without materializing per-token weight tensors.
        block_ids, order = torch.sort(block_ids, stable=True)
        token_ids = token_ids.index_select(0, order)
        unique_blocks, counts = torch.unique_consecutive(block_ids, return_counts=True)
        offsets = counts.cumsum(0).to(torch.int32)
        gate = F.gelu(pred_blocks[token_ids, block_ids])

        if self._can_grouped_mm(pred_score):
            # grouped_mm currently targets BF16 on SM80+; casts remain
            # differentiable so FP32 master weights still receive gradients.
            compute_dtype = torch.bfloat16
            sorted_x = x_value.index_select(0, token_ids).to(compute_dtype)
            value_weight = self.value.weight.view(self.groups, self.block_size, -1)
            value_weight = value_weight.transpose(1, 2).contiguous().index_select(0, unique_blocks)
            value = F.grouped_mm(
                sorted_x,
                value_weight.to(compute_dtype),
                offs=offsets,
            )
            hidden = gate.to(compute_dtype) * value
            down_weight = self.down.weight.view(self.d_model, self.groups, self.block_size)
            down_weight = down_weight.permute(1, 2, 0).contiguous().index_select(0, unique_blocks)
            contribution = F.grouped_mm(
                hidden,
                down_weight.to(compute_dtype),
                offs=offsets,
            )
        else:
            contribution, hidden = self._portable_grouped_projection(
                x_value, gate, token_ids, block_ids,
            )

        output = contribution.new_zeros(flat.size(0), self.d_model)
        output.index_add_(0, token_ids, contribution)
        y = output.view(*shape[:-1], self.d_model)
        if not collect_stats:
            return y

        usage = torch.bincount(block_ids, minlength=self.groups).to(torch.float32)
        usage = usage / max(1, token_ids.numel())
        stats = {
            "l1_act": hidden.abs().mean(),
            "act_l0": x.new_tensor(float(self.active_blocks * self.block_size)),
            "act_density": x.new_tensor(float(self.active_blocks / self.groups)),
            "pre_sparse_l1": pred_score.abs().mean(),
            "group_utilization_std": usage.std(unbiased=False),
        }
        return y, stats


class SequenceExpertFastSPFFN(nn.Module):
    """Causal sequence-routed Spark FFN built from large dense GEMMs.

    The exact token-routed implementation above skips arithmetic but pays a
    large sorting/grouped-kernel launch cost on consumer GPUs. This fast path
    stores the same full bank of predictor/value/down parameters as SPFFN and
    routes each causal token chunk to one contiguous expert. Only the selected
    expert is materialized, so forward and backward use three regular batched
    GEMMs at the active width.

    Routing a chunk from its first token is causal: every token in that chunk
    can depend on the routing token without observing the future. A
    straight-through route scale gives the small router a useful gradient
    while leaving the forward scale exactly one.
    """

    def __init__(self, cfg, ffn_cfg):
        super().__init__()
        self.d_model = int(cfg["d_model"])
        self.d_ff = int(cfg["d_ff"])
        self.r = _spark_r(ffn_cfg, self.d_model)
        self.expert_width = int(ffn_cfg.get("expert_width", ffn_cfg.get("block_size", 128)))
        if self.d_ff % self.expert_width != 0:
            raise ValueError("d_ff must be divisible by fast_sp expert_width")
        self.experts = self.d_ff // self.expert_width
        self.route_block_size = int(ffn_cfg.get("route_block_size", 128))
        self.router_jitter = float(ffn_cfg.get("router_jitter", 0.0))
        self.use_compiled = bool(ffn_cfg.get("use_compiled", True))

        self.router = nn.Linear(self.r, self.experts, bias=False)
        self.pred_weight = nn.Parameter(torch.empty(self.experts, self.expert_width, self.r))
        self.value_weight = nn.Parameter(torch.empty(
            self.experts, self.expert_width, self.d_model - self.r,
        ))
        self.down_weight = nn.Parameter(torch.empty(
            self.experts, self.d_model, self.expert_width,
        ))
        self._reset_expert_parameters()

    def _reset_expert_parameters(self):
        for weight, fan_in in (
            (self.pred_weight, self.r),
            (self.value_weight, self.d_model - self.r),
            (self.down_weight, self.expert_width),
        ):
            bound = fan_in ** -0.5
            nn.init.uniform_(weight, -bound, bound)

    def forward(self, x, collect_stats=False):
        bsz, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"expected d_model={self.d_model}, got {d_model}")
        if seq_len % self.route_block_size != 0:
            raise ValueError(
                f"FastSPFFN requires seq_len divisible by route_block_size, got "
                f"seq_len={seq_len}, route_block_size={self.route_block_size}"
            )

        global _SEQUENCE_EXPERT_COMPILE_DISABLED
        inputs = (
            x,
            self.router.weight,
            self.pred_weight,
            self.value_weight,
            self.down_weight,
            self.r,
            self.route_block_size,
            self.router_jitter,
            self.training,
        )
        if self.use_compiled and x.is_cuda and not _SEQUENCE_EXPERT_COMPILE_DISABLED:
            try:
                y, hidden, pred, expert_ids, router_probability = (
                    _compiled_sequence_expert()(*inputs)
                )
            except Exception as error:
                _SEQUENCE_EXPERT_COMPILE_DISABLED = True
                warnings.warn(
                    f"FastSPFFN compiled path unavailable; using eager fallback: {error}",
                    RuntimeWarning,
                )
                y, hidden, pred, expert_ids, router_probability = (
                    _sequence_expert_math(*inputs)
                )
        else:
            y, hidden, pred, expert_ids, router_probability = _sequence_expert_math(*inputs)
        if not collect_stats:
            return y

        usage = torch.bincount(expert_ids, minlength=self.experts).to(torch.float32)
        usage = usage / max(1, expert_ids.numel())
        entropy = -(router_probability * router_probability.clamp_min(1e-12).log()).sum(dim=-1)
        entropy = entropy / math.log(max(2, self.experts))
        stats = {
            "l1_act": hidden.abs().mean(),
            "act_l0": x.new_tensor(float(self.expert_width)),
            "act_density": x.new_tensor(float(self.expert_width / self.d_ff)),
            "pre_sparse_l1": pred.abs().mean(),
            "group_utilization_std": usage.std(unbiased=False),
            "router_entropy": entropy.mean().to(x.dtype),
        }
        return y, stats


class FastSPFFN(nn.Module):
    """Compact, GEMM-friendly Spark gate used by the Fast SP comparison.

    A fused up projection produces predictor and value channels together, then
    ``GELU(predictor) * value`` retains Spark's multiplicative activation before
    one compact down projection. The active width is physical rather than a
    zero mask over ``d_ff``, so forward, backward, and optimizer work all shrink
    together and no gather/scatter kernel is required.

    This path intentionally trades parameter capacity for wall-clock speed.
    SequenceExpertFastSPFFN preserves a full parameter bank, and
    RoutedFastSPFFN preserves exact token-dynamic routing, for ablations.
    """

    def __init__(self, cfg, ffn_cfg):
        super().__init__()
        self.d_model = int(cfg["d_model"])
        self.d_ff = int(cfg["d_ff"])
        self.active_width = int(ffn_cfg.get(
            "active_width", ffn_cfg.get("expert_width", round(self.d_ff * 0.08)),
        ))
        if not 1 <= self.active_width <= self.d_ff:
            raise ValueError("fast_sp active_width must be in [1, d_ff]")
        self.up = nn.Linear(self.d_model, 2 * self.active_width, bias=False)
        self.down = nn.Linear(self.active_width, self.d_model, bias=False)

    def forward(self, x, collect_stats=False):
        predictor, value = self.up(x).chunk(2, dim=-1)
        hidden = F.gelu(predictor) * value
        y = self.down(hidden)
        if not collect_stats:
            return y
        stats = {
            "l1_act": hidden.abs().mean(),
            "act_l0": x.new_tensor(float(self.active_width)),
            "act_density": x.new_tensor(float(self.active_width / self.d_ff)),
            "pre_sparse_l1": predictor.abs().mean(),
            "group_utilization_std": x.new_zeros(()),
        }
        return y, stats


def build_ffn(name, cfg, ffn_cfg=None):
    ffn_cfg = ffn_cfg or {}
    if name == "std":
        return StdFFN(cfg)
    if name in ("sp", "spark"):
        return SPFFN(cfg, ffn_cfg)
    if name in ("fast_sp", "fast_spark"):
        return FastSPFFN(cfg, ffn_cfg)
    if name in ("fast_sp_expert", "fast_spark_expert"):
        return SequenceExpertFastSPFFN(cfg, ffn_cfg)
    if name in ("fast_sp_token", "fast_spark_token"):
        return RoutedFastSPFFN(cfg, ffn_cfg)
    if name == "groupmix":
        return GroupedFFN(cfg, ffn_cfg)
    if name == "sparse":
        return SparseFFN(cfg, ffn_cfg)
    if name == "fixed_mask":
        return FixedMaskFFN(cfg, ffn_cfg)
    if name == "structured_sparse":
        return StructuredSparseFFN(cfg, ffn_cfg)
    raise ValueError(f"unknown ffn: {name}")
