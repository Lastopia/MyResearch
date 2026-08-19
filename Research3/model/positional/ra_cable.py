from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from model.positional.base import PositionOutput
from model.positional.cable import CABLE


class RACABLE(CABLE):
    """CABLE with a gate derived directly from QK content similarity.

    A high QK score produces a high gate value, so the CABLE distance penalty
    is reduced for that query-key pair:

        bias = cable_bias * (1 - sigmoid(scale * QK + gate_bias))

    The sparsity term prevents the trivial solution where every pair is exempt.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        gate_bias: float = -3.0,
        sparsity_weight: float = 1e-3,
    ) -> None:
        super().__init__(hidden_size, num_heads)
        self.log_gate_scale = nn.Parameter(torch.zeros(num_heads))
        self.gate_bias = nn.Parameter(
            torch.full((num_heads,), float(gate_bias))
        )
        self.sparsity_weight = float(sparsity_weight)

    def _apply_gate(
        self,
        content_logits: torch.Tensor,
        base: PositionOutput,
    ) -> PositionOutput:
        if base.bias is None:
            raise RuntimeError("RA-CABLE requires a CABLE base bias")
        gate_scale = F.softplus(self.log_gate_scale).to(
            content_logits.dtype
        )[None, :, None, None]
        gate = torch.sigmoid(
            content_logits * gate_scale
            + self.gate_bias.to(content_logits.dtype)[None, :, None, None]
        )
        bias = base.bias * (1.0 - gate)

        if gate.shape[-2] == gate.shape[-1]:
            causal = torch.tril(
                torch.ones(
                    gate.shape[-2],
                    gate.shape[-1],
                    dtype=torch.bool,
                    device=gate.device,
                )
            )[None, None, :, :]
            gate_mean = gate.masked_select(causal).mean()
        else:
            gate_mean = gate.mean()
        regularization = gate_mean * self.sparsity_weight
        return PositionOutput(
            bias=bias,
            context_distance=base.context_distance,
            relevance_gate=gate,
            regularization_loss=regularization,
            aux={
                **base.aux,
                "base_bias": base.bias,
                "gate_mean": gate_mean.detach(),
                "gate_scale_mean": gate_scale.detach().mean(),
            },
        )

    def build_bias(
        self,
        content_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> PositionOutput:
        base = super().build_bias(
            content_logits,
            hidden_states,
            positions,
            mask,
        )
        return self._apply_gate(content_logits, base)

    def build_incremental_bias(
        self,
        content_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
    ) -> tuple[PositionOutput, dict[str, torch.Tensor]]:
        base, new_cache = super().build_incremental_bias(
            content_logits,
            hidden_states,
            query_positions,
            key_positions,
            cache,
        )
        return self._apply_gate(content_logits, base), new_cache


class RACABLEStatic(RACABLE):
    """Parameter-matched ablation with a content-independent per-head gate."""

    def _apply_gate(
        self,
        content_logits: torch.Tensor,
        base: PositionOutput,
    ) -> PositionOutput:
        if base.bias is None:
            raise RuntimeError("RA-CABLE requires a CABLE base bias")
        gate_scale = F.softplus(self.log_gate_scale).to(
            content_logits.dtype
        )[None, :, None, None]
        static_logits = self.gate_bias.to(content_logits.dtype)[
            None, :, None, None
        ] + gate_scale * 0.0
        gate = torch.sigmoid(static_logits).expand_as(content_logits)
        bias = base.bias * (1.0 - gate)
        if gate.shape[-2] == gate.shape[-1]:
            causal = torch.tril(
                torch.ones(
                    gate.shape[-2],
                    gate.shape[-1],
                    dtype=torch.bool,
                    device=gate.device,
                )
            )[None, None, :, :]
            gate_mean = gate.masked_select(causal).mean()
        else:
            gate_mean = gate.mean()
        return PositionOutput(
            bias=bias,
            context_distance=base.context_distance,
            relevance_gate=gate,
            regularization_loss=gate_mean * self.sparsity_weight,
            aux={
                **base.aux,
                "base_bias": base.bias,
                "gate_mean": gate_mean.detach(),
                "gate_scale_mean": gate_scale.detach().mean(),
                "content_dependent_gate": False,
            },
        )
