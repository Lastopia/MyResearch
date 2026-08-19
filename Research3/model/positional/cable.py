from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from model.positional.base import PositionMethod, PositionOutput


class CABLE(PositionMethod):
    """Causal CABLE6 bias from the official CABLE implementation.

    For each token and head, CABLE learns a non-negative path increment and a
    query-dependent scale. The causal positional bias is the negative
    accumulated path distance multiplied by that scale.
    """

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.increment_proj = nn.Linear(hidden_size, num_heads)
        self.weight_proj = nn.Linear(hidden_size, num_heads)

    def _coordinates(self, hidden_states: torch.Tensor) -> torch.Tensor:
        increments = F.relu(self.increment_proj(hidden_states))
        return increments.cumsum(dim=1).transpose(1, 2)

    def _output(
        self,
        content_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        cumulative: torch.Tensor,
        *,
        last_query_only: bool = False,
    ) -> PositionOutput:
        query_coordinate = cumulative[:, :, -1:] if last_query_only else cumulative
        distance = (
            query_coordinate.unsqueeze(-1) - cumulative.unsqueeze(-2)
        ).clamp_min(0)
        query_weight = F.softplus(self.weight_proj(hidden_states)).transpose(1, 2)
        bias = -query_weight.unsqueeze(-1) * distance
        return PositionOutput(
            bias=bias.to(content_logits.dtype),
            context_distance=distance,
            aux={
                "query_weight_mean": query_weight.detach().mean(),
            },
        )

    def build_bias(
        self,
        content_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> PositionOutput:
        return self._output(
            content_logits,
            hidden_states,
            self._coordinates(hidden_states),
        )

    def build_cache(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"cumulative": self._coordinates(hidden_states)}

    def build_incremental_bias(
        self,
        content_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
    ) -> tuple[PositionOutput, dict[str, torch.Tensor]]:
        if hidden_states.shape[1] != 1:
            raise ValueError("CABLE incremental decoding expects one query token")
        increment = F.relu(self.increment_proj(hidden_states)).transpose(1, 2)
        if cache is None:
            cumulative = increment
        else:
            past = cache["cumulative"]
            cumulative = torch.cat((past, past[:, :, -1:] + increment), dim=-1)
        output = self._output(
            content_logits,
            hidden_states,
            cumulative,
            last_query_only=True,
        )
        return output, {"cumulative": cumulative}
