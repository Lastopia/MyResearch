from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class PositionOutput:
    bias: torch.Tensor | None = None
    context_distance: torch.Tensor | None = None
    relevance_gate: torch.Tensor | None = None
    regularization_loss: torch.Tensor | None = None
    aux: dict[str, Any] = field(default_factory=dict)


class PositionMethod(nn.Module):
    def transform_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        inference_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return q, k

    def build_bias(
        self,
        content_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> PositionOutput:
        return PositionOutput()

    def build_cache(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor] | None:
        return None

    def build_incremental_bias(
        self,
        content_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
    ) -> tuple[PositionOutput, dict[str, torch.Tensor] | None]:
        if query_positions.numel() != key_positions.numel():
            raise NotImplementedError(
                f"{type(self).__name__} does not implement incremental bias"
            )
        output = self.build_bias(
            content_logits,
            hidden_states,
            query_positions,
        )
        return output, cache
