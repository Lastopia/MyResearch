from __future__ import annotations

import torch
from torch import nn

from model.positional.base import PositionMethod, PositionOutput


class DAPEKerple(PositionMethod):
    """Official DAPE-v1 formulation with a Kerple base bias.

    Ported to this project's attention interface from
    github.com/chuanyang-zheng/dape at commit
    bde344a844f2bd1f498b2bac70240dcda41c50c1 (Apache-2.0).
    """

    def __init__(
        self,
        num_heads: int,
        *,
        mlp_width: int = 32,
        epsilon: float = 1e-2,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.epsilon = float(epsilon)
        self.bias_p = nn.Parameter(torch.rand(num_heads) * 2.0)
        self.bias_a = nn.Parameter(torch.rand(num_heads))
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_heads, mlp_width),
            nn.LeakyReLU(),
            nn.Linear(mlp_width, num_heads),
        )

    def _positive_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.bias_p.clamp_(min=self.epsilon)
            self.bias_a.clamp_(min=self.epsilon)
        return self.bias_p, self.bias_a

    def _output(
        self,
        content_logits: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> PositionOutput:
        distance = (
            query_positions[:, None] - key_positions[None, :]
        ).clamp_min(0)
        p, a = self._positive_parameters()
        base_bias = -p[None, :, None, None] * torch.log1p(
            a[None, :, None, None]
            * distance[None, None, :, :].to(content_logits.dtype)
        )
        base_bias = base_bias.to(content_logits.dtype).expand(
            content_logits.shape[0],
            -1,
            -1,
            -1,
        )
        features = torch.cat((content_logits, base_bias), dim=1)
        adaptive = self.mlp(features.permute(0, 2, 3, 1)).permute(
            0,
            3,
            1,
            2,
        )
        return PositionOutput(
            bias=base_bias + adaptive,
            context_distance=distance[None, None, :, :].expand(
                content_logits.shape[0],
                self.num_heads,
                -1,
                -1,
            ),
            aux={
                "base_bias": base_bias,
                "adaptive_bias": adaptive,
                "upstream_commit": (
                    "bde344a844f2bd1f498b2bac70240dcda41c50c1"
                ),
            },
        )

    def build_bias(
        self,
        content_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> PositionOutput:
        return self._output(content_logits, positions, positions)

    def build_incremental_bias(
        self,
        content_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
    ) -> tuple[PositionOutput, dict[str, torch.Tensor] | None]:
        return (
            self._output(
                content_logits,
                query_positions,
                key_positions,
            ),
            cache,
        )
