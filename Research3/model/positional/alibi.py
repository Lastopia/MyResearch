from __future__ import annotations

import math

import torch

from model.positional.base import PositionMethod, PositionOutput


def alibi_slopes(num_heads: int) -> torch.Tensor:
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")

    def slopes_power_of_two(heads: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(heads) - 3)))
        ratio = start
        return [start * ratio**index for index in range(heads)]

    if math.log2(num_heads).is_integer():
        values = slopes_power_of_two(num_heads)
    else:
        closest = 2 ** math.floor(math.log2(num_heads))
        values = slopes_power_of_two(closest)
        extension = slopes_power_of_two(2 * closest)[0::2]
        values.extend(extension[: num_heads - closest])
    return torch.tensor(values, dtype=torch.float32)


def build_alibi_bias(
    slopes: torch.Tensor,
    positions: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    pos = positions.to(device=device)
    distance = (pos[:, None] - pos[None, :]).clamp_min(0).to(dtype)
    return -slopes.to(device=device, dtype=dtype)[None, :, None, None] * distance[
        None, None, :, :
    ]


class ALiBi(PositionMethod):
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.register_buffer("slopes", alibi_slopes(num_heads), persistent=True)

    def build_bias(
        self,
        content_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> PositionOutput:
        bias = build_alibi_bias(
            self.slopes,
            positions,
            dtype=content_logits.dtype,
            device=content_logits.device,
        )
        return PositionOutput(
            bias=bias,
            aux={"slopes": self.slopes.detach()},
        )

    def build_incremental_bias(
        self,
        content_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
    ) -> tuple[PositionOutput, dict[str, torch.Tensor] | None]:
        distance = (
            query_positions[:, None] - key_positions[None, :]
        ).clamp_min(0)
        bias = -self.slopes.to(
            device=content_logits.device,
            dtype=content_logits.dtype,
        )[None, :, None, None] * distance.to(
            device=content_logits.device,
            dtype=content_logits.dtype,
        )[None, None, :, :]
        return (
            PositionOutput(
                bias=bias,
                aux={"slopes": self.slopes.detach()},
            ),
            cache,
        )
