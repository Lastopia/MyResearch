from __future__ import annotations

import torch

from model.positional.base import PositionMethod, PositionOutput


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RoPE(PositionMethod):
    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def transform_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        inference_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = float(inference_scale or 1.0)
        if scale <= 0:
            raise ValueError("inference_scale must be positive")
        effective_positions = positions.to(self.inv_freq.dtype) / scale
        frequencies = torch.outer(effective_positions, self.inv_freq)
        embedding = torch.repeat_interleave(frequencies, 2, dim=-1)
        cos = embedding.cos().to(dtype=q.dtype, device=q.device)[None, None, :, :]
        sin = embedding.sin().to(dtype=q.dtype, device=q.device)[None, None, :, :]
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

    def build_incremental_bias(
        self,
        content_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
    ) -> tuple[PositionOutput, dict[str, torch.Tensor] | None]:
        return PositionOutput(), cache
