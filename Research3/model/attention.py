from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from model.positional.base import PositionMethod


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        bias: bool,
        position: PositionMethod,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = float(dropout)
        self.position = position
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.output = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        return_artifacts: bool = False,
        inference_scale: float | None = None,
        past_cache: dict[str, Any] | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]:
        batch, length, channels = hidden_states.shape
        past_length = (
            int(past_cache["key"].shape[-2]) if past_cache is not None else 0
        )
        if past_cache is not None and length != 1:
            raise ValueError("Cached decoding currently expects one new token")
        qkv = self.qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(
                1, 2
            )

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        query_positions = torch.arange(
            past_length,
            past_length + length,
            device=hidden_states.device,
        )
        q, k = self.position.transform_qk(
            q,
            k,
            query_positions,
            inference_scale=inference_scale,
        )
        if past_cache is not None:
            k = torch.cat((past_cache["key"], k), dim=-2)
            v = torch.cat((past_cache["value"], v), dim=-2)
        key_positions = torch.arange(k.shape[-2], device=hidden_states.device)
        content_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if past_cache is None:
            position_output = self.position.build_bias(
                content_logits,
                hidden_states,
                query_positions,
            )
            position_cache = (
                self.position.build_cache(hidden_states) if use_cache else None
            )
        else:
            position_output, position_cache = self.position.build_incremental_bias(
                content_logits,
                hidden_states,
                query_positions,
                key_positions,
                past_cache.get("position"),
            )
        total_logits = content_logits
        if position_output.bias is not None:
            total_logits = total_logits + position_output.bias

        causal_mask = key_positions[None, :] > query_positions[:, None]
        total_logits = total_logits.masked_fill(
            causal_mask[None, None, :, :],
            torch.finfo(total_logits.dtype).min,
        )
        attention = F.softmax(total_logits, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch, length, channels)
        output = self.output(output)

        regularization = position_output.regularization_loss
        if regularization is None:
            regularization = output.new_zeros(())

        artifacts: dict[str, Any] | None = None
        if return_artifacts:
            artifacts = {
                "content_logits": content_logits.detach(),
                "position_bias": (
                    position_output.bias.detach()
                    if position_output.bias is not None
                    else None
                ),
                "total_logits": total_logits.detach(),
                "attention": attention.detach(),
                "context_distance": (
                    position_output.context_distance.detach()
                    if position_output.context_distance is not None
                    else None
                ),
                "relevance_gate": (
                    position_output.relevance_gate.detach()
                    if position_output.relevance_gate is not None
                    else None
                ),
                "aux": {
                    key: value.detach() if torch.is_tensor(value) else value
                    for key, value in position_output.aux.items()
                },
            }
        present = (
            {
                "key": k,
                "value": v,
                "position": position_cache,
            }
            if use_cache
            else None
        )
        return output, regularization, artifacts, present
