from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from model.attention import CausalSelfAttention
from model.positional.base import PositionMethod


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_dim: int, dropout: float, bias: bool) -> None:
        super().__init__()
        self.up = nn.Linear(hidden_size, ffn_dim, bias=bias)
        self.down = nn.Linear(ffn_dim, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.gelu(self.up(hidden_states), approximate="tanh")))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        bias: bool,
        position: PositionMethod,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = CausalSelfAttention(
            hidden_size,
            num_heads,
            dropout,
            bias,
            position,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, ffn_dim, dropout, bias)

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
        attention_output, regularization, artifacts, present = self.attention(
            self.attention_norm(hidden_states),
            return_artifacts=return_artifacts,
            inference_scale=inference_scale,
            past_cache=past_cache,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states, regularization, artifacts, present


class CausalLanguageModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        ffn_dim: int,
        dropout: float,
        bias: bool,
        max_seq_len: int,
        position_factory: Callable[[int], PositionMethod],
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_embd,
                    n_head,
                    ffn_dim,
                    dropout,
                    bias,
                    position_factory(layer_index),
                )
                for layer_index in range(n_layer)
            ]
        )
        self.final_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        return_artifacts: bool = False,
        inference_scale: float | None = None,
        past_key_values: list[dict[str, Any]] | None = None,
        use_cache: bool = False,
        artifact_layers: list[int] | None = None,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        past_length = (
            int(past_key_values[0]["key"].shape[-2])
            if past_key_values
            else 0
        )
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must contain one cache per layer")
        if input_ids.shape[1] + past_length > self.max_seq_len:
            raise ValueError(
                f"Sequence length {input_ids.shape[1] + past_length} exceeds max_seq_len "
                f"{self.max_seq_len}"
            )
        hidden_states = self.dropout(self.token_embedding(input_ids))
        regularization_terms: list[torch.Tensor] = []
        artifacts: list[dict[str, Any]] = []
        presents: list[dict[str, Any]] = []
        for layer_index, block in enumerate(self.blocks):
            layer_past = (
                past_key_values[layer_index]
                if past_key_values is not None
                else None
            )
            collect_layer = return_artifacts and (
                artifact_layers is None or layer_index in artifact_layers
            )
            hidden_states, regularization, block_artifacts, present = block(
                hidden_states,
                return_artifacts=collect_layer,
                inference_scale=inference_scale,
                past_cache=layer_past,
                use_cache=use_cache,
            )
            regularization_terms.append(regularization)
            if block_artifacts is not None:
                block_artifacts["layer"] = layer_index
                artifacts.append(block_artifacts)
            if present is not None:
                presents.append(present)

        logits = self.lm_head(self.final_norm(hidden_states))
        position_loss = (
            torch.stack(regularization_terms).mean()
            if regularization_terms
            else logits.new_zeros(())
        )
        lm_loss: torch.Tensor | None = None
        total_loss: torch.Tensor | None = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
            total_loss = lm_loss + position_loss
        return {
            "logits": logits,
            "loss": total_loss,
            "lm_loss": lm_loss,
            "position_loss": position_loss,
            "artifacts": artifacts if return_artifacts else None,
            "past_key_values": presents if use_cache else None,
        }
