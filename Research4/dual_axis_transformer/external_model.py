"""Task-generic classifiers and causal language models on the same backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

from .research_model import (
    ALL_METHODS,
    RMSNorm,
    ResearchBlock,
    ResearchModelConfig,
    resolve_matched_width,
)


class ByteTokenizer:
    """Fixed tokenizer: PAD/BOS/EOS plus all 256 bytes.

    It avoids training/test name leakage in CLUTRR and makes the external
    stages reproducible without a tokenizer package or downloaded vocabulary.
    """

    pad_id = 0
    bos_id = 1
    eos_id = 2
    vocab_size = 259

    def encode(self, text: str, max_length: int) -> list[int]:
        if max_length < 2:
            raise ValueError("max_length must be at least two")
        payload = list(text.encode("utf-8", errors="replace"))[: max_length - 2]
        return [self.bos_id, *(value + 3 for value in payload), self.eos_id]

    def encode_bytes(self, payload: bytes) -> Tensor:
        return torch.tensor([value + 3 for value in payload], dtype=torch.long)


@dataclass
class ClassifierOutput:
    logits: Tensor
    trace: dict[str, Tensor] | None


def resolve_external_config(config: ResearchModelConfig) -> ResearchModelConfig:
    if config.method != "parameter_matched":
        return config
    width = resolve_matched_width(
        config, config.method, sequence_length=config.max_length
    )
    values = config.to_dict()
    values["matched_ffn_width"] = width
    return ResearchModelConfig(**values)


class SequenceClassifierTransformer(nn.Module):
    def __init__(self, config: ResearchModelConfig, num_classes: int) -> None:
        super().__init__()
        if config.method not in ALL_METHODS:
            raise ValueError(f"unknown method: {config.method}")
        self.config = resolve_external_config(config)
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.blocks = nn.ModuleList(
            ResearchBlock(self.config, index)
            for index in range(self.config.num_layers)
        )
        self.final_norm = RMSNorm(self.config.d_model, self.config.norm_eps)
        self.head = nn.Linear(self.config.d_model, num_classes, bias=False)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        intervention: str | None = None,
        zero_concepts: Iterable[int] = (),
        return_trace: bool = False,
    ) -> ClassifierOutput:
        hidden = self.embedding(input_ids)
        last_trace = None
        for block in self.blocks:
            hidden, trace = block(
                hidden,
                attention_mask,
                intervention=intervention,
                zero_concepts=zero_concepts,
                collect_trace=return_trace,
            )
            if trace is not None:
                last_trace = trace
        hidden = self.final_norm(hidden)
        indices = attention_mask.long().sum(dim=-1).sub(1).clamp_min(0)
        pooled = hidden[torch.arange(len(hidden), device=hidden.device), indices]
        return ClassifierOutput(
            logits=self.head(pooled), trace=last_trace if return_trace else None
        )


@dataclass
class LanguageModelOutput:
    logits: Tensor
    trace: dict[str, Tensor] | None


class CausalLanguageModel(nn.Module):
    def __init__(self, config: ResearchModelConfig) -> None:
        super().__init__()
        if config.method not in ALL_METHODS:
            raise ValueError(f"unknown method: {config.method}")
        self.config = resolve_external_config(config)
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.blocks = nn.ModuleList(
            ResearchBlock(self.config, index)
            for index in range(self.config.num_layers)
        )
        self.final_norm = RMSNorm(self.config.d_model, self.config.norm_eps)
        self.lm_head = nn.Linear(
            self.config.d_model, self.config.vocab_size, bias=False
        )
        self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        *,
        intervention: str | None = None,
        return_trace: bool = False,
    ) -> LanguageModelOutput:
        hidden = self.embedding(input_ids)
        last_trace = None
        for block in self.blocks:
            hidden, trace = block(
                hidden,
                attention_mask,
                intervention=intervention,
                zero_concepts=(),
                collect_trace=return_trace,
            )
            if trace is not None:
                last_trace = trace
        logits = self.lm_head(self.final_norm(hidden))
        return LanguageModelOutput(
            logits=logits, trace=last_trace if return_trace else None
        )
