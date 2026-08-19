"""Decoder-only baselines and the FFN Concept Subspace Bus V2."""

from __future__ import annotations

import math
import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CONCEPT_NAMES = (
    "country",
    "france",
    "united_kingdom",
    "china",
    "japan",
    "unknown_country",
    "color",
    "red",
    "blue",
    "yellow",
    "green",
    "unknown_color",
)
CONCEPT_PARENTS = (-1, 0, 0, 0, 0, 0, -1, 6, 6, 6, 6, 6)
COUNTRY_CLASSES = ("none", "france", "united_kingdom", "china", "japan", "unknown")
COLOR_CLASSES = ("none", "red", "blue", "yellow", "green", "unknown")


@dataclass(frozen=True)
class ResearchModelConfig:
    vocab_size: int
    max_length: int
    method: str = "concept_bus_v2"
    num_layers: int = 4
    d_model: int = 128
    d_ff: int = 512
    num_heads: int = 4
    slot_dim: int = 64
    num_bus_slots: int = 2
    bus_heads: int = 4
    bus_layers: int = 1
    concept_residual_dim: int = 32
    dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    bias: bool = False
    keep_residual_attention: bool = True
    matched_ffn_width: int | None = None

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "num_heads": self.num_heads,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.slot_dim % self.bus_heads:
            raise ValueError("slot_dim must be divisible by bus_heads")
        if not 1 <= self.bus_layers <= self.num_layers:
            raise ValueError("bus_layers must be between one and num_layers")
        if self.method in CONCEPT_METHODS:
            if self.num_bus_slots != 2:
                raise ValueError(
                    "V2 uses exactly two parallel parent slots: country and color"
                )
            if self.bus_layers != 1:
                raise ValueError("V2 enables the concept lane only in the final layer")
            if not 1 <= self.concept_residual_dim < self.d_model:
                raise ValueError(
                    "concept_residual_dim must be between one and d_model - 1"
                )
            if self.dropout != 0.0:
                raise ValueError(
                    "concept methods require dropout=0 so in-forward "
                    "counterfactuals use the exact same deterministic path"
                )

    @property
    def effective_ffn_width(self) -> int:
        return self.matched_ffn_width or self.d_ff

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CONCEPT_METHODS = {"concept_projector", "concept_bus_v2"}
BUS_METHODS = set(CONCEPT_METHODS)
COMMUNICATION_METHODS = {"concept_bus_v2"}
STANDARD_METHODS = {
    "standard",
    "parameter_matched",
    "concept_aux",
}
ALL_METHODS = CONCEPT_METHODS | STANDARD_METHODS


@dataclass
class ResearchModelOutput:
    country_logits: Tensor
    color_logits: Tensor
    concept_logits: Tensor | None
    concept_probabilities: Tensor | None
    trace: dict[str, Tensor] | None
    projector_orthogonality_loss: Tensor | None = None
    country_swap_country_logits: Tensor | None = None
    country_swap_color_logits: Tensor | None = None
    color_swap_country_logits: Tensor | None = None
    color_swap_color_logits: Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, values: Tensor) -> Tensor:
        scale = values.float().pow(2).mean(dim=-1, keepdim=True)
        # Keep the reduction and reciprocal square root in fp32. The concept
        # lane receives bf16 tensors under autocast; down-casting ``scale``
        # before rsqrt needlessly weakens numerical stability.
        normalized = values.float() * torch.rsqrt(scale + self.eps)
        return normalized.to(values.dtype) * self.weight


def _rotate_half(values: Tensor) -> Tensor:
    first = values[..., 0::2]
    second = values[..., 1::2]
    return torch.stack((-second, first), dim=-1).flatten(-2)


class CausalAttention(nn.Module):
    def __init__(
        self,
        dimension: int,
        num_heads: int,
        dropout: float,
        rope_theta: float,
        bias: bool,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.num_heads = num_heads
        self.head_dim = dimension // num_heads
        self.rope_theta = rope_theta
        self.qkv = nn.Linear(dimension, 3 * dimension, bias=bias)
        self.output = nn.Linear(dimension, dimension, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.register_buffer("rope_cosine", torch.empty(0), persistent=False)
        self.register_buffer("rope_sine", torch.empty(0), persistent=False)

    def _apply_rope(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        # Cache grows monotonically for variable-length batches. It is a
        # non-persistent buffer: no parameters/checkpoints/results change.
        tokens = query.shape[-2]
        needs_refresh = (
            self.rope_cosine.device != query.device
            or self.rope_cosine.dtype != query.dtype
            or self.rope_cosine.ndim != 4
            or self.rope_cosine.shape[-2] < tokens
            or self.rope_cosine.shape[-1] != self.head_dim
        )
        if needs_refresh:
            if self.head_dim % 2:
                raise ValueError("RoPE head dimension must be even")
            positions = torch.arange(
                tokens, device=query.device, dtype=torch.float32
            )
            inverse = 1.0 / (
                self.rope_theta
                ** (
                    torch.arange(
                        0,
                        self.head_dim,
                        2,
                        device=query.device,
                        dtype=torch.float32,
                    )
                    / self.head_dim
                )
            )
            angles = torch.outer(positions, inverse).repeat_interleave(2, dim=-1)
            self.rope_cosine = angles.cos().to(query.dtype)[None, None, :, :]
            self.rope_sine = angles.sin().to(query.dtype)[None, None, :, :]
        cosine = self.rope_cosine[:, :, :tokens]
        sine = self.rope_sine[:, :, :tokens]
        return (
            query * cosine + _rotate_half(query) * sine,
            key * cosine + _rotate_half(key) * sine,
        )

    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None,
        *,
        attention_mode: str = "causal",
    ) -> Tensor:
        if attention_mode not in {"causal", "self"}:
            raise ValueError("attention_mode must be 'causal' or 'self'")
        batch, tokens, _ = hidden.shape

        if attention_mode == "self":
            # A diagonal attention mask has exactly one permitted key, hence
            # softmax=1 and the result is V at the same token. Keep all QKV/O
            # parameters registered for the matched control, but evaluate only
            # the mathematically effective V/O slices. This also avoids a T x T
            # mask and two dead matrix multiplications. Concept methods require
            # dropout=0.
            start = 2 * self.dimension
            value = F.linear(
                hidden,
                self.qkv.weight[start:],
                None if self.qkv.bias is None else self.qkv.bias[start:],
            )
            if attention_mask is not None:
                value = value * attention_mask[..., None].to(value.dtype)
            return self.output_dropout(self.output(value))

        query, key, value = self.qkv(hidden).chunk(3, dim=-1)

        def split(values: Tensor) -> Tensor:
            return values.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

        query, key, value = map(split, (query, key, value))
        query, key = self._apply_rope(query, key)
        dropout = self.attention_dropout.p if self.training else 0.0
        if attention_mask is None and attention_mode == "causal":
            # Lets CUDA select Flash/efficient SDPA without materializing T x T.
            context = F.scaled_dot_product_attention(
                query, key, value, dropout_p=dropout, is_causal=True
            )
        else:
            pattern = (
                torch.ones(
                    tokens, tokens, device=hidden.device, dtype=torch.bool
                ).tril()
                if attention_mode == "causal"
                else torch.eye(tokens, device=hidden.device, dtype=torch.bool)
            )
            allowed = pattern[None, None, :, :]
            if attention_mask is not None:
                allowed = allowed & attention_mask[:, None, None, :].bool()
            context = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=allowed,
                dropout_p=dropout,
                is_causal=False,
            )
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch, tokens, self.dimension)
        return self.output_dropout(self.output(context))


class StandardFFN(nn.Module):
    def __init__(self, dimension: int, hidden: int, dropout: float, bias: bool) -> None:
        super().__init__()
        self.up = nn.Linear(dimension, hidden, bias=bias)
        self.down = nn.Linear(hidden, dimension, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        return self.dropout(self.down(F.gelu(self.up(values))))


class ConceptProjector(nn.Module):
    """Learn two semantic low-rank subspaces from the complete FFN activation."""

    def __init__(self, config: ResearchModelConfig) -> None:
        super().__init__()
        self.num_slots = config.num_bus_slots
        self.slot_dim = config.slot_dim
        self.d_ff = config.d_ff
        self.projection = nn.Linear(
            config.d_ff,
            config.num_bus_slots * config.slot_dim,
            bias=config.bias,
        )

    def forward(
        self, intermediate: Tensor, *, compute_orthogonality: bool
    ) -> tuple[Tensor, Tensor]:
        batch, tokens, _ = intermediate.shape
        slots = self.projection(intermediate).reshape(
            batch, tokens, self.num_slots, self.slot_dim
        )
        if compute_orthogonality:
            bases = self.projection.weight.reshape(
                self.num_slots, self.slot_dim, self.d_ff
            )
            country = F.normalize(bases[0], dim=-1)
            color = F.normalize(bases[1], dim=-1)
            orthogonality = (
                torch.matmul(country, color.transpose(0, 1)).square().mean()
            )
        else:
            orthogonality = intermediate.new_zeros(())
        return slots, orthogonality


class BusTokenAttention(nn.Module):
    """No-bypass token mixer with causal or self-only connectivity."""

    def __init__(
        self, config: ResearchModelConfig, *, communication: bool
    ) -> None:
        super().__init__()
        self.communication = communication
        self.num_slots = config.num_bus_slots
        self.dimension = config.slot_dim
        self.norm = RMSNorm(config.slot_dim, config.norm_eps)
        self.attention = CausalAttention(
            config.slot_dim,
            config.bus_heads,
            config.dropout,
            config.rope_theta,
            config.bias,
        )

    def forward(self, slots: Tensor, attention_mask: Tensor | None) -> Tensor:
        batch, tokens, count, dimension = slots.shape
        flattened = slots.permute(0, 2, 1, 3).reshape(batch * count, tokens, dimension)
        mask = (
            None
            if attention_mask is None
            else attention_mask[:, None, :]
            .expand(batch, count, tokens)
            .reshape(batch * count, tokens)
        )
        # Deliberately omit an inner residual connection. Otherwise concept
        # readout can use ``flattened`` directly and learn to ignore the new
        # token-axis attention, reproducing the V1 bypass failure.
        updated = self.attention(
            self.norm(flattened),
            mask,
            attention_mode="causal" if self.communication else "self",
        )
        return updated.reshape(batch, count, tokens, dimension).permute(0, 2, 1, 3)


class GroupedConceptReadout(nn.Module):
    """Country and color are parallel, non-competing semantic groups."""

    def __init__(self, config: ResearchModelConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.slot_dim, config.norm_eps)
        self.country = nn.Linear(config.slot_dim, 6, bias=True)
        self.color = nn.Linear(config.slot_dim, 6, bias=True)
        self.write_codes = nn.Parameter(
            torch.empty(len(CONCEPT_NAMES), config.concept_residual_dim)
        )
        nn.init.normal_(self.write_codes, std=0.02)

    @staticmethod
    def intervene(
        probabilities: Tensor,
        *,
        intervention: str | None = None,
        zero_concepts: Iterable[int] = (),
    ) -> Tensor:
        selected = probabilities
        if intervention == "swap_country":
            selected = selected.clone()
            selected[..., [1, 2]] = selected[..., [2, 1]]
        elif intervention == "swap_color":
            selected = selected.clone()
            selected[..., [7, 8]] = selected[..., [8, 7]]
        if zero_concepts:
            mask = torch.ones(
                len(CONCEPT_NAMES),
                device=probabilities.device,
                dtype=probabilities.dtype,
            )
            mask[list(zero_concepts)] = 0
            selected = selected * mask
        return selected

    def logits_and_probabilities(self, states: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.norm(states)
        logits = torch.cat(
            (self.country(normalized[..., 0, :]), self.color(normalized[..., 1, :])),
            dim=-1,
        )
        return logits, logits.sigmoid()

    def write(self, probabilities: Tensor) -> Tensor:
        return torch.einsum("...n,nr->...r", probabilities, self.write_codes)


class ConceptSubspaceFFN(nn.Module):
    """Full FFN plus a small Bus-exclusive output subspace."""

    def __init__(self, config: ResearchModelConfig) -> None:
        super().__init__()
        self.config = config
        private_dim = config.d_model - config.concept_residual_dim
        self.up = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.private_down = nn.Linear(config.d_ff, private_dim, bias=config.bias)
        self.projector = ConceptProjector(config)
        # The projector control runs the same parameterized attention kernel
        # with a diagonal self-only mask. Therefore the V2 treatment changes
        # connectivity, not parameter count or projection depth.
        self.concept_attention = BusTokenAttention(
            config,
            communication=config.method in COMMUNICATION_METHODS,
        )
        self.concept_readout = GroupedConceptReadout(config)
        self.dropout = nn.Dropout(config.dropout)

    def _full_delta(self, private: Tensor, concept: Tensor) -> Tensor:
        return torch.cat((private, concept), dim=-1)

    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None,
        *,
        intervention: str | None = None,
        zero_concepts: Iterable[int] = (),
        collect_trace: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        intermediate = F.gelu(self.up(hidden))
        private_delta = self.private_down(intermediate)
        if intervention == "zero_bus" and not collect_trace:
            concept_delta = private_delta.new_zeros(
                *private_delta.shape[:-1], self.config.concept_residual_dim
            )
            return self.dropout(
                self._full_delta(private_delta, concept_delta)
            ), None
        projected, orthogonality = self.projector(
            intermediate, compute_orthogonality=collect_trace
        )
        bus = self.concept_attention(projected, attention_mask)
        if intervention == "zero_bus":
            bus = torch.zeros_like(bus)
        concept_logits, base_probabilities = (
            self.concept_readout.logits_and_probabilities(bus)
        )
        selected_probabilities = self.concept_readout.intervene(
            base_probabilities,
            intervention=intervention,
            zero_concepts=zero_concepts,
        )
        if intervention == "zero_bus":
            selected_probabilities = torch.zeros_like(selected_probabilities)
        selected_concept_delta = self.concept_readout.write(selected_probabilities)
        delta = self._full_delta(private_delta, selected_concept_delta)
        if not collect_trace:
            return self.dropout(delta), None
        trace = {
            "projected_slots": projected,
            "bus_states": bus,
            "concept_logits": concept_logits,
            "concept_probabilities": selected_probabilities,
            "concept_delta": selected_concept_delta,
            "projector_orthogonality_loss": orthogonality,
        }
        return self.dropout(delta), trace


class ResearchBlock(nn.Module):
    def __init__(self, config: ResearchModelConfig, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalAttention(
            config.d_model,
            config.num_heads,
            config.dropout,
            config.rope_theta,
            config.bias,
        )
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.uses_bus = (
            config.method in BUS_METHODS
            and layer_index >= config.num_layers - config.bus_layers
        )
        if self.uses_bus:
            self.ffn: nn.Module = ConceptSubspaceFFN(config)
        else:
            self.ffn = StandardFFN(
                config.d_model,
                config.effective_ffn_width,
                config.dropout,
                config.bias,
            )

    def forward(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None,
        *,
        intervention: str | None,
        zero_concepts: Iterable[int],
        collect_trace: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        if self.config.keep_residual_attention:
            hidden = hidden + self.attention(self.attention_norm(hidden), attention_mask)
        normalized = self.ffn_norm(hidden)
        if self.uses_bus:
            delta, trace = self.ffn(
                normalized,
                attention_mask,
                intervention=intervention,
                zero_concepts=zero_concepts,
                collect_trace=collect_trace,
            )
        else:
            delta, trace = self.ffn(normalized), None
        return hidden + delta, trace


class DualTagTransformer(nn.Module):
    def __init__(self, config: ResearchModelConfig) -> None:
        super().__init__()
        if config.method not in ALL_METHODS:
            raise ValueError(f"unknown method: {config.method}")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            ResearchBlock(config, index) for index in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.country_head = nn.Linear(config.d_model, len(COUNTRY_CLASSES), bias=False)
        self.color_head = nn.Linear(config.d_model, len(COLOR_CLASSES), bias=False)
        if config.method == "concept_aux":
            self.probe = nn.Linear(config.d_model, len(CONCEPT_NAMES), bias=False)
        else:
            self.probe = None
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        intervention: str | None = None,
        zero_concepts: Iterable[int] = (),
        return_trace: bool = False,
        compute_counterfactuals: bool = True,
    ) -> ResearchModelOutput:
        hidden = self.embedding_dropout(self.embedding(input_ids))
        last_trace = None
        collect_concepts = intervention != "zero_bus" or return_trace
        for block in self.blocks:
            hidden, trace = block(
                hidden,
                attention_mask,
                intervention=intervention,
                zero_concepts=zero_concepts,
                collect_trace=collect_concepts,
            )
            if trace is not None:
                last_trace = trace
        normalized_hidden = self.final_norm(hidden)
        final_indices = attention_mask.long().sum(dim=-1).sub(1).clamp_min(0)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        final_hidden = normalized_hidden[
            rows, final_indices
        ]
        concept_logits = concept_probabilities = None
        country_swap_country_logits = country_swap_color_logits = None
        color_swap_country_logits = color_swap_color_logits = None
        if last_trace is not None:
            concept_logits = last_trace["concept_logits"][
                rows, final_indices
            ]
            concept_probabilities = last_trace["concept_probabilities"][
                rows, final_indices
            ]
            if intervention is None and compute_counterfactuals:
                bus_ffn = next(
                    block.ffn for block in reversed(self.blocks) if block.uses_bus
                )
                readout = bus_ffn.concept_readout
                normal_delta = readout.write(concept_probabilities)
                country_delta = readout.write(
                    readout.intervene(
                        concept_probabilities, intervention="swap_country"
                    )
                )
                color_delta = readout.write(
                    readout.intervene(
                        concept_probabilities, intervention="swap_color"
                    )
                )
                private_zeros = hidden.new_zeros(
                    hidden.shape[0],
                    self.config.d_model - self.config.concept_residual_dim,
                )
                raw_final = hidden[rows, final_indices]
                country_swapped = self.final_norm(
                    raw_final
                    + torch.cat(
                        (private_zeros, country_delta - normal_delta), dim=-1
                    )
                )
                color_swapped = self.final_norm(
                    raw_final
                    + torch.cat(
                        (private_zeros, color_delta - normal_delta), dim=-1
                    )
                )
                country_swap_country_logits = self.country_head(country_swapped)
                country_swap_color_logits = self.color_head(country_swapped)
                color_swap_country_logits = self.country_head(color_swapped)
                color_swap_color_logits = self.color_head(color_swapped)
        elif self.probe is not None:
            concept_logits = self.probe(final_hidden)
            concept_probabilities = concept_logits.sigmoid()
        country_logits = self.country_head(final_hidden)
        color_logits = self.color_head(final_hidden)
        return ResearchModelOutput(
            country_logits=country_logits,
            color_logits=color_logits,
            concept_logits=concept_logits,
            concept_probabilities=concept_probabilities,
            trace=last_trace if return_trace else None,
            projector_orthogonality_loss=(
                last_trace["projector_orthogonality_loss"]
                if last_trace is not None
                else None
            ),
            country_swap_country_logits=country_swap_country_logits,
            country_swap_color_logits=country_swap_color_logits,
            color_swap_country_logits=color_swap_country_logits,
            color_swap_color_logits=color_swap_color_logits,
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def initialize_named_parameters(model: nn.Module, seed: int) -> None:
    """Give identically named tensors identical initial values across methods."""

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim == 1:
                if name.endswith("norm.weight") or ".norm.weight" in name:
                    parameter.fill_(1.0)
                else:
                    parameter.zero_()
            else:
                digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
                local_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
                generator = torch.Generator(device=parameter.device).manual_seed(local_seed)
                parameter.normal_(mean=0.0, std=0.02, generator=generator)


def _estimated_block_macs(config: ResearchModelConfig, sequence_length: int) -> int:
    tokens = sequence_length
    dimension = config.d_model
    attention = 4 * tokens * dimension**2 + 2 * tokens**2 * dimension
    standard_ffn = 2 * tokens * dimension * config.effective_ffn_width
    if config.method not in BUS_METHODS:
        return attention + standard_ffn
    bus = config.num_bus_slots
    width = config.slot_dim
    residual = config.concept_residual_dim
    private_ffn = tokens * config.d_ff * (2 * dimension - residual)
    projection = tokens * config.d_ff * bus * width
    # The self-only projector control evaluates only V and O because Q/K are
    # mathematically dead under a diagonal mask. The communicating treatment
    # uses the complete QKV+O operator.
    bus_attention = 2 * tokens * bus * width**2
    if config.method in COMMUNICATION_METHODS:
        bus_attention += 2 * tokens * bus * width**2
        bus_attention += 2 * bus * tokens**2 * width
    concepts = tokens * bus * 6 * width
    # Core forward cost. DualTag's two supervised swaps are now computed only
    # at the final summary token and are negligible/task-specific overhead.
    writeback = tokens * len(CONCEPT_NAMES) * residual
    return attention + private_ffn + projection + bus_attention + concepts + writeback


def estimated_model_macs(config: ResearchModelConfig, sequence_length: int) -> int:
    if config.method not in BUS_METHODS:
        return config.num_layers * _estimated_block_macs(config, sequence_length)
    standard = _replace(config, method="standard", matched_ffn_width=None)
    return (
        (config.num_layers - config.bus_layers)
        * _estimated_block_macs(standard, sequence_length)
        + config.bus_layers * _estimated_block_macs(config, sequence_length)
    )


def _replace(config: ResearchModelConfig, **changes: Any) -> ResearchModelConfig:
    values = config.to_dict()
    values.update(changes)
    return ResearchModelConfig(**values)


def resolve_matched_width(
    base: ResearchModelConfig,
    method: str,
    *,
    sequence_length: int,
) -> int:
    target_config = _replace(base, method="concept_bus_v2", matched_ffn_width=None)
    lower = max(8, base.d_model // 2)
    upper = base.d_ff * 4
    if method == "parameter_matched":
        # Width resolution needs shapes only. Meta modules avoid allocating
        # several temporary 124M-parameter CPU models before a formal run.
        with torch.device("meta"):
            target = parameter_count(DualTagTransformer(target_config))

        def candidate_parameters(width: int) -> int:
            candidate = _replace(base, method=method, matched_ffn_width=width)
            with torch.device("meta"):
                return parameter_count(DualTagTransformer(candidate))

        # A standard FFN's parameter count is affine in its hidden width.  Two
        # samples locate the optimum, avoiding construction of ~1000 separate
        # 38M-parameter models when the paper configuration is resolved.
        next_width = min(upper, lower + 1)
        candidate_lower = candidate_parameters(lower)
        candidate_next = candidate_parameters(next_width)
        slope = (candidate_next - candidate_lower) / max(1, next_width - lower)
        raw_estimate = (
            lower
            if slope <= 0.0
            else lower + (target - candidate_lower) / slope
        )
        candidates = sorted(
            {
                lower,
                upper,
                min(upper, max(lower, int(math.floor(raw_estimate)))),
                min(upper, max(lower, int(math.ceil(raw_estimate)))),
            }
        )

        def score(width: int) -> float:
            estimated_parameters = candidate_lower + slope * (width - lower)
            return abs(estimated_parameters - target)

    else:
        raise ValueError("matching is defined only for parameter_matched")
    return min(candidates, key=score)


def build_model(config: ResearchModelConfig) -> DualTagTransformer:
    if config.method == "parameter_matched" and config.matched_ffn_width is None:
        width = resolve_matched_width(
            config, config.method, sequence_length=config.max_length
        )
        config = _replace(config, matched_ffn_width=width)
    return DualTagTransformer(config)
