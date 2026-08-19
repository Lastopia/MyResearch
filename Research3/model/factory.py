from __future__ import annotations

from functools import partial
from typing import Any, Callable

import torch
from torch import nn

from model.positional import (
    ALiBi,
    CABLE,
    DAPEKerple,
    RACABLE,
    RACABLEStatic,
    RoPE,
)
from model.positional.base import PositionMethod
from model.transformer import CausalLanguageModel
from tools.seed import seed_everything


METHODS = {
    "rope",
    "alibi",
    "cable",
    "ra_cable",
    "ra_cable_lite",
    "ra_cable_static",
    "dape_kerple",
}


def _position_factory(cfg: dict[str, Any]) -> Callable[[int], PositionMethod]:
    model_cfg = cfg["model"]
    position_cfg = cfg["position"]
    method = str(cfg["run"]["method"])
    hidden_size = int(model_cfg["n_embd"])
    num_heads = int(model_cfg["n_head"])
    head_dim = hidden_size // num_heads

    if method == "rope":
        constructor = partial(
            RoPE, head_dim=head_dim, base=float(position_cfg["rope_base"])
        )
        return lambda _layer_index: constructor()
    if method == "alibi":
        constructor = partial(ALiBi, num_heads=num_heads)
        return lambda _layer_index: constructor()
    if method == "cable":
        constructor = partial(
            CABLE, hidden_size=hidden_size, num_heads=num_heads
        )
        return lambda _layer_index: constructor()
    if method == "ra_cable":
        constructor = partial(
            RACABLE,
            hidden_size=hidden_size,
            num_heads=num_heads,
            gate_bias=float(position_cfg["ra_gate_bias"]),
            sparsity_weight=float(position_cfg["ra_sparsity_weight"]),
        )
        return lambda _layer_index: constructor()
    if method == "ra_cable_lite":
        adaptive_start = int(model_cfg["n_layer"]) - int(
            position_cfg["ra_lite_layers"]
        )

        def build_lite_position(layer_index: int) -> PositionMethod:
            if layer_index < adaptive_start:
                return CABLE(hidden_size=hidden_size, num_heads=num_heads)
            return RACABLE(
                hidden_size=hidden_size,
                num_heads=num_heads,
                gate_bias=float(position_cfg["ra_gate_bias"]),
                sparsity_weight=float(position_cfg["ra_sparsity_weight"]),
            )

        return build_lite_position
    if method == "ra_cable_static":
        constructor = partial(
            RACABLEStatic,
            hidden_size=hidden_size,
            num_heads=num_heads,
            gate_bias=float(position_cfg["ra_gate_bias"]),
            sparsity_weight=float(position_cfg["ra_sparsity_weight"]),
        )
        return lambda _layer_index: constructor()
    if method == "dape_kerple":
        constructor = partial(
            DAPEKerple,
            num_heads=num_heads,
            mlp_width=int(position_cfg["dape_mlp_width"]),
            epsilon=float(position_cfg["dape_kerple_epsilon"]),
        )
        return lambda _layer_index: constructor()
    raise ValueError(f"Unknown method: {method}")


def _reset_shared_parameters(model: nn.Module, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".position." in name:
                continue
            if name.endswith("bias"):
                parameter.zero_()
            elif "norm" in name and name.endswith("weight"):
                parameter.fill_(1.0)
            else:
                values = torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=parameter.dtype,
                    device="cpu",
                )
                parameter.copy_(values.to(parameter.device) * 0.02)


def _reset_cable_base_parameters(model: nn.Module, seed: int) -> None:
    """Match the official N(0, 0.02) CABLE initialization in both methods."""
    for layer_index, block in enumerate(model.blocks):
        position = block.attention.position
        if not isinstance(position, CABLE):
            continue
        generator = torch.Generator(device="cpu").manual_seed(seed + layer_index)
        with torch.no_grad():
            for linear in (position.increment_proj, position.weight_proj):
                nn.init.normal_(
                    linear.weight,
                    mean=0.0,
                    std=0.02,
                    generator=generator,
                )
                if linear.bias is not None:
                    linear.bias.zero_()


def build_model(cfg: dict[str, Any]) -> CausalLanguageModel:
    method = str(cfg["run"]["method"])
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {sorted(METHODS)}")

    seed = int(cfg["run"]["seed"])
    seed_everything(seed)
    model_cfg = cfg["model"]
    model = CausalLanguageModel(
        vocab_size=int(cfg["data"]["vocab_size"]),
        n_layer=int(model_cfg["n_layer"]),
        n_head=int(model_cfg["n_head"]),
        n_embd=int(model_cfg["n_embd"]),
        ffn_dim=int(model_cfg["ffn_dim"]),
        dropout=float(model_cfg["dropout"]),
        bias=bool(model_cfg["bias"]),
        max_seq_len=int(model_cfg["max_seq_len"]),
        position_factory=_position_factory(cfg),
    )
    _reset_shared_parameters(model, seed)
    _reset_cable_base_parameters(model, seed + 10_000)
    return model
