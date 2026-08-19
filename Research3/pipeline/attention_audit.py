from __future__ import annotations

import hashlib
from typing import Any

import torch

from pipeline.data import (
    SPECIAL_TOKENS,
    RetrievalBatch,
    generate_nested_retrieval_batches,
    load_tokens,
)
from pipeline.train import load_pretrained_model
from tools.io import write_json
from tools.log import log_resources, stage_banner
from tools.metrics import (
    attention_radius,
    attention_sink_ratio_mask,
    bias_far_near_pair_violation_rate,
    bias_monotonic_violation_rate,
    context_adaptivity_score,
    false_exemption_rate,
    geometric_head_fit,
    mean_attention_distance,
    normalized_attention_distance,
    relevant_attention_advantage,
    relative_head_distance_drift,
    semantic_exemption_success_rate,
)
from tools.paths import audit_dir, large_dir
from tools.runtime import autocast_context, resolve_device, resolve_dtype


def _head_values(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().float().cpu().tolist()]


def _sample_head_distances(
    attention: torch.Tensor, query_fraction: float
) -> torch.Tensor:
    return torch.stack(
        [
            mean_attention_distance(
                attention[index : index + 1],
                query_fraction=query_fraction,
            )
            for index in range(attention.shape[0])
        ]
    )


def _mean_present(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record[key] for record in records if record.get(key) is not None]
    return sum(values) / len(values) if values else None


def _save_artifacts(
    cfg: dict[str, Any],
    *,
    length: int,
    layer: int,
    condition: str,
    artifacts: dict[str, Any],
) -> None:
    sample_limit = max(1, int(cfg["audit"]["artifact_sample_limit"]))
    fields = {
        "attention": "attention",
        "position_bias": "bias",
        "relevance_gate": "gates",
        "content_logits": "logits",
        "context_distance": "context_distance",
    }
    base_bias = artifacts.get("aux", {}).get("base_bias")
    if torch.is_tensor(base_bias):
        fields["aux.base_bias"] = "bias"
    for field, category in fields.items():
        if field == "aux.base_bias":
            tensor = base_bias
            filename_field = "base_bias"
        else:
            tensor = artifacts.get(field)
            filename_field = field
        if tensor is None:
            continue
        path = (
            large_dir(cfg, category)
            / (
                f"{condition}_length{length}_layer{layer}_"
                f"{filename_field}.pt"
            )
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor[:sample_limit].detach().cpu(), path)


def _artifact_record(
    artifacts: dict[str, Any],
    retrieval: RetrievalBatch | None,
    *,
    query_fraction: float,
    gate_threshold: float,
    sink_masks: dict[str, torch.Tensor],
) -> dict[str, Any]:
    layer_index = int(artifacts["layer"])
    attention = artifacts["attention"].float()
    bias = artifacts["position_bias"]
    gate = artifacts["relevance_gate"]
    mad = mean_attention_distance(attention, query_fraction)
    nad = normalized_attention_distance(attention, query_fraction)
    radius90 = attention_radius(attention, 0.9, query_fraction)
    per_sample_mad = _sample_head_distances(attention, query_fraction)
    content = artifacts["content_logits"].float()
    context_distance = artifacts.get("context_distance")
    if torch.is_tensor(context_distance):
        context_distance = context_distance.float()
    base_bias = artifacts.get("aux", {}).get("base_bias")
    sink_ratios = {
        name: attention_sink_ratio_mask(
            attention,
            mask,
            query_fraction=query_fraction,
        )
        for name, mask in sink_masks.items()
    }
    record: dict[str, Any] = {
        "layer": layer_index,
        "_weight": attention.shape[0],
        "_sample_mad": per_sample_mad.cpu(),
        "mad_by_head": _head_values(mad),
        "nad_by_head": _head_values(nad),
        "r90_by_head": _head_values(radius90),
        "mad_mean": float(mad.mean().item()),
        "nad_mean": float(nad.mean().item()),
        "r90_mean": float(radius90.mean().item()),
        "geometric_head_fit": geometric_head_fit(mad),
        "context_adaptivity_score": context_adaptivity_score(per_sample_mad),
        "relevant_attention_advantage": (
            relevant_attention_advantage(
                attention,
                retrieval.relevant_positions,
                retrieval.distractor_positions,
            )
            if retrieval is not None
            else None
        ),
        "attention_sink_ratio": sink_ratios.get("special_combined"),
        "attention_sink_bos_or_eos_ratio": sink_ratios.get("bos_or_eos"),
        "attention_sink_newline_ratio": sink_ratios.get("newline"),
        "attention_sink_separator_ratio": sink_ratios.get("separator"),
        "attention_sink_query_ratio": sink_ratios.get("query"),
        "content_logits_abs_mean": float(content.abs().mean().item()),
        "position_bias_abs_mean": None,
        "bias_to_content_ratio": None,
        "bias_monotonic_violation_rate": None,
        "bias_adjacent_monotonic_violation_rate": None,
        "semantic_exemption_success_rate": None,
        "false_exemption_rate": None,
        "gate_mean": None,
        "base_bias_abs_mean": None,
        "adaptive_minus_base_abs_mean": None,
        "base_bias_monotonic_violation_rate": None,
        "context_distance_abs_mean": (
            float(context_distance.abs().mean().item())
            if torch.is_tensor(context_distance)
            else None
        ),
        "context_distance_monotonic_violation_rate": (
            bias_monotonic_violation_rate(
                -context_distance,
                query_fraction=query_fraction,
            )
            if torch.is_tensor(context_distance)
            else None
        ),
    }
    if torch.is_tensor(base_bias):
        base_bias = base_bias.float()
        record.update(
            {
                "base_bias_abs_mean": float(base_bias.abs().mean().item()),
                "base_bias_monotonic_violation_rate": (
                    bias_monotonic_violation_rate(
                        base_bias,
                        query_fraction=query_fraction,
                    )
                ),
            }
        )
    if bias is not None:
        bias = bias.float()
        if bias.shape[0] == 1 and attention.shape[0] > 1:
            bias = bias.expand(attention.shape[0], -1, -1, -1)
        bias_abs_mean = float(bias.abs().mean().item())
        content_abs_mean = record["content_logits_abs_mean"]
        record.update(
            {
                "position_bias_abs_mean": bias_abs_mean,
                "bias_to_content_ratio": bias_abs_mean
                / max(content_abs_mean, 1e-12),
                "bias_monotonic_violation_rate": bias_monotonic_violation_rate(
                    bias,
                    query_fraction=query_fraction,
                ),
                "bias_adjacent_monotonic_violation_rate": (
                    bias_monotonic_violation_rate(
                        bias,
                        query_fraction=query_fraction,
                    )
                ),
                "semantic_exemption_success_rate": (
                    semantic_exemption_success_rate(
                        bias,
                        retrieval.relevant_positions,
                        retrieval.distractor_positions,
                    )
                    if retrieval is not None
                    else None
                ),
            }
        )
        record["bias_monotonic_violation_rate"] = (
            bias_far_near_pair_violation_rate(
                bias,
                retrieval.relevant_positions,
                retrieval.distractor_positions,
            )
            if retrieval is not None
            else None
        )
        if torch.is_tensor(base_bias):
            record["adaptive_minus_base_abs_mean"] = float(
                (bias - base_bias).abs().mean().item()
            )
    if gate is not None:
        gate = gate.float()
        record.update(
            {
                "false_exemption_rate": false_exemption_rate(
                    gate,
                    retrieval.irrelevant_mask,
                    threshold=gate_threshold,
                )
                if retrieval is not None
                and retrieval.irrelevant_mask is not None
                else None,
                "gate_mean": float(gate.mean().item()),
            }
        )
    return record


def _merge_layer_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot merge an empty record list")
    total_weight = sum(int(record["_weight"]) for record in records)

    def weighted_scalar(key: str) -> float | None:
        available = [
            (record.get(key), int(record["_weight"]))
            for record in records
            if record.get(key) is not None
        ]
        if not available:
            return None
        denominator = sum(weight for _, weight in available)
        return sum(float(value) * weight for value, weight in available) / denominator

    def weighted_heads(key: str) -> list[float]:
        tensors = [
            torch.tensor(record[key], dtype=torch.float32)
            * int(record["_weight"])
            for record in records
        ]
        return _head_values(sum(tensors) / total_weight)

    merged: dict[str, Any] = {"layer": int(records[0]["layer"])}
    for key in ("mad_by_head", "nad_by_head", "r90_by_head"):
        merged[key] = weighted_heads(key)
    for key in (
        "mad_mean",
        "nad_mean",
        "r90_mean",
        "relevant_attention_advantage",
        "attention_sink_ratio",
        "attention_sink_bos_or_eos_ratio",
        "attention_sink_newline_ratio",
        "attention_sink_separator_ratio",
        "attention_sink_query_ratio",
        "content_logits_abs_mean",
        "position_bias_abs_mean",
        "bias_monotonic_violation_rate",
        "bias_adjacent_monotonic_violation_rate",
        "semantic_exemption_success_rate",
        "false_exemption_rate",
        "gate_mean",
        "base_bias_abs_mean",
        "adaptive_minus_base_abs_mean",
        "base_bias_monotonic_violation_rate",
        "context_distance_abs_mean",
        "context_distance_monotonic_violation_rate",
    ):
        merged[key] = weighted_scalar(key)
    merged["bias_to_content_ratio"] = (
        merged["position_bias_abs_mean"]
        / max(merged["content_logits_abs_mean"], 1e-12)
        if merged["position_bias_abs_mean"] is not None
        else None
    )
    mad = torch.tensor(merged["mad_by_head"])
    merged["geometric_head_fit"] = geometric_head_fit(mad)
    merged["context_adaptivity_score"] = context_adaptivity_score(
        torch.cat([record["_sample_mad"] for record in records], dim=0)
    )
    return merged


_SUMMARY_KEYS = (
    "mad_mean",
    "nad_mean",
    "r90_mean",
    "geometric_head_fit",
    "context_adaptivity_score",
    "relevant_attention_advantage",
    "attention_sink_ratio",
    "attention_sink_bos_or_eos_ratio",
    "attention_sink_newline_ratio",
    "attention_sink_separator_ratio",
    "attention_sink_query_ratio",
    "bias_monotonic_violation_rate",
    "bias_adjacent_monotonic_violation_rate",
    "semantic_exemption_success_rate",
    "false_exemption_rate",
    "gate_mean",
    "base_bias_abs_mean",
    "adaptive_minus_base_abs_mean",
    "base_bias_monotonic_violation_rate",
    "context_distance_abs_mean",
    "context_distance_monotonic_violation_rate",
)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _natural_inputs(
    tokens: torch.Tensor,
    *,
    length: int,
    samples: int,
) -> torch.Tensor:
    if len(tokens) < length:
        raise ValueError(f"not enough natural tokens for audit length {length}")
    max_start = max(0, len(tokens) - length)
    starts = [
        round(index * max_start / max(1, samples - 1))
        for index in range(samples)
    ]
    return torch.stack(
        [
            tokens[start : start + length].to(torch.long).clone()
            for start in starts
        ]
    )


def _shuffle_natural_inputs(
    input_ids: torch.Tensor,
    *,
    protected_token_ids: set[int],
    seed: int,
) -> torch.Tensor:
    shuffled = input_ids.clone()
    for sample in range(shuffled.shape[0]):
        generator = torch.Generator().manual_seed(seed + sample)
        row = shuffled[sample]
        movable = torch.tensor(
            [
                index
                for index, token in enumerate(row.tolist())
                if int(token) not in protected_token_ids
            ],
            dtype=torch.long,
        )
        if movable.numel() <= 1:
            continue
        permutation = torch.randperm(movable.numel(), generator=generator)
        row[movable] = row[movable[permutation]].clone()
    return shuffled


def _sink_masks(
    cfg: dict[str, Any],
    input_ids: torch.Tensor,
    *,
    synthetic: bool,
) -> dict[str, torch.Tensor]:
    token_groups: dict[str, list[int]]
    if synthetic:
        token_groups = {
            "bos_or_eos": [SPECIAL_TOKENS["bos"]],
            "separator": [SPECIAL_TOKENS["separator"]],
            "query": [SPECIAL_TOKENS["query"]],
        }
    else:
        token_groups = {
            str(name): [int(token) for token in tokens]
            for name, tokens in cfg["audit"]
            .get("natural_sink_token_ids", {})
            .items()
        }
    masks: dict[str, torch.Tensor] = {}
    combined = torch.zeros_like(input_ids, dtype=torch.bool)
    for name, token_ids in token_groups.items():
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in token_ids:
            mask |= input_ids.eq(int(token_id))
        masks[name] = mask
        combined |= mask
    masks["special_combined"] = combined
    return masks


def _audit_condition(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    *,
    condition: str,
    input_ids: torch.Tensor,
    retrieval: RetrievalBatch | None,
    length: int,
    artifact_layers: list[int],
    batch_size: int,
    query_fraction: float,
    gate_threshold: float,
    device: torch.device,
    dtype: torch.dtype,
    synthetic: bool,
) -> dict[str, Any]:
    records_by_layer: dict[int, list[dict[str, Any]]] = {}
    for start in range(0, input_ids.shape[0], batch_size):
        end = start + batch_size
        retrieval_chunk = (
            retrieval.select(start, end).to(device)
            if retrieval is not None
            else None
        )
        chunk_input_ids = (
            retrieval_chunk.input_ids
            if retrieval_chunk is not None
            else input_ids[start:end].to(device)
        )
        sink_masks = _sink_masks(
            cfg,
            chunk_input_ids,
            synthetic=synthetic,
        )
        for requested_layer in artifact_layers:
            with autocast_context(device, dtype):
                output = model(
                    chunk_input_ids,
                    return_artifacts=True,
                    artifact_layers=[requested_layer],
                )
            if len(output["artifacts"]) != 1:
                raise RuntimeError(
                    "Expected one artifact layer, got "
                    f"{len(output['artifacts'])}"
                )
            artifacts = output["artifacts"][0]
            layer_index = int(artifacts["layer"])
            records_by_layer.setdefault(layer_index, []).append(
                _artifact_record(
                    artifacts,
                    retrieval_chunk,
                    query_fraction=query_fraction,
                    gate_threshold=gate_threshold,
                    sink_masks=sink_masks,
                )
            )
            if bool(cfg["audit"]["save_full_artifacts"]) and start == 0:
                _save_artifacts(
                    cfg,
                    length=length,
                    layer=layer_index,
                    condition=condition,
                    artifacts=artifacts,
                )
            del output
        del chunk_input_ids
        if retrieval_chunk is not None:
            del retrieval_chunk

    layer_results = [
        _merge_layer_records(records)
        for _, records in sorted(records_by_layer.items())
    ]
    return {
        "input_sha256": _tensor_sha256(input_ids),
        "samples": int(input_ids.shape[0]),
        "layers": layer_results,
        "summary": {
            key: _mean_present(layer_results, key)
            for key in _SUMMARY_KEYS
        },
    }


def _paired_condition_deltas(
    natural: dict[str, Any],
    shuffled: dict[str, Any],
) -> list[dict[str, Any]]:
    shuffled_by_layer = {
        int(layer["layer"]): layer for layer in shuffled["layers"]
    }
    keys = (
        "mad_mean",
        "nad_mean",
        "r90_mean",
        "context_adaptivity_score",
        "attention_sink_ratio",
    )
    result: list[dict[str, Any]] = []
    for natural_layer in natural["layers"]:
        layer_index = int(natural_layer["layer"])
        shuffled_layer = shuffled_by_layer[layer_index]
        row: dict[str, Any] = {"layer": layer_index}
        for key in keys:
            natural_value = natural_layer.get(key)
            shuffled_value = shuffled_layer.get(key)
            row[f"{key}_shuffled_minus_natural"] = (
                float(shuffled_value) - float(natural_value)
                if natural_value is not None and shuffled_value is not None
                else None
            )
        result.append(row)
    return result


@torch.no_grad()
def run(cfg: dict[str, Any]) -> dict[str, Any]:
    stage_banner("ATTENTION AUDIT", cfg=cfg)
    device = resolve_device(cfg)
    dtype = resolve_dtype(cfg)
    checkpoint_kind = str(cfg["audit"].get("checkpoint", "adapt")).lower()
    if checkpoint_kind not in {"pretrain", "adapt", "auto"}:
        raise ValueError("audit.checkpoint must be pretrain, adapt, or auto")
    model, checkpoint = load_pretrained_model(
        cfg,
        prefer_adapted=checkpoint_kind in {"adapt", "auto"},
        require_adapted=checkpoint_kind == "adapt",
    )
    model.eval()
    query_fraction = float(cfg["audit"]["query_fraction"])
    gate_threshold = float(cfg["audit"]["gate_threshold"])
    batch_size = max(1, int(cfg["audit"]["batch_size"]))
    layer_setting = cfg["audit"]["layers"]
    if layer_setting == "all":
        artifact_layers = list(range(int(cfg["model"]["n_layer"])))
    else:
        artifact_layers = [int(layer) for layer in layer_setting]
        invalid_layers = [
            layer
            for layer in artifact_layers
            if layer < 0 or layer >= int(cfg["model"]["n_layer"])
        ]
        if invalid_layers:
            raise ValueError(f"Invalid audit layers: {invalid_layers}")
    lengths = [
        int(length)
        for length in cfg["audit"]["lengths"]
        if int(length) <= int(cfg["model"]["max_seq_len"])
    ]
    result: dict[str, Any] = {
        "method": cfg["run"]["method"],
        "seed": cfg["run"]["seed"],
        "checkpoint": str(checkpoint),
        "checkpoint_kind": checkpoint_kind,
        "query_fraction": query_fraction,
        "batch_size": batch_size,
        "audit_layers": artifact_layers,
        "conditions": list(cfg["audit"].get("conditions", [])),
        "data_sources": {
            "natural": {
                "dataset": cfg["data"]["fineweb_dataset"],
                "config": cfg["data"]["fineweb_config"],
                "revision": cfg["data"]["fineweb_revision"],
                "local_split": "test",
                "program_generated": False,
            },
            "shuffled": {
                "source": "paired permutation of the natural condition",
                "program_generated": True,
                "claim_scope": "content-order control only",
            },
            "synthetic_remote_target": {
                "generator": "generate_nested_retrieval_batches",
                "generator_version": 2,
                "seed": int(cfg["data"]["seed"]) + 10_000,
                "program_generated": True,
                "claim_scope": "synthetic mechanism control only",
            },
        },
        "lengths": {},
    }
    requested_conditions = set(result["conditions"])
    supported_conditions = {
        "natural",
        "shuffled",
        "synthetic_remote_target",
    }
    unknown_conditions = requested_conditions - supported_conditions
    if unknown_conditions:
        raise ValueError(
            f"Unsupported audit conditions: {sorted(unknown_conditions)}"
        )
    natural_tokens = load_tokens(cfg, "test")
    natural_sink_ids = {
        int(token)
        for tokens in cfg["audit"]
        .get("natural_sink_token_ids", {})
        .values()
        for token in tokens
    }
    distances_by_condition: dict[str, dict[int, list[torch.Tensor]]] = {}
    nested_retrieval = generate_nested_retrieval_batches(
        samples=int(cfg["audit"]["samples"]),
        lengths=lengths,
        vocab_size=int(cfg["data"]["vocab_size"]),
        num_pairs=int(cfg["data"]["num_key_value_pairs"]),
        seed=int(cfg["data"]["seed"]) + 10_000,
    )

    for length in lengths:
        retrieval = nested_retrieval[length]
        natural_input_ids = _natural_inputs(
            natural_tokens,
            length=length,
            samples=int(cfg["audit"]["samples"]),
        )
        shuffled_input_ids = _shuffle_natural_inputs(
            natural_input_ids,
            protected_token_ids=natural_sink_ids,
            seed=int(cfg["data"]["seed"]) + 20_000 + length,
        )
        condition_results: dict[str, dict[str, Any]] = {}
        if "natural" in requested_conditions:
            condition_results["natural"] = _audit_condition(
                model,
                cfg,
                condition="natural",
                input_ids=natural_input_ids,
                retrieval=None,
                length=length,
                artifact_layers=artifact_layers,
                batch_size=batch_size,
                query_fraction=query_fraction,
                gate_threshold=gate_threshold,
                device=device,
                dtype=dtype,
                synthetic=False,
            )
        if "shuffled" in requested_conditions:
            condition_results["shuffled"] = _audit_condition(
                model,
                cfg,
                condition="shuffled",
                input_ids=shuffled_input_ids,
                retrieval=None,
                length=length,
                artifact_layers=artifact_layers,
                batch_size=batch_size,
                query_fraction=query_fraction,
                gate_threshold=gate_threshold,
                device=device,
                dtype=dtype,
                synthetic=False,
            )
        if "synthetic_remote_target" in requested_conditions:
            condition_results["synthetic_remote_target"] = _audit_condition(
                model,
                cfg,
                condition="synthetic_remote_target",
                input_ids=retrieval.input_ids,
                retrieval=retrieval,
                length=length,
                artifact_layers=artifact_layers,
                batch_size=batch_size,
                query_fraction=query_fraction,
                gate_threshold=gate_threshold,
                device=device,
                dtype=dtype,
                synthetic=True,
            )

        for condition, condition_result in condition_results.items():
            for record in condition_result["layers"]:
                distances_by_condition.setdefault(condition, {}).setdefault(
                    int(record["layer"]),
                    [],
                ).append(torch.tensor(record["mad_by_head"]))

        length_result: dict[str, Any] = {
            "conditions": condition_results,
        }
        if {"natural", "shuffled"} <= set(condition_results):
            length_result["paired_natural_shuffled_deltas"] = (
                _paired_condition_deltas(
                    condition_results["natural"],
                    condition_results["shuffled"],
                )
            )
        result["lengths"][str(length)] = length_result

    result["cross_length_by_condition"] = {}
    for condition, by_layer in sorted(distances_by_condition.items()):
        rows = [
            {
                "layer": layer_index,
                "relative_head_distance_drift": relative_head_distance_drift(
                    torch.stack(distance_rows)
                ),
            }
            for layer_index, distance_rows in sorted(by_layer.items())
        ]
        result["cross_length_by_condition"][condition] = {
            "layers": rows,
            "summary": {
                "relative_head_distance_drift": _mean_present(
                    rows,
                    "relative_head_distance_drift",
                )
            },
        }
    primary_cross_length = result["cross_length_by_condition"].get(
        "synthetic_remote_target"
    )
    if primary_cross_length is None and result["cross_length_by_condition"]:
        primary_cross_length = next(
            iter(result["cross_length_by_condition"].values())
        )
    result["cross_length"] = (
        primary_cross_length["layers"] if primary_cross_length else []
    )
    result["cross_length_summary"] = (
        primary_cross_length["summary"]
        if primary_cross_length
        else {"relative_head_distance_drift": None}
    )
    write_json(audit_dir(cfg) / "attention_audit.json", result)
    log_resources(cfg, "attention_audit", device=device)
    stage_banner("ATTENTION AUDIT", "DONE", cfg=cfg)
    return result
