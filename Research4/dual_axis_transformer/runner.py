"""One-run training, exact recovery, evaluation and causal auditing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .research_model import (
    BUS_METHODS,
    CONCEPT_METHODS,
    COLOR_CLASSES,
    CONCEPT_NAMES,
    COUNTRY_CLASSES,
    ResearchModelConfig,
    ResearchModelOutput,
    build_model,
    estimated_model_macs,
    initialize_named_parameters,
    parameter_count,
)
from .metrics import gini, mean_jaccard, multilabel_metrics, unknown_metrics
from .locking import RunLock
from .resources import detect_resources
from .synthetic_data import (
    DeterministicBatcher,
    SyntheticDataBundle,
    ensure_synthetic_dataset,
    iter_eval_batches,
    collate_examples,
    load_examples,
)
from .training_log import (
    CumulativeTrainingTimer,
    FixedWidthTrainingLogger,
    checkpoint_due,
    current_device_log_status,
)
from .storage import prepare_run_checkpoint_dir, prepare_storage


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_run_config(
    cfg: dict[str, Any],
    size_name: str,
    method: str,
    seed: int,
    phase_name: str | None = None,
) -> dict[str, Any]:
    if size_name not in cfg["sizes"]:
        raise ValueError(f"unknown size: {size_name}")
    size = cfg["sizes"][size_name]
    phase: dict[str, Any] = {}
    if phase_name is not None:
        matches = [
            item for item in size.get("suite", []) if item.get("name") == phase_name
        ]
        if not matches:
            raise ValueError(f"unknown phase {phase_name!r} for size {size_name!r}")
        phase = matches[0]
        size = _deep_merge(size, phase)
    if method not in size["methods"]:
        raise ValueError(f"method {method!r} is not registered in size {size_name!r}")
    model = _deep_merge(cfg["model_defaults"], size["model"])
    return {
        "size": size_name,
        "phase": phase_name,
        "stage": size["stage"],
        "method": method,
        "seed": int(seed),
        "data": dict(size["data"]),
        "model": model,
        "train": dict(size["train"]),
        "loss": _deep_merge(cfg["loss"], size.get("loss", {})),
        "optimizer": dict(cfg["optimizer"]),
        "audit": dict(cfg.get("audit", {})),
        "logging": dict(cfg["logging"]),
        "run": dict(cfg["run"]),
    }


_SEED_LIST_PATTERN = re.compile(r'("seeds"\s*:\s*)\[[^\]]*\]')

# Reporting/UI/orchestration-only edits must not invalidate multi-day model
# results. These files define the learned parameters, batches, objectives,
# numerical kernels or adaptive effective-batch execution.
_SCIENTIFIC_SOURCE_FILES = (
    "data_download.py",
    "external_model.py",
    "external_tasks.py",
    "formal_data.py",
    "language_model.py",
    "metrics.py",
    "research_model.py",
    "resources.py",
    "runner.py",
    "synthetic_data.py",
)


def _normalize_cfg_for_fingerprint(payload: str) -> str:
    """Ignore orchestration-only seed arrays in the code fingerprint.

    The scalar seed is already part of every resolved run configuration.  This
    lets an existing seed remain reusable when the user later appends new seeds
    to the one CFG list, without hiding changes to model/data/loss settings.
    """

    return _SEED_LIST_PATTERN.sub(r'\1["<orchestration-only>"]', payload)


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    package = Path(__file__).resolve().parent
    project = package.parent
    paths = [
        project / "cfg.py",
        project / "requirements.txt",
        project / "pyproject.toml",
        *(package / name for name in _SCIENTIFIC_SOURCE_FILES),
    ]
    for path in paths:
        if not path.exists():
            continue
        digest.update(path.name.encode("utf-8"))
        content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if path.name == "cfg.py":
            normalized = _normalize_cfg_for_fingerprint(content)
            digest.update(normalized.encode("utf-8"))
        else:
            digest.update(content.encode("utf-8"))
    return digest.hexdigest()


def _semantic_hash(config: dict[str, Any], vocabulary_size: int) -> str:
    payload = {
        **config,
        "vocabulary_size": vocabulary_size,
        "source_fingerprint": _source_fingerprint(),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def _probe_device(device: torch.device) -> tuple[bool, str | None]:
    if device.type != "cuda":
        return True, None
    try:
        left = torch.randn(8, 8, device=device)
        value = float((left @ left).sum().item())
        del left
        torch.cuda.synchronize(device)
        if not math.isfinite(value):
            raise RuntimeError("CUDA probe produced a non-finite result")
        return True, None
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def select_device(required_dtype: str = "auto") -> tuple[torch.device, dict[str, Any]]:
    requested = os.environ.get("CONCEPT_BUS_DEVICE")
    if requested:
        candidate = torch.device(requested)
        usable, error = _probe_device(candidate)
        if usable:
            return candidate, {"requested": requested, "fallback_reason": None}
        if candidate.type == "cuda":
            raise RuntimeError(
                f"scheduled CUDA device {requested} failed its runtime probe: {error}"
            )
        return torch.device("cpu"), {"requested": requested, "fallback_reason": error}
    snapshot = detect_resources(required_dtype=required_dtype)
    usable_gpus = [gpu for gpu in snapshot.gpus if gpu.torch_usable]
    if usable_gpus:
        candidate = torch.device("cuda", usable_gpus[0].index)
        usable, error = _probe_device(candidate)
        if usable:
            return candidate, {
                "requested": "auto",
                "fallback_reason": None,
                "resource_snapshot": snapshot.to_dict(),
            }
        fallback = error
    else:
        fallback = "; ".join(
            gpu.probe_error or "unusable CUDA device" for gpu in snapshot.gpus
        ) or "no CUDA device detected"
    return torch.device("cpu"), {
        "requested": "auto",
        "fallback_reason": fallback,
        "resource_snapshot": snapshot.to_dict(),
    }


def _move(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()}


def _autocast(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


class SmoothMaxLossBalancer:
    """A resumable, scale-normalized soft maximum over training objectives."""

    def __init__(self, *, temperature: float, ema_decay: float) -> None:
        if temperature <= 0:
            raise ValueError("balance_temperature must be positive")
        if not 0 <= ema_decay < 1:
            raise ValueError("balance_ema_decay must be in [0, 1)")
        self.temperature = float(temperature)
        self.ema_decay = float(ema_decay)
        self.ema: dict[str, float] = {}

    def update_ema(self, values: dict[str, float]) -> None:
        for name, raw_value in values.items():
            value = float(raw_value)
            previous = self.ema.get(name, max(value, 1e-6))
            self.ema[name] = (
                self.ema_decay * previous + (1.0 - self.ema_decay) * value
            )

    def balance(
        self,
        losses: dict[str, Tensor],
        *,
        update_ema: bool = True,
    ) -> tuple[Tensor, dict[str, float]]:
        if not losses:
            raise ValueError("at least one objective is required")
        normalized = []
        names = list(losses)
        scales: dict[str, float] = {}
        for name in names:
            value = float(losses[name].detach())
            previous = self.ema.get(name, max(value, 1e-6))
            scale = max(previous, 1e-6)
            scales[name] = scale
            normalized.append(losses[name] / scale)
        stacked = torch.stack(normalized)
        temperature = self.temperature
        dimensionless = temperature * (
            torch.logsumexp(stacked / temperature, dim=0)
            - math.log(len(normalized))
        )
        # Restore the task-loss unit after balancing. In the one-objective
        # baseline this makes ``total`` exactly equal to the original task CE,
        # rather than silently changing the baseline to CE / EMA(CE).
        reference_scale = scales.get("task", scales[names[0]])
        total = reference_scale * dimensionless
        weights = torch.softmax(stacked.detach() / temperature, dim=0)
        diagnostics = {
            f"balance_weight_{name}": float(weight)
            for name, weight in zip(names, weights)
        }
        diagnostics.update(
            {f"normalized_{name}": float(value.detach()) for name, value in zip(names, normalized)}
        )
        diagnostics["balance_reference_scale"] = reference_scale
        if update_ema:
            self.update_ema(
                {name: float(losses[name].detach()) for name in names}
            )
        return total, diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "ema_decay": self.ema_decay,
            "ema": dict(self.ema),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.ema = {str(key): float(value) for key, value in state.get("ema", {}).items()}


def _losses(
    output: ResearchModelOutput,
    batch: dict[str, Tensor],
    config: dict[str, Any],
    method: str,
    balancer: SmoothMaxLossBalancer | None = None,
    *,
    update_balancer: bool = True,
) -> tuple[Tensor, dict[str, float]]:
    country = F.cross_entropy(output.country_logits, batch["country_targets"])
    color = F.cross_entropy(output.color_logits, batch["color_targets"])
    task = 0.5 * (country + color)
    concept = task.new_zeros(())
    if output.concept_logits is not None:
        targets = batch["concept_targets"]
        concept = F.binary_cross_entropy_with_logits(output.concept_logits, targets)
    causal = task.new_zeros(())
    if method in CONCEPT_METHODS:
        counterfactuals = (
            output.country_swap_country_logits,
            output.country_swap_color_logits,
            output.color_swap_country_logits,
            output.color_swap_color_logits,
        )
        if any(value is None for value in counterfactuals):
            raise RuntimeError("concept methods require in-forward counterfactual logits")
        expected_country = batch["country_targets"].clone()
        expected_country[batch["country_targets"] == 1] = 2
        expected_country[batch["country_targets"] == 2] = 1
        expected_color = batch["color_targets"].clone()
        expected_color[batch["color_targets"] == 1] = 2
        expected_color[batch["color_targets"] == 2] = 1
        causal = 0.25 * (
            F.cross_entropy(output.country_swap_country_logits, expected_country)
            + F.cross_entropy(output.country_swap_color_logits, batch["color_targets"])
            + F.cross_entropy(output.color_swap_country_logits, batch["country_targets"])
            + F.cross_entropy(output.color_swap_color_logits, expected_color)
        )
    orthogonality = (
        output.projector_orthogonality_loss
        if output.projector_orthogonality_loss is not None
        else task.new_zeros(())
    )
    objectives = {"task": task}
    if output.concept_logits is not None:
        objectives["concept"] = concept
    if method in CONCEPT_METHODS:
        objectives["causal"] = causal
    if balancer is None:
        total = torch.stack(list(objectives.values())).mean()
        balance_diagnostics: dict[str, float] = {}
    else:
        total, balance_diagnostics = balancer.balance(
            objectives, update_ema=update_balancer
        )
    total = total + float(config["loss"]["orthogonality_weight"]) * orthogonality
    components = {
        "task": float(task.detach()),
        "concept": float(concept.detach()),
        "causal": float(causal.detach()),
        "projector_orthogonality": float(orthogonality.detach()),
    }
    components.update(balance_diagnostics)
    return total, components


def _macro_f1(targets: Tensor, predictions: Tensor) -> float:
    scores = []
    for column in range(targets.shape[1]):
        truth = targets[:, column].bool()
        guess = predictions[:, column].bool()
        true_positive = int((truth & guess).sum())
        false_positive = int((~truth & guess).sum())
        false_negative = int((truth & ~guess).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / len(scores)


def _binary_auroc(targets: Tensor, scores: Tensor) -> float:
    targets = targets.bool().cpu()
    scores = scores.float().cpu()
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty(len(scores), dtype=torch.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(ranks[targets].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    examples: list[Any],
    bundle: SyntheticDataBundle,
    config: dict[str, Any],
    device: torch.device,
    *,
    intervention: str | None = None,
    zero_concepts: Iterable[int] = (),
    return_predictions: bool = False,
) -> dict[str, float] | tuple[dict[str, float], tuple[Tensor, Tensor]]:
    model.eval()
    total_loss = total_items = country_correct = color_correct = exact_correct = 0
    concept_targets: list[Tensor] = []
    concept_probabilities: list[Tensor] = []
    country_predictions: list[Tensor] = []
    color_predictions: list[Tensor] = []
    batch_size = int(
        config.get("runtime", {}).get(
            "micro_batch_size", config["train"]["batch_size"]
        )
    )
    for raw_batch in iter_eval_batches(
        examples,
        bundle.vocabulary,
        batch_size=batch_size,
        max_length=int(config["data"]["max_length"]),
    ):
        batch = _move(raw_batch, device)
        with _autocast(device):
            output = model(
                batch["input_ids"],
                batch["attention_mask"],
                intervention=intervention,
                zero_concepts=zero_concepts,
                compute_counterfactuals=False,
            )
            # Evaluation is deliberately task-only. Training auxiliaries and
            # regularizers must not make losses across methods incomparable.
            country_loss = F.cross_entropy(
                output.country_logits, batch["country_targets"]
            )
            color_loss = F.cross_entropy(
                output.color_logits, batch["color_targets"]
            )
            loss = 0.5 * (country_loss + color_loss)
        count = len(batch["input_ids"])
        country_prediction = output.country_logits.argmax(dim=-1)
        color_prediction = output.color_logits.argmax(dim=-1)
        if return_predictions:
            country_predictions.append(country_prediction.cpu())
            color_predictions.append(color_prediction.cpu())
        country_hit = country_prediction == batch["country_targets"]
        color_hit = color_prediction == batch["color_targets"]
        total_loss += float(loss) * count
        total_items += count
        country_correct += int(country_hit.sum())
        color_correct += int(color_hit.sum())
        exact_correct += int((country_hit & color_hit).sum())
        if output.concept_probabilities is not None:
            concept_targets.append(batch["concept_targets"].cpu())
            concept_probabilities.append(output.concept_probabilities.float().cpu())
    metrics = {
        "loss": total_loss / max(1, total_items),
        "country_accuracy": country_correct / max(1, total_items),
        "color_accuracy": color_correct / max(1, total_items),
        "exact_accuracy": exact_correct / max(1, total_items),
    }
    if concept_probabilities:
        targets = torch.cat(concept_targets)
        probabilities = torch.cat(concept_probabilities)
        metrics.update(multilabel_metrics(targets, probabilities))
        unknown_targets = torch.cat((targets[:, 5], targets[:, 11]))
        unknown_scores = torch.cat((probabilities[:, 5], probabilities[:, 11]))
        metrics.update(unknown_metrics(unknown_targets, unknown_scores))
    model.train()
    if return_predictions:
        return metrics, (
            torch.cat(country_predictions),
            torch.cat(color_predictions),
        )
    return metrics


@torch.no_grad()
def _predict_classes(
    model: nn.Module,
    examples: list[Any],
    bundle: SyntheticDataBundle,
    config: dict[str, Any],
    device: torch.device,
    *,
    intervention: str | None = None,
    zero_concepts: Iterable[int] = (),
) -> tuple[Tensor, Tensor]:
    model.eval()
    countries: list[Tensor] = []
    colors: list[Tensor] = []
    batch_size = int(
        config.get("runtime", {}).get(
            "micro_batch_size", config["train"]["batch_size"]
        )
    )
    for raw_batch in iter_eval_batches(
        examples,
        bundle.vocabulary,
        batch_size=batch_size,
        max_length=int(config["data"]["max_length"]),
    ):
        batch = _move(raw_batch, device)
        with _autocast(device):
            output = model(
                batch["input_ids"],
                batch["attention_mask"],
                intervention=intervention,
                zero_concepts=zero_concepts,
            )
        countries.append(output.country_logits.argmax(dim=-1).cpu())
        colors.append(output.color_logits.argmax(dim=-1).cpu())
    model.train()
    return torch.cat(countries), torch.cat(colors)


def _conditional_accuracy(
    predictions: Tensor, targets: Tensor, selected: Tensor
) -> float:
    count = int(selected.sum())
    return float((predictions[selected] == targets[selected]).float().mean()) if count else float("nan")


@torch.no_grad()
def _trace_statistics(
    model: nn.Module,
    examples: list[Any],
    bundle: SyntheticDataBundle,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    if not examples:
        return {}
    batch_size = min(
        int(config.get("runtime", {}).get("micro_batch_size", 64)), len(examples)
    )
    projected_rows: list[Tensor] = []
    bus_rows: list[Tensor] = []
    probability_rows: list[Tensor] = []
    orthogonality_rows: list[float] = []
    route_sets: list[set[int]] = []
    labels: list[tuple[int, int]] = []
    model.eval()
    for raw_batch in iter_eval_batches(
        examples,
        bundle.vocabulary,
        batch_size=batch_size,
        max_length=int(config["data"]["max_length"]),
    ):
        batch = _move(raw_batch, device)
        with _autocast(device):
            output = model(
                batch["input_ids"],
                batch["attention_mask"],
                return_trace=True,
                compute_counterfactuals=False,
            )
        if output.trace is None or output.concept_probabilities is None:
            continue
        final_indices = batch["attention_mask"].sum(dim=-1).long() - 1
        rows = torch.arange(len(final_indices), device=device)
        projected = output.trace["projected_slots"][
            rows, final_indices
        ].float().cpu()
        bus = output.trace["bus_states"][rows, final_indices].float().cpu()
        probabilities = output.concept_probabilities.float().cpu()
        projected_rows.append(projected)
        bus_rows.append(bus)
        probability_rows.append(probabilities)
        orthogonality_rows.append(
            float(output.trace["projector_orthogonality_loss"].float().cpu())
        )
        route_sets.extend(
            set(torch.nonzero(row >= 0.5).flatten().tolist()) for row in probabilities
        )
        labels.extend(
            zip(
                batch["country_targets"].cpu().tolist(),
                batch["color_targets"].cpu().tolist(),
            )
        )
    model.train()
    if not projected_rows:
        return {}
    projected = torch.cat(projected_rows)
    bus = torch.cat(bus_rows)
    probabilities = torch.cat(probability_rows)
    projector_energy = projected.square().mean(dim=(0, 2))
    normalized_bus = F.normalize(bus, dim=-1)
    similarities = torch.matmul(normalized_bus, normalized_bus.transpose(-2, -1))
    slot_count = similarities.shape[-1]
    off_diagonal = ~torch.eye(slot_count, dtype=torch.bool)[None]
    mean_slot_similarity = float(similarities[off_diagonal.expand_as(similarities)].mean())

    grouped_probabilities: dict[tuple[int, int], list[Tensor]] = {}
    grouped_routes: dict[tuple[int, int], list[set[int]]] = {}
    for label, probability, route in zip(labels, probabilities, route_sets):
        grouped_probabilities.setdefault(label, []).append(probability)
        grouped_routes.setdefault(label, []).append(route)
    context_stds = []
    route_jaccards = []
    for label, rows in grouped_probabilities.items():
        if len(rows) < 2:
            continue
        context_stds.append(float(torch.stack(rows).std(dim=0).mean()))
        routes = grouped_routes[label]
        route_jaccards.append(
            mean_jaccard(routes[1:], [routes[0]] * (len(routes) - 1))
        )
    return {
        "projector_group_energy_gini": gini(projector_energy),
        "projector_orthogonality": sum(orthogonality_rows)
        / len(orthogonality_rows),
        "bus_slot_cosine_similarity": mean_slot_similarity,
        "same_label_concept_std": sum(context_stds) / len(context_stds)
        if context_stds
        else float("nan"),
        "same_label_route_jaccard": sum(route_jaccards) / len(route_jaccards)
        if route_jaccards
        else float("nan"),
    }


def _causal_audit(
    model: nn.Module,
    examples: list[Any],
    bundle: SyntheticDataBundle,
    config: dict[str, Any],
    device: torch.device,
    full_metrics: dict[str, float],
) -> dict[str, float]:
    limit = int(config.get("audit", {}).get("max_examples", len(examples)))
    audited = examples[: min(limit, len(examples))]
    baseline_result = evaluate(
        model,
        audited,
        bundle,
        config,
        device,
        return_predictions=True,
    )
    baseline, (base_country, base_color) = baseline_result
    zero_bus = evaluate(
        model, audited, bundle, config, device, intervention="zero_bus"
    )
    country_zero = evaluate(
        model, audited, bundle, config, device, zero_concepts=range(0, 6)
    )
    color_zero = evaluate(
        model, audited, bundle, config, device, zero_concepts=range(6, 12)
    )
    swap_country, swap_country_color = _predict_classes(
        model, audited, bundle, config, device, intervention="swap_country"
    )
    swap_color_country, swap_color = _predict_classes(
        model, audited, bundle, config, device, intervention="swap_color"
    )
    country_targets = torch.tensor([item.country_target for item in audited])
    color_targets = torch.tensor([item.color_target for item in audited])
    country_selected = (country_targets == 1) | (country_targets == 2)
    color_selected = (color_targets == 1) | (color_targets == 2)
    expected_country = country_targets.clone()
    expected_country[country_targets == 1] = 2
    expected_country[country_targets == 2] = 1
    expected_color = color_targets.clone()
    expected_color[color_targets == 1] = 2
    expected_color[color_targets == 2] = 1
    audit = {
        "audit_examples": float(len(audited)),
        "zero_bus_exact_drop": baseline["exact_accuracy"] - zero_bus["exact_accuracy"],
        "country_zero_country_drop": baseline["country_accuracy"] - country_zero["country_accuracy"],
        "country_zero_color_side_effect": baseline["color_accuracy"] - country_zero["color_accuracy"],
        "color_zero_color_drop": baseline["color_accuracy"] - color_zero["color_accuracy"],
        "color_zero_country_side_effect": baseline["country_accuracy"] - color_zero["country_accuracy"],
        "country_keep_sufficiency": color_zero["country_accuracy"] / max(1e-9, baseline["country_accuracy"]),
        "color_keep_sufficiency": country_zero["color_accuracy"] / max(1e-9, baseline["color_accuracy"]),
        "country_counterfactual_success": _conditional_accuracy(
            swap_country, expected_country, country_selected
        ),
        "color_counterfactual_success": _conditional_accuracy(
            swap_color, expected_color, color_selected
        ),
        "country_counterfactual_change": float(
            (swap_country[country_selected] != base_country[country_selected]).float().mean()
        )
        if int(country_selected.sum())
        else float("nan"),
        "color_counterfactual_change": float(
            (swap_color[color_selected] != base_color[color_selected]).float().mean()
        )
        if int(color_selected.sum())
        else float("nan"),
        "country_counterfactual_color_side_effect": float(
            (swap_country_color != base_color).float().mean()
        ),
        "color_counterfactual_country_side_effect": float(
            (swap_color_country != base_country).float().mean()
        ),
    }
    for class_index, concept_index in enumerate(range(1, 6), start=1):
        selected = country_targets == class_index
        if int(selected.sum()):
            zero_prediction, _ = _predict_classes(
                model,
                audited,
                bundle,
                config,
                device,
                zero_concepts=(concept_index,),
            )
            before = _conditional_accuracy(base_country, country_targets, selected)
            after = _conditional_accuracy(zero_prediction, country_targets, selected)
            audit[f"node_necessity_{CONCEPT_NAMES[concept_index]}"] = before - after
    for class_index, concept_index in enumerate(range(7, 12), start=1):
        selected = color_targets == class_index
        if int(selected.sum()):
            _, zero_prediction = _predict_classes(
                model,
                audited,
                bundle,
                config,
                device,
                zero_concepts=(concept_index,),
            )
            before = _conditional_accuracy(base_color, color_targets, selected)
            after = _conditional_accuracy(zero_prediction, color_targets, selected)
            audit[f"node_necessity_{CONCEPT_NAMES[concept_index]}"] = before - after
    audit.update(_trace_statistics(model, audited, bundle, config, device))
    return audit


def _optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(config["train"]["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(config["train"]["learning_rate"]),
        betas=tuple(float(value) for value in config["optimizer"]["betas"]),
    )


def _scheduler(
    optimizer: torch.optim.Optimizer, config: dict[str, Any]
) -> torch.optim.lr_scheduler.LambdaLR:
    total = int(config["train"]["max_steps"])
    warmup = max(1, int(total * float(config["train"]["warmup_fraction"])))

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _save_checkpoint(
    run_dir: Path,
    *,
    checkpoint_dir: Path,
    name: str,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batcher: DeterministicBatcher,
    timer: CumulativeTrainingTimer,
    best_validation: float,
    config_hash: str,
    loss_balancer: SmoothMaxLossBalancer | None = None,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "batcher": batcher.state_dict(),
            "timer": timer.state_dict(),
            "rng": _rng_state(),
            "best_validation": best_validation,
            "config_hash": config_hash,
            "loss_balancer": (
                loss_balancer.state_dict() if loss_balancer is not None else None
            ),
        },
        temporary,
    )
    temporary.replace(path)
    latest = checkpoint_dir / "latest.json"
    latest.write_text(
        json.dumps({"path": path.name, "step": step}, indent=2), encoding="utf-8"
    )
    # A 38M AdamW checkpoint is hundreds of MB. Keep only the newest periodic
    # recovery point; after successful completion, ``final.pt`` supersedes it.
    if name.startswith("step_"):
        for stale in checkpoint_dir.glob("step_*.pt"):
            if stale != path:
                stale.unlink(missing_ok=True)
    elif name == "final.pt":
        for stale in checkpoint_dir.glob("step_*.pt"):
            stale.unlink(missing_ok=True)
    return path


def _load_latest(
    run_dir: Path,
    *,
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batcher: DeterministicBatcher,
    timer: CumulativeTrainingTimer,
    config_hash: str,
    device: torch.device,
    loss_balancer: SmoothMaxLossBalancer | None = None,
) -> tuple[int, float]:
    latest = checkpoint_dir / "latest.json"
    if not latest.exists():
        return 0, float("inf")
    metadata = json.loads(latest.read_text(encoding="utf-8"))
    checkpoint_path = latest.parent / metadata["path"]
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if state.get("config_hash") != config_hash:
        raise RuntimeError("checkpoint config hash does not match this run")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    batcher.load_state_dict(state["batcher"])
    timer.load_state_dict(state["timer"])
    if loss_balancer is not None and state.get("loss_balancer") is not None:
        loss_balancer.load_state_dict(state["loss_balancer"])
    _restore_rng(state["rng"])
    return int(state["step"]), float(state["best_validation"])


def _best_model_metadata(
    checkpoint_dir: Path, *, config_hash: str
) -> dict[str, Any] | None:
    metadata_path = checkpoint_dir / "best.json"
    model_path = checkpoint_dir / "best.pt"
    if not metadata_path.exists() or not model_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("config_hash") != config_hash:
        return None
    return metadata


def _save_best_model(
    *,
    checkpoint_dir: Path,
    model: nn.Module,
    step: int,
    validation_loss: float,
    config_hash: str,
) -> Path:
    """Atomically save the model selected by validation, separate from resume state."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / "best.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "step": int(step),
            "validation_loss": float(validation_loss),
            "config_hash": config_hash,
        },
        temporary,
    )
    temporary.replace(path)
    metadata_path = checkpoint_dir / "best.json"
    metadata_temporary = metadata_path.with_suffix(".json.tmp")
    metadata_temporary.write_text(
        json.dumps(
            {
                "path": path.name,
                "step": int(step),
                "validation_loss": float(validation_loss),
                "config_hash": config_hash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metadata_temporary.replace(metadata_path)
    return path


def _load_best_model(
    *,
    checkpoint_dir: Path,
    model: nn.Module,
    config_hash: str,
    device: torch.device,
) -> tuple[int, float]:
    metadata = _best_model_metadata(checkpoint_dir, config_hash=config_hash)
    if metadata is None:
        raise RuntimeError("no compatible best-validation model checkpoint exists")
    path = checkpoint_dir / str(metadata["path"])
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("config_hash") != config_hash:
        raise RuntimeError("best model config hash does not match this run")
    model.load_state_dict(state["model"])
    return int(state["step"]), float(state["validation_loss"])


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_one(
    cfg: dict[str, Any],
    *,
    size_name: str,
    phase_name: str | None = None,
    method: str,
    seed: int,
    output_root: str | Path,
) -> Path:
    output_root = Path(output_root).resolve()
    roots = prepare_storage(cfg, output_root)
    config = resolve_run_config(cfg, size_name, method, seed, phase_name)
    bundle = ensure_synthetic_dataset(roots.data, config["data"])
    global_batch_size = int(config["train"]["batch_size"])
    micro_batch_size = int(
        os.environ.get("CONCEPT_BUS_MICRO_BATCH", global_batch_size)
    )
    accumulation_steps = int(
        os.environ.get(
            "CONCEPT_BUS_GRAD_ACCUM",
            max(1, global_batch_size // micro_batch_size),
        )
    )
    if micro_batch_size * accumulation_steps != global_batch_size:
        raise RuntimeError(
            "micro_batch_size * accumulation_steps must equal the fixed global batch"
        )
    # Micro-batch partitioning is part of run identity. This is conservative
    # for nonlinear multi-objective balancing and prevents a checkpoint from
    # silently resuming under a different accumulation layout.
    config["runtime"] = {
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": global_batch_size,
    }
    config_hash = _semantic_hash(config, len(bundle.vocabulary.tokens))
    run_dir = (
        output_root
        / "runs"
        / config["stage"]
        / method
        / f"seed{seed}"
        / config_hash
        / f"attempt{int(config['run']['attempt'])}"
    )
    checkpoint_dir = prepare_run_checkpoint_dir(cfg, output_root, run_dir)
    final_path = run_dir / "metrics" / "final.json"
    if final_path.exists() and bool(config["run"]["skip_completed"]):
        print(f"skip completed | {method} | seed {seed} | {final_path}")
        return run_dir
    for folder in ("metrics", "logs", "audits"):
        (run_dir / folder).mkdir(parents=True, exist_ok=True)
    run_lock = RunLock(run_dir, final_path)
    if not run_lock.acquire():
        print(f"completed by another process | {method} | seed {seed}")
        return run_dir

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device, device_diagnostics = select_device()
    if device.type == "cpu":
        requested_threads = int(os.environ.get("OMP_NUM_THREADS", min(8, os.cpu_count() or 1)))
        torch.set_num_threads(max(1, requested_threads))

    model_values = dict(config["model"])
    model_config = ResearchModelConfig(
        vocab_size=len(bundle.vocabulary.tokens),
        max_length=int(config["data"]["max_length"]),
        method=method,
        num_layers=int(model_values["num_layers"]),
        d_model=int(model_values["d_model"]),
        d_ff=int(model_values["d_ff"]),
        num_heads=int(model_values["num_heads"]),
        slot_dim=int(model_values["slot_dim"]),
        num_bus_slots=int(model_values["num_bus_slots"]),
        bus_heads=int(model_values["bus_heads"]),
        bus_layers=int(model_values["bus_layers"]),
        concept_residual_dim=int(model_values["concept_residual_dim"]),
        dropout=float(model_values["dropout"]),
        norm_eps=float(model_values["norm_eps"]),
        rope_theta=float(model_values["rope_theta"]),
        bias=bool(model_values["bias"]),
        keep_residual_attention=bool(model_values["keep_residual_attention"]),
    )
    model = build_model(model_config)
    initialize_named_parameters(model, seed)
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if method == "parameter_matched":
        target_config_values = model.config.to_dict()
        target_config_values.update(
            {"method": "concept_bus_v2", "matched_ffn_width": None}
        )
        with torch.device("meta"):
            target_model = build_model(ResearchModelConfig(**target_config_values))
        target_parameters = parameter_count(target_model)
        actual_parameters = parameter_count(model)
        target_macs = estimated_model_macs(target_model.config, model.config.max_length)
        actual_macs = estimated_model_macs(model.config, model.config.max_length)
        matching_report = {
            "method": method,
            "matched_ffn_width": model.config.effective_ffn_width,
            "target_parameters": target_parameters,
            "actual_parameters": actual_parameters,
            "parameter_difference_fraction": abs(actual_parameters - target_parameters) / target_parameters,
            "target_macs": target_macs,
            "actual_macs": actual_macs,
            "mac_difference_fraction": abs(actual_macs - target_macs) / target_macs,
        }
        if method == "parameter_matched" and matching_report["parameter_difference_fraction"] > 0.01:
            raise RuntimeError("parameter-matched baseline exceeds the 1% tolerance")
        (run_dir / "matching_report.json").write_text(
            json.dumps(matching_report, indent=2), encoding="utf-8"
        )
        del target_model
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config)
    loss_balancer = SmoothMaxLossBalancer(
        temperature=float(config["loss"]["balance_temperature"]),
        ema_decay=float(config["loss"]["balance_ema_decay"]),
    )
    train_examples = load_examples(bundle.split_paths["train"])
    validation_examples = load_examples(bundle.split_paths["validation"])
    test_examples = load_examples(bundle.split_paths["test"])
    config["runtime"]["device"] = str(device)
    batcher = DeterministicBatcher(
        train_examples,
        bundle.vocabulary,
        batch_size=micro_batch_size,
        max_length=int(config["data"]["max_length"]),
        seed=10_000 + seed,
    )
    # The batcher owns compact tensorized inputs/targets; release the much
    # larger Python dataclass/string training corpus before GPU training.
    del train_examples
    timer = CumulativeTrainingTimer()
    start_step, best_validation = (0, float("inf"))
    if bool(config["run"]["resume"]):
        start_step, best_validation = _load_latest(
            run_dir,
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            batcher=batcher,
            timer=timer,
            config_hash=config_hash,
            device=device,
            loss_balancer=loss_balancer,
        )
    best_metadata = _best_model_metadata(checkpoint_dir, config_hash=config_hash)
    if best_metadata is not None:
        best_validation = min(
            best_validation, float(best_metadata["validation_loss"])
        )

    resolved = {
        **config,
        "config_hash": config_hash,
        "model_resolved": model.config.to_dict(),
        "parameter_count": parameter_count(model),
        "data_manifest": str(bundle.manifest_path),
        "source_fingerprint": _source_fingerprint(),
    }
    (run_dir / "resolved_config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "device": str(device),
                "device_diagnostics": device_diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger_settings = config["logging"]
    total_steps = int(config["train"]["max_steps"])
    log_interval = int(config["train"]["log_interval_steps"])
    eval_interval = int(config["train"]["eval_interval_steps"])
    checkpoint_interval_minutes = float(
        config["train"]["checkpoint_interval_minutes"]
    )
    last_checkpoint_seconds = timer.elapsed_seconds
    metrics_path = run_dir / "metrics" / "train.jsonl"
    task_name = f"{config['stage']}/{method}"
    with FixedWidthTrainingLogger(
        run_dir / "logs" / str(logger_settings["file_name"]),
        timer=timer,
        device_provider=lambda: current_device_log_status(device),
        widths=logger_settings["column_widths"],
        console_widths=logger_settings["console_column_widths"],
        console_mode=str(logger_settings["console_mode"]),
        flush_each_line=bool(logger_settings["flush_each_line"]),
    ) as logger:
        logger.log("system", step=start_step, total_steps=total_steps, seed=seed, task=task_name)
        model.train()
        for step in range(start_step + 1, total_steps + 1):
            timer.start()
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            component_sums: dict[str, float] = {}
            for _ in range(accumulation_steps):
                batch = _move(batcher.next_batch(), device)
                with _autocast(device):
                    output = model(batch["input_ids"], batch["attention_mask"])
                    micro_loss, components = _losses(
                        output,
                        batch,
                        config,
                        method,
                        loss_balancer,
                        update_balancer=False,
                    )
                    scaled_loss = micro_loss / accumulation_steps
                if not torch.isfinite(micro_loss):
                    timer.pause()
                    logger.log("error", step=step, total_steps=total_steps, loss=float(micro_loss), seed=seed, task=task_name)
                    raise RuntimeError(f"non-finite loss at step {step}")
                scaled_loss.backward()
                step_loss += float(micro_loss.detach()) / accumulation_steps
                for key, value in components.items():
                    component_sums[key] = component_sums.get(key, 0.0) + value / accumulation_steps
            active_objectives = {"task": component_sums["task"]}
            if method in CONCEPT_METHODS or method == "concept_aux":
                active_objectives["concept"] = component_sums["concept"]
            if method in CONCEPT_METHODS:
                active_objectives["causal"] = component_sums["causal"]
            # One EMA update per effective global batch. Adaptive micro-batch
            # and gradient accumulation must not alter the optimization rule.
            loss_balancer.update_ema(active_objectives)
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["optimizer"]["gradient_clip"])
                )
            )
            optimizer.step()
            scheduler.step()
            timer.pause()

            if step % log_interval == 0 or step == 1 or step == total_steps:
                logger.log("train", step=step, total_steps=total_steps, loss=step_loss, seed=seed, task=task_name)
                _append_jsonl(
                    metrics_path,
                    {
                        "step": step,
                        "loss": step_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "gradient_norm": gradient_norm,
                        "training_seconds": timer.elapsed_seconds,
                        **component_sums,
                    },
                )

            if step % eval_interval == 0 or step == total_steps:
                validation_limit = int(
                    config["train"].get(
                        "monitor_validation_examples", len(validation_examples)
                    )
                )
                validation = evaluate(
                    model,
                    validation_examples[:validation_limit],
                    bundle,
                    config,
                    device,
                )
                logger.log("valid", step=step, total_steps=total_steps, loss=validation["loss"], seed=seed, task=task_name)
                _append_jsonl(
                    run_dir / "metrics" / "validation.jsonl",
                    {"step": step, **validation},
                )
                if validation["loss"] < best_validation:
                    best_validation = validation["loss"]
                    _save_best_model(
                        checkpoint_dir=checkpoint_dir,
                        model=model,
                        step=step,
                        validation_loss=best_validation,
                        config_hash=config_hash,
                    )

            if checkpoint_due(
                timer,
                last_checkpoint_seconds,
                checkpoint_interval_minutes,
                final_step=step == total_steps,
            ):
                _save_checkpoint(
                    run_dir,
                    checkpoint_dir=checkpoint_dir,
                    name=(
                        "final.pt"
                        if step == total_steps
                        else f"step_{step:08d}.pt"
                    ),
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    batcher=batcher,
                    timer=timer,
                    best_validation=best_validation,
                    config_hash=config_hash,
                    loss_balancer=loss_balancer,
                )
                last_checkpoint_seconds = timer.elapsed_seconds
                logger.log("checkpt", step=step, total_steps=total_steps, seed=seed, task=task_name)

        # The held-out test and all causal audits must evaluate the model chosen
        # only by validation loss, never the potentially overfit last step.
        selected_step, selected_validation_loss = _load_best_model(
            checkpoint_dir=checkpoint_dir,
            model=model,
            config_hash=config_hash,
            device=device,
        )
        logger.log(
            "select",
            step=selected_step,
            total_steps=total_steps,
            loss=selected_validation_loss,
            seed=seed,
            task=task_name,
        )
        test_result = evaluate(
            model,
            test_examples,
            bundle,
            config,
            device,
            return_predictions=True,
        )
        test, (test_country_predictions, test_color_predictions) = test_result
        final_metrics = {f"test_{key}": value for key, value in test.items()}
        torch.save(
            {
                "example_ids": [example.example_id for example in test_examples],
                "country_targets": torch.tensor(
                    [example.country_target for example in test_examples]
                ),
                "color_targets": torch.tensor(
                    [example.color_target for example in test_examples]
                ),
                "country_predictions": test_country_predictions,
                "color_predictions": test_color_predictions,
            },
            run_dir / "audits" / "test_predictions.pt",
        )
        if method in BUS_METHODS:
            final_metrics.update(
                _causal_audit(
                    model,
                    test_examples,
                    bundle,
                    config,
                    device,
                    test,
                )
            )
            audit_batch = _move(
                collate_examples(
                    test_examples[
                        : min(
                            int(config.get("audit", {}).get("trace_examples", 64)),
                            len(test_examples),
                        )
                    ],
                    bundle.vocabulary,
                    int(config["data"]["max_length"]),
                ),
                device,
            )
            model.eval()
            with torch.no_grad(), _autocast(device):
                audit_output = model(
                    audit_batch["input_ids"],
                    audit_batch["attention_mask"],
                    return_trace=True,
                    compute_counterfactuals=False,
                )
            trace = audit_output.trace or {}
            final_indices = audit_batch["attention_mask"].sum(dim=-1).long() - 1
            projected = trace.get("projected_slots")
            audit_rows = []
            for row, example in enumerate(test_examples[: len(audit_batch["input_ids"])]):
                probabilities = audit_output.concept_probabilities[row].float().cpu()
                active = [
                    {"node": CONCEPT_NAMES[index], "value": float(value)}
                    for index, value in enumerate(probabilities)
                    if value >= 0.5
                ]
                evidence = []
                if projected is not None:
                    slots = projected[row, final_indices[row]].float().cpu()
                    group_names = ("country", "color")
                    for slot, values in enumerate(slots):
                        magnitudes, indices = values.abs().topk(
                            min(3, values.shape[-1])
                        )
                        evidence.append(
                            {
                                "concept_group": group_names[slot],
                                "top_dimensions": indices.tolist(),
                                "absolute_values": magnitudes.tolist(),
                                "rms": float(values.square().mean().sqrt()),
                            }
                        )
                audit_rows.append(
                    {
                        "example_id": example.example_id,
                        "text": example.text,
                        "target_country": COUNTRY_CLASSES[example.country_target],
                        "target_color": COLOR_CLASSES[example.color_target],
                        "predicted_country": COUNTRY_CLASSES[int(audit_output.country_logits[row].argmax())],
                        "predicted_color": COLOR_CLASSES[int(audit_output.color_logits[row].argmax())],
                        "active_nodes": active,
                        "projection_evidence": evidence,
                    }
                )
            (run_dir / "audits" / "trace_sample.json").write_text(
                json.dumps(audit_rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        final_metrics.update(
            {
                "parameters": parameter_count(model),
                "training_seconds": timer.elapsed_seconds,
                "steps": total_steps,
                "selected_checkpoint_step": selected_step,
                "selected_validation_loss": selected_validation_loss,
                "training_examples_per_second": (
                    total_steps * global_batch_size / max(1e-9, timer.elapsed_seconds)
                ),
                "estimated_macs_per_example": estimated_model_macs(
                    model.config, int(config["data"]["max_length"])
                ),
                "peak_cuda_memory_bytes": float(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0.0,
            }
        )
        criteria: dict[str, bool] = {}
        if config["stage"] == "smoke" and method == "concept_bus_v2":
            criteria = {
                # Smoke proves the complete numerical path, not scientific
                # quality. Five optimization steps cannot support accuracy or
                # causal-effect thresholds.
                "finite_test_loss": math.isfinite(final_metrics["test_loss"]),
                "finite_concept_f1": math.isfinite(
                    final_metrics["test_concept_macro_f1"]
                ),
            }
        feasible = all(criteria.values()) if criteria else True
        payload = {
            "stage": config["stage"],
            "method": method,
            "seed": seed,
            "config_hash": config_hash,
            "source_fingerprint": _source_fingerprint(),
            "metrics": final_metrics,
            "criteria": criteria,
            "feasible": feasible,
        }
        final_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # The final optimization step is unconditionally checkpointed inside
        # the loop. Do not serialize the same multi-GB state a second time.
        logger.log("final", step=total_steps, total_steps=total_steps, loss=test["loss"], seed=seed, task=task_name)
    if not feasible:
        run_lock.release()
        raise RuntimeError(f"pre-registered feasibility gate failed: {criteria}")
    run_lock.release()
    return run_dir
