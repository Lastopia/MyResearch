from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import string
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pipeline.data import (
    SPECIAL_TOKENS,
    MultiQueryRetrievalBatch,
    RetrievalBatch,
    TokenBlockDataset,
    generate_multi_query_retrieval_batches,
    generate_position_swap_batches,
    load_qasper_examples,
    load_tokens,
    load_wikitext_tokens,
    prepare_tokenizer,
)
from pipeline.train import load_pretrained_model
from tools.io import read_json, write_json
from tools.log import log_fields, log_resources, stage_banner
from tools.metrics import perplexity, target_nll
from tools.paths import data_dir, metric_dir, wikitext_dir
from tools.runtime import (
    autocast_context,
    resolve_device,
    resolve_dtype,
)


def evaluation_dataloader_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Avoid nested worker processes in short-lived evaluation loaders."""
    return {
        "num_workers": 0,
        "pin_memory": bool(
            cfg["resources"].get("resolved_pin_memory", False)
        ),
    }


ProgressCallback = Callable[[int, int], None]


def _notify_progress(
    callback: ProgressCallback | None,
    current: int,
    total: int,
) -> None:
    if callback is None:
        return
    interval = max(1, total // 8)
    if current == total or current % interval == 0:
        callback(current, total)


@torch.no_grad()
def _evaluate_lm(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    length: int,
    batches: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    loader_kwargs: dict[str, Any] | None = None,
    inference_scale: float | None = None,
    progress: ProgressCallback | None = None,
) -> float:
    dataset = TokenBlockDataset(tokens, length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **(loader_kwargs or {"num_workers": 0}),
    )
    total_nll = 0.0
    total_tokens = 0
    for batch_index, (batch_input_ids, batch_targets) in enumerate(loader):
        if batch_index >= batches:
            break
        batch_input_ids = batch_input_ids.to(device)
        batch_targets = batch_targets.to(device)
        with autocast_context(device, dtype):
            output = model(
                batch_input_ids,
                batch_targets,
                inference_scale=inference_scale,
            )
        valid = batch_targets.ne(-100).sum().item()
        total_nll += float(output["lm_loss"].item()) * valid
        total_tokens += valid
        _notify_progress(progress, batch_index + 1, batches)
    return perplexity(total_nll, total_tokens)


@torch.no_grad()
def _evaluate_retrieval(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    *,
    length: int,
    retrieval: RetrievalBatch,
    device: torch.device,
    dtype: torch.dtype,
    inference_scale: float | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, float]:
    batch_size = int(cfg["eval"]["batch_size"])
    predictions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    nll_gains: list[torch.Tensor] = []
    total_chunks = math.ceil(retrieval.input_ids.shape[0] / batch_size)
    for chunk_index, start in enumerate(
        range(0, retrieval.input_ids.shape[0], batch_size),
        start=1,
    ):
        chunk = retrieval.select(start, start + batch_size).to(device)
        with autocast_context(device, dtype):
            output = model(chunk.input_ids, inference_scale=inference_scale)
        final_logits = output["logits"][:, -1, :]
        full_nll = target_nll(final_logits, chunk.labels)

        removed = chunk.input_ids.clone()
        batch_index = torch.arange(removed.shape[0], device=device)
        removed[batch_index, chunk.relevant_positions] = SPECIAL_TOKENS["filler"]
        with autocast_context(device, dtype):
            removed_output = model(removed, inference_scale=inference_scale)
        removed_nll = target_nll(
            removed_output["logits"][:, -1, :],
            chunk.labels,
        )
        predictions.append(final_logits.argmax(dim=-1).cpu())
        labels.append(chunk.labels.cpu())
        nll_gains.append((removed_nll - full_nll).cpu())
        _notify_progress(progress, chunk_index, total_chunks)

    all_predictions = torch.cat(predictions)
    all_labels = torch.cat(labels)
    accuracy = float((all_predictions == all_labels).float().mean().item())
    rcug = float(torch.cat(nll_gains).mean().item())
    distances = retrieval.target_distances.float().cpu()
    median = distances.median()
    near = distances <= median
    far = distances > median

    def subset_accuracy(mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return accuracy
        return float(
            (all_predictions[mask] == all_labels[mask]).float().mean().item()
        )

    near_accuracy = subset_accuracy(near)
    far_accuracy = subset_accuracy(far)

    def accuracy_bins(values: torch.Tensor, edges: list[float], prefix: str) -> dict[str, float]:
        ordered = sorted(float(edge) for edge in edges)
        if len(ordered) < 2:
            return {}
        result: dict[str, float] = {}
        for lower, upper in itertools.pairwise(ordered):
            mask = (values >= lower) & (
                (values < upper) if upper != ordered[-1] else (values <= upper)
            )
            if bool(mask.any()):
                result[f"{prefix}[{lower:g},{upper:g}{')' if upper != ordered[-1] else ']'}"] = subset_accuracy(mask)
        overflow = values > ordered[-1]
        if bool(overflow.any()):
            result[f"{prefix}({ordered[-1]:g},inf)"] = subset_accuracy(overflow)
        return result

    position_fraction = (
        retrieval.relevant_positions.float().cpu() / max(1, length - 1)
    )
    position_edges = [
        float(edge) for edge in cfg["eval"].get("target_position_bins", [])
    ]
    return {
        "accuracy": accuracy,
        "near_accuracy": near_accuracy,
        "far_accuracy": far_accuracy,
        "rcug": rcug,
        "distance_conditioned_accuracy": accuracy_bins(
            distances,
            [float(edge) for edge in cfg["eval"].get("distance_bins", [])],
            "distance",
        ),
        "target_position_conditioned_accuracy": accuracy_bins(
            position_fraction,
            position_edges,
            "position_fraction",
        ),
    }


@torch.no_grad()
def _evaluate_multi_query_retrieval(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    *,
    retrieval: MultiQueryRetrievalBatch,
    device: torch.device,
    dtype: torch.dtype,
    inference_scale: float | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    batch_size = int(cfg["eval"]["batch_size"])
    predictions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    total_chunks = math.ceil(retrieval.input_ids.shape[0] / batch_size)
    for chunk_index, start in enumerate(
        range(0, retrieval.input_ids.shape[0], batch_size),
        start=1,
    ):
        chunk = retrieval.select(start, start + batch_size).to(device)
        with autocast_context(device, dtype):
            output = model(chunk.input_ids, inference_scale=inference_scale)
        batch_index = torch.arange(
            chunk.input_ids.shape[0],
            device=device,
        )[:, None]
        query_logits = output["logits"][batch_index, chunk.query_positions]
        predictions.append(query_logits.argmax(dim=-1).cpu())
        labels.append(chunk.labels.cpu())
        _notify_progress(progress, chunk_index, total_chunks)

    all_predictions = torch.cat(predictions)
    all_labels = torch.cat(labels)
    correct = all_predictions.eq(all_labels)
    distances = retrieval.target_distances.float().cpu()
    median = distances.median()
    near = distances <= median
    far = distances > median

    def masked_accuracy(mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return float(correct.float().mean().item())
        return float(correct[mask].float().mean().item())

    return {
        "association_accuracy": float(correct.float().mean().item()),
        "sample_exact_match": float(correct.all(dim=1).float().mean().item()),
        "near_accuracy": masked_accuracy(near),
        "far_accuracy": masked_accuracy(far),
        "queries_per_sample": int(all_labels.shape[1]),
        "similar_distractors_per_query": int(
            retrieval.distractor_positions.shape[2]
        ),
        "per_query_accuracy": {
            str(index): float(correct[:, index].float().mean().item())
            for index in range(correct.shape[1])
        },
    }


def _normalize_qa_answer(text: str) -> list[str]:
    lowered = str(text).casefold()
    without_punctuation = "".join(
        " " if character in string.punctuation else character
        for character in lowered
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split()).split()


def _qa_token_f1(prediction: str, answers: list[str]) -> float:
    predicted = _normalize_qa_answer(prediction)
    best = 0.0
    for answer in answers:
        expected = _normalize_qa_answer(answer)
        if not predicted and not expected:
            best = max(best, 1.0)
            continue
        overlap = sum((Counter(predicted) & Counter(expected)).values())
        if overlap == 0:
            continue
        precision = overlap / max(1, len(predicted))
        recall = overlap / max(1, len(expected))
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _qa_exact_match(prediction: str, answers: list[str]) -> float:
    normalized_prediction = _normalize_qa_answer(prediction)
    return float(
        any(
            normalized_prediction == _normalize_qa_answer(answer)
            for answer in answers
        )
    )


@torch.no_grad()
def _continuation_nll(
    model: torch.nn.Module,
    *,
    prompt: torch.Tensor,
    answers: list[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    inference_scale: float | None,
) -> float:
    losses: list[float] = []
    prompt = prompt.to(device)
    for candidate_answer in answers:
        candidate_answer = candidate_answer.to(device)
        sequence = torch.cat((prompt, candidate_answer), dim=0).unsqueeze(0)
        with autocast_context(device, dtype):
            output = model(
                sequence[:, :-1],
                inference_scale=inference_scale,
            )
        start = prompt.numel() - 1
        logits = output["logits"][
            :, start : start + candidate_answer.numel(), :
        ]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            candidate_answer,
            reduction="mean",
        )
        losses.append(float(loss.item()))
    return min(losses)


@torch.no_grad()
def _greedy_continuation(
    model: torch.nn.Module,
    *,
    prompt: torch.Tensor,
    max_tokens: int,
    eos_token_id: int | None,
    device: torch.device,
    dtype: torch.dtype,
    inference_scale: float | None,
) -> torch.Tensor:
    prompt = prompt.to(device).unsqueeze(0)
    with autocast_context(device, dtype):
        output = model(
            prompt,
            use_cache=True,
            inference_scale=inference_scale,
        )
    past = output["past_key_values"]
    current = output["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [current.squeeze(0).squeeze(0).cpu()]
    for _ in range(max(0, max_tokens - 1)):
        if eos_token_id is not None and int(current.item()) == eos_token_id:
            break
        with autocast_context(device, dtype):
            output = model(
                current,
                past_key_values=past,
                use_cache=True,
                inference_scale=inference_scale,
            )
        past = output["past_key_values"]
        current = output["logits"][:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )
        generated.append(current.squeeze(0).squeeze(0).cpu())
    return torch.stack(generated).to(torch.long)


@torch.no_grad()
def _evaluate_qasper(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    *,
    examples: list[dict[str, Any]],
    tokenizer: Any,
    length: int,
    device: torch.device,
    dtype: torch.dtype,
    inference_scale: float | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    max_answer_tokens = int(cfg["eval"]["qasper_max_answer_tokens"])
    generation_tokens = int(cfg["eval"]["qasper_generation_tokens"])
    newline_token = 198
    full_nlls: list[float] = []
    evidence_removed_nlls: list[float] = []
    f1_scores: list[float] = []
    exact_matches: list[float] = []
    evidence_distances: list[int] = []
    evaluated_ids: list[str] = []
    sample_hashes: list[str] = []
    for example_index, example in enumerate(examples, start=1):
        _notify_progress(progress, example_index, len(examples))
        suffix = torch.tensor(
            tokenizer.encode(
                f"\n\nQuestion: {example['question']}\nAnswer:",
                add_special_tokens=False,
            ),
            dtype=torch.long,
        )
        document_budget = length - suffix.numel() - max_answer_tokens
        if document_budget <= 0:
            continue
        absolute_evidence_start = int(example["evidence_token_start"])
        absolute_evidence_end = int(example["evidence_token_end"])
        full_document = example["document_token_ids"]
        desired_evidence_start = max(0, document_budget // 8)
        window_start = max(
            0,
            absolute_evidence_start - desired_evidence_start,
        )
        window_start = min(
            window_start,
            max(0, full_document.numel() - document_budget),
        )
        window_end = window_start + document_budget
        if (
            absolute_evidence_start < window_start
            or absolute_evidence_end > window_end
        ):
            continue
        evidence_start = absolute_evidence_start - window_start
        evidence_end = absolute_evidence_end - window_start
        document = full_document[window_start:window_end].clone()
        if document.numel() < document_budget:
            continue
        prompt = torch.cat((document, suffix), dim=0)
        answers = [
            tokens[:max_answer_tokens].clone()
            for tokens in example["answer_token_ids"]
            if tokens.numel() > 0
        ]
        if not answers:
            continue
        full_nll = _continuation_nll(
            model,
            prompt=prompt,
            answers=answers,
            device=device,
            dtype=dtype,
            inference_scale=inference_scale,
        )
        removed_document = document.clone()
        removed_document[evidence_start:evidence_end] = newline_token
        removed_prompt = torch.cat((removed_document, suffix), dim=0)
        removed_nll = _continuation_nll(
            model,
            prompt=removed_prompt,
            answers=answers,
            device=device,
            dtype=dtype,
            inference_scale=inference_scale,
        )
        generated = _greedy_continuation(
            model,
            prompt=prompt,
            max_tokens=generation_tokens,
            eos_token_id=tokenizer.eos_token_id,
            device=device,
            dtype=dtype,
            inference_scale=inference_scale,
        )
        prediction = tokenizer.decode(
            generated.tolist(),
            skip_special_tokens=True,
        )
        answer_texts = [str(value) for value in example["answer_texts"]]
        full_nlls.append(full_nll)
        evidence_removed_nlls.append(removed_nll)
        f1_scores.append(_qa_token_f1(prediction, answer_texts))
        exact_matches.append(_qa_exact_match(prediction, answer_texts))
        evidence_distances.append(prompt.numel() - evidence_end)
        evaluated_ids.append(str(example["question_id"]))
        sample_hashes.append(str(example["sample_sha256"]))

    if not full_nlls:
        raise RuntimeError(
            f"No QASPER examples fit evaluation length {length}"
        )
    count = len(full_nlls)
    return {
        "answer_nll": sum(full_nlls) / count,
        "answer_perplexity": math.exp(
            min(50.0, sum(full_nlls) / count)
        ),
        "token_f1": sum(f1_scores) / count,
        "exact_match": sum(exact_matches) / count,
        "evidence_utilization_gain": sum(
            removed - full
            for removed, full in zip(
                evidence_removed_nlls,
                full_nlls,
                strict=True,
            )
        )
        / count,
        "mean_evidence_distance_tokens": sum(evidence_distances) / count,
        "samples": count,
        "question_ids": evaluated_ids,
        "sample_sha256": sample_hashes,
        "program_generated": False,
        "program_modified": True,
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_fingerprint(
    cfg: dict[str, Any],
    *,
    checkpoint_sha256: str,
    checkpoint_kind: str,
) -> str:
    payload = {
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_sha256": checkpoint_sha256,
        "run": {
            "method": cfg["run"]["method"],
            "seed": cfg["run"]["seed"],
            "dtype": cfg["run"]["dtype"],
        },
        "data": cfg["data"],
        "model": cfg["model"],
        "position": cfg["position"],
        "eval": cfg["eval"],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_evaluation_cache(
    path: Path,
    *,
    fingerprint: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    cached = read_json(path)
    if cached.get("evaluation_fingerprint") != fingerprint:
        return None
    return cached


def _data_source_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    fineweb_meta_path = data_dir(cfg) / "meta.json"
    wikitext_meta_path = wikitext_dir(cfg) / "meta.json"
    fineweb_meta = (
        read_json(fineweb_meta_path) if fineweb_meta_path.exists() else {}
    )
    wikitext_meta = (
        read_json(wikitext_meta_path) if wikitext_meta_path.exists() else {}
    )
    leakage_audit = fineweb_meta.get("evaluation_leakage_audit", {})
    return {
        "natural_language": {
            "fineweb_edu_held_out": {
                "dataset": cfg["data"]["fineweb_dataset"],
                "config": cfg["data"]["fineweb_config"],
                "revision": cfg["data"]["fineweb_revision"],
                "upstream_split": "train",
                "local_split": "test",
                "sample_unit": "token",
                "sample_count": int(cfg["data"]["test_tokens"]),
                "program_generated": False,
                "selection": (
                    "deterministic streamed shuffle followed by disjoint local "
                    "train/valid/test token-budget partition"
                ),
                "prepared_documents_seen": fineweb_meta.get("documents_seen"),
                "evaluation_leakage_audit": leakage_audit,
            },
            "wikitext103": {
                "dataset": cfg["data"]["wikitext_dataset"],
                "config": cfg["data"]["wikitext_config"],
                "revision": cfg["data"]["wikitext_revision"],
                "upstream_split": "test",
                "local_split": "test",
                "sample_unit": "token",
                "sample_count": wikitext_meta.get("token_count"),
                "token_sha256": wikitext_meta.get("token_sha256"),
                "program_generated": False,
                "selection": "all non-empty test rows in upstream order",
                "fineweb_near_duplicate_exclusion_complete": (
                    leakage_audit.get(
                        "complete_for_all_prepared_documents",
                        False,
                    )
                ),
            },
        },
        "synthetic_control": {
            "single_query": {
                "generator": "generate_nested_retrieval_batches",
                "generator_version": 2,
                "samples_per_length": int(cfg["eval"]["retrieval_samples"]),
                "seed": int(cfg["data"]["seed"]) + 1_000,
                "program_generated": True,
            },
            "multi_query_associative_recall": {
                "generator": "generate_multi_query_retrieval_batches",
                "generator_version": 1,
                "samples_per_length": int(cfg["eval"]["retrieval_samples"]),
                "queries_per_sample": int(
                    cfg["data"]["retrieval_queries_per_sample"]
                ),
                "similar_distractors_per_query": int(
                    cfg["data"]["retrieval_similar_distractors"]
                ),
                "seed": int(cfg["data"]["seed"]) + 2_000,
                "program_generated": True,
            },
            "target_distractor_position_swap": {
                "generator": "generate_position_swap_batches",
                "generator_version": 1,
                "samples_per_length": int(cfg["eval"]["retrieval_samples"]),
                "seed": int(cfg["data"]["seed"]) + 1_000,
                "program_generated": True,
            },
        },
    }


def _length_degradation_rate(results: dict[str, dict[str, float]]) -> float:
    if len(results) < 2:
        return 0.0
    lengths = torch.tensor([float(key) for key in results], dtype=torch.float32)
    accuracies = torch.tensor(
        [results[key]["accuracy"] for key in results], dtype=torch.float32
    )
    x = torch.log2(lengths)
    centered = x - x.mean()
    denominator = centered.square().sum()
    if denominator <= 0:
        return 0.0
    slope = (centered * (accuracies - accuracies.mean())).sum() / denominator
    return float((-slope).item())


def _run_checkpoint(
    cfg: dict[str, Any],
    checkpoint_kind: str,
) -> dict[str, Any]:
    stage_banner("EVALUATE", cfg=cfg)
    if checkpoint_kind not in {"pretrain", "adapt"}:
        raise ValueError("checkpoint_kind must be pretrain or adapt")
    device = resolve_device(cfg)
    dtype = resolve_dtype(cfg)
    model, checkpoint = load_pretrained_model(
        cfg,
        prefer_adapted=checkpoint_kind in {"adapt", "auto"},
        require_adapted=checkpoint_kind == "adapt",
    )
    model.eval()
    checkpoint_sha256 = _file_sha256(checkpoint)
    fingerprint = _evaluation_fingerprint(
        cfg,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_kind=checkpoint_kind,
    )
    metrics_root = metric_dir(cfg)
    final_path = metrics_root / f"evaluation_{checkpoint_kind}.json"
    partial_path = metrics_root / f"evaluation_{checkpoint_kind}.partial.json"
    force = bool(cfg["run"].get("force", False))
    cached_final = (
        None
        if force
        else _load_evaluation_cache(final_path, fingerprint=fingerprint)
    )
    if cached_final is not None and cached_final.get("status") == "completed":
        log_fields(
            "eval",
            cfg=cfg,
            method=cfg["run"]["method"],
            checkpoint=checkpoint_kind,
            state="reused",
        )
        stage_banner("EVALUATE", "REUSED", cfg=cfg)
        return cached_final

    test_tokens = load_tokens(cfg, "test")
    external_tokens = (
        load_wikitext_tokens(cfg)
        if bool(cfg["data"].get("prepare_wikitext", False))
        and str(cfg["data"]["source"]) == "fineweb_edu"
        else None
    )
    qasper_examples: list[dict[str, Any]] | None = None
    qasper_meta: dict[str, Any] | None = None
    tokenizer: Any | None = None
    if bool(cfg["data"].get("prepare_qasper", False)):
        qasper_examples, qasper_meta = load_qasper_examples(cfg)
        qasper_limit = min(
            len(qasper_examples),
            int(cfg["eval"]["qasper_samples"]),
        )
        qasper_examples = qasper_examples[:qasper_limit]
        qasper_meta = deepcopy(qasper_meta)
        qasper_meta["selected_samples"] = len(qasper_examples)
        qasper_meta["selected_question_ids"] = [
            str(example["question_id"]) for example in qasper_examples
        ]
        qasper_meta["selected_sample_sha256"] = [
            str(example["sample_sha256"]) for example in qasper_examples
        ]
        tokenizer = prepare_tokenizer(cfg)
    # Evaluation constructs short-lived loaders for several context lengths.
    # Keep them in the GPU job process instead of nesting DataLoader worker
    # processes inside the outer per-GPU ProcessPoolExecutor process.
    loader_options = evaluation_dataloader_kwargs(cfg)
    lengths = [
        int(length)
        for length in cfg["eval"]["lengths"]
        if int(length) <= int(cfg["model"]["max_seq_len"])
    ]
    result = (
        None
        if force
        else _load_evaluation_cache(partial_path, fingerprint=fingerprint)
    )
    if result is None:
        result = {
            "status": "partial",
            "evaluation_fingerprint": fingerprint,
            "checkpoint_sha256": checkpoint_sha256,
            "method": cfg["run"]["method"],
            "seed": cfg["run"]["seed"],
            "checkpoint": str(checkpoint),
            "checkpoint_kind": checkpoint_kind,
            "data_sources": _data_source_metadata(cfg),
            "lengths": {},
        }
    else:
        log_fields(
            "eval",
            cfg=cfg,
            method=cfg["run"]["method"],
            checkpoint=checkpoint_kind,
            state="resumed",
        )
    if qasper_meta is not None:
        result["data_sources"]["real_long_document_qa"] = {
            "qasper": {
                "dataset": qasper_meta["dataset"],
                "data_file": qasper_meta["data_file"],
                "revision": qasper_meta["dataset_revision"],
                "split": qasper_meta["split"],
                "sample_count": qasper_meta["selected_samples"],
                "question_ids": qasper_meta["selected_question_ids"],
                "sample_sha256": qasper_meta["selected_sample_sha256"],
                "program_generated": False,
                "program_modified": True,
                "processing_rule": qasper_meta["processing_rule"],
                "fineweb_documents_excluded_for_evaluation_overlap": (
                    qasper_meta[
                        "fineweb_documents_excluded_for_evaluation_overlap"
                    ]
                ),
                "leakage_check_complete": qasper_meta[
                    "leakage_check_complete"
                ],
                "leakage_check_rules": qasper_meta[
                    "leakage_check_rules"
                ],
            }
        }
    paired_position_controls = generate_position_swap_batches(
        samples=int(cfg["eval"]["retrieval_samples"]),
        lengths=lengths,
        vocab_size=int(cfg["data"]["vocab_size"]),
        num_pairs=int(cfg["data"]["num_key_value_pairs"]),
        seed=int(cfg["data"]["seed"]) + 1_000,
    )
    nested_retrieval = {
        length: controls[0]
        for length, controls in paired_position_controls.items()
    }
    multi_query_retrieval = generate_multi_query_retrieval_batches(
        samples=int(cfg["eval"]["retrieval_samples"]),
        lengths=lengths,
        vocab_size=int(cfg["data"]["vocab_size"]),
        queries_per_sample=int(
            cfg["data"]["retrieval_queries_per_sample"]
        ),
        similar_distractors=int(
            cfg["data"]["retrieval_similar_distractors"]
        ),
        seed=int(cfg["data"]["seed"]) + 2_000,
    )

    def persist() -> None:
        result["status"] = "partial"
        write_json(partial_path, result)

    def run_part(
        *,
        length: int,
        part: str,
        container: dict[str, Any],
        key: str,
        compute: Callable[[ProgressCallback], Any],
    ) -> Any:
        if key in container:
            log_fields(
                "eval",
                cfg=cfg,
                method=cfg["run"]["method"],
                checkpoint=checkpoint_kind,
                length=length,
                part=part,
                state="reused",
            )
            return container[key]

        log_fields(
            "eval",
            cfg=cfg,
            method=cfg["run"]["method"],
            checkpoint=checkpoint_kind,
            length=length,
            part=part,
            state="start",
        )

        def progress(current: int, total: int) -> None:
            log_fields(
                "eval",
                cfg=cfg,
                method=cfg["run"]["method"],
                checkpoint=checkpoint_kind,
                length=length,
                part=part,
                state="progress",
                batch=f"{current}/{total}",
            )

        value = compute(progress)
        container[key] = value
        persist()
        log_fields(
            "eval",
            cfg=cfg,
            method=cfg["run"]["method"],
            checkpoint=checkpoint_kind,
            length=length,
            part=part,
            state="done",
        )
        return value

    for length in lengths:
        length_result = result["lengths"].setdefault(
            str(length),
            {
                "natural_language": {},
                "real_long_document_qa": {},
                "synthetic_control": {},
            },
        )
        natural = length_result.setdefault("natural_language", {})
        real_qa = length_result.setdefault("real_long_document_qa", {})
        synthetic = length_result.setdefault("synthetic_control", {})

        run_part(
            length=length,
            part="fineweb_ppl",
            container=natural,
            key="fineweb_edu_held_out_ppl",
            compute=lambda progress: _evaluate_lm(
                model,
                test_tokens,
                length=length,
                batches=int(cfg["eval"]["lm_batches"]),
                batch_size=int(cfg["eval"]["batch_size"]),
                device=device,
                dtype=dtype,
                loader_kwargs=loader_options,
                progress=progress,
            ),
        )
        run_part(
            length=length,
            part="wikitext_ppl",
            container=natural,
            key="wikitext103_ppl",
            compute=lambda progress: (
                _evaluate_lm(
                    model,
                    external_tokens,
                    length=length,
                    batches=int(cfg["eval"]["lm_batches"]),
                    batch_size=int(cfg["eval"]["batch_size"]),
                    device=device,
                    dtype=dtype,
                    loader_kwargs=loader_options,
                    progress=progress,
                )
                if external_tokens is not None and len(external_tokens) > length
                else None
            ),
        )

        retrieval = run_part(
            length=length,
            part="single_query",
            container=synthetic,
            key="single_query",
            compute=lambda progress: {
                **_evaluate_retrieval(
                    model,
                    cfg,
                    length=length,
                    retrieval=nested_retrieval[length],
                    device=device,
                    dtype=dtype,
                    progress=progress,
                ),
                "input_sha256": _tensor_sha256(
                    nested_retrieval[length].input_ids
                ),
            },
        )

        run_part(
            length=length,
            part="multi_query",
            container=synthetic,
            key="multi_query_associative_recall",
            compute=lambda progress: {
                **_evaluate_multi_query_retrieval(
                    model,
                    cfg,
                    retrieval=multi_query_retrieval[length],
                    device=device,
                    dtype=dtype,
                    progress=progress,
                ),
                "input_sha256": _tensor_sha256(
                    multi_query_retrieval[length].input_ids
                ),
            },
        )

        def position_swap(progress: ProgressCallback) -> dict[str, Any]:
            swapped = _evaluate_retrieval(
                model,
                cfg,
                length=length,
                retrieval=paired_position_controls[length][1],
                device=device,
                dtype=dtype,
                progress=progress,
            )
            return {
                "original": retrieval,
                "swapped": swapped,
                "accuracy_delta_swapped_minus_original": (
                    swapped["accuracy"] - retrieval["accuracy"]
                ),
                "original_input_sha256": _tensor_sha256(
                    paired_position_controls[length][0].input_ids
                ),
                "swapped_input_sha256": _tensor_sha256(
                    paired_position_controls[length][1].input_ids
                ),
            }

        run_part(
            length=length,
            part="position_swap",
            container=synthetic,
            key="target_distractor_position_swap",
            compute=position_swap,
        )
        if qasper_examples is not None and tokenizer is not None:
            run_part(
                length=length,
                part="qasper",
                container=real_qa,
                key="qasper",
                compute=lambda progress: _evaluate_qasper(
                    model,
                    cfg,
                    examples=qasper_examples,
                    tokenizer=tokenizer,
                    length=length,
                    device=device,
                    dtype=dtype,
                    progress=progress,
                ),
            )

        if (
            str(cfg["run"]["method"]) == "rope"
            and bool(cfg["eval"].get("rope_pi_enabled", False))
        ):
            training_length = int(cfg["eval"]["rope_pi_train_length"])
            rope_pi_scale = max(1.0, length / max(1, training_length))
            rope_pi = length_result.setdefault(
                "rope_pi",
                {
                    "mode": "linear_position_interpolation",
                    "inference_scale": rope_pi_scale,
                    "training_length": training_length,
                    "natural_language": {},
                    "real_long_document_qa": {},
                    "synthetic_control": {},
                },
            )
            rope_natural = rope_pi.setdefault("natural_language", {})
            rope_real = rope_pi.setdefault("real_long_document_qa", {})
            rope_synthetic = rope_pi.setdefault("synthetic_control", {})
            run_part(
                length=length,
                part="rope_pi_fineweb_ppl",
                container=rope_natural,
                key="fineweb_edu_held_out_ppl",
                compute=lambda progress: _evaluate_lm(
                    model,
                    test_tokens,
                    length=length,
                    batches=int(cfg["eval"]["lm_batches"]),
                    batch_size=int(cfg["eval"]["batch_size"]),
                    device=device,
                    dtype=dtype,
                    loader_kwargs=loader_options,
                    inference_scale=rope_pi_scale,
                    progress=progress,
                ),
            )
            run_part(
                length=length,
                part="rope_pi_wikitext_ppl",
                container=rope_natural,
                key="wikitext103_ppl",
                compute=lambda progress: (
                    _evaluate_lm(
                        model,
                        external_tokens,
                        length=length,
                        batches=int(cfg["eval"]["lm_batches"]),
                        batch_size=int(cfg["eval"]["batch_size"]),
                        device=device,
                        dtype=dtype,
                        loader_kwargs=loader_options,
                        inference_scale=rope_pi_scale,
                        progress=progress,
                    )
                    if external_tokens is not None
                    and len(external_tokens) > length
                    else None
                ),
            )
            run_part(
                length=length,
                part="rope_pi_single_query",
                container=rope_synthetic,
                key="single_query",
                compute=lambda progress: _evaluate_retrieval(
                    model,
                    cfg,
                    length=length,
                    retrieval=nested_retrieval[length],
                    device=device,
                    dtype=dtype,
                    inference_scale=rope_pi_scale,
                    progress=progress,
                ),
            )
            run_part(
                length=length,
                part="rope_pi_multi_query",
                container=rope_synthetic,
                key="multi_query_associative_recall",
                compute=lambda progress: _evaluate_multi_query_retrieval(
                    model,
                    cfg,
                    retrieval=multi_query_retrieval[length],
                    device=device,
                    dtype=dtype,
                    inference_scale=rope_pi_scale,
                    progress=progress,
                ),
            )
            if qasper_examples is not None and tokenizer is not None:
                run_part(
                    length=length,
                    part="rope_pi_qasper",
                    container=rope_real,
                    key="qasper",
                    compute=lambda progress: _evaluate_qasper(
                        model,
                        cfg,
                        examples=qasper_examples,
                        tokenizer=tokenizer,
                        length=length,
                        device=device,
                        dtype=dtype,
                        inference_scale=rope_pi_scale,
                        progress=progress,
                    ),
                )
        log_resources(cfg, "evaluate", device=device, length=length)
        log_fields(
            "eval",
            cfg=cfg,
            method=cfg["run"]["method"],
            checkpoint=checkpoint_kind,
            length=length,
            state="length_done",
        )

    retrieval_rows = {
        key: value["synthetic_control"]["single_query"]
        for key, value in result["lengths"].items()
    }
    result["length_degradation_rate"] = _length_degradation_rate(
        retrieval_rows
    )
    if str(cfg["run"]["method"]) == "rope":
        rope_pi_rows = {
            key: {
                "accuracy": value["rope_pi"]["synthetic_control"][
                    "single_query"
                ]["accuracy"]
            }
            for key, value in result["lengths"].items()
            if "rope_pi" in value
        }
        if rope_pi_rows:
            result["rope_pi_length_degradation_rate"] = (
                _length_degradation_rate(rope_pi_rows)
            )
    result["status"] = "completed"
    write_json(final_path, result)
    partial_path.unlink(missing_ok=True)
    stage_banner("EVALUATE", "DONE", cfg=cfg)
    return result


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for checkpoint_kind in cfg["eval"]["checkpoints"]:
        checkpoint_cfg = deepcopy(cfg)
        results[str(checkpoint_kind)] = _run_checkpoint(
            checkpoint_cfg,
            str(checkpoint_kind),
        )
    combined = {
        "method": cfg["run"]["method"],
        "seed": cfg["run"]["seed"],
        "checkpoints": results,
    }
    write_json(metric_dir(cfg) / "evaluation.json", combined)
    return combined
