"""CLUTRR acquisition and relation-classification experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data_download import download_file, safe_extract_zip_recursive, sha256_file
from .external_model import ByteTokenizer, SequenceClassifierTransformer
from .locking import RunLock
from .reporting import build_report
from .research_model import (
    BUS_METHODS,
    ResearchModelConfig,
    estimated_model_macs,
    initialize_named_parameters,
    parameter_count,
)
from .resources import (
    ResourcePolicy,
    WorkloadSpec,
    detect_resources,
    plan_jobs,
    write_resource_outputs,
)
from .runner import (
    _autocast,
    _optimizer,
    _restore_rng,
    _rng_state,
    _scheduler,
    _source_fingerprint,
    select_device,
)
from .scheduler import ScheduledCommand, execute_schedule
from .training_log import (
    CumulativeTrainingTimer,
    FixedWidthTrainingLogger,
    checkpoint_due,
    current_device_log_status,
)
from .storage import prepare_run_checkpoint_dir, prepare_storage, storage_roots


@dataclass(frozen=True)
class RelationExample:
    text: str
    label: int
    relation: str
    length: int


@dataclass(frozen=True)
class ClutrrBundle:
    root: Path
    manifest_path: Path
    train_path: Path
    validation_path: Path
    test_path: Path
    labels: tuple[str, ...]


@dataclass(frozen=True)
class _ParsedRelationRow:
    text: str
    target: str
    length: int | None
    split: str | None


def _relation_length(path: Path) -> int | None:
    # The official ZIP stores Windows backslashes in member names. On Linux,
    # zipfile preserves those characters instead of treating them as path
    # separators, so normalize both styles before matching task lengths.
    text = "/".join(path.parts).lower().replace("\\", "/")
    patterns = (
        r"(?:^|[/_.-])(?:task)?[1-7][._-]([2-6])(?:[/_.-]|$)",
        r"(?:^|[/_.-])k(?:=|_|-)?([2-6])(?:[/_.-]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _normalize_split(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    if normalized in {"train", "training", "trian"}:
        return "train"
    if normalized in {"validation", "valid", "val", "dev", "development"}:
        return "validation"
    if normalized in {"test", "testing"}:
        return "test"
    return None


def _split_from_path(path: Path) -> str | None:
    lower = "/".join(path.parts).lower().replace("\\", "/")
    for name in ("validation", "valid", "val", "dev"):
        if re.search(rf"(?:^|[/_.-]){name}(?:[/_.-]|$)", lower):
            return "validation"
    for name in ("train", "test"):
        if re.search(rf"(?:^|[/_.-]){name}(?:[/_.-]|$)", lower):
            return name
    return None


def _read_csv_rows(
    path: Path,
    *,
    default_length: int | None = None,
    default_split: str | None = None,
) -> Iterable[_ParsedRelationRow]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return
        names = {name.lower().strip(): name for name in reader.fieldnames}
        story_key = next(
            (names[key] for key in ("story", "text", "clean_story") if key in names),
            None,
        )
        query_key = next(
            (names[key] for key in ("query", "question") if key in names), None
        )
        # The official archive has used both text-valued ``target`` and the
        # Hugging Face schema's integer ``target`` + text ``target_text``.
        target_key = next(
            (
                names[key]
                for key in ("target_text", "relation", "answer", "target")
                if key in names
            ),
            None,
        )
        task_key = next(
            (names[key] for key in ("task_name", "task") if key in names), None
        )
        split_key = next(
            (names[key] for key in ("task_split", "split", "set") if key in names),
            None,
        )
        if story_key is None or target_key is None:
            return
        for row in reader:
            story = str(row.get(story_key, "")).strip()
            query = str(row.get(query_key, "")).strip() if query_key else ""
            target = str(row.get(target_key, "")).strip().lower()
            if story and target:
                task = str(row.get(task_key, "")).strip() if task_key else ""
                row_length = _relation_length(Path(task)) if task else None
                split_value = str(row.get(split_key, "")).strip() if split_key else ""
                yield _ParsedRelationRow(
                    text=f"{story} Query: {query} Answer:",
                    target=target,
                    length=row_length if row_length is not None else default_length,
                    split=_normalize_split(split_value)
                    or _normalize_split(default_split),
                )


def _collect_clutrr_rows(
    paths: Iterable[Path],
    *,
    root: Path,
    dataset_id: str,
    train_lengths: set[int],
    test_lengths: set[int],
) -> tuple[
    list[tuple[str, str, int]],
    list[tuple[str, str, int]],
    list[tuple[str, str, int]],
    list[dict[str, Any]],
]:
    candidates = sorted(set(paths))
    selected = [path for path in candidates if dataset_id.lower() in str(path).lower()]
    if selected:
        candidates = selected
    outputs: dict[str, list[tuple[str, str, int]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    inspected: list[dict[str, Any]] = []
    for path in candidates:
        default_length = _relation_length(path)
        default_split = _split_from_path(path)
        read = accepted = 0
        seen_lengths: set[int] = set()
        seen_splits: set[str] = set()
        for row in _read_csv_rows(
            path,
            default_length=default_length,
            default_split=default_split,
        ):
            read += 1
            if row.length is None or row.split is None:
                continue
            allowed = test_lengths if row.split == "test" else train_lengths
            if row.length not in allowed:
                continue
            outputs[row.split].append((row.text, row.target, row.length))
            accepted += 1
            seen_lengths.add(row.length)
            seen_splits.add(row.split)
        try:
            relative = str(path.relative_to(root))
        except ValueError:
            relative = str(path)
        inspected.append(
            {
                "path": relative,
                "rows_read": read,
                "rows_accepted": accepted,
                "lengths": sorted(seen_lengths),
                "splits": sorted(seen_splits),
            }
        )
    return outputs["train"], outputs["validation"], outputs["test"], inspected


def _download_clutrr_mirror(
    cfg: dict[str, Any], root: Path
) -> tuple[list[Path], list[dict[str, str]]]:
    settings = cfg["external"]["clutrr"]
    urls = settings.get("fallback_csv_urls", {})
    if set(urls) != {"train", "validation", "test"}:
        return [], []
    mirror = root / "mirror" / str(settings.get("fallback_split", "published"))
    paths: list[Path] = []
    sources: list[dict[str, str]] = []
    for split in ("train", "validation", "test"):
        path = mirror / f"{split}.csv"
        download_file(
            str(urls[split]),
            path,
            timeout=float(cfg["external"]["download_timeout_seconds"]),
        )
        paths.append(path)
        sources.append(
            {"split": split, "url": str(urls[split]), "sha256": sha256_file(path)}
        )
    return paths, sources


def _write_examples(path: Path, examples: list[RelationExample]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")


def _load_examples(path: Path) -> list[RelationExample]:
    with path.open("r", encoding="utf-8") as handle:
        return [RelationExample(**json.loads(line)) for line in handle if line.strip()]


def _normalization_config(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "dataset_id": str(settings.get("dataset_id", "data_089907f8")),
        "fallback_split": str(settings.get("fallback_split", "")),
        "train_rows": int(settings["train_rows"]),
        "validation_rows": int(settings["validation_rows"]),
        "test_rows_per_length": int(settings["test_rows_per_length"]),
        "train_lengths": sorted(int(value) for value in settings["train_lengths"]),
        "test_lengths": sorted(int(value) for value in settings["test_lengths"]),
        "shuffle_seed": 20260806,
    }


def ensure_clutrr_dataset(cfg: dict[str, Any], data_root: Path) -> ClutrrBundle:
    settings = cfg["external"]["clutrr"]
    root = data_root / "clutrr"
    normalized = root / "normalized"
    manifest_path = normalized / "manifest.json"
    paths = {name: normalized / f"{name}.jsonl" for name in ("train", "validation", "test")}
    normalization_config = _normalization_config(settings)
    if manifest_path.exists() and all(path.exists() for path in paths.values()):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_manifest = manifest.get("splits", {})
        valid_cache = (
            manifest.get("normalization_config") == normalization_config
            and bool(manifest.get("labels"))
            and all(
                split_manifest.get(name, {}).get("sha256")
                == sha256_file(path)
                for name, path in paths.items()
            )
        )
        if valid_cache:
            return ClutrrBundle(
                normalized,
                manifest_path,
                paths["train"],
                paths["validation"],
                paths["test"],
                tuple(manifest["labels"]),
            )
    if not bool(cfg["external"].get("allow_download", True)):
        raise FileNotFoundError(
            f"CLUTRR is not cached at {root} and external.allow_download is false"
        )
    archive = root / "data_publish.zip"
    download_file(
        str(settings["archive_url"]),
        archive,
        timeout=float(cfg["external"]["download_timeout_seconds"]),
    )
    extracted = root / "extracted"
    safe_extract_zip_recursive(archive, extracted)
    train_lengths = {int(value) for value in settings["train_lengths"]}
    test_lengths = {int(value) for value in settings["test_lengths"]}
    dataset_id = normalization_config["dataset_id"]
    raw_train, raw_validation, raw_test, inspected = _collect_clutrr_rows(
        extracted.rglob("*.csv"),
        root=extracted,
        dataset_id=dataset_id,
        train_lengths=train_lengths,
        test_lengths=test_lengths,
    )
    source_mode = "official_archive"
    fallback_sources: list[dict[str, str]] = []
    if not raw_train or not raw_test:
        mirror_paths, fallback_sources = _download_clutrr_mirror(cfg, root)
        mirror_train, mirror_validation, mirror_test, mirror_inspected = (
            _collect_clutrr_rows(
                mirror_paths,
                root=root,
                dataset_id=dataset_id,
                train_lengths=train_lengths,
                test_lengths=test_lengths,
            )
        )
        inspected.extend(mirror_inspected)
        if mirror_train and mirror_test:
            raw_train, raw_validation, raw_test = (
                mirror_train,
                mirror_validation,
                mirror_test,
            )
            source_mode = "huggingface_loader_mirror"
    if not raw_train or not raw_test:
        raise RuntimeError(
            "CLUTRR sources were downloaded but no compatible train/test rows "
            "were found after nested-ZIP, task_name and mirror parsing; "
            f"inspected={inspected[:20]}"
        )
    labels = tuple(
        sorted(
            {
                target
                for _, target, _ in raw_train + raw_validation + raw_test
            }
        )
    )
    label_to_id = {label: index for index, label in enumerate(labels)}
    generator = random.Random(20260806)
    generator.shuffle(raw_train)
    generator.shuffle(raw_validation)
    generator.shuffle(raw_test)
    validation_rows = int(settings["validation_rows"])
    if raw_validation:
        train_raw = raw_train[: int(settings["train_rows"])]
        validation_raw = raw_validation[:validation_rows]
    else:
        reserve = min(validation_rows, max(1, len(raw_train) // 5))
        train_rows = min(
            int(settings["train_rows"]), max(0, len(raw_train) - reserve)
        )
        validation_raw = raw_train[train_rows : train_rows + reserve]
        train_raw = raw_train[:train_rows]
    per_length = int(settings["test_rows_per_length"])
    test_raw = []
    for length in sorted(test_lengths):
        test_raw.extend([row for row in raw_test if row[2] == length][:per_length])
    available_test_lengths = {row[2] for row in test_raw}
    missing_test_lengths = sorted(test_lengths - available_test_lengths)
    if not train_raw or not validation_raw or missing_test_lengths:
        raise RuntimeError(
            "CLUTRR normalized split is incomplete: "
            f"train={len(train_raw)}, validation={len(validation_raw)}, "
            f"test={len(test_raw)}, missing_test_lengths={missing_test_lengths}"
        )

    def convert(rows: list[tuple[str, str, int]]) -> list[RelationExample]:
        return [
            RelationExample(text, label_to_id[target], target, length)
            for text, target, length in rows
        ]

    splits = {
        "train": convert(train_raw),
        "validation": convert(validation_raw),
        "test": convert(test_raw),
    }
    normalized.mkdir(parents=True, exist_ok=True)
    for name, examples in splits.items():
        _write_examples(paths[name], examples)
    manifest = {
        "benchmark": "CLUTRR",
        "normalization_config": normalization_config,
        "source": str(settings["archive_url"]),
        "source_mode": source_mode,
        "dataset_id": dataset_id,
        "license": "CC-BY-NC-4.0",
        "archive_sha256": sha256_file(archive),
        "fallback_sources": fallback_sources,
        "labels": labels,
        "files_inspected": inspected,
        "splits": {
            name: {"size": len(values), "sha256": sha256_file(paths[name])}
            for name, values in splits.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ClutrrBundle(
        normalized,
        manifest_path,
        paths["train"],
        paths["validation"],
        paths["test"],
        labels,
    )


@lru_cache(maxsize=100_000)
def _cached_byte_encoding(text: str, max_length: int) -> Tensor:
    """Tokenize each immutable CLUTRR story only once per worker process."""

    return torch.tensor(ByteTokenizer().encode(text, max_length), dtype=torch.long)


def _collate(
    examples: list[RelationExample], tokenizer: ByteTokenizer, max_length: int
) -> dict[str, Tensor]:
    del tokenizer  # ByteTokenizer has a fixed, parameter-free vocabulary.
    encoded = [_cached_byte_encoding(example.text, max_length) for example in examples]
    width = max(row.numel() for row in encoded)
    input_ids = torch.zeros(len(examples), width, dtype=torch.long)
    mask = torch.zeros_like(input_ids)
    for index, row in enumerate(encoded):
        input_ids[index, : row.numel()] = row
        mask[index, : row.numel()] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": mask,
        "targets": torch.tensor([example.label for example in examples]),
        "lengths": torch.tensor([example.length for example in examples]),
    }


def _batches(
    examples: list[RelationExample],
    tokenizer: ByteTokenizer,
    max_length: int,
    batch_size: int,
) -> Iterable[dict[str, Tensor]]:
    for start in range(0, len(examples), batch_size):
        yield _collate(examples[start : start + batch_size], tokenizer, max_length)


@torch.no_grad()
def _evaluate_classifier(
    model: nn.Module,
    examples: list[RelationExample],
    tokenizer: ByteTokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
    *,
    intervention: str | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = total = correct = 0
    by_length: dict[int, list[int]] = {}
    for raw in _batches(examples, tokenizer, max_length, batch_size):
        batch = {key: value.to(device) for key, value in raw.items()}
        with _autocast(device):
            output = model(
                batch["input_ids"],
                batch["attention_mask"],
                intervention=intervention,
            )
            loss = F.cross_entropy(output.logits, batch["targets"])
        predictions = output.logits.argmax(-1)
        hits = predictions == batch["targets"]
        count = len(predictions)
        total_loss += float(loss) * count
        total += count
        correct += int(hits.sum())
        for length in batch["lengths"].unique().tolist():
            selected = batch["lengths"] == length
            slot = by_length.setdefault(int(length), [0, 0])
            slot[0] += int(hits[selected].sum())
            slot[1] += int(selected.sum())
    metrics = {"loss": total_loss / max(1, total), "accuracy": correct / max(1, total)}
    for length, (hits, count) in sorted(by_length.items()):
        metrics[f"accuracy_length_{length}"] = hits / max(1, count)
    model.train()
    return metrics


def _model_config(
    cfg: dict[str, Any], method: str, max_length: int, size_name: str = "large"
) -> ResearchModelConfig:
    values = cfg["sizes"][size_name]["model"]
    defaults = cfg["model_defaults"]
    return ResearchModelConfig(
        vocab_size=ByteTokenizer.vocab_size,
        max_length=max_length,
        method=method,
        num_layers=int(values["num_layers"]),
        d_model=int(values["d_model"]),
        d_ff=int(values["d_ff"]),
        num_heads=int(values["num_heads"]),
        slot_dim=int(values["slot_dim"]),
        num_bus_slots=int(values["num_bus_slots"]),
        bus_heads=int(values["bus_heads"]),
        bus_layers=int(values["bus_layers"]),
        concept_residual_dim=int(values["concept_residual_dim"]),
        dropout=float(values["dropout"]),
        norm_eps=float(defaults["norm_eps"]),
        rope_theta=float(defaults["rope_theta"]),
        bias=bool(defaults["bias"]),
        keep_residual_attention=bool(defaults["keep_residual_attention"]),
    )


def run_clutrr_one(
    cfg: dict[str, Any], *, method: str, seed: int, output_root: Path,
    size_name: str = "large", stage: str | None = None,
) -> Path:
    roots = prepare_storage(cfg, output_root)
    bundle = ensure_clutrr_dataset(cfg, roots.data)
    settings = cfg["external"]["clutrr"]
    size = cfg["sizes"][size_name]
    stage = stage or ("pilot_clutrr" if size_name == "medium" else "clutrr")
    max_length = int(settings["max_length"])
    max_steps = int(settings["max_steps"])
    global_batch = int(settings["batch_size"])
    micro_batch = int(os.environ.get("CONCEPT_BUS_MICRO_BATCH", global_batch))
    accumulation = int(
        os.environ.get("CONCEPT_BUS_GRAD_ACCUM", max(1, global_batch // micro_batch))
    )
    if micro_batch * accumulation != global_batch:
        raise RuntimeError("CLUTRR micro batch and accumulation do not match global batch")
    config_payload = {
        "task": stage,
        "size": size_name,
        "method": method,
        "seed": seed,
        "settings": settings,
        "model": size["model"],
        "manifest": sha256_file(bundle.manifest_path),
        "source": _source_fingerprint(),
        "runtime_batching": {
            "global_batch_size": global_batch,
            "micro_batch_size": micro_batch,
            "gradient_accumulation_steps": accumulation,
        },
    }
    digest = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    run_dir = output_root / "runs" / stage / method / f"seed{seed}" / digest / "attempt1"
    checkpoint_dir = prepare_run_checkpoint_dir(cfg, output_root, run_dir)
    final_path = run_dir / "metrics" / "final.json"
    if final_path.exists() and cfg["run"]["skip_completed"]:
        print(f"skip completed | {stage}/{method}/seed{seed}")
        return run_dir
    for name in ("metrics", "logs"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    run_lock = RunLock(run_dir, final_path)
    if not run_lock.acquire():
        return run_dir
    random.seed(seed)
    torch.manual_seed(seed)
    device, diagnostics = select_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    tokenizer = ByteTokenizer()
    train = _load_examples(bundle.train_path)
    validation = _load_examples(bundle.validation_path)
    test = _load_examples(bundle.test_path)
    model = SequenceClassifierTransformer(
        _model_config(cfg, method, max_length, size_name), len(bundle.labels)
    )
    initialize_named_parameters(model, seed)
    model.to(device)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                **config_payload,
                "model_resolved": model.config.to_dict(),
                "max_steps": max_steps,
                "global_batch_size": global_batch,
                "micro_batch_size": micro_batch,
                "gradient_accumulation_steps": accumulation,
                "data_manifest": str(bundle.manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "device": str(device),
                "device_diagnostics": diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if method == "parameter_matched":
        target_values = model.config.to_dict()
        target_values.update({"method": "concept_bus_v2", "matched_ffn_width": None})
        with torch.device("meta"):
            target = SequenceClassifierTransformer(
                ResearchModelConfig(**target_values), len(bundle.labels)
            )
        target_parameters = parameter_count(target)
        actual_parameters = parameter_count(model)
        target_macs = estimated_model_macs(target.config, max_length)
        actual_macs = estimated_model_macs(model.config, max_length)
        matching = {
            "target_parameters": target_parameters,
            "actual_parameters": actual_parameters,
            "parameter_difference_fraction": abs(actual_parameters - target_parameters)
            / target_parameters,
            "target_macs": target_macs,
            "actual_macs": actual_macs,
            "mac_difference_fraction": abs(actual_macs - target_macs) / target_macs,
        }
        if method == "parameter_matched" and matching["parameter_difference_fraction"] > 0.01:
            raise RuntimeError("CLUTRR parameter match exceeds 1% tolerance")
        (run_dir / "matching_report.json").write_text(
            json.dumps(matching, indent=2), encoding="utf-8"
        )
        del target
    train_config = {
        "train": {
            "max_steps": max_steps,
            "learning_rate": float(size["train"]["learning_rate"]),
            "weight_decay": float(size["train"]["weight_decay"]),
            "warmup_fraction": float(size["train"]["warmup_fraction"]),
        },
        "optimizer": cfg["optimizer"],
    }
    optimizer = _optimizer(model, train_config)
    scheduler = _scheduler(optimizer, train_config)
    generator = torch.Generator().manual_seed(20_000 + seed)
    timer = CumulativeTrainingTimer()
    start = 0
    latest = checkpoint_dir / "latest.pt"
    if latest.exists() and cfg["run"]["resume"]:
        state = torch.load(latest, map_location=device, weights_only=False)
        if state.get("config_hash") != digest:
            raise RuntimeError("CLUTRR checkpoint config hash mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["generator"])
        timer.load_state_dict(state["timer"])
        if "rng" in state:
            _restore_rng(state["rng"])
        start = int(state["step"])
    logger_cfg = cfg["logging"]
    log_interval = int(size["train"]["log_interval_steps"])
    eval_interval = int(size["train"]["eval_interval_steps"])
    checkpoint_interval_minutes = float(
        size["train"]["checkpoint_interval_minutes"]
    )
    last_checkpoint_seconds = timer.elapsed_seconds

    def save(step: int) -> None:
        temporary = latest.with_suffix(".tmp")
        torch.save(
            {
                "step": step,
                "config_hash": digest,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "generator": generator.get_state(),
                "timer": timer.state_dict(),
                "rng": _rng_state(),
            },
            temporary,
        )
        temporary.replace(latest)

    with FixedWidthTrainingLogger(
        run_dir / "logs" / str(logger_cfg["file_name"]),
        timer=timer,
        device_provider=lambda: current_device_log_status(device),
        widths=logger_cfg["column_widths"],
        console_widths=logger_cfg["console_column_widths"],
        console_mode=str(logger_cfg["console_mode"]),
        flush_each_line=bool(logger_cfg["flush_each_line"]),
    ) as logger:
        logger.log("system", step=start, total_steps=max_steps, seed=seed, task=f"{stage}/{method}")
        model.train()
        for step in range(start + 1, max_steps + 1):
            timer.start()
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for _ in range(accumulation):
                indices = torch.randint(len(train), (micro_batch,), generator=generator).tolist()
                raw = _collate([train[index] for index in indices], tokenizer, max_length)
                batch = {key: value.to(device) for key, value in raw.items()}
                with _autocast(device):
                    output = model(batch["input_ids"], batch["attention_mask"])
                    loss = F.cross_entropy(output.logits, batch["targets"])
                (loss / accumulation).backward()
                loss_value += float(loss.detach()) / accumulation
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["optimizer"]["gradient_clip"]))
            optimizer.step()
            scheduler.step()
            timer.pause()
            if step == 1 or step % log_interval == 0 or step == max_steps:
                logger.log("train", step=step, total_steps=max_steps, loss=loss_value, seed=seed, task=f"{stage}/{method}")
            if step % eval_interval == 0 or step == max_steps:
                metrics = _evaluate_classifier(model, validation, tokenizer, max_length, micro_batch, device)
                logger.log("valid", step=step, total_steps=max_steps, loss=metrics["loss"], seed=seed, task=f"{stage}/{method}")
            if checkpoint_due(
                timer,
                last_checkpoint_seconds,
                checkpoint_interval_minutes,
                final_step=step == max_steps,
            ):
                save(step)
                last_checkpoint_seconds = timer.elapsed_seconds
                logger.log("checkpt", step=step, total_steps=max_steps, seed=seed, task=f"{stage}/{method}")
        metrics = _evaluate_classifier(model, test, tokenizer, max_length, micro_batch, device)
        final_metrics = {f"test_{key}": value for key, value in metrics.items()}
        if method in BUS_METHODS:
            zero = _evaluate_classifier(model, test, tokenizer, max_length, micro_batch, device, intervention="zero_bus")
            final_metrics["zero_bus_accuracy_drop"] = metrics["accuracy"] - zero["accuracy"]
        final_metrics.update(
            {
                "parameters": parameter_count(model),
                "estimated_macs_per_example": estimated_model_macs(model.config, max_length),
                "training_seconds": timer.elapsed_seconds,
                "training_examples_per_second": max_steps * global_batch / max(1e-9, timer.elapsed_seconds),
                "peak_cuda_memory_bytes": float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0,
            }
        )
        final_path.write_text(
            json.dumps(
                {
                    "stage": stage,
                    "method": method,
                    "seed": seed,
                    "config_hash": digest,
                    "source_fingerprint": config_payload["source"],
                    "metrics": final_metrics,
                    "data_manifest": str(bundle.manifest_path),
                    "device_diagnostics": diagnostics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        # ``checkpoint_due(..., final_step=True)`` already persisted this
        # exact state at the final optimization step.
        logger.log("final", step=max_steps, total_steps=max_steps, loss=metrics["loss"], seed=seed, task=f"{stage}/{method}")
    run_lock.release()
    return run_dir


def run_clutrr_phase(
    cfg: dict[str, Any],
    *,
    size_name: str,
    phase: dict[str, Any],
    output_root: Path,
    project_root: Path,
    monitor_interval: float,
) -> dict[str, Any]:
    stage = str(phase.get("stage", "pilot_clutrr" if size_name == "medium" else "clutrr"))
    try:
        bundle = ensure_clutrr_dataset(
            cfg, storage_roots(cfg, output_root).data
        )
    except Exception as error:
        return {"status": "skipped", "reason": f"CLUTRR unavailable: {type(error).__name__}: {error}"}
    methods = list(phase["methods"])
    seeds = [
        int(value)
        for value in phase.get("seeds", cfg["sizes"][size_name]["seeds"])
    ]
    settings = cfg["resources"]
    jobs = []
    commands = []
    for method in methods:
        for seed in seeds:
            name = f"{stage}_{method}_seed{seed}"
            jobs.append(
                WorkloadSpec(
                    name=name,
                    global_batch_size=int(cfg["external"]["clutrr"]["batch_size"]),
                    max_micro_batch_size=max(
                        1,
                        int(
                            int(cfg["external"]["clutrr"]["batch_size"])
                            * float(phase.get("_micro_batch_scale", 1.0))
                        ),
                    ),
                    estimated_model_memory_gb=float(settings["estimated_model_memory_gb"]),
                    estimated_activation_memory_per_sample_gb=float(settings["estimated_activation_memory_per_sample_gb"]),
                )
            )
            commands.append(
                ScheduledCommand(
                    name=name,
                    command=(sys.executable, str(project_root / "main.py"), "external-one", "--task", "clutrr", "--size", size_name, "--stage", stage, "--method", method, "--seed", str(seed), "--output-root", str(output_root)),
                    cwd=str(project_root),
                )
            )
    snapshot = detect_resources(required_dtype=str(settings["required_dtype"]))
    plans = plan_jobs(jobs, snapshot, ResourcePolicy.from_dict(settings))
    write_resource_outputs(output_root, snapshot, plans)
    schedule = execute_schedule(plans, commands, output_root, monitor=True, monitor_interval_seconds=monitor_interval)
    statuses = json.loads((schedule / "status.json").read_text(encoding="utf-8"))
    failed = sum(item["status"] != "completed" for item in statuses)
    report = build_report(
        output_root,
        stage=stage,
        max_static_figures=int(cfg["report"]["max_static_figures"]),
        source_fingerprint=_source_fingerprint(),
    )
    return {
        "status": "completed" if not failed else "completed_with_failures",
        "runs": len(jobs),
        "failed_runs": failed,
        "data_manifest": str(bundle.manifest_path),
        "scheduler": str(schedule),
        "report": str(report),
    }
