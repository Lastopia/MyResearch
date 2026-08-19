from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.bootstrap import run as bootstrap_dependencies
from tools.io import read_json, write_json
from tools.log import log_fields, stage_banner
from tools.memory import process_rss_gb
from tools.paths import (
    data_dir,
    huggingface_cache_dir,
    qasper_dir,
    tokenizer_dir,
    wikitext_dir,
)


SPECIAL_TOKENS = {
    "bos": 0,
    "separator": 1,
    "query": 2,
    "filler": 3,
}


def _compact_token_count(value: int) -> str:
    count = int(value)
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.2f}K"
    return str(count)


def _split_progress(written: int, target: int, *, percentage: bool = False) -> str:
    value = f"{_compact_token_count(written)}/{_compact_token_count(target)}"
    if percentage:
        value += f" ({100.0 * int(written) / max(1, int(target)):.2f}%)"
    return value


class TokenBlockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, tokens: torch.Tensor, block_size: int) -> None:
        if tokens.ndim != 1:
            raise ValueError("tokens must be one-dimensional")
        if len(tokens) < block_size + 1:
            raise ValueError("not enough tokens for one block")
        self.tokens = tokens
        self.block_size = int(block_size)
        self.num_blocks = (len(tokens) - 1) // block_size

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.block_size
        block = self.tokens[start : start + self.block_size + 1].to(torch.long)
        return block[:-1], block[1:]


class MemmapTokenBlockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        path: str | Path,
        *,
        token_count: int,
        block_size: int,
        dtype: str = "uint16",
    ) -> None:
        self.path = str(Path(path))
        self.token_count = int(token_count)
        self.block_size = int(block_size)
        self.dtype = np.dtype(dtype)
        if self.token_count < self.block_size + 1:
            raise ValueError("not enough tokens for one block")
        expected_bytes = self.token_count * self.dtype.itemsize
        if Path(self.path).stat().st_size != expected_bytes:
            raise ValueError(
                f"Token file size mismatch: {self.path}; "
                f"expected {expected_bytes} bytes"
            )
        self.num_blocks = (self.token_count - 1) // self.block_size
        self._tokens: np.memmap | None = None

    def _array(self) -> np.memmap:
        if self._tokens is None:
            self._tokens = np.memmap(
                self.path,
                dtype=self.dtype,
                mode="r",
                shape=(self.token_count,),
            )
        return self._tokens

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.block_size
        array = np.asarray(
            self._array()[start : start + self.block_size + 1],
            dtype=np.int64,
        )
        block = torch.from_numpy(array)
        return block[:-1], block[1:]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_tokens"] = None
        return state


@dataclass
class RetrievalBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    relevant_positions: torch.Tensor
    distractor_positions: torch.Tensor
    target_distances: torch.Tensor
    irrelevant_mask: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "RetrievalBatch":
        return RetrievalBatch(
            input_ids=self.input_ids.to(device),
            labels=self.labels.to(device),
            relevant_positions=self.relevant_positions.to(device),
            distractor_positions=self.distractor_positions.to(device),
            target_distances=self.target_distances.to(device),
            irrelevant_mask=(
                self.irrelevant_mask.to(device)
                if self.irrelevant_mask is not None
                else None
            ),
        )

    def select(self, start: int, end: int) -> "RetrievalBatch":
        selection = slice(start, end)
        return RetrievalBatch(
            input_ids=self.input_ids[selection],
            labels=self.labels[selection],
            relevant_positions=self.relevant_positions[selection],
            distractor_positions=self.distractor_positions[selection],
            target_distances=self.target_distances[selection],
            irrelevant_mask=(
                self.irrelevant_mask[selection]
                if self.irrelevant_mask is not None
                else None
            ),
        )


@dataclass
class MultiQueryRetrievalBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    query_positions: torch.Tensor
    relevant_positions: torch.Tensor
    distractor_positions: torch.Tensor
    target_distances: torch.Tensor

    def to(self, device: torch.device | str) -> "MultiQueryRetrievalBatch":
        return MultiQueryRetrievalBatch(
            input_ids=self.input_ids.to(device),
            labels=self.labels.to(device),
            query_positions=self.query_positions.to(device),
            relevant_positions=self.relevant_positions.to(device),
            distractor_positions=self.distractor_positions.to(device),
            target_distances=self.target_distances.to(device),
        )

    def select(self, start: int, end: int) -> "MultiQueryRetrievalBatch":
        selection = slice(start, end)
        return MultiQueryRetrievalBatch(
            input_ids=self.input_ids[selection],
            labels=self.labels[selection],
            query_positions=self.query_positions[selection],
            relevant_positions=self.relevant_positions[selection],
            distractor_positions=self.distractor_positions[selection],
            target_distances=self.target_distances[selection],
        )


class RetrievalDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, batch: RetrievalBatch) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return self.batch.input_ids.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.batch.input_ids[index],
            "label": self.batch.labels[index],
            "relevant_position": self.batch.relevant_positions[index],
            "distractor_position": self.batch.distractor_positions[index],
            "target_distance": self.batch.target_distances[index],
        }


def generate_synthetic_tokens(
    token_count: int, vocab_size: int, seed: int
) -> torch.Tensor:
    if vocab_size < 32:
        raise ValueError("vocab_size must be at least 32")
    generator = torch.Generator().manual_seed(seed)
    content_vocab = vocab_size - 4
    steps = torch.randint(1, 8, (token_count,), generator=generator)
    tokens = torch.empty(token_count, dtype=torch.long)
    tokens[0] = 4 + seed % content_vocab
    for index in range(1, token_count):
        if index % 32 < 8 and index >= 32:
            tokens[index] = tokens[index - 32]
        else:
            tokens[index] = 4 + ((tokens[index - 1] - 4 + steps[index]) % content_vocab)
    return tokens


def generate_retrieval_batch(
    *,
    samples: int,
    length: int,
    vocab_size: int,
    num_pairs: int,
    seed: int,
) -> RetrievalBatch:
    if length < 16:
        raise ValueError("retrieval length must be at least 16")
    num_keys = min(32, (vocab_size - 4) // 2)
    if num_keys < max(4, num_pairs):
        raise ValueError("vocab_size is too small for retrieval keys and values")
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.full(
        (samples, length),
        SPECIAL_TOKENS["filler"],
        dtype=torch.long,
    )
    labels = torch.empty(samples, dtype=torch.long)
    relevant_positions = torch.empty(samples, dtype=torch.long)
    distractor_positions = torch.empty(samples, dtype=torch.long)
    irrelevant_mask = torch.zeros((samples, length), dtype=torch.bool)

    key_base = 4
    value_base = key_base + num_keys
    earliest = 2
    latest_target = max(earliest + 1, length // 2)
    for sample in range(samples):
        inputs[sample, 0] = SPECIAL_TOKENS["bos"]
        target_key_index = int(
            torch.randint(0, num_keys, (1,), generator=generator).item()
        )
        target_key = key_base + target_key_index
        target_value = value_base + target_key_index
        target_key_position = earliest + (
            sample * max(1, latest_target - earliest)
        ) // max(1, samples - 1)
        target_key_position = min(target_key_position, latest_target)
        inputs[sample, target_key_position] = target_key
        inputs[sample, target_key_position + 1] = target_value

        distractor_key_index = (target_key_index + 1) % num_keys
        distractor_key = key_base + distractor_key_index
        distractor_value = value_base + distractor_key_index
        distractor_key_position = length - 6
        inputs[sample, distractor_key_position] = distractor_key
        inputs[sample, distractor_key_position + 1] = distractor_value
        irrelevant_mask[
            sample,
            distractor_key_position : distractor_key_position + 2,
        ] = True

        available_positions = list(range(2, max(2, length - 8), 2))
        pair_count = min(num_pairs - 2, len(available_positions))
        for _ in range(pair_count):
            position = available_positions[
                int(torch.randint(0, len(available_positions), (1,), generator=generator))
            ]
            if abs(position - target_key_position) <= 2:
                continue
            random_key_index = int(
                torch.randint(0, num_keys, (1,), generator=generator).item()
            )
            inputs[sample, position] = key_base + random_key_index
            inputs[sample, position + 1] = value_base + random_key_index
            if (length - 1) - position >= max(1, length // 4):
                irrelevant_mask[sample, position : position + 2] = True

        inputs[sample, -2] = SPECIAL_TOKENS["query"]
        inputs[sample, -1] = target_key
        labels[sample] = target_value
        relevant_positions[sample] = target_key_position + 1
        distractor_positions[sample] = distractor_key_position + 1
        irrelevant_mask[sample, target_key_position : target_key_position + 2] = False

    target_distances = (length - 1) - relevant_positions
    return RetrievalBatch(
        input_ids=inputs,
        labels=labels,
        relevant_positions=relevant_positions,
        distractor_positions=distractor_positions,
        target_distances=target_distances,
        irrelevant_mask=irrelevant_mask,
    )


def generate_nested_retrieval_batches(
    *,
    samples: int,
    lengths: list[int],
    vocab_size: int,
    num_pairs: int,
    seed: int,
) -> dict[int, RetrievalBatch]:
    """Build length-matched controls with stable associations across lengths.

    Samples cycle through early, middle and late target regions at every
    length. The target key/value identity is stable across lengths, while its
    absolute position scales with the evaluated context length.
    """
    ordered_lengths = sorted({int(length) for length in lengths})
    if not ordered_lengths or ordered_lengths[0] < 16:
        raise ValueError("all retrieval lengths must be at least 16")
    num_keys = min(32, (vocab_size - 4) // 2)
    if num_keys < max(4, num_pairs):
        raise ValueError("vocab_size is too small for retrieval keys and values")
    generator = torch.Generator().manual_seed(seed)
    key_base = 4
    value_base = key_base + num_keys
    target_indices = torch.randint(
        0,
        num_keys,
        (samples,),
        generator=generator,
    )
    batches: dict[int, RetrievalBatch] = {}
    for length in ordered_lengths:
        inputs = torch.full(
            (samples, length),
            SPECIAL_TOKENS["filler"],
            dtype=torch.long,
        )
        inputs[:, 0] = SPECIAL_TOKENS["bos"]
        labels = value_base + target_indices
        relevant_positions = torch.empty(samples, dtype=torch.long)
        distractor_positions = torch.empty(samples, dtype=torch.long)
        irrelevant_mask = torch.zeros((samples, length), dtype=torch.bool)
        target_fractions = (0.125, 0.5, 0.875)
        target_start = 2
        target_end = max(target_start, length - 8)
        for sample in range(samples):
            target_index = int(target_indices[sample].item())
            fraction = target_fractions[sample % len(target_fractions)]
            target_key_position = target_start + round(
                fraction * max(0, target_end - target_start)
            )
            target_key_position = min(target_key_position, length - 8)
            inputs[sample, target_key_position] = key_base + target_index
            inputs[sample, target_key_position + 1] = value_base + target_index
            relevant_positions[sample] = target_key_position + 1

            distractor_index = (target_index + 1) % num_keys
            distractor_key_position = length - 6
            inputs[sample, distractor_key_position] = key_base + distractor_index
            inputs[sample, distractor_key_position + 1] = (
                value_base + distractor_index
            )
            distractor_positions[sample] = distractor_key_position + 1

            candidate_positions = [
                position
                for position in range(2, max(2, length - 8), 3)
                if abs(position - target_key_position) > 2
                and abs(position - distractor_key_position) > 2
            ]
            pair_count = min(
                max(0, num_pairs - 2),
                len(candidate_positions),
            )
            for pair_index in range(pair_count):
                position = candidate_positions[
                    (sample * 5 + pair_index * 7) % len(candidate_positions)
                ]
                random_index = int(
                    torch.randint(
                        0,
                        num_keys,
                        (1,),
                        generator=generator,
                    ).item()
                )
                if random_index in {target_index, distractor_index}:
                    random_index = (random_index + 2) % num_keys
                inputs[sample, position] = key_base + random_index
                inputs[sample, position + 1] = value_base + random_index
                if (length - 1) - position >= max(1, length // 4):
                    irrelevant_mask[sample, position : position + 2] = True

            inputs[sample, -2] = SPECIAL_TOKENS["query"]
            inputs[sample, -1] = key_base + target_index
        batches[length] = RetrievalBatch(
            input_ids=inputs,
            labels=labels.clone(),
            relevant_positions=relevant_positions.clone(),
            distractor_positions=distractor_positions,
            target_distances=(length - 1) - relevant_positions,
            irrelevant_mask=irrelevant_mask,
        )
    return batches


def generate_position_swap_batches(
    *,
    samples: int,
    lengths: list[int],
    vocab_size: int,
    num_pairs: int,
    seed: int,
) -> dict[int, tuple[RetrievalBatch, RetrievalBatch]]:
    """Return paired controls that exchange target and near-distractor records."""
    originals = generate_nested_retrieval_batches(
        samples=samples,
        lengths=lengths,
        vocab_size=vocab_size,
        num_pairs=num_pairs,
        seed=seed,
    )
    result: dict[int, tuple[RetrievalBatch, RetrievalBatch]] = {}
    for length, original in originals.items():
        swapped_inputs = original.input_ids.clone()
        for sample in range(samples):
            target_value = int(original.relevant_positions[sample].item())
            distractor_value = int(original.distractor_positions[sample].item())
            target_slice = swapped_inputs[
                sample,
                target_value - 1 : target_value + 1,
            ].clone()
            distractor_slice = swapped_inputs[
                sample,
                distractor_value - 1 : distractor_value + 1,
            ].clone()
            swapped_inputs[
                sample,
                target_value - 1 : target_value + 1,
            ] = distractor_slice
            swapped_inputs[
                sample,
                distractor_value - 1 : distractor_value + 1,
            ] = target_slice
        swapped = RetrievalBatch(
            input_ids=swapped_inputs,
            labels=original.labels.clone(),
            relevant_positions=original.distractor_positions.clone(),
            distractor_positions=original.relevant_positions.clone(),
            target_distances=(
                (length - 1) - original.distractor_positions
            ).clone(),
            irrelevant_mask=(
                original.irrelevant_mask.clone()
                if original.irrelevant_mask is not None
                else None
            ),
        )
        result[length] = (original, swapped)
    return result


def generate_multi_query_retrieval_batches(
    *,
    samples: int,
    lengths: list[int],
    vocab_size: int,
    queries_per_sample: int,
    similar_distractors: int,
    seed: int,
) -> dict[int, MultiQueryRetrievalBatch]:
    """Generate multi-query associative recall with shared-prefix distractors.

    Every key is a two-token ``(family, member)`` tuple. Distractors for a
    query share its family token and differ only in the member token, so this
    is an explicit near-key interference condition rather than a claim about
    natural-language semantic similarity.
    """
    ordered_lengths = sorted({int(length) for length in lengths})
    queries = int(queries_per_sample)
    distractors = int(similar_distractors)
    if queries <= 1:
        raise ValueError("queries_per_sample must be greater than one")
    if distractors <= 0:
        raise ValueError("similar_distractors must be positive")
    group_count = max(2, min(8, queries))
    associations = queries * (distractors + 1)
    suffix_count = max(16, associations + 4)
    prefix_base = 4
    suffix_base = prefix_base + group_count
    value_base = suffix_base + suffix_count
    if value_base + suffix_count > vocab_size:
        raise ValueError("vocab_size is too small for multi-query retrieval")

    result: dict[int, MultiQueryRetrievalBatch] = {}
    for length in ordered_lengths:
        query_region = 3 * queries
        if length < max(32, query_region + 3 * associations + 4):
            raise ValueError(
                "retrieval length is too short for the requested multi-query "
                "and distractor counts"
            )
        query_start = length - query_region
        slots = list(range(2, query_start - 2, 3))
        if len(slots) < associations:
            raise ValueError("not enough context slots for multi-query retrieval")
        input_ids = torch.full(
            (samples, length),
            SPECIAL_TOKENS["filler"],
            dtype=torch.long,
        )
        input_ids[:, 0] = SPECIAL_TOKENS["bos"]
        labels = torch.empty((samples, queries), dtype=torch.long)
        query_positions = torch.empty((samples, queries), dtype=torch.long)
        relevant_positions = torch.empty((samples, queries), dtype=torch.long)
        distractor_positions = torch.empty(
            (samples, queries, distractors),
            dtype=torch.long,
        )

        for sample in range(samples):
            generator = torch.Generator().manual_seed(
                int(seed) + length * 10_000 + sample
            )
            family = prefix_base + (sample % group_count)
            suffix_order = torch.randperm(
                suffix_count,
                generator=generator,
            )[:associations].tolist()
            used_slots: set[int] = set()
            target_slot_indices: list[int] = []
            centers = (0.125, 0.5, 0.875)
            for query_index in range(queries):
                desired = round(
                    centers[(sample + query_index) % len(centers)]
                    * (len(slots) - 1)
                )
                candidate = min(
                    (
                        index
                        for index in range(len(slots))
                        if index not in used_slots
                    ),
                    key=lambda index: abs(index - desired),
                )
                used_slots.add(candidate)
                target_slot_indices.append(candidate)
            remaining_slots = [
                index for index in range(len(slots)) if index not in used_slots
            ]
            suffix_cursor = 0
            distractor_cursor = 0
            for query_index, target_slot_index in enumerate(target_slot_indices):
                target_suffix_index = int(suffix_order[suffix_cursor])
                suffix_cursor += 1
                target_position = slots[target_slot_index]
                input_ids[
                    sample,
                    target_position : target_position + 3,
                ] = torch.tensor(
                    [
                        family,
                        suffix_base + target_suffix_index,
                        value_base + target_suffix_index,
                    ]
                )
                labels[sample, query_index] = value_base + target_suffix_index
                relevant_positions[sample, query_index] = target_position + 2

                for distractor_index in range(distractors):
                    suffix_index = int(suffix_order[suffix_cursor])
                    suffix_cursor += 1
                    slot_index = remaining_slots[distractor_cursor]
                    distractor_cursor += 1
                    position = slots[slot_index]
                    input_ids[
                        sample,
                        position : position + 3,
                    ] = torch.tensor(
                        [
                            family,
                            suffix_base + suffix_index,
                            value_base + suffix_index,
                        ]
                    )
                    distractor_positions[
                        sample,
                        query_index,
                        distractor_index,
                    ] = position + 2

                query_position = query_start + 3 * query_index
                input_ids[
                    sample,
                    query_position : query_position + 3,
                ] = torch.tensor(
                    [
                        SPECIAL_TOKENS["query"],
                        family,
                        suffix_base + target_suffix_index,
                    ]
                )
                query_positions[sample, query_index] = query_position + 2

        result[length] = MultiQueryRetrievalBatch(
            input_ids=input_ids,
            labels=labels,
            query_positions=query_positions,
            relevant_positions=relevant_positions,
            distractor_positions=distractor_positions,
            target_distances=query_positions - relevant_positions,
        )
    return result


def _split_local_tokens(tokens: torch.Tensor, cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    requested = [
        int(cfg["data"]["train_tokens"]),
        int(cfg["data"]["valid_tokens"]),
        int(cfg["data"]["test_tokens"]),
    ]
    total = sum(requested)
    if len(tokens) < total:
        raise ValueError(f"Local token file has {len(tokens)} tokens, expected {total}")
    train_end = requested[0]
    valid_end = train_end + requested[1]
    return {
        "train": tokens[:train_end],
        "valid": tokens[train_end:valid_end],
        "test": tokens[valid_end:total],
    }


def _split_counts(cfg: dict[str, Any]) -> dict[str, int]:
    return {
        "train": int(cfg["data"]["train_tokens"]),
        "valid": int(cfg["data"]["valid_tokens"]),
        "test": int(cfg["data"]["test_tokens"]),
    }


def _token_dtype(cfg: dict[str, Any]) -> np.dtype[Any]:
    dtype = np.dtype(str(cfg["data"]["token_dtype"]))
    vocab_size = int(cfg["data"]["vocab_size"])
    if dtype == np.dtype("uint16") and vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("uint16 token storage requires vocab_size <= 65536")
    return dtype


def _config_fingerprint(cfg: dict[str, Any]) -> str:
    relevant = {
        "source": cfg["data"]["source"],
        "local_tokens_path": cfg["data"]["local_tokens_path"],
        "seed": cfg["data"]["seed"],
        "tokenizer_name": cfg["data"]["tokenizer_name"],
        "tokenizer_revision": cfg["data"]["tokenizer_revision"],
        "fineweb_dataset": cfg["data"]["fineweb_dataset"],
        "fineweb_config": cfg["data"]["fineweb_config"],
        "fineweb_revision": cfg["data"]["fineweb_revision"],
        "token_dtype": cfg["data"]["token_dtype"],
        "vocab_size": cfg["data"]["vocab_size"],
        "block_size": cfg["data"]["block_size"],
        "splits": _split_counts(cfg),
        "leakage_shingle_width": cfg["data"]["leakage_shingle_width"],
        "leakage_shingle_stride": cfg["data"]["leakage_shingle_stride"],
        "leakage_min_shared_shingles": cfg["data"][
            "leakage_min_shared_shingles"
        ],
        "leakage_overlap_threshold": cfg["data"][
            "leakage_overlap_threshold"
        ],
    }
    encoded = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _binary_tokens_ready(cfg: dict[str, Any]) -> bool:
    target = data_dir(cfg)
    meta_path = target / "meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = read_json(meta_path)
        if meta.get("fingerprint") != _config_fingerprint(cfg):
            return False
        dtype = _token_dtype(cfg)
        for split, count in _split_counts(cfg).items():
            path = target / f"{split}.bin"
            if not path.exists() or path.stat().st_size != count * dtype.itemsize:
                return False
        return True
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _binary_data_ready(cfg: dict[str, Any]) -> bool:
    requires_external_assets = str(cfg["data"]["source"]) == "fineweb_edu"
    return (
        _binary_tokens_ready(cfg)
        and _retrieval_ready(cfg)
        and (
            not requires_external_assets
            or not bool(cfg["data"].get("prepare_wikitext", False))
            or _wikitext_ready(cfg)
        )
        and (
            not requires_external_assets
            or not bool(cfg["data"].get("prepare_qasper", False))
            or _qasper_ready(cfg)
        )
    )


def _wikitext_ready(cfg: dict[str, Any]) -> bool:
    target = wikitext_dir(cfg)
    meta_path = target / "meta.json"
    token_path = target / "test.bin"
    if not meta_path.exists() or not token_path.exists():
        return False
    try:
        meta = read_json(meta_path)
        return (
            meta.get("dataset_revision") == cfg["data"]["wikitext_revision"]
            and token_path.stat().st_size
            == int(meta["token_count"]) * _token_dtype(cfg).itemsize
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _qasper_fingerprint(cfg: dict[str, Any]) -> str:
    relevant = {
        "dataset": cfg["data"]["qasper_dataset"],
        "data_file": cfg["data"]["qasper_data_file"],
        "revision": cfg["data"]["qasper_revision"],
        "split": cfg["data"]["qasper_split"],
        "requested_samples": cfg["data"]["qasper_samples"],
        "tokenizer": cfg["data"]["tokenizer_name"],
        "tokenizer_revision": cfg["data"]["tokenizer_revision"],
        "eval_lengths": cfg["eval"]["lengths"],
        "selection_version": 2,
    }
    encoded = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _qasper_ready(cfg: dict[str, Any]) -> bool:
    target = qasper_dir(cfg)
    meta_path = target / "meta.json"
    examples_path = target / "examples.pt"
    if not meta_path.exists() or not examples_path.exists():
        return False
    try:
        return (
            read_json(meta_path).get("fingerprint")
            == _qasper_fingerprint(cfg)
        )
    except (OSError, json.JSONDecodeError):
        return False


def _retrieval_fingerprint(cfg: dict[str, Any]) -> str:
    relevant = {
        "seed": cfg["data"]["seed"],
        "vocab_size": cfg["data"]["vocab_size"],
        "samples": cfg["data"]["retrieval_train_samples"],
        "num_key_value_pairs": cfg["data"]["num_key_value_pairs"],
        "length": cfg["adapt"]["max_seq_len"],
    }
    encoded = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _retrieval_ready(cfg: dict[str, Any]) -> bool:
    target = data_dir(cfg)
    data_path = target / "retrieval_adapt.pt"
    meta_path = target / "retrieval_adapt.meta.json"
    if not data_path.exists() or not meta_path.exists():
        return False
    try:
        return (
            read_json(meta_path).get("fingerprint")
            == _retrieval_fingerprint(cfg)
        )
    except (OSError, json.JSONDecodeError):
        return False


def data_ready(cfg: dict[str, Any]) -> bool:
    return _binary_data_ready(cfg)


def _remove_obsolete_token_files(cfg: dict[str, Any]) -> None:
    for split in _split_counts(cfg):
        (data_dir(cfg) / f"{split}.pt").unlink(missing_ok=True)


def prepare_tokenizer(cfg: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    target = tokenizer_dir(cfg)
    if target.exists():
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                target,
                local_files_only=True,
            )
            if len(tokenizer) != int(cfg["data"]["vocab_size"]):
                raise ValueError(
                    f"Tokenizer vocab is {len(tokenizer)}, "
                    f"expected {cfg['data']['vocab_size']}"
                )
            log_fields(
                "asset",
                cfg=cfg,
                name="tokenizer",
                status="verified",
                path=target,
            )
            return tokenizer
        except (OSError, ValueError):
            log_fields(
                "asset",
                cfg=cfg,
                name="tokenizer",
                status="invalid_rebuild",
                path=target,
            )

    target.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(cfg["data"]["tokenizer_name"]),
        revision=str(cfg["data"]["tokenizer_revision"]),
        cache_dir=huggingface_cache_dir(cfg),
    )
    if len(tokenizer) != int(cfg["data"]["vocab_size"]):
        raise ValueError(
            f"Downloaded tokenizer vocab is {len(tokenizer)}, "
            f"expected {cfg['data']['vocab_size']}"
        )
    tokenizer.save_pretrained(target)
    AutoTokenizer.from_pretrained(target, local_files_only=True)
    log_fields(
        "asset",
        cfg=cfg,
        name="tokenizer",
        status="downloaded_and_verified",
        path=target,
    )
    return tokenizer


def _normalized_text_sha256(text: str) -> str:
    normalized = " ".join(str(text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _token_shingles(
    tokens: list[int] | np.ndarray,
    *,
    width: int,
    stride: int,
) -> set[int]:
    values = [int(value) + 1 for value in tokens]
    if len(values) < width:
        return set()
    modulus = 1 << 64
    mask = modulus - 1
    base = 1_000_003
    leading = pow(base, width - 1, modulus)
    current = 0
    for value in values[:width]:
        current = ((current * base) + value) & mask
    shingles = {current} if current % stride == 0 else set()
    for start in range(1, len(values) - width + 1):
        current = (
            ((current - values[start - 1] * leading) * base)
            + values[start + width - 1]
        ) & mask
        # Content-based sampling is invariant to inserted/deleted prefixes;
        # sampling by window index would miss the same passage after a shift.
        if current % stride == 0:
            shingles.add(current)
    return shingles


def _evaluation_shingle_index(
    cfg: dict[str, Any],
) -> tuple[dict[int, list[str]], dict[str, int]]:
    width = int(cfg["data"]["leakage_shingle_width"])
    stride = int(cfg["data"]["leakage_shingle_stride"])
    targets: dict[str, set[int]] = {}
    qasper_path = qasper_dir(cfg) / "examples.pt"
    if qasper_path.exists():
        payload = torch.load(
            qasper_path,
            map_location="cpu",
            weights_only=False,
        )
        for example in payload["examples"]:
            name = f"qasper:{example['paper_id']}"
            if name not in targets:
                targets[name] = _token_shingles(
                    example["document_token_ids"].tolist(),
                    width=width,
                    stride=stride,
                )
    wikitext_path = wikitext_dir(cfg) / "test.bin"
    if wikitext_path.exists():
        tokens = np.memmap(
            wikitext_path,
            dtype=_token_dtype(cfg),
            mode="r",
        )
        targets["wikitext103:test"] = _token_shingles(
            tokens,
            width=width,
            stride=stride,
        )
    inverted: dict[int, list[str]] = {}
    for target_name, shingles in targets.items():
        for shingle in shingles:
            inverted.setdefault(shingle, []).append(target_name)
    return inverted, {
        name: len(shingles) for name, shingles in targets.items()
    }


def _qasper_document(example: dict[str, Any]) -> str:
    parts = [str(example.get("title") or ""), str(example.get("abstract") or "")]
    full_text = example.get("full_text") or {}
    section_names = list(full_text.get("section_name") or [])
    paragraphs = list(full_text.get("paragraphs") or [])
    for index, section_paragraphs in enumerate(paragraphs):
        section = (
            section_names[index]
            if index < len(section_names)
            else None
        )
        if section:
            parts.append(str(section))
        if isinstance(section_paragraphs, list):
            parts.extend(str(item) for item in section_paragraphs if item)
        elif section_paragraphs:
            parts.append(str(section_paragraphs))
    return "\n\n".join(part.strip() for part in parts if part.strip())


def _qasper_annotation_texts(
    answer_group: dict[str, Any],
) -> tuple[list[str], list[str]]:
    answers: list[str] = []
    evidence: list[str] = []
    for annotation in answer_group.get("answer", []) or []:
        if bool(annotation.get("unanswerable", False)):
            continue
        free_form = str(annotation.get("free_form_answer") or "").strip()
        if free_form:
            answers.append(free_form)
        yes_no = annotation.get("yes_no")
        if yes_no is not None:
            answers.append("yes" if bool(yes_no) else "no")
        answers.extend(
            str(span).strip()
            for span in annotation.get("extractive_spans", []) or []
            if str(span).strip()
        )
        evidence.extend(
            str(item).strip()
            for item in annotation.get("evidence", []) or []
            if str(item).strip()
        )
    return list(dict.fromkeys(answers)), list(dict.fromkeys(evidence))


def prepare_qasper(cfg: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Prepare fixed, human-authored QASPER validation examples."""
    from datasets import load_dataset

    target = qasper_dir(cfg)
    meta_path = target / "meta.json"
    examples_path = target / "examples.pt"
    if _qasper_ready(cfg):
        meta = read_json(meta_path)
        log_fields("asset", cfg=cfg, name="qasper", status="verified")
        return meta

    target.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        str(cfg["data"]["qasper_dataset"]),
        data_files={
            str(cfg["data"]["qasper_split"]): str(
                cfg["data"]["qasper_data_file"]
            )
        },
        split=str(cfg["data"]["qasper_split"]),
        revision=str(cfg["data"]["qasper_revision"]),
        streaming=True,
        cache_dir=str(huggingface_cache_dir(cfg)),
    )
    requested = int(cfg["data"]["qasper_samples"])
    candidate_target = requested * 4
    maximum_length = max(int(value) for value in cfg["eval"]["lengths"])
    examples: list[dict[str, Any]] = []
    document_hashes: dict[str, str] = {}
    paper_ids: list[str] = []
    for paper in dataset:
        paper_id = str(paper["id"])
        document = _qasper_document(paper)
        document_tokens = tokenizer.encode(
            document,
            add_special_tokens=False,
        )
        if len(document_tokens) < maximum_length:
            continue
        document_hash = _normalized_text_sha256(document)
        qas = paper.get("qas") or {}
        questions = list(qas.get("question") or [])
        question_ids = list(qas.get("question_id") or [])
        answer_groups = list(qas.get("answers") or [])
        for index, question in enumerate(questions):
            if index >= len(question_ids) or index >= len(answer_groups):
                continue
            answer_texts, evidence_texts = _qasper_annotation_texts(
                answer_groups[index]
            )
            if not answer_texts or not evidence_texts:
                continue
            evidence_matches = [
                (document.find(text), text)
                for text in evidence_texts
                if document.find(text) >= 0
            ]
            if not evidence_matches:
                continue
            evidence_char_start, evidence_text = min(
                evidence_matches,
                key=lambda item: item[0],
            )
            evidence_char_end = evidence_char_start + len(evidence_text)
            evidence_token_start = len(
                tokenizer.encode(
                    document[:evidence_char_start],
                    add_special_tokens=False,
                )
            )
            evidence_token_end = len(
                tokenizer.encode(
                    document[:evidence_char_end],
                    add_special_tokens=False,
                )
            )
            encoded_answers = [
                tokenizer.encode(
                    " " + answer,
                    add_special_tokens=False,
                )
                for answer in answer_texts
            ]
            encoded_answers = [tokens for tokens in encoded_answers if tokens]
            if not encoded_answers:
                continue
            question_id = str(question_ids[index])
            sample_identity = {
                "paper_id": paper_id,
                "question_id": question_id,
                "document_sha256": document_hash,
                "evidence_sha256": _normalized_text_sha256(evidence_text),
            }
            examples.append(
                {
                    **sample_identity,
                    "sample_sha256": hashlib.sha256(
                        json.dumps(
                            sample_identity,
                            sort_keys=True,
                        ).encode("utf-8")
                    ).hexdigest(),
                    "title": str(paper.get("title") or ""),
                    "question": str(question),
                    "answer_texts": answer_texts,
                    "answer_token_ids": [
                        torch.tensor(tokens, dtype=torch.long)
                        for tokens in encoded_answers
                    ],
                    "document_token_ids": torch.tensor(
                        document_tokens,
                        dtype=torch.long,
                    ),
                    "evidence_token_start": evidence_token_start,
                    "evidence_token_end": evidence_token_end,
                }
            )
            document_hashes[paper_id] = document_hash
            if paper_id not in paper_ids:
                paper_ids.append(paper_id)
            if len(examples) >= candidate_target:
                break
        if len(examples) >= candidate_target:
            break
    if len(examples) < requested:
        raise RuntimeError(
            "QASPER preparation found only "
            f"{len(examples)} eligible examples, requested {requested}"
        )

    torch.save({"examples": examples}, examples_path)
    meta = {
        "fingerprint": _qasper_fingerprint(cfg),
        "dataset": cfg["data"]["qasper_dataset"],
        "data_file": cfg["data"]["qasper_data_file"],
        "dataset_revision": cfg["data"]["qasper_revision"],
        "split": cfg["data"]["qasper_split"],
        "license": "CC BY 4.0",
        "requested_samples": requested,
        "candidate_samples": len(examples),
        "paper_ids": paper_ids,
        "document_sha256_by_paper": document_hashes,
        "question_ids": [item["question_id"] for item in examples],
        "sample_sha256": [item["sample_sha256"] for item in examples],
        "tokenizer": cfg["data"]["tokenizer_name"],
        "tokenizer_revision": cfg["data"]["tokenizer_revision"],
        "processing_rule": (
            "validation questions with human answers and exact supporting "
            "evidence; documents must cover the maximum evaluation length; "
            "each evaluation length uses a contiguous real-paper window with "
            "the annotated evidence retained near the first eighth"
        ),
        "program_generated": False,
        "program_modified": True,
        "modification": (
            "paper text is tokenized and truncated by context budget; question "
            "and answer labels remain human-authored"
        ),
    }
    write_json(meta_path, meta)
    log_fields(
        "asset",
        cfg=cfg,
        name="qasper",
        status="downloaded_and_verified",
        samples=len(examples),
    )
    return meta


def _progress_path(cfg: dict[str, Any]) -> Path:
    return data_dir(cfg) / "prepare_progress.json"


def _initial_progress(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "fingerprint": _config_fingerprint(cfg),
        "documents_seen": 0,
        "tokens_written": dict.fromkeys(_split_counts(cfg), 0),
        "evaluation_leakage_audit": {
            "documents_checked": 0,
            "documents_excluded": 0,
            "matched_paper_ids": [],
            "matched_evaluation_targets": [],
            "match_details": [],
            "rules": [
                "QASPER arXiv paper ID appears in FineWeb URL or source ID",
                "normalized full-document SHA-256 exact match",
                "token-shingle near-duplicate overlap against QASPER and "
                "WikiText-103 evaluation documents",
            ],
        },
        "complete": False,
    }


def _load_progress(cfg: dict[str, Any]) -> dict[str, Any]:
    path = _progress_path(cfg)
    if path.exists():
        progress = read_json(path)
        if progress.get("fingerprint") == _config_fingerprint(cfg):
            progress.setdefault(
                "evaluation_leakage_audit",
                {
                    "documents_checked": 0,
                    "documents_excluded": 0,
                    "matched_paper_ids": [],
                    "matched_evaluation_targets": [],
                    "match_details": [],
                    "rules": [
                        "QASPER arXiv paper ID appears in FineWeb URL or source ID",
                        "normalized full-document SHA-256 exact match",
                        "token-shingle near-duplicate overlap against QASPER "
                        "and WikiText-103 evaluation documents",
                    ],
                },
            )
            return progress
    progress = _initial_progress(cfg)
    write_json(path, progress)
    return progress


def _open_partial_memmaps(
    cfg: dict[str, Any],
) -> dict[str, np.memmap]:
    target = data_dir(cfg)
    dtype = _token_dtype(cfg)
    required = sum(_split_counts(cfg).values()) * dtype.itemsize
    existing = sum(
        path.stat().st_size
        for path in target.glob("*.bin.partial")
        if path.is_file()
    )
    free = shutil.disk_usage(target).free
    safety = float(cfg["resources"]["disk_safety_fraction"])
    additional = max(0, required - existing)
    if additional > free * safety:
        raise RuntimeError(
            "Insufficient disk space for token files: "
            f"need {additional / (1024**3):.2f}GB additional, "
            f"free {free / (1024**3):.2f}GB"
        )
    result: dict[str, np.memmap] = {}
    for split, count in _split_counts(cfg).items():
        path = target / f"{split}.bin.partial"
        expected_bytes = count * dtype.itemsize
        if path.exists() and path.stat().st_size != expected_bytes:
            raise ValueError(
                f"Partial token file has wrong size: {path}. "
                "Delete this one partial file and rerun preparation."
            )
        mode = "r+" if path.exists() else "w+"
        result[split] = np.memmap(path, dtype=dtype, mode=mode, shape=(count,))
    return result


def _prepare_fineweb(cfg: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    from datasets import load_dataset

    if _binary_tokens_ready(cfg):
        meta = read_json(data_dir(cfg) / "meta.json")
        log_fields("asset", cfg=cfg, name="fineweb_edu", status="verified")
        return meta

    target = data_dir(cfg)
    target.mkdir(parents=True, exist_ok=True)
    progress = _load_progress(cfg)
    arrays = _open_partial_memmaps(cfg)
    dataset = load_dataset(
        str(cfg["data"]["fineweb_dataset"]),
        name=str(cfg["data"]["fineweb_config"]),
        split="train",
        revision=str(cfg["data"]["fineweb_revision"]),
        streaming=bool(cfg["data"]["streaming"]),
        cache_dir=str(huggingface_cache_dir(cfg)),
    )
    if bool(cfg["data"]["streaming"]):
        dataset = dataset.shuffle(
            seed=int(cfg["data"]["seed"]),
            buffer_size=int(cfg["data"]["shuffle_buffer"]),
        )
        already_seen = int(progress["documents_seen"])
        if already_seen:
            dataset = dataset.skip(already_seen)

    targets = _split_counts(cfg)
    order = ("train", "valid", "test")
    interval = max(1, int(cfg["data"]["progress_interval_documents"]))
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must provide eos_token_id")
    qasper_meta_path = qasper_dir(cfg) / "meta.json"
    qasper_meta = (
        read_json(qasper_meta_path) if qasper_meta_path.exists() else {}
    )
    qasper_paper_ids = set(qasper_meta.get("paper_ids", []))
    qasper_hash_to_paper = {
        document_hash: paper_id
        for paper_id, document_hash in qasper_meta.get(
            "document_sha256_by_paper",
            {},
        ).items()
    }
    leakage = progress["evaluation_leakage_audit"]
    shingle_index, target_shingle_counts = _evaluation_shingle_index(cfg)
    shingle_width = int(cfg["data"]["leakage_shingle_width"])
    shingle_stride = int(cfg["data"]["leakage_shingle_stride"])
    minimum_shared = int(cfg["data"]["leakage_min_shared_shingles"])
    overlap_threshold = float(cfg["data"]["leakage_overlap_threshold"])

    for example in dataset:
        current = next(
            (
                split
                for split in order
                if int(progress["tokens_written"][split]) < targets[split]
            ),
            None,
        )
        if current is None:
            break
        text = str(example["text"])
        source_identity = " ".join(
            str(example.get(field) or "")
            for field in ("id", "url", "file_path")
        )
        candidate_ids = set(
            re.findall(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", source_identity)
        )
        candidate_ids |= {
            item.removesuffix(match)
            for item in candidate_ids
            for match in re.findall(r"v\d+$", item)
        }
        matched_ids = sorted(qasper_paper_ids & candidate_ids)
        exact_hash = _normalized_text_sha256(text)
        exact_paper = qasper_hash_to_paper.get(exact_hash)
        if exact_paper and exact_paper not in matched_ids:
            matched_ids.append(exact_paper)
        identity_or_exact_matches = list(matched_ids)
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        document_shingles = _token_shingles(
            token_ids,
            width=shingle_width,
            stride=shingle_stride,
        )
        shared_counts: dict[str, int] = {}
        for shingle in document_shingles:
            for target_name in shingle_index.get(shingle, []):
                shared_counts[target_name] = (
                    shared_counts.get(target_name, 0) + 1
                )
        near_matches: list[dict[str, Any]] = []
        for target_name, shared in shared_counts.items():
            denominator = min(
                len(document_shingles),
                target_shingle_counts[target_name],
            )
            overlap = shared / max(1, denominator)
            if shared >= minimum_shared and overlap >= overlap_threshold:
                near_matches.append(
                    {
                        "target": target_name,
                        "shared_shingles": shared,
                        "overlap_coefficient": overlap,
                    }
                )
                if target_name.startswith("qasper:"):
                    paper_id = target_name.removeprefix("qasper:")
                    if paper_id not in matched_ids:
                        matched_ids.append(paper_id)
        matched_targets = {
            *(f"qasper:{paper_id}" for paper_id in matched_ids),
            *(match["target"] for match in near_matches),
        }
        for paper_id in matched_ids:
            if paper_id not in leakage["matched_paper_ids"]:
                leakage["matched_paper_ids"].append(paper_id)
        for target_name in sorted(matched_targets):
            if target_name not in leakage["matched_evaluation_targets"]:
                leakage["matched_evaluation_targets"].append(target_name)
        if matched_targets:
            leakage["documents_excluded"] = (
                int(leakage.get("documents_excluded", 0)) + 1
            )
            leakage["match_details"].append(
                {
                    "fineweb_source_identity": source_identity,
                    "matched_targets": sorted(matched_targets),
                    "source_id_or_exact_matches": sorted(
                        identity_or_exact_matches
                    ),
                    "exact_document_hash_target": exact_paper,
                    "near_duplicate_matches": near_matches,
                }
            )
        leakage["documents_checked"] = (
            int(leakage.get("documents_checked", 0)) + 1
        )
        progress["documents_seen"] = int(progress["documents_seen"]) + 1
        if matched_targets:
            write_json(_progress_path(cfg), progress)
            continue
        token_ids.append(int(eos_id))
        offset = int(progress["tokens_written"][current])
        take = min(len(token_ids), targets[current] - offset)
        if take:
            arrays[current][offset : offset + take] = np.asarray(
                token_ids[:take],
                dtype=_token_dtype(cfg),
            )
            progress["tokens_written"][current] = offset + take
        if int(progress["documents_seen"]) % interval == 0:
            for array in arrays.values():
                array.flush()
            write_json(_progress_path(cfg), progress)
            written = progress["tokens_written"]
            process_ram = process_rss_gb()
            log_fields(
                "data",
                source="FineWeb-Edu",
                docs=f"{int(progress['documents_seen']):,}",
                train=_split_progress(
                    int(written["train"]),
                    int(targets["train"]),
                    percentage=True,
                ),
                valid=_split_progress(
                    int(written["valid"]),
                    int(targets["valid"]),
                ),
                test=_split_progress(
                    int(written["test"]),
                    int(targets["test"]),
                ),
                ram=(
                    f"{process_ram:.2f}GB"
                    if process_ram is not None
                    else "n/a"
                ),
            )
    else:
        missing = {
            split: targets[split] - int(progress["tokens_written"][split])
            for split in order
            if int(progress["tokens_written"][split]) < targets[split]
        }
        if missing:
            raise RuntimeError(
                f"FineWeb-Edu stream ended before requested tokens: {missing}"
            )

    for array in arrays.values():
        array.flush()
    del arrays
    for split in order:
        partial = target / f"{split}.bin.partial"
        final = target / f"{split}.bin"
        os.replace(partial, final)
    progress["complete"] = True
    write_json(_progress_path(cfg), progress)
    meta = {
        "source": "fineweb_edu",
        "fingerprint": _config_fingerprint(cfg),
        "dataset": cfg["data"]["fineweb_dataset"],
        "dataset_config": cfg["data"]["fineweb_config"],
        "dataset_revision": cfg["data"]["fineweb_revision"],
        "tokenizer": cfg["data"]["tokenizer_name"],
        "token_dtype": str(_token_dtype(cfg)),
        "vocab_size": int(cfg["data"]["vocab_size"]),
        "block_size": int(cfg["data"]["block_size"]),
        "split_tokens": targets,
        "documents_seen": int(progress["documents_seen"]),
        "seed": int(cfg["data"]["seed"]),
        "document_boundary_policy": "a source document never crosses a split",
        "evaluation_leakage_audit": {
            **leakage,
            "complete_for_all_prepared_documents": (
                int(leakage.get("documents_checked", 0))
                == int(progress["documents_seen"])
            ),
        },
    }
    write_json(target / "meta.json", meta)
    return meta


def prepare_wikitext(cfg: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    from datasets import load_dataset

    target = wikitext_dir(cfg)
    meta_path = target / "meta.json"
    token_path = target / "test.bin"
    if meta_path.exists() and token_path.exists():
        meta = read_json(meta_path)
        expected_bytes = int(meta.get("token_count", -1)) * _token_dtype(cfg).itemsize
        if (
            meta.get("tokenizer") == cfg["data"]["tokenizer_name"]
            and meta.get("dataset_revision")
            == cfg["data"]["wikitext_revision"]
            and token_path.stat().st_size == expected_bytes
        ):
            log_fields("asset", cfg=cfg, name="wikitext103", status="verified")
            return meta

    target.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        str(cfg["data"]["wikitext_dataset"]),
        str(cfg["data"]["wikitext_config"]),
        split="test",
        revision=str(cfg["data"]["wikitext_revision"]),
        streaming=True,
        cache_dir=str(huggingface_cache_dir(cfg)),
    )
    tokens: list[int] = []
    for example in dataset:
        text = str(example["text"])
        if not text.strip():
            continue
        tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        tokens.append(int(tokenizer.eos_token_id))
    array = np.asarray(tokens, dtype=_token_dtype(cfg))
    array.tofile(token_path)
    meta = {
        "dataset": cfg["data"]["wikitext_dataset"],
        "dataset_config": cfg["data"]["wikitext_config"],
        "dataset_revision": cfg["data"]["wikitext_revision"],
        "split": "test",
        "tokenizer": cfg["data"]["tokenizer_name"],
        "token_dtype": str(_token_dtype(cfg)),
        "token_count": int(array.size),
        "token_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "processing_rule": (
            "drop empty rows, GPT-2 BPE without added special tokens, append "
            "tokenizer EOS after each retained row"
        ),
    }
    write_json(meta_path, meta)
    log_fields(
        "asset",
        cfg=cfg,
        name="wikitext103",
        status="downloaded_and_verified",
        tokens=int(array.size),
    )
    return meta


def _prepare_retrieval(cfg: dict[str, Any]) -> None:
    target = data_dir(cfg)
    path = target / "retrieval_adapt.pt"
    if _retrieval_ready(cfg) and not bool(cfg["run"].get("force", False)):
        return
    retrieval = generate_retrieval_batch(
        samples=int(cfg["data"]["retrieval_train_samples"]),
        length=int(cfg["adapt"]["max_seq_len"]),
        vocab_size=int(cfg["data"]["vocab_size"]),
        num_pairs=int(cfg["data"]["num_key_value_pairs"]),
        seed=int(cfg["data"]["seed"]) + 100,
    )
    torch.save(retrieval.__dict__, path)
    write_json(
        target / "retrieval_adapt.meta.json",
        {
            "fingerprint": _retrieval_fingerprint(cfg),
            "samples": int(cfg["data"]["retrieval_train_samples"]),
            "length": int(cfg["adapt"]["max_seq_len"]),
            "num_key_value_pairs": int(cfg["data"]["num_key_value_pairs"]),
            "seed": int(cfg["data"]["seed"]) + 100,
        },
    )


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    stage_banner("DATA", cfg=cfg)
    target_dir = data_dir(cfg)
    target_dir.mkdir(parents=True, exist_ok=True)
    source = str(cfg["data"]["source"])
    seed = int(cfg["data"]["seed"])
    vocab_size = int(cfg["data"]["vocab_size"])

    if data_ready(cfg) and not bool(cfg["run"].get("force", False)):
        _remove_obsolete_token_files(cfg)
        meta = read_json(target_dir / "meta.json")
        log_fields("asset", cfg=cfg, name="experiment_data", status="verified")
        stage_banner("DATA", "REUSED", cfg=cfg)
        return meta

    if source == "fineweb_edu":
        bootstrap_dependencies(cfg)
        tokenizer = prepare_tokenizer(cfg)
        if bool(cfg["data"].get("prepare_qasper", False)):
            prepare_qasper(cfg, tokenizer)
        if bool(cfg["data"]["prepare_wikitext"]):
            prepare_wikitext(cfg, tokenizer)
        meta = _prepare_fineweb(cfg, tokenizer)
        _prepare_retrieval(cfg)
        if not _binary_data_ready(cfg):
            raise RuntimeError("Prepared FineWeb-Edu files failed final validation")
        _remove_obsolete_token_files(cfg)
        stage_banner("DATA", "DONE", cfg=cfg)
        return meta
    if source == "synthetic":
        splits = {
            "train": generate_synthetic_tokens(
                int(cfg["data"]["train_tokens"]), vocab_size, seed
            ),
            "valid": generate_synthetic_tokens(
                int(cfg["data"]["valid_tokens"]), vocab_size, seed + 1
            ),
            "test": generate_synthetic_tokens(
                int(cfg["data"]["test_tokens"]), vocab_size, seed + 2
            ),
        }
    elif source == "local_tokens":
        local_path = cfg["data"].get("local_tokens_path")
        if not local_path:
            raise ValueError(
                "data.local_tokens_path is required for local_tokens. "
                "Formal FineWeb-Edu download is intentionally not automatic."
            )
        loaded = torch.load(Path(local_path), map_location="cpu", weights_only=False)
        tokens = loaded["tokens"] if isinstance(loaded, dict) else loaded
        splits = _split_local_tokens(tokens.to(torch.long), cfg)
    else:
        raise ValueError(f"Unsupported data source: {source}")

    dtype = _token_dtype(cfg)
    split_hashes: dict[str, str] = {}
    for split, tensor in splits.items():
        array = np.asarray(tensor.cpu().numpy(), dtype=dtype)
        expected = _split_counts(cfg)[split]
        if array.size != expected:
            raise ValueError(
                f"{split} contains {array.size} tokens, expected {expected}"
            )
        array.tofile(target_dir / f"{split}.bin")
        split_hashes[split] = hashlib.sha256(array.tobytes()).hexdigest()

    _prepare_retrieval(cfg)

    meta = {
        "source": source,
        "fingerprint": _config_fingerprint(cfg),
        "vocab_size": vocab_size,
        "block_size": cfg["data"]["block_size"],
        "token_dtype": str(dtype),
        "split_tokens": {key: len(value) for key, value in splits.items()},
        "split_sha256": split_hashes,
        "seed": seed,
        "special_tokens": SPECIAL_TOKENS,
    }
    write_json(target_dir / "meta.json", meta)
    if not _binary_data_ready(cfg):
        raise RuntimeError("Prepared token files failed final validation")
    _remove_obsolete_token_files(cfg)
    stage_banner("DATA", "DONE", cfg=cfg)
    return meta


def load_tokens(cfg: dict[str, Any], split: str) -> torch.Tensor:
    binary = data_dir(cfg) / f"{split}.bin"
    if not binary.exists():
        run(cfg)
    count = _split_counts(cfg)[split]
    array = np.memmap(
        binary,
        dtype=_token_dtype(cfg),
        mode="c",
        shape=(count,),
    )
    return torch.from_numpy(array)


def load_token_dataset(
    cfg: dict[str, Any],
    split: str,
    *,
    block_size: int | None = None,
) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
    binary = data_dir(cfg) / f"{split}.bin"
    if not binary.exists():
        run(cfg)
    resolved_block = int(block_size or cfg["data"]["block_size"])
    return MemmapTokenBlockDataset(
        binary,
        token_count=_split_counts(cfg)[split],
        block_size=resolved_block,
        dtype=str(_token_dtype(cfg)),
    )


def load_retrieval_adapt(cfg: dict[str, Any]) -> RetrievalBatch:
    path = data_dir(cfg) / "retrieval_adapt.pt"
    if not path.exists():
        run(cfg)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return RetrievalBatch(**payload)


def load_wikitext_tokens(cfg: dict[str, Any]) -> torch.Tensor:
    target = wikitext_dir(cfg)
    meta_path = target / "meta.json"
    token_path = target / "test.bin"
    if not meta_path.exists() or not token_path.exists():
        bootstrap_dependencies(cfg)
        tokenizer = prepare_tokenizer(cfg)
        prepare_wikitext(cfg, tokenizer)
    meta = read_json(meta_path)
    array = np.memmap(
        token_path,
        dtype=np.dtype(str(meta["token_dtype"])),
        mode="c",
        shape=(int(meta["token_count"]),),
    )
    return torch.from_numpy(array)


def load_qasper_examples(
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = qasper_dir(cfg)
    meta_path = target / "meta.json"
    examples_path = target / "examples.pt"
    if not _qasper_ready(cfg):
        bootstrap_dependencies(cfg)
        tokenizer = prepare_tokenizer(cfg)
        prepare_qasper(cfg, tokenizer)
    meta = read_json(meta_path)
    payload = torch.load(
        examples_path,
        map_location="cpu",
        weights_only=False,
    )
    examples = list(payload["examples"])
    fineweb_meta_path = data_dir(cfg) / "meta.json"
    fineweb_meta = (
        read_json(fineweb_meta_path) if fineweb_meta_path.exists() else {}
    )
    leakage = fineweb_meta.get("evaluation_leakage_audit", {})
    leakage_complete = bool(
        leakage.get("complete_for_all_prepared_documents", False)
    )
    if str(cfg["data"]["source"]) == "fineweb_edu" and not leakage_complete:
        raise RuntimeError(
            "FineWeb leakage audit is incomplete; QASPER evaluation is blocked"
        )
    requested = int(cfg["data"]["qasper_samples"])
    if len(examples) < requested:
        raise RuntimeError(
            f"Only {len(examples)} QASPER candidates are available; "
            f"{requested} required"
        )
    selected = examples[:requested]
    returned_meta = {
        **meta,
        "selected_samples": len(selected),
        "selected_question_ids": [
            example["question_id"] for example in selected
        ],
        "selected_sample_sha256": [
            example["sample_sha256"] for example in selected
        ],
        "fineweb_documents_excluded_for_evaluation_overlap": int(
            leakage.get("documents_excluded", 0)
        ),
        "matched_evaluation_targets": leakage.get(
            "matched_evaluation_targets",
            [],
        ),
        "leakage_check_complete": leakage_complete,
        "leakage_check_rules": leakage.get("rules", []),
    }
    return selected, returned_meta
