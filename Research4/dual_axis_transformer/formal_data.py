"""Reproducible GPT-2-tokenized corpora for the formal language experiments."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor


class GPT2Tokenizer:
    """Thin, fixed wrapper around the public GPT-2 BPE vocabulary."""

    pad_id = 50256
    bos_id = 50256
    eos_id = 50256
    vocab_size = 50257

    def __init__(self, cache_dir: Path | None = None) -> None:
        try:
            import tiktoken
        except ImportError as error:  # pragma: no cover - exercised on servers
            raise RuntimeError(
                "formal experiments require tiktoken; run pip install -r requirements.txt"
            ) from error
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Keep the downloaded GPT-2 vocabulary inside data/, never output/.
            os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)
        self.encoding = tiktoken.get_encoding("gpt2")

    def encode(self, text: str, max_length: int) -> list[int]:
        if max_length < 2:
            raise ValueError("max_length must be at least two")
        body = self.encoding.encode_ordinary(text)[: max_length - 2]
        return [self.bos_id, *body, self.eos_id]

    def encode_documents(self, texts: list[str]) -> list[list[int]]:
        return [
            [*tokens, self.eos_id]
            for tokens in self.encoding.encode_ordinary_batch(texts)
        ]


@dataclass(frozen=True)
class FormalLanguageBundle:
    root: Path
    manifest_path: Path
    train_path: Path
    validation_path: Path
    external_test_path: Path


class TokenCorpus:
    """Read-only memory map; a 1B-token corpus never enters host RAM at once."""

    def __init__(self, path: Path) -> None:
        if not path.exists() or path.stat().st_size % 4:
            raise RuntimeError(f"invalid uint32 token corpus: {path}")
        self.path = path
        self.token_count = path.stat().st_size // 4
        if self.token_count < 1024:
            raise RuntimeError(f"token corpus is unexpectedly small: {path}")
        self.values = torch.from_file(
            str(path), shared=False, size=self.token_count, dtype=torch.int32
        )

    def __len__(self) -> int:
        return self.token_count

    def __getitem__(self, index: Tensor) -> Tensor:
        return self.values[index].long()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _append_tokens(handle: Any, tokens: Iterable[int], remaining: int) -> int:
    import numpy as np

    if remaining <= 0:
        return 0
    values = list(tokens)
    if len(values) > remaining:
        values = values[:remaining]
    np.asarray(values, dtype=np.dtype("<u4")).tofile(handle)
    return len(values)


def _prepare_fineweb(settings: dict[str, Any], root: Path) -> tuple[Path, Path, int]:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - exercised on servers
        raise RuntimeError(
            "formal experiments require datasets; run pip install -r requirements.txt"
        ) from error

    train_target = int(settings["train_tokens"])
    validation_target = int(settings["validation_tokens"])
    train_path = root / "fineweb_edu_train.uint32"
    validation_path = root / "fineweb_edu_validation.uint32"
    if (
        train_path.exists()
        and validation_path.exists()
        and train_path.stat().st_size == train_target * 4
        and validation_path.stat().st_size == validation_target * 4
    ):
        return train_path, validation_path, -1

    train_part = train_path.with_suffix(train_path.suffix + ".part")
    validation_part = validation_path.with_suffix(validation_path.suffix + ".part")
    root.mkdir(parents=True, exist_ok=True)
    print(
        "data | preparing pinned FineWeb-Edu sample-10BT | "
        f"train {train_target:,} | validation {validation_target:,} tokens",
        flush=True,
    )
    dataset = load_dataset(
        str(settings["dataset_id"]),
        name=str(settings["dataset_config"]),
        split="train",
        streaming=True,
        revision=str(settings["dataset_revision"]),
        cache_dir=str(root / "hf_cache"),
    )
    tokenizer = GPT2Tokenizer(root / "tokenizer_cache")
    train_written = validation_written = documents = 0
    next_report = int(settings.get("preparation_log_tokens", 10_000_000))
    with train_part.open("wb") as train_handle, validation_part.open("wb") as val_handle:
        for batch in dataset.iter(batch_size=int(settings.get("tokenizer_batch_size", 128))):
            texts = [str(value) for value in batch["text"] if value]
            for tokens in tokenizer.encode_documents(texts):
                documents += 1
                if validation_written < validation_target:
                    validation_written += _append_tokens(
                        val_handle, tokens, validation_target - validation_written
                    )
                elif train_written < train_target:
                    train_written += _append_tokens(
                        train_handle, tokens, train_target - train_written
                    )
                if train_written + validation_written >= next_report:
                    print(
                        "data | FineWeb-Edu tokenized "
                        f"{train_written + validation_written:,}/"
                        f"{train_target + validation_target:,} tokens",
                        flush=True,
                    )
                    next_report += int(
                        settings.get("preparation_log_tokens", 10_000_000)
                    )
                if train_written == train_target and validation_written == validation_target:
                    break
            if train_written == train_target and validation_written == validation_target:
                break
    if train_written != train_target or validation_written != validation_target:
        raise RuntimeError(
            "FineWeb-Edu stream ended before the fixed token budget was filled: "
            f"train={train_written}/{train_target}, "
            f"validation={validation_written}/{validation_target}"
        )
    os.replace(train_part, train_path)
    os.replace(validation_part, validation_path)
    return train_path, validation_path, documents


def _prepare_wikitext(settings: dict[str, Any], root: Path) -> tuple[Path, int]:
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "formal experiments require datasets; run pip install -r requirements.txt"
        ) from error
    output = root / "wikitext103_test.uint32"
    if output.exists() and output.stat().st_size >= 4096:
        return output, output.stat().st_size // 4
    print("data | preparing pinned WikiText-103 external test", flush=True)
    dataset = load_dataset(
        str(settings["external_dataset_id"]),
        name=str(settings["external_dataset_config"]),
        split=str(settings.get("external_split", "test")),
        revision=str(settings["external_dataset_revision"]),
        cache_dir=str(root / "hf_cache"),
    )
    tokenizer = GPT2Tokenizer(root / "tokenizer_cache")
    temporary = output.with_suffix(output.suffix + ".part")
    count = 0
    with temporary.open("wb") as handle:
        texts = [str(row["text"]) for row in dataset if row.get("text")]
        for start in range(0, len(texts), 128):
            for tokens in tokenizer.encode_documents(texts[start : start + 128]):
                count += _append_tokens(handle, tokens, 2**63 - 1)
    if count < 1024:
        raise RuntimeError("WikiText-103 external test corpus is unexpectedly small")
    os.replace(temporary, output)
    return output, count


def ensure_formal_language_dataset(
    cfg: dict[str, Any], data_root: Path
) -> FormalLanguageBundle:
    settings = cfg["external"]["formal_language"]
    os.environ.setdefault(
        "HF_HUB_DOWNLOAD_TIMEOUT",
        str(int(cfg["external"].get("download_timeout_seconds", 120))),
    )
    root = data_root / "formal_language"
    manifest_path = root / "manifest.json"
    train_target = int(settings["train_tokens"])
    validation_target = int(settings["validation_tokens"])
    expected = {
        "schema_version": 1,
        "dataset_id": settings["dataset_id"],
        "dataset_config": settings["dataset_config"],
        "dataset_revision": settings["dataset_revision"],
        "train_tokens": train_target,
        "validation_tokens": validation_target,
        "external_dataset_id": settings["external_dataset_id"],
        "external_dataset_config": settings["external_dataset_config"],
        "external_dataset_revision": settings["external_dataset_revision"],
        "tokenizer": "tiktoken:gpt2",
        "vocab_size": GPT2Tokenizer.vocab_size,
        "dtype": "uint32-little-endian",
    }
    train_path = root / "fineweb_edu_train.uint32"
    validation_path = root / "fineweb_edu_validation.uint32"
    external_path = root / "wikitext103_test.uint32"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            all(manifest.get(key) == value for key, value in expected.items())
            and train_path.exists()
            and train_path.stat().st_size == train_target * 4
            and validation_path.exists()
            and validation_path.stat().st_size == validation_target * 4
            and external_path.exists()
            and external_path.stat().st_size >= 4096
        ):
            return FormalLanguageBundle(
                root, manifest_path, train_path, validation_path, external_path
            )
    if not bool(cfg["external"].get("allow_download", True)):
        raise FileNotFoundError(
            f"formal language data is not cached at {root} and downloads are disabled"
        )
    train_path, validation_path, documents = _prepare_fineweb(settings, root)
    external_path, external_tokens = _prepare_wikitext(settings, root)
    manifest = {
        **expected,
        "licenses": {
            "FineWeb-Edu": "ODC-By 1.0",
            "WikiText-103": "CC BY-SA 3.0 / GFDL",
        },
        "fineweb_documents_consumed": documents,
        "external_test_tokens": external_tokens,
        "files": {
            "train": train_path.name,
            "validation": validation_path.name,
            "external_test": external_path.name,
        },
        "split_policy": (
            "fixed stream prefix: validation first, then train; document EOS retained"
        ),
        "fingerprint": hashlib.sha256(
            json.dumps(expected, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    _atomic_json(manifest_path, manifest)
    return FormalLanguageBundle(
        root, manifest_path, train_path, validation_path, external_path
    )
