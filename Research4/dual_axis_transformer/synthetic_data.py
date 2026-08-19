"""Deterministic country/color multi-label data and a tiny reproducible tokenizer."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from torch import Tensor

from .research_model import COLOR_CLASSES, CONCEPT_NAMES, COUNTRY_CLASSES


GENERATOR_VERSION = "dual-tag-v2"
TOKEN_PATTERN = re.compile(r"\[CONCEPT\]|[A-Za-z]+|[0-9]+|[^\w\s]", re.UNICODE)

KNOWN_COUNTRIES = ("france", "united_kingdom", "china", "japan")
TRAIN_UNKNOWN_COUNTRIES = ("canada", "brazil")
TEST_UNKNOWN_COUNTRIES = ("india", "egypt")
KNOWN_COLORS = ("red", "blue", "yellow", "green")
TRAIN_UNKNOWN_COLORS = ("purple", "white")
TEST_UNKNOWN_COLORS = ("orange", "black")

COUNTRY_DISPLAY = {
    "france": "France",
    "united_kingdom": "the United Kingdom",
    "china": "China",
    "japan": "Japan",
    "canada": "Canada",
    "brazil": "Brazil",
    "india": "India",
    "egypt": "Egypt",
}
COUNTRY_DESCRIPTION = {
    "france": "a Parisian bakery tradition",
    "united_kingdom": "a London tea-house tradition",
    "china": "a Beijing porcelain tradition",
    "japan": "a Tokyo origami tradition",
    "canada": "a maple-leaf workshop tradition",
    "brazil": "a Rio carnival workshop tradition",
    "india": "a Delhi textile tradition",
    "egypt": "a Cairo papyrus tradition",
}
COLOR_DESCRIPTION = {
    "red": "the shade of a ripe tomato",
    "blue": "the shade of a clear midday sky",
    "yellow": "the shade of a sunflower",
    "green": "the shade of fresh grass",
    "purple": "the shade of a violet flower",
    "white": "the shade of fresh snow",
    "orange": "the shade of citrus peel",
    "black": "the shade of coal",
}


@dataclass(frozen=True)
class DualTagExample:
    example_id: str
    split: str
    template_family: str
    text: str
    country_target: int
    color_target: int
    concept_targets: tuple[int, ...]
    country_value: str | None
    color_value: str | None

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "DualTagExample":
        values = dict(values)
        values["concept_targets"] = tuple(values["concept_targets"])
        return cls(**values)


class Vocabulary:
    PAD = "[PAD]"
    UNK = "[UNK]"
    CONCEPT = "[CONCEPT]"

    def __init__(self, tokens: Iterable[str]) -> None:
        unique = sorted(set(tokens) - {self.PAD, self.UNK, self.CONCEPT})
        self.tokens = (self.PAD, self.UNK, self.CONCEPT, *unique)
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [
            token if token == Vocabulary.CONCEPT else token.lower()
            for token in TOKEN_PATTERN.findall(text)
        ]

    @classmethod
    def build(cls, examples: Iterable[DualTagExample]) -> "Vocabulary":
        return cls(
            token
            for example in examples
            for token in cls.tokenize(example.text)
        )

    def encode(self, text: str, max_length: int) -> list[int]:
        values = [self.token_to_id.get(token, 1) for token in self.tokenize(text)]
        if not values or values[-1] != self.token_to_id[self.CONCEPT]:
            values.append(self.token_to_id[self.CONCEPT])
        # Keep the causal summary token even when a generated sentence is long.
        if len(values) > max_length:
            values = values[: max_length - 1] + [self.token_to_id[self.CONCEPT]]
        return values

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": list(self.tokens)}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "Vocabulary":
        instance = cls(())
        instance.tokens = tuple(values["tokens"])
        instance.token_to_id = {
            token: index for index, token in enumerate(instance.tokens)
        }
        return instance


@dataclass(frozen=True)
class SyntheticDataBundle:
    root: Path
    manifest_path: Path
    vocabulary_path: Path
    split_paths: dict[str, Path]
    vocabulary: Vocabulary


def _country_target(value: str | None) -> int:
    if value is None:
        return 0
    return KNOWN_COUNTRIES.index(value) + 1 if value in KNOWN_COUNTRIES else 5


def _color_target(value: str | None) -> int:
    if value is None:
        return 0
    return KNOWN_COLORS.index(value) + 1 if value in KNOWN_COLORS else 5


def _concept_targets(country: str | None, color: str | None) -> tuple[int, ...]:
    targets = [0] * len(CONCEPT_NAMES)
    if country is not None:
        targets[0] = 1
        targets[_country_target(country)] = 1
    if color is not None:
        targets[6] = 1
        targets[6 + _color_target(color)] = 1
    return tuple(targets)


def _choose_value(
    rng: random.Random,
    known: tuple[str, ...],
    unknown: tuple[str, ...],
    unknown_fraction: float,
) -> str:
    pool = unknown if rng.random() < unknown_fraction else known
    return rng.choice(pool)


def _render(
    rng: random.Random,
    template_family: str,
    country: str | None,
    color: str | None,
) -> str:
    split_style, family = template_family.split("_", 1)
    country_name = None if country is None else COUNTRY_DISPLAY[country]
    color_name = color
    if family == "direct":
        if split_style == "train":
            origin = "No origin was stated." if country is None else f"The target was made in {country_name}."
            shade = "No color was stated." if color is None else f"The target is {color_name}."
        elif split_style == "validation":
            origin = "The production record is blank." if country is None else f"Production records identify {country_name}."
            shade = "The appearance record is blank." if color is None else f"Observation records identify {color_name}."
        else:
            origin = "The origin field is unspecified." if country is None else f"The origin field reads {country_name}."
            shade = "The color field is unspecified." if color is None else f"The color field reads {color_name}."
        body = f"{origin} {shade}"
    elif family == "descriptive":
        if split_style == "train":
            origin = "The description gives no geographic tradition." if country is None else f"The target follows {COUNTRY_DESCRIPTION[country]}."
            shade = "The description gives no visible shade." if color is None else f"Its surface has {COLOR_DESCRIPTION[color]}."
        elif split_style == "validation":
            origin = "Experts find no regional craft clue." if country is None else f"Experts link its craft to {COUNTRY_DESCRIPTION[country]}."
            shade = "Experts find no shade clue." if color is None else f"Experts compare its appearance to {COLOR_DESCRIPTION[color]}."
        else:
            origin = "No cultural style can be verified." if country is None else f"The verified style resembles {COUNTRY_DESCRIPTION[country]}."
            shade = "No visual tone can be verified." if color is None else f"Witnesses liken the surface to {COLOR_DESCRIPTION[color]}."
        body = f"{origin} {shade}"
    elif family == "binding":
        distractor_country = COUNTRY_DISPLAY[rng.choice(KNOWN_COUNTRIES)]
        distractor_color = rng.choice(KNOWN_COLORS)
        origin = (
            "The target itself has no stated origin."
            if country is None
            else f"The target itself was made in {country_name}."
        )
        shade = (
            "Its own color is not stated."
            if color is None
            else f"Its own surface is {color_name}."
        )
        if split_style == "train":
            body = f"A {distractor_color} notebook belongs to a visitor from {distractor_country}. {origin} {shade}"
        elif split_style == "validation":
            body = f"The owner carries a {distractor_color} card from {distractor_country}. For the target alone: {origin} {shade}"
        else:
            body = f"Ignore a {distractor_color} brochure brought from {distractor_country}. The verified target record says: {origin} {shade}"
    else:  # counterfactual
        distractor_country = COUNTRY_DISPLAY[rng.choice(KNOWN_COUNTRIES)]
        distractor_color = rng.choice(KNOWN_COLORS)
        origin = (
            "Despite the rumor, no origin is actually given for the target."
            if country is None
            else f"Despite a rumor about {distractor_country}, the target came from {country_name}."
        )
        shade = (
            "The note mentions a colored box but gives no target color."
            if color is None
            else f"Although a label mentions {distractor_color}, the target is {color_name}."
        )
        if split_style == "train":
            body = f"{origin} {shade}"
        elif split_style == "validation":
            body = f"Separate rumor from evidence. {origin} {shade}"
        else:
            body = f"Only the corrected claim applies. {origin} {shade}"
    query = {
        "train": "Report the target origin and color.",
        "validation": "Identify both attributes of the target.",
        "test": "State the verified origin and shade.",
    }[split_style]
    return f"{body} {query} [CONCEPT]"


def generate_split(
    split: str,
    size: int,
    *,
    seed: int,
    unknown_fraction: float,
    template_ood: bool = True,
) -> list[DualTagExample]:
    rng = random.Random(seed)
    unknown_countries = (
        TRAIN_UNKNOWN_COUNTRIES if split == "train" else TEST_UNKNOWN_COUNTRIES
    )
    unknown_colors = TRAIN_UNKNOWN_COLORS if split == "train" else TEST_UNKNOWN_COLORS
    templates = ("direct", "descriptive", "binding", "counterfactual")
    template_weights = (0.30, 0.30, 0.25, 0.15)

    examples = []
    for index in range(size):
        bucket = index % 10
        has_country = bucket in {2, 3, 6, 7, 8, 9}
        has_color = bucket in {4, 5, 6, 7, 8, 9}
        country = (
            _choose_value(
                rng, KNOWN_COUNTRIES, unknown_countries, unknown_fraction
            )
            if has_country
            else None
        )
        color = (
            _choose_value(rng, KNOWN_COLORS, unknown_colors, unknown_fraction)
            if has_color
            else None
        )
        template_split = split if template_ood else "train"
        template = f"{template_split}_{rng.choices(templates, weights=template_weights, k=1)[0]}"
        examples.append(
            DualTagExample(
                example_id=f"{split}-{index:08d}",
                split=split,
                template_family=template,
                text=_render(rng, template, country, color),
                country_target=_country_target(country),
                color_target=_color_target(color),
                concept_targets=_concept_targets(country, color),
                country_value=country,
                color_value=color,
            )
        )
    return examples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, examples: Iterable[DualTagExample]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")


def _data_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(
        {"generator": GENERATOR_VERSION, **config}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def ensure_synthetic_dataset(
    data_root: str | Path, config: dict[str, Any]
) -> SyntheticDataBundle:
    fingerprint = _data_fingerprint(config)
    root = Path(data_root) / "dual_tag" / fingerprint
    manifest_path = root / "manifest.json"
    vocabulary_path = root / "vocabulary.json"
    split_paths = {
        "train": root / "train.jsonl",
        "validation": root / "validation.jsonl",
        "test": root / "test.jsonl",
    }
    if manifest_path.exists() and vocabulary_path.exists() and all(
        path.exists() for path in split_paths.values()
    ):
        vocabulary = Vocabulary.from_dict(
            json.loads(vocabulary_path.read_text(encoding="utf-8"))
        )
        return SyntheticDataBundle(
            root, manifest_path, vocabulary_path, split_paths, vocabulary
        )

    root.mkdir(parents=True, exist_ok=True)
    base_seed = int(config["generation_seed"])
    unknown_fraction = float(config["unknown_fraction"])
    template_ood = bool(config.get("template_ood", True))
    splits = {
        "train": generate_split(
            "train",
            int(config["train_size"]),
            seed=base_seed,
            unknown_fraction=unknown_fraction,
            template_ood=template_ood,
        ),
        "validation": generate_split(
            "validation",
            int(config["validation_size"]),
            seed=base_seed + 1,
            unknown_fraction=unknown_fraction,
            template_ood=template_ood,
        ),
        "test": generate_split(
            "test",
            int(config["test_size"]),
            seed=base_seed + 2,
            unknown_fraction=unknown_fraction,
            template_ood=template_ood,
        ),
    }
    vocabulary = Vocabulary.build(splits["train"])
    vocabulary_path.write_text(
        json.dumps(vocabulary.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for name, examples in splits.items():
        _write_jsonl(split_paths[name], examples)
    manifest = {
        "generator": GENERATOR_VERSION,
        "fingerprint": fingerprint,
        "config": config,
        "vocabulary_size": len(vocabulary.tokens),
        "splits": {
            name: {
                "path": path.name,
                "size": len(splits[name]),
                "sha256": _sha256(path),
            }
            for name, path in split_paths.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return SyntheticDataBundle(
        root, manifest_path, vocabulary_path, split_paths, vocabulary
    )


def load_examples(path: str | Path) -> list[DualTagExample]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [DualTagExample.from_dict(json.loads(line)) for line in handle if line.strip()]


def collate_examples(
    examples: list[DualTagExample], vocabulary: Vocabulary, max_length: int
) -> dict[str, Tensor]:
    encoded = [vocabulary.encode(example.text, max_length) for example in examples]
    width = max(len(values) for values in encoded)
    input_ids = torch.zeros(len(examples), width, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, values in enumerate(encoded):
        input_ids[row, : len(values)] = torch.tensor(values, dtype=torch.long)
        attention_mask[row, : len(values)] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "country_targets": torch.tensor(
            [example.country_target for example in examples], dtype=torch.long
        ),
        "color_targets": torch.tensor(
            [example.color_target for example in examples], dtype=torch.long
        ),
        "concept_targets": torch.tensor(
            [example.concept_targets for example in examples], dtype=torch.float32
        ),
    }


class DeterministicBatcher:
    """Replacement-sampling batcher with a checkpointable generator state."""

    def __init__(
        self,
        examples: list[DualTagExample],
        vocabulary: Vocabulary,
        *,
        batch_size: int,
        max_length: int,
        seed: int,
    ) -> None:
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)
        # Tokenization is deterministic and independent of augmentation, so do
        # it once. Re-running regex tokenization every optimizer step can leave
        # a fast GPU idle on this deliberately small model.
        encoded = [
            vocabulary.encode(example.text, max_length) for example in examples
        ]
        width = max(len(values) for values in encoded)
        self.input_ids = torch.zeros(len(examples), width, dtype=torch.long)
        self.lengths = torch.tensor(
            [len(values) for values in encoded], dtype=torch.long
        )
        for row, values in enumerate(encoded):
            self.input_ids[row, : len(values)] = torch.tensor(values)
        self.country_targets = torch.tensor(
            [example.country_target for example in examples], dtype=torch.long
        )
        self.color_targets = torch.tensor(
            [example.color_target for example in examples], dtype=torch.long
        )
        self.concept_targets = torch.tensor(
            [example.concept_targets for example in examples], dtype=torch.float32
        )

    def next_batch(self) -> dict[str, Tensor]:
        indices = torch.randint(
            len(self.input_ids), (self.batch_size,), generator=self.generator
        )
        lengths = self.lengths.index_select(0, indices)
        width = int(lengths.max())
        input_ids = self.input_ids.index_select(0, indices)[:, :width]
        attention_mask = (
            torch.arange(width, dtype=torch.long)[None, :] < lengths[:, None]
        ).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "country_targets": self.country_targets.index_select(0, indices),
            "color_targets": self.color_targets.index_select(0, indices),
            "concept_targets": self.concept_targets.index_select(0, indices),
        }

    def state_dict(self) -> dict[str, Tensor]:
        return {"generator_state": self.generator.get_state()}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.generator.set_state(state["generator_state"])


def iter_eval_batches(
    examples: list[DualTagExample],
    vocabulary: Vocabulary,
    *,
    batch_size: int,
    max_length: int,
) -> Iterator[dict[str, Tensor]]:
    for start in range(0, len(examples), batch_size):
        yield collate_examples(
            examples[start : start + batch_size], vocabulary, max_length
        )
