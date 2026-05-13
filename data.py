import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from para import PATH, SECRETS
from utils import ensure_dir


TINY_TEXTS = [
    "Sparse autoencoders can reveal features in transformer activations.",
    "Position encodings decide how models represent order and distance.",
    "RoPE rotates query and key vectors by relative position.",
    "PoPE separates magnitude-like content from phase-like position.",
]


class TinyTokenizer:
    eos_token = "<eos>"
    pad_token = "<eos>"

    def __init__(self):
        alphabet = sorted(set("".join(TINY_TEXTS) + self.eos_token))
        self.stoi = {ch: idx for idx, ch in enumerate(alphabet)}

    def encode(self, text):
        return [self.stoi.get(ch, 0) for ch in text]

    def __len__(self):
        return len(self.stoi)


class BlockDataset(Dataset):
    def __init__(self, blocks: torch.Tensor):
        self.blocks = blocks.long()

    def __len__(self):
        return self.blocks.size(0)

    def __getitem__(self, idx):
        x = self.blocks[idx, :-1]
        y = self.blocks[idx, 1:]
        return x, y


class GenerateData:
    def __init__(self, data_cfg):
        self.data_cfg = data_cfg
        self.configure_hf_cache()
        dataset_cache_name = data_cfg.dataset.replace("/", "__")
        self.cache_path = Path(PATH.cache_dir) / "tokens" / (
            f"{dataset_cache_name}_{data_cfg.tokenizer}_seq{data_cfg.seq_len}_"
            f"tr{data_cfg.train_blocks}_va{data_cfg.valid_blocks}.pt"
        )

    @property
    def hf_token(self):
        return getattr(SECRETS, "hf_token", None)

    @property
    def hf_cache_dir(self):
        return getattr(self, "_hf_cache_dir", getattr(self.data_cfg, "hf_cache_dir", None))

    @property
    def local_files_only(self):
        return bool(getattr(self.data_cfg, "local_files_only", False))

    def configure_hf_cache(self):
        if self.data_cfg.dataset == "tiny":
            return
        cache_dir = getattr(self.data_cfg, "hf_cache_dir", None)
        if not cache_dir:
            return
        root = Path(cache_dir).resolve()
        ensure_dir(root)
        ensure_dir(root / "hub")
        ensure_dir(root / "xet")
        ensure_dir(root / "datasets")
        self._hf_cache_dir = str(root)
        os.environ["HF_HOME"] = str(root)
        os.environ["HF_HUB_CACHE"] = str(root / "hub")
        os.environ["HF_XET_CACHE"] = str(root / "xet")
        os.environ["HF_DATASETS_CACHE"] = str(root / "datasets")
        if not self.local_files_only:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("HF_DATASETS_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)

    def load_tokenizer(self):
        try:
            from transformers import AutoTokenizer

            kwargs = {}
            if self.hf_cache_dir:
                kwargs["cache_dir"] = self.hf_cache_dir
            if self.local_files_only or self.data_cfg.dataset == "tiny":
                kwargs["local_files_only"] = True
            if self.hf_token and self.data_cfg.dataset != "tiny":
                kwargs["token"] = self.hf_token
            tokenizer = AutoTokenizer.from_pretrained(self.data_cfg.tokenizer, **kwargs)
        except Exception:
            if self.data_cfg.dataset != "tiny":
                raise
            tokenizer = TinyTokenizer()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def text_stream(self):
        if self.data_cfg.dataset == "tiny":
            while True:
                for text in TINY_TEXTS:
                    yield text
        else:
            from datasets import load_dataset

            kwargs = {"split": "train", "streaming": self.data_cfg.streaming}
            if self.hf_cache_dir:
                kwargs["cache_dir"] = self.hf_cache_dir
            if self.hf_token:
                kwargs["token"] = self.hf_token
            if self.data_cfg.dataset_config:
                ds = load_dataset(self.data_cfg.dataset, self.data_cfg.dataset_config, **kwargs)
            else:
                ds = load_dataset(self.data_cfg.dataset, **kwargs)
            for row in ds:
                text = row.get(self.data_cfg.text_key, "")
                if text and text.strip():
                    yield text

    def build_blocks(self, tokenizer):
        needed = (self.data_cfg.train_blocks + self.data_cfg.valid_blocks) * (self.data_cfg.seq_len + 1)
        token_ids = []
        for text in self.text_stream():
            token_ids.extend(tokenizer.encode(text + tokenizer.eos_token))
            if len(token_ids) >= needed:
                break
        if len(token_ids) < needed:
            repeats = (needed // max(len(token_ids), 1)) + 1
            token_ids = (token_ids * repeats)[:needed]
        ids = torch.tensor(token_ids[:needed], dtype=torch.long)
        return ids.view(-1, self.data_cfg.seq_len + 1)

    def save_cache(self, data_res):
        ensure_dir(self.cache_path.parent)
        torch.save(
            {
                "train_blocks": data_res["train"].blocks,
                "valid_blocks": data_res["valid"].blocks,
                "meta": data_res["meta"],
            },
            self.cache_path,
        )

    def load_cache(self, tokenizer):
        if not self.cache_path.exists():
            return None
        cached = torch.load(self.cache_path, map_location="cpu")
        return {
            "train": BlockDataset(cached["train_blocks"]),
            "valid": BlockDataset(cached["valid_blocks"]),
            "tokenizer": tokenizer,
            "meta": cached["meta"],
        }

    def run(self):
        tokenizer = self.load_tokenizer()
        if self.data_cfg.use_cache:
            cached = self.load_cache(tokenizer)
            if cached is not None:
                return cached

        blocks = self.build_blocks(tokenizer)
        train_blocks = blocks[: self.data_cfg.train_blocks]
        valid_blocks = blocks[self.data_cfg.train_blocks :]
        data_res = {
            "train": BlockDataset(train_blocks),
            "valid": BlockDataset(valid_blocks),
            "tokenizer": tokenizer,
            "meta": {
                "dataset": self.data_cfg.dataset,
                "tokenizer": self.data_cfg.tokenizer,
                "seq_len": self.data_cfg.seq_len,
                "vocab_size": len(tokenizer),
                "num_train_blocks": len(train_blocks),
                "num_valid_blocks": len(valid_blocks),
            },
        }
        if self.data_cfg.use_cache:
            self.save_cache(data_res)
        return data_res
