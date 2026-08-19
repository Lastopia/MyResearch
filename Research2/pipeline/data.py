from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset

from model.factory import active_aliases, is_pretrain_mode, validate_aliases
from model.premodel import premodel_entry, tokenizer_for_premodel
from tools.io import block_dir, data_blocks_alias, data_tokenized_alias, data_tokenizer_alias, ensure_dir, save_config, write_json
from tools.io import read_json
from tools.log import stage_title


def _append_tiktoken(buffer, text, enc):
    if text:
        buffer.extend(enc.encode_ordinary(text))
        buffer.append(enc.eot_token)


def _append_hf_tokens(buffer, text, tokenizer):
    if text:
        buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        if tokenizer.eos_token_id is not None:
            buffer.append(tokenizer.eos_token_id)


def token_dtype(vocab_size):
    return "uint16" if vocab_size <= np.iinfo(np.uint16).max else "uint32"


def tokenizer_info(cfg, alias=None):
    if is_pretrain_mode(cfg):
        tokenizer = tokenizer_for_premodel(cfg, alias)
        entry = premodel_entry(cfg, alias)
        return {
            "tokenizer": tokenizer,
            "source": "transformers",
            "tokenizer_id": entry.get("tokenizer_id") or entry["hf_id"],
            "vocab_size": len(tokenizer),
            "append": _append_hf_tokens,
        }
    enc = tiktoken.get_encoding(cfg["data"]["tokenizer_alias"])
    return {
        "tokenizer": enc,
        "source": "tiktoken",
        "tokenizer_id": cfg["data"]["tokenizer_alias"],
        "vocab_size": enc.n_vocab,
        "append": _append_tiktoken,
    }


def prepare_blocks(cfg, alias=None):
    data_cfg = cfg["data"]
    tok = tokenizer_info(cfg, alias)
    tokenizer = tok["tokenizer"]
    dtype = token_dtype(tok["vocab_size"])
    block_size = data_cfg["block_size"]
    row_len = block_size + 1
    total_blocks = data_cfg["train_blocks"] + data_cfg["valid_blocks"]
    total_tokens = total_blocks * row_len

    raw_dir = Path("data") / "raw" / data_cfg["raw_alias"]
    tokenizer_dir = Path("data") / "tokenizer" / data_tokenizer_alias(cfg, alias)
    tok_dir = Path("data") / "tokenized" / data_tokenized_alias(cfg, alias)
    blocks_path = block_dir(cfg, alias)
    ensure_dir(raw_dir)
    ensure_dir(tokenizer_dir)
    ensure_dir(tok_dir)
    ensure_dir(blocks_path)

    meta = {
        "mode": cfg["run"].get("mode", "retrain"),
        "model_alias": alias,
        "corpus": data_cfg["corpus"],
        "raw_alias": data_cfg["raw_alias"],
        "tokenizer_alias": data_tokenizer_alias(cfg, alias),
        "tokenizer_source": tok["source"],
        "tokenizer_id": tok["tokenizer_id"],
        "tokenized_alias": data_tokenized_alias(cfg, alias),
        "blocks_alias": data_blocks_alias(cfg, alias),
        "block_size": block_size,
        "row_len": row_len,
        "train_blocks": data_cfg["train_blocks"],
        "valid_blocks": data_cfg["valid_blocks"],
        "seed": data_cfg["seed"],
        "token_dtype": dtype,
    }
    train_path = blocks_path / "train.bin"
    valid_path = blocks_path / "valid.bin"
    meta_path = blocks_path / "meta.json"
    if train_path.exists() and valid_path.exists() and meta_path.exists() and read_json(meta_path) == meta:
        label = alias or data_cfg["tokenizer_alias"]
        print(f"[data] {label} | blocks exists | train_blocks={data_cfg['train_blocks']} | valid_blocks={data_cfg['valid_blocks']}")
        return

    tokens = np.memmap(tok_dir / "tokens.bin", dtype=np.dtype(dtype), mode="w+", shape=(total_tokens,))

    ds = load_dataset(data_cfg["hf_dataset"], split="train")
    ds = ds.shuffle(seed=data_cfg["seed"])
    buf = []
    written = 0
    for row in ds:
        tok["append"](buf, row.get("text", ""), tokenizer)
        while len(buf) >= 8192 and written < total_tokens:
            n = min(len(buf), total_tokens - written)
            tokens[written:written + n] = np.asarray(buf[:n], dtype=np.dtype(dtype))
            del buf[:n]
            written += n
        if written >= total_tokens:
            break

    if written < total_tokens and buf:
        n = min(len(buf), total_tokens - written)
        tokens[written:written + n] = np.asarray(buf[:n], dtype=np.dtype(dtype))
        written += n

    if written < total_tokens:
        raise RuntimeError(f"OpenWebText ended early: got {written}, need {total_tokens} tokens")

    tokens.flush()
    arr = np.memmap(tok_dir / "tokens.bin", dtype=np.dtype(dtype), mode="r", shape=(total_tokens,))
    blocks = np.asarray(arr).reshape(total_blocks, row_len)
    train = np.memmap(blocks_path / "train.bin", dtype=np.dtype(dtype), mode="w+", shape=(data_cfg["train_blocks"], row_len))
    valid = np.memmap(blocks_path / "valid.bin", dtype=np.dtype(dtype), mode="w+", shape=(data_cfg["valid_blocks"], row_len))
    train[:] = blocks[:data_cfg["train_blocks"]]
    valid[:] = blocks[data_cfg["train_blocks"]:]
    train.flush()
    valid.flush()

    write_json(raw_dir / "meta.json", {"corpus": data_cfg["corpus"], "hf_dataset": data_cfg["hf_dataset"]})
    write_json(tokenizer_dir / "meta.json", {"tokenizer_alias": data_tokenizer_alias(cfg, alias), "source": tok["source"], "tokenizer_id": tok["tokenizer_id"]})
    write_json(tok_dir / "meta.json", meta)
    write_json(blocks_path / "meta.json", meta)
    label = alias or data_cfg["tokenizer_alias"]
    print(f"[data] {label} | train_blocks={data_cfg['train_blocks']} | valid_blocks={data_cfg['valid_blocks']}")


def run(cfg):
    stage_title("data")
    if is_pretrain_mode(cfg):
        aliases = active_aliases(cfg)
        validate_aliases(cfg, aliases)
        for alias in aliases:
            prepare_blocks(cfg, alias)
    else:
        prepare_blocks(cfg)
    save_config(cfg)
