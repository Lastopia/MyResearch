import math
import string

import torch

from model.factory import is_pretrain_mode
from model.premodel import tokenizer_for_premodel


def token_decoder(cfg, alias=None):
    """Return a robust single-token and token-sequence decoder."""
    if is_pretrain_mode(cfg):
        tokenizer = tokenizer_for_premodel(cfg, alias)

        def decode_token(token_id):
            return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)

        def decode_tokens(token_ids):
            return tokenizer.decode([int(value) for value in token_ids], clean_up_tokenization_spaces=False)

        return decode_token, decode_tokens

    import tiktoken

    encoding = tiktoken.get_encoding(cfg["data"]["tokenizer_alias"])

    def decode_token(token_id):
        try:
            return encoding.decode_single_token_bytes(int(token_id)).decode("utf-8", errors="replace")
        except KeyError:
            return ""

    def decode_tokens(token_ids):
        try:
            return encoding.decode([int(value) for value in token_ids])
        except (KeyError, ValueError):
            return "".join(decode_token(value) for value in token_ids)

    return decode_token, decode_tokens


def _surface_flags(text):
    stripped = text.strip()
    lower = stripped.lower()
    nonempty = bool(stripped)
    return {
        "alphabetic": nonempty and stripped.isalpha(),
        "numeric": nonempty and any(char.isdigit() for char in stripped) and all(char.isdigit() or char in ".,:%+-/" for char in stripped),
        "punctuation": nonempty and all(char in string.punctuation or char in "…–—‘’“”" for char in stripped),
        "uppercase_initial": nonempty and stripped[0].isupper(),
        "leading_space": bool(text) and text[0].isspace() and "\n" not in text[:1],
        "newline": "\n" in text or "\r" in text,
        "whitespace": bool(text) and text.isspace(),
        "short_word": stripped.isalpha() and len(stripped) <= 3,
        "long_word": stripped.isalpha() and len(stripped) >= 8,
        "starts_vowel": nonempty and lower[0] in "aeiou",
        "suffix_ing": len(lower) > 3 and lower.endswith("ing"),
        "suffix_ed": len(lower) > 2 and lower.endswith("ed"),
        "suffix_ly": len(lower) > 2 and lower.endswith("ly"),
        "url_like": "http" in lower or "www." in lower or ".com" in lower,
        "code_symbol": any(symbol in text for symbol in ("{", "}", "=>", "==", "!=", "();", "</", "::")),
    }


def build_concept_masks(cfg, alias, vocab_size):
    """Build auditable token-level concepts for next-token prediction.

    The labels apply to y[t], while the evaluated hidden state is h[t]. This
    alignment makes the same concepts usable by both probes and interventions.
    """
    concept_cfg = cfg["interpretability"]["concepts"]
    decode_token, decode_tokens = token_decoder(cfg, alias)
    texts = [decode_token(token_id) for token_id in range(int(vocab_size))]
    normalized = [text.strip().lower() for text in texts]
    masks = {}

    if concept_cfg.get("include_surface", True):
        names = list(_surface_flags("example"))
        values = {name: [] for name in names}
        for text in texts:
            flags = _surface_flags(text)
            for name in names:
                values[name].append(flags[name])
        masks.update({name: torch.tensor(rows, dtype=torch.bool) for name, rows in values.items()})

    if concept_cfg.get("include_first_letter", True):
        for letter in string.ascii_lowercase:
            masks[f"first_letter_{letter}"] = torch.tensor(
                [bool(value) and value[0] == letter for value in normalized], dtype=torch.bool
            )

    if concept_cfg.get("include_keyword_concepts", True):
        for name, words in concept_cfg.get("keyword_concepts", {}).items():
            vocabulary = {str(word).strip().lower() for word in words}
            masks[f"keyword_{name}"] = torch.tensor([value in vocabulary for value in normalized], dtype=torch.bool)

    # Concepts with no vocabulary support can never pass the positive-count
    # gate, so removing them early keeps result files unambiguous.
    masks = {name: mask for name, mask in masks.items() if bool(mask.any())}
    metadata = {
        name: {
            "vocab_positive": int(mask.sum()),
            "vocab_fraction": float(mask.float().mean()),
        }
        for name, mask in masks.items()
    }
    return masks, metadata, decode_token, decode_tokens


def concept_log_odds(logits, vocab_mask):
    """Log probability odds of a (possibly multi-token) vocabulary concept."""
    mask = torch.as_tensor(vocab_mask, dtype=torch.bool, device=logits.device)
    if not bool(mask.any()) or bool(mask.all()):
        return torch.full(logits.shape[:-1], float("nan"), device=logits.device)
    positive = torch.logsumexp(logits[..., mask].float(), dim=-1)
    negative = torch.logsumexp(logits[..., ~mask].float(), dim=-1)
    return positive - negative


def normalized_entropy(counts):
    counts = torch.as_tensor(counts, dtype=torch.float64)
    counts = counts[counts > 0]
    if counts.numel() <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * probabilities.log()).sum() / math.log(counts.numel()))
