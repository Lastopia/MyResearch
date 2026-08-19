import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .loss import ce_loss


def premodel_entry(cfg, alias=None):
    pre_cfg = cfg["premodel"]
    alias = alias or pre_cfg.get("default")
    entry = dict(pre_cfg["models"][alias])
    entry["alias"] = alias
    return entry


def premodel_aliases(cfg):
    aliases = cfg["run"].get("models") or [cfg["premodel"]["default"]]
    if isinstance(aliases, str):
        aliases = [aliases]
    return aliases


def torch_dtype_from_cfg(cfg):
    dtype = cfg.get("premodel", {}).get("torch_dtype", "auto")
    if dtype in (None, "auto"):
        return "auto"
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"unknown premodel.torch_dtype: {dtype}")


def _optional(value):
    return None if value in (None, "", "none", "None") else value


def tokenizer_for_premodel(cfg, alias):
    pre_cfg = cfg["premodel"]
    entry = premodel_entry(cfg, alias)
    tokenizer_id = entry.get("tokenizer_id") or entry["hf_id"]
    kwargs = {
        "revision": _optional(entry.get("revision")),
        "cache_dir": _optional(pre_cfg.get("cache_dir")),
        "local_files_only": bool(pre_cfg.get("local_files_only", False)),
        "trust_remote_code": bool(entry.get("trust_remote_code", pre_cfg.get("trust_remote_code", False))),
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class PretrainedCausalLM(nn.Module):
    def __init__(self, cfg, alias):
        super().__init__()
        self.alias = alias
        self.entry = premodel_entry(cfg, alias)
        self.loss_fn = ce_loss
        self.last_aux = {}
        pre_cfg = cfg["premodel"]
        kwargs = {
            "revision": _optional(self.entry.get("revision")),
            "cache_dir": _optional(pre_cfg.get("cache_dir")),
            "local_files_only": bool(pre_cfg.get("local_files_only", False)),
            "trust_remote_code": bool(self.entry.get("trust_remote_code", pre_cfg.get("trust_remote_code", False))),
            "torch_dtype": torch_dtype_from_cfg(cfg),
        }
        attn_impl = _optional(self.entry.get("attn_implementation", pre_cfg.get("attn_implementation")))
        if attn_impl is not None:
            kwargs["attn_implementation"] = attn_impl
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        self.model = AutoModelForCausalLM.from_pretrained(self.entry["hf_id"], **kwargs)
        if pre_cfg.get("gradient_checkpointing", False) and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False
        config = self.model.config
        self.hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", None))
        self.n_layer = getattr(config, "num_hidden_layers", getattr(config, "n_layer", None))
        self.vocab_size = getattr(config, "vocab_size", None)

    def forward(self, idx, targets=None, return_hidden=False, hook_layer=None, return_attn=False, return_aux=False):
        out = self.model(
            input_ids=idx,
            output_hidden_states=return_hidden,
            output_attentions=return_attn,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits
        hidden = None
        if return_hidden:
            states = out.hidden_states
            if hook_layer == -1 or hook_layer is None:
                hook_layer = len(states) - 1
            hidden = states[int(hook_layer)]
        attn_weights = list(out.attentions) if return_attn and out.attentions is not None else []
        if targets is None:
            if return_aux:
                return logits, hidden, attn_weights, {}
            if return_hidden or return_attn:
                return logits, hidden, attn_weights
            return logits
        loss = self.loss_fn(logits, targets)
        self.last_aux = {"ce_loss": loss.detach()}
        if return_aux:
            return logits, loss, hidden, attn_weights, self.last_aux
        if return_hidden or return_attn:
            return logits, loss, hidden, attn_weights
        return logits, loss

    def logits_from_hidden(self, hidden):
        core = getattr(self.model, "model", None)
        norm = getattr(core, "norm", None)
        if norm is None:
            norm = getattr(core, "final_layer_norm", None)
        if norm is not None:
            hidden = norm(hidden)
        return self.model.get_output_embeddings()(hidden)

    def forward_from_hidden(self, hidden, hook_layer):
        if int(hook_layer) not in (-1, int(self.n_layer)):
            raise NotImplementedError("intermediate-layer continuation is only implemented for the local Transformer")
        return self.logits_from_hidden(hidden)
