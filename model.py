import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import count_parameters


POSITION_ENCODING_REGISTRY = {
    "std": {
        "description": "Fixed additive sinusoidal position embedding before the first block.",
        "uses_qk_rotation": False,
        "trainable_position_parameters": 0,
    },
    "rope": {
        "description": "RoPE rotates query/key pairs with fixed position-dependent phases.",
        "uses_qk_rotation": True,
        "trainable_position_parameters": 0,
    },
    "pope": {
        "description": "PoPE keeps pair magnitudes and replaces pair directions with position phases plus per-head Q/K phase bias.",
        "uses_qk_rotation": True,
        "trainable_position_parameters": "2 * n_layers * n_heads * head_dim / 2",
    },
    "alibi": {
        "description": "ALiBi adds fixed per-head linear relative-distance bias directly to attention logits.",
        "uses_qk_rotation": False,
        "trainable_position_parameters": 0,
    },
}

POPE_TYPES = {"pope"}
ALIBI_TYPES = {"alibi"}


def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryCache(nn.Module):
    def __init__(self, head_dim, max_seq_len, base):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    def rope(self, x):
        cos = self.cos[:, :, : x.size(-2), :].to(dtype=x.dtype, device=x.device)
        sin = self.sin[:, :, : x.size(-2), :].to(dtype=x.dtype, device=x.device)
        return (x * cos) + (rotate_half(x) * sin)


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len, dropout):
        super().__init__()
        positions = torch.arange(max_seq_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :].to(dtype=x.dtype, device=x.device))


class PoPEApplier(nn.Module):
    def __init__(self, head_dim, n_heads, max_seq_len, base, pope_type="modify"):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"PoPE requires an even head_dim, got {head_dim}")
        if pope_type not in {"origin", "modify"}:
            raise ValueError(f"pope_type must be 'origin' or 'modify', got {pope_type}")
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.pope_type = pope_type
        pair_dim = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("freqs", freqs, persistent=False)
        self.bias_q = nn.Parameter(torch.zeros(n_heads, pair_dim))
        self.bias_k = nn.Parameter(torch.zeros(n_heads, pair_dim))

    def _magnitude(self, x):
        x0, x1 = x[..., 0::2], x[..., 1::2]
        mag = torch.sqrt(x0.square() + x1.square() + 1e-9)
        if self.pope_type == "origin":
            mag = F.softplus(mag)
        return mag

    def _apply_phase(self, x, bias):
        bsz, heads, seq_len, dim = x.shape
        pair_dim = dim // 2
        mag = self._magnitude(x)
        freqs = self.freqs[:seq_len].to(dtype=x.dtype, device=x.device)
        phase = freqs.unsqueeze(0) + bias.clamp(-math.pi, math.pi).to(x.dtype).unsqueeze(1)
        phase = phase.view(1, heads, seq_len, pair_dim)
        out = torch.stack((mag * phase.cos(), mag * phase.sin()), dim=-1)
        return out.reshape(bsz, heads, seq_len, dim)

    def forward(self, q, k):
        return self._apply_phase(q, self.bias_q), self._apply_phase(k, self.bias_k)


class ALiBiBias(nn.Module):
    def __init__(self, n_heads, max_seq_len):
        super().__init__()
        slopes = self._build_slopes(n_heads)
        positions = torch.arange(max_seq_len)
        distance = (positions[:, None] - positions[None, :]).clamp(min=0).float()
        bias = -slopes[:, None, None] * distance[None, :, :]
        self.register_buffer("bias", bias.unsqueeze(0), persistent=False)

    @staticmethod
    def _build_slopes(n_heads):
        def slopes_power_of_2(n):
            start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio**idx) for idx in range(n)]

        if math.log2(n_heads).is_integer():
            slopes = slopes_power_of_2(n_heads)
        else:
            closest_power = 2 ** math.floor(math.log2(n_heads))
            slopes = slopes_power_of_2(closest_power)
            extra = slopes_power_of_2(2 * closest_power)[0::2]
            slopes.extend(extra[: n_heads - closest_power])
        return torch.tensor(slopes, dtype=torch.float32)

    def forward(self, seq_len, dtype, device):
        return self.bias[:, :, :seq_len, :seq_len].to(dtype=dtype, device=device)


class CausalSelfAttention(nn.Module):
    def __init__(self, model_cfg, position_type):
        super().__init__()
        if model_cfg.d_model % model_cfg.n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads, got {model_cfg.d_model} and {model_cfg.n_heads}"
            )
        self.n_heads = model_cfg.n_heads
        self.d_model = model_cfg.d_model
        self.head_dim = self.d_model // self.n_heads
        self.position_type = position_type
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.out = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(model_cfg.dropout)
        base = model_cfg.pope_base if position_type in POPE_TYPES else model_cfg.rope_base
        self.rotary = RotaryCache(self.head_dim, model_cfg.seq_len, base)
        self.pope = (
            PoPEApplier(
                self.head_dim,
                self.n_heads,
                model_cfg.seq_len,
                model_cfg.pope_base,
                getattr(model_cfg, "pope_type", "modify"),
            )
            if position_type in POPE_TYPES
            else None
        )
        self.alibi = ALiBiBias(self.n_heads, model_cfg.seq_len) if position_type in ALIBI_TYPES else None

    def forward(self, x, return_attention=False):
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.position_type == "rope":
            q, k = self.rotary.rope(q), self.rotary.rope(k)
        elif self.position_type in POPE_TYPES:
            if self.pope is None:
                raise RuntimeError("PoPE applier was not initialized")
            q, k = self.pope(q, k)
        elif self.position_type not in {"std", "alibi"}:
            raise ValueError(f"Unknown position type: {self.position_type}")

        if not return_attention and self.alibi is None:
            dropout_p = self.dropout.p if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
            y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
            return self.out(y), None, None

        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.alibi is not None:
            logits = logits + self.alibi(seq_len, logits.dtype, logits.device)
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        attn = F.softmax(logits.float(), dim=-1).to(v.dtype)
        y = self.dropout(attn) @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        y = self.out(y)
        if return_attention:
            return y, attn, logits
        return y, None, None


class MLP(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_cfg.d_model, model_cfg.d_ff),
            nn.GELU(),
            nn.Linear(model_cfg.d_ff, model_cfg.d_model),
            nn.Dropout(model_cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, model_cfg, position_type):
        super().__init__()
        self.ln1 = nn.LayerNorm(model_cfg.d_model)
        self.attn = CausalSelfAttention(model_cfg, position_type)
        self.ln2 = nn.LayerNorm(model_cfg.d_model)
        self.mlp = MLP(model_cfg)

    def forward(self, x, return_attention=False):
        attn_out, attn, logits = self.attn(self.ln1(x), return_attention=return_attention)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, attn, logits


class GPTLikeTransformer(nn.Module):
    def __init__(self, model_cfg, position_type):
        super().__init__()
        if position_type not in POSITION_ENCODING_REGISTRY:
            raise ValueError(f"Unknown position type: {position_type}")
        self.model_cfg = model_cfg
        self.position_type = position_type
        self.token_embedding = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        if position_type == "std":
            self.position_embedding = SinusoidalPositionEmbedding(
                model_cfg.d_model,
                model_cfg.seq_len,
                model_cfg.dropout,
            )
        else:
            self.position_embedding = None
        self.drop = nn.Dropout(model_cfg.dropout)
        self.blocks = nn.ModuleList([Block(model_cfg, position_type) for _ in range(model_cfg.n_layers)])
        self.ln_f = nn.LayerNorm(model_cfg.d_model)
        self.lm_head = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)
        if model_cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, return_attention=False, capture_layers=None):
        if input_ids.size(1) > self.model_cfg.seq_len:
            raise ValueError(
                f"Input sequence length {input_ids.size(1)} exceeds configured seq_len {self.model_cfg.seq_len}"
            )
        x = self.drop(self.token_embedding(input_ids))
        if self.position_embedding is not None:
            x = self.position_embedding(x)
        attentions, attention_logits, activations = [], [], {}
        capture_layers = set(capture_layers or [])
        for idx, block in enumerate(self.blocks):
            x, attn, logits = block(x, return_attention=return_attention)
            if idx in capture_layers:
                activations[idx] = x
            if return_attention:
                attentions.append(attn)
                attention_logits.append(logits)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1))
        return {
            "loss": loss,
            "logits": logits,
            "attentions": attentions,
            "attention_logits": attention_logits,
            "activations": activations,
        }


class SelfTransformer:
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg

    @property
    def head_dim(self):
        return self.model_cfg.d_model // self.model_cfg.n_heads

    def validate_config(self):
        if self.model_cfg.d_model % self.model_cfg.n_heads != 0:
            raise ValueError(
                f"d_model must be divisible by n_heads, got {self.model_cfg.d_model} and {self.model_cfg.n_heads}"
            )
        if self.model_cfg.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.model_cfg.seq_len}")
        unknown = [name for name in self.model_cfg.model_names if name not in POSITION_ENCODING_REGISTRY]
        if unknown:
            raise ValueError(f"Unknown model_names: {unknown}")
        if any(name in POPE_TYPES for name in self.model_cfg.model_names) and self.head_dim % 2 != 0:
            raise ValueError(f"PoPE requires an even head_dim, got {self.head_dim}")

    def build_model(self, model_name):
        if model_name in POSITION_ENCODING_REGISTRY:
            return GPTLikeTransformer(self.model_cfg, model_name)
        raise ValueError(f"Unknown model name: {model_name}")

    def position_meta(self, model_name):
        meta = dict(POSITION_ENCODING_REGISTRY[model_name])
        if model_name in POPE_TYPES:
            meta["trainable_position_parameters"] = (
                2 * self.model_cfg.n_layers * self.model_cfg.n_heads * (self.head_dim // 2)
            )
        if model_name in ALIBI_TYPES:
            meta["alibi_bias_parameters"] = 0
        return meta

    def architecture_signature(self):
        return {
            "n_layers": self.model_cfg.n_layers,
            "d_model": self.model_cfg.d_model,
            "n_heads": self.model_cfg.n_heads,
            "head_dim": self.head_dim,
            "d_ff": self.model_cfg.d_ff,
            "dropout": self.model_cfg.dropout,
            "vocab_size": self.model_cfg.vocab_size,
            "seq_len": self.model_cfg.seq_len,
            "tie_embeddings": self.model_cfg.tie_embeddings,
        }

    @torch.no_grad()
    def smoke_check_model(self, model):
        seq_len = min(8, self.model_cfg.seq_len)
        input_ids = torch.arange(seq_len).unsqueeze(0) % self.model_cfg.vocab_size
        model.eval()
        out = model(input_ids, labels=input_ids, return_attention=True)
        logits_ok = out["logits"].shape == (1, seq_len, self.model_cfg.vocab_size)
        loss_ok = out["loss"] is not None and torch.isfinite(out["loss"]).item()
        attention_layers_ok = len(out["attentions"]) == self.model_cfg.n_layers
        causal_mask_ok = True
        attention_shape_ok = True
        for attn, raw_logits in zip(out["attentions"], out["attention_logits"]):
            expected = (1, self.model_cfg.n_heads, seq_len, seq_len)
            attention_shape_ok = attention_shape_ok and attn.shape == expected and raw_logits.shape == expected
            upper = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
            causal_mask_ok = causal_mask_ok and torch.all(attn[:, :, upper] == 0).item()
        return {
            "forward_logits_shape_ok": logits_ok,
            "finite_loss_ok": bool(loss_ok),
            "attention_layer_count_ok": attention_layers_ok,
            "attention_shape_ok": attention_shape_ok,
            "causal_mask_ok": causal_mask_ok,
            "checked_seq_len": seq_len,
        }

    def parameter_audit(self, model_meta):
        baseline_name = getattr(self.model_cfg, "baseline_model_name", self.model_cfg.model_names[0])
        if baseline_name not in model_meta:
            baseline_name = self.model_cfg.model_names[0]
        baseline_count = model_meta[baseline_name]["num_parameters"]
        max_ratio_delta = getattr(self.model_cfg, "max_parameter_ratio_delta", 0.01)
        audit = {}
        for name, meta in model_meta.items():
            delta = meta["num_parameters"] - baseline_count
            ratio_delta = delta / max(baseline_count, 1)
            audit[name] = {
                "baseline": baseline_name,
                "parameter_delta": delta,
                "parameter_ratio_delta": ratio_delta,
                "within_tolerance": abs(ratio_delta) <= max_ratio_delta,
            }
        return audit

    def run(self):
        self.validate_config()
        models = {name: self.build_model(name) for name in self.model_cfg.model_names}
        model_meta = {
            name: {
                "num_parameters": count_parameters(model),
                "position_encoding": self.position_meta(name),
                "architecture_signature": self.architecture_signature(),
            }
            for name, model in models.items()
        }
        checks = {}
        if getattr(self.model_cfg, "run_model_checks", True):
            checks = {name: self.smoke_check_model(model) for name, model in models.items()}
        parameter_audit = self.parameter_audit(model_meta)
        return {
            "config": self.model_cfg,
            "models": models,
            "meta": model_meta,
            "checks": checks,
            "parameter_audit": parameter_audit,
        }
