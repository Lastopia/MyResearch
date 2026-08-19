import torch
import torch.nn as nn

from .attn import build_attn
from .ffn import build_ffn
from .loss import build_loss


class Block(nn.Module):
    def __init__(self, model_cfg, variant):
        super().__init__()
        self.ln1 = nn.LayerNorm(model_cfg["d_model"])
        self.attn = build_attn(variant["attn"]["name"], model_cfg, variant["attn"])
        self.ln2 = nn.LayerNorm(model_cfg["d_model"])
        self.ffn = build_ffn(variant["ffn"]["name"], model_cfg, variant["ffn"])

    def forward(self, x, need_attn=False, collect_aux=False):
        if need_attn:
            attn_out, weights = self.attn(self.ln1(x), need_weights=True)
        else:
            attn_out, weights = self.attn(self.ln1(x)), None
        x = x + attn_out
        if collect_aux:
            ffn_out, stats = self.ffn(self.ln2(x), collect_stats=True)
        else:
            ffn_out, stats = self.ffn(self.ln2(x)), None
        x = x + ffn_out
        if need_attn or collect_aux:
            return x, weights, stats
        return x


class Transformer(nn.Module):
    def __init__(self, model_cfg, variant):
        super().__init__()
        self.cfg = dict(model_cfg)
        self.variant = dict(variant)
        self.loss_cfg = dict(variant.get("loss", {"name": "ce"}))
        self.loss_fn = build_loss(self.loss_cfg["name"])
        self.last_aux = {}
        self.tok_emb = nn.Embedding(model_cfg["vocab_size"], model_cfg["d_model"])
        self.drop = nn.Dropout(model_cfg.get("dropout", 0.0))
        self.blocks = nn.ModuleList([Block(model_cfg, variant) for _ in range(model_cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(model_cfg["d_model"])
        self.head = nn.Linear(model_cfg["d_model"], model_cfg["vocab_size"], bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_hidden=False, hook_layer=None, return_attn=False, return_aux=False):
        x = self.tok_emb(idx)
        x = self.drop(x)
        hidden = None
        attn_weights = []
        aux_stats = []
        collect_aux = return_aux or (targets is not None and self.loss_cfg["name"] == "l1_act")
        for layer, block in enumerate(self.blocks, start=1):
            if return_attn or collect_aux:
                x, weights, stats = block(x, need_attn=return_attn, collect_aux=collect_aux)
                attn_weights.append(weights)
                if stats is not None:
                    aux_stats.append(stats)
            else:
                x = block(x)
            if return_hidden and layer == hook_layer:
                hidden = x
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            if return_aux:
                aux = self._merge_aux(aux_stats)
                return logits, hidden, attn_weights, aux
            if return_hidden or return_attn:
                return logits, hidden, attn_weights
            return logits
        ce_loss = self.loss_fn(logits, targets)
        aux = self._merge_aux(aux_stats)
        loss = self._apply_aux_loss(ce_loss, aux)
        if return_aux:
            return logits, loss, hidden, attn_weights, aux
        if return_hidden or return_attn:
            return logits, loss, hidden, attn_weights
        return logits, loss

    def _merge_aux(self, stats):
        if not stats:
            self.last_aux = {}
            return {}
        keys = sorted({key for row in stats for key in row})
        merged = {}
        for key in keys:
            vals = [row[key] for row in stats if key in row]
            if vals:
                merged[key] = torch.stack(vals).mean()
        self.last_aux = merged
        return merged

    def _apply_aux_loss(self, ce_loss, aux):
        if self.loss_cfg["name"] != "l1_act":
            self.last_aux = {"ce_loss": ce_loss.detach(), **{k: v.detach() for k, v in aux.items()}}
            return ce_loss
        l1 = aux.get("l1_act")
        if l1 is None:
            raise RuntimeError("loss.name=l1_act requires FFN l1_act stats")
        weight = self.loss_cfg.get("lambda", 1e-4)
        total = ce_loss + weight * l1
        self.last_aux = {
            "ce_loss": ce_loss.detach(),
            "l1_act": l1.detach(),
            "aux_loss": (weight * l1).detach(),
            **{k: v.detach() for k, v in aux.items() if k != "l1_act"},
        }
        return total

    def logits_from_hidden(self, hidden):
        return self.head(self.ln_f(hidden))

    def forward_from_hidden(self, hidden, hook_layer):
        """Continue a forward pass from a residual stream after hook_layer.

        hook_layer uses the same 1-based, post-block convention as forward().
        This is the intervention path used by causal SAE evaluations.
        """
        layer = int(hook_layer)
        if layer < 0:
            layer = len(self.blocks)
        if layer < 0 or layer > len(self.blocks):
            raise ValueError(f"hook_layer must be in [0, {len(self.blocks)}], got {hook_layer}")
        x = hidden
        for block in self.blocks[layer:]:
            x = block(x)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg["block_size"]:]
            logits = self(idx_cond)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat((idx, next_id), dim=1)
        return idx
