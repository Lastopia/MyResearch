import torch.nn.functional as F


def ce_loss(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def build_loss(name):
    if name in ("ce", "l1_act"):
        return ce_loss
    raise ValueError(f"unknown loss: {name}")
