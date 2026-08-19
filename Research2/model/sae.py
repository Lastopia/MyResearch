import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, d_in, d_sae, k, tied_init=True, normalize_decoder=True):
        super().__init__()
        self.k = k
        self.normalize_decoder = normalize_decoder
        self.encoder = nn.Linear(d_in, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_in, bias=True)

        nn.init.kaiming_uniform_(self.decoder.weight, a=5 ** 0.5)
        if self.normalize_decoder:
            self.normalize_decoder_()
        if tied_init:
            with torch.no_grad():
                self.encoder.weight.copy_(self.decoder.weight.T)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def pre_activations(self, x):
        return F.relu(self.encoder(x - self.decoder.bias))

    def encode(self, x, return_pre=False):
        pre = self.pre_activations(x)
        keep = min(max(1, int(self.k)), pre.size(-1))
        values, idx = torch.topk(pre, keep, dim=-1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(-1, idx, values)
        if return_pre:
            return sparse, pre
        return sparse

    def forward(self, x, return_pre=False):
        encoded = self.encode(x, return_pre=return_pre)
        if return_pre:
            z, pre = encoded
        else:
            z, pre = encoded, None
        recon = self.decoder(z)
        if return_pre:
            return recon, z, pre
        return recon, z

    @torch.no_grad()
    def normalize_decoder_(self):
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def remove_decoder_gradient_parallel_component_(self):
        if self.decoder.weight.grad is None:
            return
        weight = F.normalize(self.decoder.weight, dim=0)
        parallel = (self.decoder.weight.grad * weight).sum(dim=0, keepdim=True) * weight
        self.decoder.weight.grad.sub_(parallel)

    def dead_latent_aux_loss(self, x, recon, pre, dead_mask, aux_k):
        """Reconstruct the detached residual with currently dead latents.

        This follows the TopK auxiliary-loss idea while keeping the primary
        reconstruction path unchanged. It returns zero if no dead latent is
        available, so callers can enable it safely for every checkpoint.
        """
        if dead_mask is None or not bool(dead_mask.any()):
            return x.new_tensor(0.0)
        scores = pre.masked_fill(~dead_mask.view(1, -1), float("-inf"))
        keep = min(max(1, int(aux_k)), int(dead_mask.sum().item()))
        values, indices = torch.topk(scores, keep, dim=-1)
        values = values.masked_fill(~torch.isfinite(values), 0.0)
        aux_z = torch.zeros_like(pre).scatter_(-1, indices, values)
        aux_recon = F.linear(aux_z, self.decoder.weight, bias=None)
        residual = (x - recon).detach()
        return F.mse_loss(aux_recon, residual)
