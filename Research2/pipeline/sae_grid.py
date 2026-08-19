from itertools import product

import torch

from model.sae import TopKSAE
from tools.io import checkpoint_dir, output_dir


def _as_list(value):
    return value if isinstance(value, list) else [value]


def sae_layers(cfg):
    values = cfg["sae"].get("layers", [cfg["sae"].get("hook_layer", -1)])
    return [int(value) for value in _as_list(values)]


def sae_ks(cfg):
    values = cfg["sae"].get("k_grid", [cfg["sae"].get("k", 32)])
    return [int(value) for value in _as_list(values)]


def sae_expansions(cfg):
    values = cfg["sae"].get("expansion_grid", [cfg["sae"].get("expansion", 8)])
    return [int(value) for value in _as_list(values)]


def sae_seeds(cfg):
    values = cfg["sae"].get("seed", [0])
    return [int(value) for value in _as_list(values)]


def sae_specs(cfg):
    rows = []
    for layer, expansion, k, sae_seed in product(sae_layers(cfg), sae_expansions(cfg), sae_ks(cfg), sae_seeds(cfg)):
        rows.append({
            "layer": layer,
            "expansion": expansion,
            "k": k,
            "sae_seed": sae_seed,
            "sae_id": sae_run_id(layer, expansion, k, sae_seed),
        })
    return rows


def sae_run_id(layer, expansion, k, sae_seed):
    return f"layer{int(layer)}_e{int(expansion)}x_k{int(k)}_sseed{int(sae_seed)}"


def sae_dir(cfg, alias, model_seed, spec):
    return output_dir(cfg) / "sae" / f"{alias}_mseed{model_seed}_{spec['sae_id']}"


def sae_checkpoint_dir(cfg, alias, model_seed, spec):
    return checkpoint_dir(cfg) / "sae" / f"{alias}_mseed{model_seed}_{spec['sae_id']}"


def sae_summary_path(cfg, alias, model_seed, spec):
    return output_dir(cfg) / "metrics" / f"[{alias}]mseed{model_seed}_{spec['sae_id']}_sae_summary.json"


def load_sae(cfg, alias, model_seed, spec, d_in, device="cpu"):
    path = sae_checkpoint_dir(cfg, alias, model_seed, spec) / "checkpoint.pt"
    if not path.exists():
        raise FileNotFoundError(f"missing SAE checkpoint: {path}. Run `run sae` first.")
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(path, map_location=device)
    d_sae = int(state.get("d_sae", d_in * spec["expansion"]))
    sae = TopKSAE(
        d_in,
        d_sae,
        spec["k"],
        tied_init=cfg["sae"].get("tied_init", True),
        normalize_decoder=cfg["sae"].get("normalize_decoder", True),
    ).to(device)
    sae.load_state_dict(state["sae"])
    sae.eval()
    return sae, state
