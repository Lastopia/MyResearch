import json

import torch

from model.factory import active_aliases, build_model, validate_aliases
from pipeline.train import BlockData, latest_checkpoint, load_checkpoint, run_seeds
from tools.io import block_dir, checkpoint_dir, ensure_dir, output_dir, save_config, write_json
from tools.log import stage_title
from tools.plot import plot_metric_bars
from tools.resource import configured_gpus, cuda_device, gpu_label, run_gpu_jobs


def normalize_attention_for_metrics(weights):
    """Return row-normalized probabilities without changing model behavior.

    Some sparse attention variants intentionally return unnormalized, gated
    weights. Entropy, top-1 mass, and effective support are probability
    statistics, so evaluation must normalize each non-empty query row first.
    Empty rows are marked invalid and excluded rather than treated as uniform.
    """
    nonnegative = torch.nan_to_num(
        weights.float(), nan=0.0, posinf=0.0, neginf=0.0,
    ).clamp_min(0.0)
    mass = nonnegative.sum(dim=-1)
    valid = mass > 0
    probabilities = nonnegative / mass.unsqueeze(-1).clamp_min(1e-12)
    return probabilities, mass, valid


def load_model(cfg, alias, seed, device):
    ckpt = latest_checkpoint(checkpoint_dir(cfg), alias, seed)
    if ckpt is None:
        raise FileNotFoundError(f"missing checkpoint for {alias} seed={seed}. Run `run train` first.")
    model = build_model(cfg, alias).to(device)
    state = load_checkpoint(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def eval_attn_one(cfg, alias, seed, gpu_index):
    if cfg["run"].get("num_threads"):
        torch.set_num_threads(cfg["run"]["num_threads"])
    device = torch.device(cuda_device(gpu_index)) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"[attn] {gpu_label(gpu_index)} | {alias} | seed={seed} | start", flush=True)
    dtype = torch.float16 if cfg["run"]["dtype"] == "float16" else torch.bfloat16
    model = load_model(cfg, alias, seed, device)
    blocks_path = block_dir(cfg, alias)
    valid_data = BlockData(blocks_path / "valid.bin", cfg["data"]["valid_blocks"], cfg["data"]["block_size"])
    batches = cfg["attn"].get("eval_batches", cfg["train"]["eval_batches"])
    eval_seed = int(cfg["attn"].get("eval_seed", cfg["train"].get("eval_seed", 424242)))

    entropy_sum = 0.0
    top1_sum = 0.0
    eff_sum = 0.0
    mass_sum = 0.0
    valid_rows = 0
    total_rows = 0
    for batch_index in range(batches):
        x, _ = valid_data.deterministic_batch(
            cfg["train"]["batch_size"], device, eval_seed, batch_index,
        )
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            _, _, weights = model(x, return_attn=True)
        for w in weights:
            p, mass, valid = normalize_attention_for_metrics(w)
            entropy = torch.where(p > 0, -(p * p.log()), torch.zeros_like(p)).sum(dim=-1)
            top1 = p.max(dim=-1).values
            eff = entropy.exp()
            entropy_sum += entropy[valid].sum().item()
            top1_sum += top1[valid].sum().item()
            eff_sum += eff[valid].sum().item()
            mass_sum += mass.sum().item()
            valid_rows += int(valid.sum())
            total_rows += valid.numel()

    if valid_rows == 0:
        raise RuntimeError(f"no non-empty attention rows for {alias} seed={seed}")

    row = {
        "model": alias,
        "seed": seed,
        "gpu": gpu_label(gpu_index),
        "eval_seed": eval_seed,
        "evaluation_rows": valid_rows,
        "metric_distribution_normalized": True,
        "attention_mass_mean": round(mass_sum / max(1, total_rows), 6),
        "zero_mass_row_rate": round((total_rows - valid_rows) / max(1, total_rows), 6),
        "attn_entropy": round(entropy_sum / valid_rows, 6),
        "top1_mass": round(top1_sum / valid_rows, 6),
        "effective_tokens": round(eff_sum / valid_rows, 6),
    }
    write_json(output_dir(cfg) / "metrics" / f"[{alias}]seed{seed}attn_summary.json", row)
    print(
        f"[attn] {gpu_label(gpu_index)} | {alias} | "
        f"seed={seed} | "
        f"attn_entropy={row['attn_entropy']:.2f} | "
        f"top1_mass={row['top1_mass']:.2f} | "
        f"effective_tokens={row['effective_tokens']:.2f}",
        flush=True,
    )
    return row


def run(cfg):
    stage_title("attn")
    aliases = active_aliases(cfg)
    validate_aliases(cfg, aliases)
    seeds = run_seeds(cfg)
    jobs = [(alias, seed) for seed in seeds for alias in aliases]
    gpus = configured_gpus(cfg)
    failed = run_gpu_jobs(
        cfg,
        jobs,
        eval_attn_one,
        lambda job, gpu_index: (cfg, job[0], job[1], gpu_index),
        gpus,
        stage="attn",
    )
    if failed:
        raise RuntimeError(f"attn subprocess failed: {failed}")
    rows = []
    for alias, seed in jobs:
        path = output_dir(cfg) / "metrics" / f"[{alias}]seed{seed}attn_summary.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    metrics_dir = output_dir(cfg) / "metrics"
    ensure_dir(metrics_dir)
    with (metrics_dir / "attn.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(metrics_dir / "attn_summary.json", {"models": rows})
    plot_metric_bars(output_dir(cfg), rows, "attn_entropy", "attn_entropy.png")
    plot_metric_bars(output_dir(cfg), rows, "top1_mass", "attn_top1_mass.png")
    plot_metric_bars(output_dir(cfg), rows, "effective_tokens", "attn_effective_tokens.png")
    plot_metric_bars(output_dir(cfg), rows, "attention_mass_mean", "attn_raw_mass.png")
    plot_metric_bars(output_dir(cfg), rows, "zero_mass_row_rate", "attn_zero_mass_row_rate.png")
    save_config(cfg)
    print(f"[attn] done | metrics={metrics_dir / 'attn_summary.json'} | eval={output_dir(cfg) / 'eval'}", flush=True)
