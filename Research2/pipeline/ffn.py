import json

import torch

from model.factory import active_aliases, build_model, validate_aliases
from pipeline.train import BlockData, latest_checkpoint, load_checkpoint, run_seeds
from tools.io import block_dir, checkpoint_dir, ensure_dir, output_dir, save_config, write_json
from tools.log import stage_title
from tools.plot import plot_metric_bars
from tools.resource import configured_gpus, cuda_device, gpu_label, run_gpu_jobs


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
def eval_ffn_one(cfg, alias, seed, gpu_index):
    if cfg["run"].get("num_threads"):
        torch.set_num_threads(cfg["run"]["num_threads"])
    device = torch.device(cuda_device(gpu_index)) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"[ffn] {gpu_label(gpu_index)} | {alias} | seed={seed} | start", flush=True)
    dtype = torch.float16 if cfg["run"]["dtype"] == "float16" else torch.bfloat16
    model = load_model(cfg, alias, seed, device)
    blocks_path = block_dir(cfg, alias)
    valid_data = BlockData(blocks_path / "valid.bin", cfg["data"]["valid_blocks"], cfg["data"]["block_size"])
    batches = cfg.get("ffn", {}).get("eval_batches", cfg["train"].get("eval_batches", 20))
    eval_seed = int(cfg.get("ffn", {}).get("eval_seed", cfg["train"].get("eval_seed", 424242)))

    sums = {}
    count = 0
    for batch_index in range(batches):
        x, y = valid_data.deterministic_batch(
            cfg["train"]["batch_size"], device, eval_seed, batch_index,
        )
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            _, loss, _, _, _ = model(x, y, return_aux=True)
        aux = getattr(model, "last_aux", {})
        for key, val in aux.items():
            if key == "ce_loss":
                continue
            sums[key] = sums.get(key, 0.0) + float(val.detach())
        sums["lm_loss"] = sums.get("lm_loss", 0.0) + float(aux.get("ce_loss", loss).detach())
        count += 1

    row = {
        "model": alias,
        "seed": seed,
        "gpu": gpu_label(gpu_index),
        "eval_seed": eval_seed,
        "evaluation_batches": int(batches),
    }
    for key, val in sums.items():
        row[key] = round(val / max(1, count), 6)
    write_json(output_dir(cfg) / "metrics" / f"[{alias}]seed{seed}ffn_summary.json", row)
    print(
        f"[ffn] {gpu_label(gpu_index)} | {alias} | "
        f"seed={seed} | "
        f"lm_loss={row.get('lm_loss', 0):.2f} | "
        f"act_l0={row.get('act_l0', 0):.2f} | "
        f"act_density={row.get('act_density', 0):.2f}",
        flush=True,
    )
    return row


def run(cfg):
    stage_title("ffn")
    aliases = active_aliases(cfg)
    validate_aliases(cfg, aliases)
    seeds = run_seeds(cfg)
    jobs = [(alias, seed) for seed in seeds for alias in aliases]
    gpus = configured_gpus(cfg)
    failed = run_gpu_jobs(
        cfg,
        jobs,
        eval_ffn_one,
        lambda job, gpu_index: (cfg, job[0], job[1], gpu_index),
        gpus,
        stage="ffn",
    )
    if failed:
        raise RuntimeError(f"ffn subprocess failed: {failed}")
    rows = []
    for alias, seed in jobs:
        path = output_dir(cfg) / "metrics" / f"[{alias}]seed{seed}ffn_summary.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))

    metrics_dir = output_dir(cfg) / "metrics"
    ensure_dir(metrics_dir)
    with (metrics_dir / "ffn.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(metrics_dir / "ffn_summary.json", {"models": rows})
    for metric in ["lm_loss", "l1_act", "act_l0", "act_density", "group_utilization_std", "pre_sparse_l1"]:
        plot_metric_bars(output_dir(cfg), rows, metric, f"ffn_{metric}.png")
    save_config(cfg)
    print(f"[ffn] done | metrics={metrics_dir / 'ffn_summary.json'} | eval={output_dir(cfg) / 'eval'}", flush=True)
