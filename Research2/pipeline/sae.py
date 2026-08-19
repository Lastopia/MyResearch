import json
import math
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from model.factory import active_aliases, build_model, checkpoint_signature, hidden_size, is_pretrain_mode, resolve_experiment, validate_aliases
from model.sae import TopKSAE
from pipeline.sae_grid import sae_checkpoint_dir, sae_dir, sae_specs, sae_summary_path
from pipeline.train import BlockData, latest_checkpoint, load_checkpoint, run_seeds
from tools.io import block_dir, checkpoint_dir, ensure_dir, output_dir, save_config, write_json
from tools.log import elapsed_seconds, event_line, now, stage_title, train_line
from tools.plot import plot_metric_by_k
from tools.resource import configured_gpus, cuda_device, gpu_label, mem_usage, run_gpu_jobs, vram_usage


def resolved_hook_layer(base, cfg, spec):
    layer = int(spec["layer"])
    if layer == -1:
        layer = int(getattr(base, "n_layer", cfg["model"]["n_layer"]))
    return layer


def load_base(cfg, alias, seed, device):
    ckpt = latest_checkpoint(checkpoint_dir(cfg), alias, seed)
    if ckpt is None:
        raise FileNotFoundError(f"missing base checkpoint for {alias} seed={seed}. Run `run train` first.")
    model = build_model(cfg, alias).to(device)
    state = load_checkpoint(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def activation_batch(base, data, cfg, spec, device, dtype):
    x, _ = data.batch(cfg["train"]["batch_size"], device)
    hook_layer = resolved_hook_layer(base, cfg, spec)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        _, hidden, _ = base(x, return_hidden=True, hook_layer=hook_layer)
    if hidden is None:
        raise RuntimeError(f"model did not return hidden state for hook_layer={hook_layer}")
    return hidden.reshape(-1, hidden.size(-1)).float()


def _continue_logits(base, hidden, hook_layer):
    if hasattr(base, "forward_from_hidden"):
        return base.forward_from_hidden(hidden, hook_layer)
    final_layer = int(getattr(base, "n_layer", hook_layer))
    if hook_layer == final_layer and hasattr(base, "logits_from_hidden"):
        return base.logits_from_hidden(hidden)
    raise NotImplementedError("model cannot continue from this hook layer")


def _decoder_duplication(decoder_weight, chosen, chunk_size=1024):
    if chosen.numel() <= 1:
        return {}
    vectors = F.normalize(decoder_weight[:, chosen].T.float(), dim=-1)
    maxima = []
    for start in range(0, vectors.size(0), chunk_size):
        block = vectors[start:start + chunk_size] @ vectors.T
        row = torch.arange(block.size(0), device=block.device)
        col = torch.arange(start, start + block.size(0), device=block.device)
        block[row, col] = 0.0
        maxima.append(block.abs().max(dim=-1).values.cpu())
    values = torch.cat(maxima)
    return {
        "decoder_duplication_proxy": values.mean().item(),
        "decoder_max_cosine_p50": values.quantile(0.50).item(),
        "decoder_max_cosine_p95": values.quantile(0.95).item(),
        "decoder_max_cosine_p99": values.quantile(0.99).item(),
    }


@torch.no_grad()
def eval_sae_metrics(base, sae, valid_data, cfg, spec, device, dtype):
    base.eval()
    sae.eval()
    hook_layer = resolved_hook_layer(base, cfg, spec)
    eval_batches = int(cfg["sae"].get("eval_batches", cfg["train"].get("eval_batches", 20)))
    eval_seed = int(cfg["sae"].get("eval_seed", 9102))
    batch_size = int(cfg["train"]["batch_size"])

    mse_values = []
    normalized_mse_values = []
    explained_variances = []
    cosine_values = []
    reconstruction_biases = []
    l0s = []
    l1s = []
    activation_entropies = []
    freq_sum = torch.zeros(sae.encoder.out_features, device=device)
    magnitude_sum = torch.zeros_like(freq_sum)
    token_count = 0
    base_losses = []
    recon_lm_losses = []
    zero_lm_losses = []
    recon_kls = []
    zero_kls = []
    ablation_cache = []
    can_patch = True

    for batch_index in range(eval_batches):
        x, y = valid_data.deterministic_batch(batch_size, device, eval_seed, batch_index)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits, base_loss, hidden, _ = base(x, y, return_hidden=True, hook_layer=hook_layer)
        flat = hidden.reshape(-1, hidden.size(-1)).float()
        recon, z = sae(flat)
        error = recon - flat
        mse = error.square().mean()
        centered_energy = (flat - flat.mean(dim=0, keepdim=True)).square().mean().clamp_min(1e-12)
        error_variance = (error - error.mean(dim=0, keepdim=True)).square().mean()
        mse_values.append(mse.item())
        normalized_mse_values.append((mse / centered_energy).item())
        explained_variances.append((1.0 - error_variance / centered_energy).item())
        cosine_values.append(F.cosine_similarity(recon, flat, dim=-1).mean().item())
        bias = (recon.mean(dim=0) - flat.mean(dim=0)).norm() / flat.std(dim=0, unbiased=False).norm().clamp_min(1e-12)
        reconstruction_biases.append(bias.item())

        active = z > 0
        l0s.append(active.float().sum(dim=-1).mean().item())
        l1s.append(z.abs().sum(dim=-1).mean().item())
        activation_mass = z.clamp_min(0)
        activation_probability = activation_mass / activation_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        activation_entropy = -(activation_probability * activation_probability.clamp_min(1e-12).log()).sum(dim=-1)
        activation_entropy = activation_entropy / math.log(max(2, min(int(spec["k"]), sae.encoder.out_features)))
        activation_entropies.append(activation_entropy.mean().item())
        freq_sum += active.float().sum(dim=0)
        magnitude_sum += z.abs().sum(dim=0)
        token_count += active.size(0)

        if can_patch:
            try:
                recon_hidden = recon.to(hidden.dtype).view_as(hidden)
                zero_hidden = torch.zeros_like(hidden)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                    recon_logits = _continue_logits(base, recon_hidden, hook_layer)
                    zero_logits = _continue_logits(base, zero_hidden, hook_layer)
                recon_lm_losses.append(base.loss_fn(recon_logits, y).item())
                zero_lm_losses.append(base.loss_fn(zero_logits, y).item())
                base_losses.append(base_loss.item())
                flat_logits = logits.float().reshape(-1, logits.size(-1))
                flat_recon_logits = recon_logits.float().reshape(-1, recon_logits.size(-1))
                flat_zero_logits = zero_logits.float().reshape(-1, zero_logits.size(-1))
                base_prob = torch.softmax(flat_logits, dim=-1)
                recon_kl = F.kl_div(torch.log_softmax(flat_recon_logits, dim=-1), base_prob, reduction="batchmean")
                zero_kl = F.kl_div(torch.log_softmax(flat_zero_logits, dim=-1), base_prob, reduction="batchmean")
                recon_kls.append(recon_kl.item())
                zero_kls.append(zero_kl.item())
                if len(ablation_cache) < int(cfg["sae"].get("ablation_batches", 2)):
                    ablation_cache.append((hidden.detach(), y.detach(), z.detach(), logits.detach()))
            except (NotImplementedError, RuntimeError):
                can_patch = False
                base_losses.clear()
                recon_lm_losses.clear()
                zero_lm_losses.clear()
                recon_kls.clear()
                zero_kls.clear()
                ablation_cache.clear()

    freq = freq_sum / max(1, token_count)
    magnitude = magnitude_sum / max(1, token_count)
    dead_rate = (freq == 0).float().mean().item()
    rare_threshold = float(cfg["sae"].get("rare_feature_threshold", 1e-5))
    rare_rate = (freq < rare_threshold).float().mean().item()
    active_freq = freq[freq > 0]
    feature_usage_entropy = 0.0
    if active_freq.numel():
        usage = active_freq / active_freq.sum().clamp_min(1e-12)
        feature_usage_entropy = (-(usage * usage.log()).sum() / math.log(max(2, usage.numel()))).item()

    active_order = torch.argsort(freq, descending=True)
    cap = min(int(cfg["sae"].get("duplication_eval_features", 4096)), sae.encoder.out_features)
    duplication = _decoder_duplication(sae.decoder.weight.detach(), active_order[:cap])
    encoder = F.normalize(sae.encoder.weight.detach().float(), dim=-1)
    decoder = F.normalize(sae.decoder.weight.detach().T.float(), dim=-1)
    encoder_decoder_alignment = (encoder * decoder).sum(dim=-1)

    summary = {
        "sae_valid_loss": round(sum(mse_values) / len(mse_values), 8),
        "normalized_mse": round(sum(normalized_mse_values) / len(normalized_mse_values), 8),
        "explained_variance": round(sum(explained_variances) / len(explained_variances), 8),
        "reconstruction_cosine": round(sum(cosine_values) / len(cosine_values), 8),
        "relative_reconstruction_bias": round(sum(reconstruction_biases) / len(reconstruction_biases), 8),
        "active": round(sum(l0s) / len(l0s), 4),
        "actual_l0": round(sum(l0s) / len(l0s), 4),
        "latent_density": round((sum(l0s) / len(l0s)) / sae.encoder.out_features, 8),
        "sae_l1": round(sum(l1s) / len(l1s), 6),
        "activation_entropy": round(sum(activation_entropies) / len(activation_entropies), 6),
        "dead_feature_rate": round(dead_rate, 6),
        "fraction_alive": round(1.0 - dead_rate, 6),
        "rare_feature_rate": round(rare_rate, 6),
        "feature_usage_entropy": round(feature_usage_entropy, 6),
        "feature_activation_mean": round(freq.mean().item(), 8),
        "feature_activation_std": round(freq.std(unbiased=False).item(), 8),
        "feature_activation_max": round(freq.max().item(), 8),
        "encoder_decoder_alignment_mean": round(encoder_decoder_alignment.mean().item(), 6),
        "encoder_decoder_alignment_p05": round(encoder_decoder_alignment.quantile(0.05).item(), 6),
        **{key: round(value, 6) for key, value in duplication.items()},
    }

    if base_losses:
        base_lm = sum(base_losses) / len(base_losses)
        recon_lm = sum(recon_lm_losses) / len(recon_lm_losses)
        zero_lm = sum(zero_lm_losses) / len(zero_lm_losses)
        denom = max(1e-8, zero_lm - base_lm)
        mean_recon_kl = sum(recon_kls) / len(recon_kls)
        mean_zero_kl = sum(zero_kls) / len(zero_kls)
        summary.update({
            "base_lm_loss": round(base_lm, 6),
            "recon_lm_loss": round(recon_lm, 6),
            "zero_ablation_lm_loss": round(zero_lm, 6),
            "loss_recovered": round((zero_lm - recon_lm) / denom, 6),
            "reconstruction_kl": round(mean_recon_kl, 6),
            "zero_ablation_kl": round(mean_zero_kl, 6),
            "kl_recovered": round(1.0 - mean_recon_kl / max(1e-8, mean_zero_kl), 6),
        })

    ablation_deltas = []
    ablation_coverages = []
    if ablation_cache:
        top_n = min(int(cfg["sae"].get("ablation_features", 16)), sae.encoder.out_features)
        chosen_features = active_order[:top_n]
        for feature in chosen_features:
            if freq[feature].item() <= 0:
                continue
            deltas = []
            for hidden, y, z, original_logits in ablation_cache:
                contribution = z[:, feature:feature + 1] * sae.decoder.weight[:, feature].float().view(1, -1)
                ablated_hidden = hidden.float().reshape(-1, hidden.size(-1)) - contribution
                ablated_hidden = ablated_hidden.to(hidden.dtype).view_as(hidden)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                    ablated_logits = _continue_logits(base, ablated_hidden, hook_layer)
                original_loss = base.loss_fn(original_logits, y).item()
                deltas.append(base.loss_fn(ablated_logits, y).item() - original_loss)
            ablation_deltas.append(sum(deltas) / len(deltas))
            ablation_coverages.append(freq[feature].item())
    if ablation_deltas:
        deltas = torch.tensor(ablation_deltas)
        coverages = torch.tensor(ablation_coverages)
        summary.update({
            "feature_ablation_loss_delta_mean": round(deltas.mean().item(), 6),
            "feature_ablation_loss_delta_max": round(deltas.max().item(), 6),
            "feature_ablation_coverage_mean": round(coverages.mean().item(), 6),
        })

    feature_stats = {
        "frequency": freq.detach().cpu(),
        "mean_magnitude": magnitude.detach().cpu(),
        "encoder_decoder_alignment": encoder_decoder_alignment.detach().cpu(),
        "tokens": int(token_count),
    }
    return summary, feature_stats


def train_sae_one(cfg, alias, model_seed, spec, gpu_index):
    if cfg["run"].get("num_threads"):
        torch.set_num_threads(cfg["run"]["num_threads"])
    out = output_dir(cfg)
    target = sae_dir(cfg, alias, model_seed, spec)
    ckpt_target = sae_checkpoint_dir(cfg, alias, model_seed, spec)
    if cfg["sae"].get("retrain", True):
        for stale_dir in (target, ckpt_target):
            if stale_dir.exists():
                shutil.rmtree(stale_dir)
    ensure_dir(target)
    ensure_dir(ckpt_target)
    ckpt_path = ckpt_target / "checkpoint.pt"
    config_path = ckpt_target / "config.json"
    validate_aliases(cfg, [alias])
    current = {
        "mode": cfg["run"].get("mode", "retrain"),
        "model": alias,
        "model_seed": model_seed,
        "sae_spec": spec,
        "sae": cfg["sae"],
        "model_cfg": cfg.get("model"),
        "signature": checkpoint_signature(cfg, alias),
    }
    if is_pretrain_mode(cfg):
        current["premodel"] = cfg["premodel"]["models"][alias]
    else:
        exp = resolve_experiment(cfg, alias)
        current["model_alias"] = exp["model_alias"]
        current["position_encoding"] = exp["position_encoding"]
        current["variant"] = cfg["models"][exp["model_alias"]]
    if not cfg["sae"].get("retrain", True) and ckpt_path.exists() and config_path.exists():
        try:
            old = json.loads(config_path.read_text(encoding="utf-8"))
            if old == current:
                event_line("sae", gpu_label(gpu_index), alias, model_seed, "checkpoint exists", spec["sae_id"])
                return {"model": alias, "model_seed": model_seed, **spec, "skipped": True}
        except json.JSONDecodeError:
            pass

    device = torch.device(cuda_device(gpu_index)) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = torch.float16 if cfg["run"]["dtype"] == "float16" else torch.bfloat16
    torch.manual_seed(int(spec["sae_seed"]))
    np.random.seed(int(cfg["sae"].get("data_seed", 404)))

    base, base_ckpt = load_base(cfg, alias, model_seed, device)
    blocks_path = block_dir(cfg, alias)
    train_data = BlockData(blocks_path / "train.bin", cfg["data"]["train_blocks"], cfg["data"]["block_size"])
    valid_data = BlockData(blocks_path / "valid.bin", cfg["data"]["valid_blocks"], cfg["data"]["block_size"])
    d_in = hidden_size(base, cfg)
    d_sae = d_in * int(spec["expansion"])
    sae = TopKSAE(
        d_in,
        d_sae,
        int(spec["k"]),
        tied_init=cfg["sae"].get("tied_init", True),
        normalize_decoder=cfg["sae"].get("normalize_decoder", True),
    ).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=cfg["sae"].get("lr", 1e-3))
    max_steps = int(cfg["sae"].get("max_steps", 2000))
    log_interval = int(cfg["sae"].get("log_interval", cfg["train"]["log_interval"]))
    aux_weight = float(cfg["sae"].get("aux_dead_loss_weight", 0.0))
    aux_k = int(cfg["sae"].get("aux_k", 512))
    dead_steps_threshold = int(cfg["sae"].get("dead_steps_threshold", 200))
    last_fired = torch.zeros(d_sae, dtype=torch.long, device=device)
    start = now()
    last_log_seconds = 0
    metrics_path = out / "metrics" / f"[{alias}]mseed{model_seed}_{spec['sae_id']}_sae.jsonl"
    ensure_dir(metrics_path.parent)
    accumulation_steps = max(1, int(cfg["sae"].get("gradient_accumulation_steps", 1)))

    for step in range(1, max_steps + 1):
        dead_mask = (step - last_fired) >= dead_steps_threshold
        opt.zero_grad(set_to_none=True)
        accumulated_loss = torch.zeros((), device=device)
        accumulated_reconstruction = torch.zeros((), device=device)
        accumulated_aux = torch.zeros((), device=device)
        accumulated_active = torch.zeros((), device=device)
        fired = torch.zeros(d_sae, dtype=torch.bool, device=device)
        for micro_step in range(accumulation_steps):
            act = activation_batch(base, train_data, cfg, spec, device, dtype)
            if (
                step == 1
                and micro_step == 0
                and cfg["sae"].get("initialize_decoder_bias", True)
            ):
                with torch.no_grad():
                    sae.decoder.bias.copy_(act.mean(dim=0))
            recon, z, pre = sae(act, return_pre=True)
            reconstruction_loss = F.mse_loss(recon, act)
            aux_loss = (
                sae.dead_latent_aux_loss(act, recon, pre, dead_mask, aux_k)
                if aux_weight else act.new_tensor(0.0)
            )
            loss = reconstruction_loss + aux_weight * aux_loss
            (loss / accumulation_steps).backward()
            accumulated_loss += loss.detach()
            accumulated_reconstruction += reconstruction_loss.detach()
            accumulated_aux += aux_loss.detach()
            accumulated_active += (z > 0).float().sum(dim=-1).mean().detach()
            fired |= (z > 0).any(dim=0)
        last_fired[fired] = step
        loss = accumulated_loss / accumulation_steps
        reconstruction_loss = accumulated_reconstruction / accumulation_steps
        aux_loss = accumulated_aux / accumulation_steps
        active = (accumulated_active / accumulation_steps).item()
        if sae.normalize_decoder:
            sae.remove_decoder_gradient_parallel_component_()
        opt.step()
        if sae.normalize_decoder:
            sae.normalize_decoder_()
        if step % log_interval == 0:
            total_seconds = elapsed_seconds(start)
            delta_seconds = total_seconds - last_log_seconds
            last_log_seconds = total_seconds
            train_line("sae", gpu_label(gpu_index), f"{alias} m{model_seed} {spec['sae_id']}", step, loss.item(), None, delta_seconds, total_seconds, mem_usage(), vram_usage(device))
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "model": alias,
                    "model_seed": model_seed,
                    **spec,
                    "step": step,
                    "sae_loss": round(loss.item(), 8),
                    "reconstruction_loss": round(reconstruction_loss.item(), 8),
                    "aux_dead_loss": round(aux_loss.item(), 8),
                    "dead_in_batch_window": int(dead_mask.sum().item()),
                    "active": round(active, 4),
                    "micro_batch_size": int(cfg["train"]["batch_size"]),
                    "gradient_accumulation_steps": accumulation_steps,
                    "tokens_per_step": int(cfg["train"]["batch_size"])
                    * accumulation_steps * int(cfg["data"]["block_size"]),
                    "time": total_seconds,
                    "time_delta": delta_seconds,
                }, ensure_ascii=False) + "\n")

    torch.save({
        "sae": sae.state_dict(),
        "cfg": cfg,
        "spec": spec,
        "d_sae": d_sae,
        "base_ckpt": str(base_ckpt),
    }, ckpt_path)
    write_json(config_path, current)
    summary, feature_stats = eval_sae_metrics(base, sae, valid_data, cfg, spec, device, dtype)
    summary.update({"model": alias, "model_seed": model_seed, **spec, "time": elapsed_seconds(start)})
    write_json(target / "config.json", current)
    write_json(target / "metrics.json", summary)
    torch.save(feature_stats, target / "feature_stats.pt")
    write_json(sae_summary_path(cfg, alias, model_seed, spec), summary)
    return summary


def _frontier_summary(cfg, rows):
    target = float(cfg.get("eval", {}).get("loss_recovered_target", 0.95))
    grouped = {}
    for row in rows:
        key = (row.get("model"), row.get("model_seed"), row.get("layer"), row.get("expansion"), row.get("sae_seed"))
        grouped.setdefault(key, []).append(row)
    output = []
    for key, values in grouped.items():
        values = sorted(values, key=lambda row: row.get("k", 10 ** 9))
        eligible = [row for row in values if row.get("loss_recovered", -float("inf")) >= target]
        best = min(eligible, key=lambda row: row["actual_l0"]) if eligible else None
        output.append({
            "model": key[0],
            "model_seed": key[1],
            "layer": key[2],
            "expansion": key[3],
            "sae_seed": key[4],
            "loss_recovered_target": target,
            "k_at_95_loss_recovered": None if best is None else best["k"],
            "actual_l0_at_target": None if best is None else best["actual_l0"],
            "normalized_mse_at_target": None if best is None else best["normalized_mse"],
        })
    return output


def run(cfg):
    stage_title("sae")
    aliases = active_aliases(cfg)
    validate_aliases(cfg, aliases)
    model_seeds = run_seeds(cfg)
    specs = sae_specs(cfg)
    jobs = [(alias, model_seed, spec) for model_seed in model_seeds for alias in aliases for spec in specs]
    gpus = configured_gpus(cfg)
    failed = run_gpu_jobs(
        cfg,
        jobs,
        train_sae_one,
        lambda job, gpu_index: (cfg, job[0], job[1], job[2], gpu_index),
        gpus,
        stage="sae",
    )
    if failed:
        raise RuntimeError(f"sae subprocess failed: {failed}")
    rows = []
    for alias, model_seed, spec in jobs:
        path = sae_summary_path(cfg, alias, model_seed, spec)
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))

    frontiers = _frontier_summary(cfg, rows)
    write_json(output_dir(cfg) / "metrics" / "sae_summary.json", {"models": rows, "frontiers": frontiers})
    plotted_rows = [row for row in rows if not row.get("skipped")]
    for metric in [
        "normalized_mse",
        "explained_variance",
        "reconstruction_cosine",
        "loss_recovered",
        "kl_recovered",
        "actual_l0",
        "fraction_alive",
        "rare_feature_rate",
        "feature_usage_entropy",
        "activation_entropy",
        "decoder_duplication_proxy",
        "feature_ablation_loss_delta_mean",
    ]:
        plot_metric_by_k(output_dir(cfg), plotted_rows, metric, f"sae_{metric}.png")
    save_config(cfg)
