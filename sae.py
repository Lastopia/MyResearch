import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import ExperimentLogger
from metrics import compute_dead_feature_rate, compute_l0
from para import PATH
from utils import (
    ensure_dir,
    get_device,
    load_json,
    manifest_is_current,
    mean_std,
    save_json,
    save_manifest,
    set_seed,
    valid_torch_checkpoint,
    write_csv,
)
from visualize import plot_metric_curves, plot_sae_health_curves, plot_sae_training_curves


class TopKSAE(nn.Module):
    def __init__(self, input_dim, dict_size, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)

    def encode(self, x):
        acts = F.relu(self.encoder(x))
        values, indices = torch.topk(acts, k=min(self.k, acts.size(-1)), dim=-1)
        sparse = torch.zeros_like(acts)
        sparse.scatter_(-1, indices, values)
        return sparse

    def forward(self, x):
        acts = self.encode(x)
        recon = self.decoder(acts)
        return recon, acts


def load_sae_item(sae_item, device):
    """Load an SAE checkpoint only when a downstream stage actually needs it."""
    if "sae" in sae_item and "normalization" in sae_item:
        sae_item["sae"] = sae_item["sae"].to(device)
        return sae_item

    ckpt_path = sae_item.get("checkpoint_path")
    if not ckpt_path or not valid_torch_checkpoint(ckpt_path):
        raise FileNotFoundError(f"Missing or invalid SAE checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = sae_item["meta"]
    normalization = ckpt["normalization"]
    sae = TopKSAE(
        normalization["mean"].size(-1),
        meta["dict_size"],
        meta["k"],
    ).to(device)
    sae.load_state_dict(ckpt["sae"])
    return {
        **sae_item,
        "sae": sae,
        "normalization": normalization,
        "metrics": sae_item.get("metrics", ckpt.get("metrics", {})),
    }


class SelfSAE:
    def __init__(self, sae_cfg, train_res, data_res):
        self.sae_cfg = sae_cfg
        self.train_res = train_res
        self.data_res = data_res
        self.device = get_device(getattr(sae_cfg, "device", "cuda"))
        self.logger = ExperimentLogger()

    def loader(self, split):
        return torch.utils.data.DataLoader(
            self.data_res[split],
            batch_size=getattr(self.sae_cfg, "activation_batch_size", 1),
            shuffle=False,
            drop_last=False,
        )

    def stage_config(self):
        train_summary = {}
        for model_name, seeds in self.train_res.items():
            train_summary[model_name] = {}
            for seed, item in seeds.items():
                state = item["train_state"]
                checkpoint_selection = state.get("checkpoint_selection", {})
                train_summary[model_name][str(seed)] = self.selected_model_checkpoint(checkpoint_selection)
        return {
            "sae": self.sae_cfg,
            "train_checkpoints": train_summary,
            "data_meta": self.data_res.get("meta", {}),
        }

    def selected_model_checkpoint(self, checkpoint_selection):
        return (
            checkpoint_selection.get("best_validation")
            or checkpoint_selection.get("primary")
        )

    def load_model_checkpoint(self, model, checkpoint):
        if checkpoint is None:
            raise FileNotFoundError("Missing selected model checkpoint metadata")
        ckpt_path = checkpoint.get("checkpoint_path")
        if not ckpt_path:
            raise FileNotFoundError("Missing selected model checkpoint path")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(ckpt["model"])

    def stage_outputs(self):
        return [
            Path(PATH.raw_metrics_dir) / "sae_res.json",
            Path(PATH.raw_metrics_dir) / "phase4a_summary.json",
        ]

    def stage_manifest_path(self):
        return Path(PATH.raw_metrics_dir) / "sae_manifest.json"

    def load_completed_sae_res(self):
        self.logger.log_stage_start("load completed SAE results")
        serializable_path = Path(PATH.raw_metrics_dir) / "sae_res.json"
        if not serializable_path.exists():
            return None
        serializable = load_json(serializable_path)
        sae_res = {}
        for model_name, model_item in serializable.items():
            sae_res[model_name] = {}
            for model_seed_text, seed_item in model_item.items():
                model_seed = int(model_seed_text)
                sae_res[model_name][model_seed] = {}
                for layer_text, items in seed_item.items():
                    layer = int(layer_text)
                    layer_res = []
                    for item in items:
                        ckpt_path = item.get("checkpoint_path")
                        if not ckpt_path or not valid_torch_checkpoint(ckpt_path):
                            return None
                        meta = item["meta"]
                        layer_res.append(
                            {
                                "meta": meta,
                                "metrics": item["metrics"],
                                "normalization_summary": item["normalization_summary"],
                                "checkpoint_path": ckpt_path,
                            }
                        )
                    sae_res[model_name][model_seed][layer] = layer_res
        self.logger.log_stage_end("load completed SAE results metadata only")
        return sae_res

    @torch.no_grad()
    def collect_activations(self, model, layer, split, max_tokens, model_name=None, model_seed=None):
        model.eval()
        acts = []
        seen = 0
        label = f"{model_name or 'model'} seed {model_seed} layer {layer} {split}"
        self.logger.log_stage_start(f"collect SAE activations {label} target_tokens={max_tokens}")
        for batch_idx, (x, _) in enumerate(self.loader(split), start=1):
            x = x.to(self.device)
            out = model(x, capture_layers=[layer])
            if layer not in out["activations"]:
                raise ValueError(f"Layer {layer} was not captured; model has fewer layers than requested")
            flat = out["activations"][layer].reshape(-1, out["activations"][layer].size(-1)).float().cpu()
            take = min(flat.size(0), max_tokens - seen)
            if take > 0:
                acts.append(flat[:take])
                seen += take
            if batch_idx == 1 or seen >= max_tokens or batch_idx % 10 == 0:
                self.logger.write(
                    f"[progress] collect SAE activations {label}: "
                    f"tokens={seen}/{max_tokens} batches={batch_idx}"
                )
            if seen >= max_tokens:
                break
        if not acts:
            raise RuntimeError(f"No activations collected for split={split}, layer={layer}")
        self.logger.log_stage_end(f"collect SAE activations {label} tokens={seen}")
        return torch.cat(acts, dim=0)

    def activation_stats(self, activations):
        mean = activations.mean(dim=0, keepdim=True)
        std = activations.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        return {"mean": mean, "std": std}

    def normalize(self, activations, stats):
        if not getattr(self.sae_cfg, "normalize_activations", True):
            return activations
        return (activations - stats["mean"]) / stats["std"]

    def explained_variance(self, x, recon):
        total_var = x.var(dim=0, unbiased=False).sum().clamp_min(1e-9)
        residual_var = (x - recon).var(dim=0, unbiased=False).sum()
        return 1.0 - (residual_var / total_var).item()

    def feature_health(self, feature_acts):
        active = feature_acts != 0
        frequencies = active.float().mean(dim=0)
        active_features = frequencies > getattr(self.sae_cfg, "dead_feature_threshold", 0.0)
        reuse_threshold = getattr(self.sae_cfg, "feature_reuse_frequency_threshold", 0.001)
        reused_features = frequencies >= reuse_threshold
        probs = frequencies / frequencies.sum().clamp_min(1e-9)
        entropy = -(probs * probs.clamp_min(1e-9).log()).sum().item()
        max_entropy = math.log(max(feature_acts.size(-1), 2))
        q = torch.tensor([0.5, 0.9, 0.99], device=frequencies.device)
        quantiles = torch.quantile(frequencies.float(), q)
        return {
            "l0": compute_l0(feature_acts).item(),
            "average_active_features_per_token": active.float().sum(dim=-1).mean().item(),
            "dead_feature_rate": compute_dead_feature_rate(feature_acts).item(),
            "active_feature_rate": active_features.float().mean().item(),
            "feature_reuse_rate": reused_features.float().mean().item(),
            "top_feature_activation_frequency": frequencies.max().item(),
            "feature_frequency_entropy": entropy,
            "feature_frequency_entropy_normalized": entropy / max_entropy,
            "feature_density_distribution": {
                "mean": frequencies.mean().item(),
                "std": frequencies.std(unbiased=False).item(),
                "p50": quantiles[0].item(),
                "p90": quantiles[1].item(),
                "p99": quantiles[2].item(),
                "max": frequencies.max().item(),
            },
        }

    @torch.no_grad()
    def evaluate_sae(self, sae, data):
        sae.eval()
        recon, feature_acts = sae(data)
        mse = F.mse_loss(recon, data).item()
        health = self.feature_health(feature_acts)
        return {
            "validation_mse": mse,
            "reconstruction_loss": mse,
            "normalized_reconstruction_mse": mse,
            "explained_variance": self.explained_variance(data, recon),
            **health,
        }

    def train_one_sae(
        self,
        train_acts,
        valid_acts,
        model_name,
        model_seed,
        layer,
        dict_size,
        k,
        sae_seed,
        stats,
        model_checkpoint_selection,
    ):
        set_seed(sae_seed)
        sae = TopKSAE(train_acts.size(-1), dict_size, k).to(self.device)
        opt = torch.optim.Adam(sae.parameters(), lr=self.sae_cfg.lr)
        train_data = train_acts.to(self.device)
        valid_data = valid_acts.to(self.device)
        history = []
        start_time = time.perf_counter()
        run_name = (
            f"SAE {model_name} modelseed {model_seed} layer {layer} "
            f"dict {dict_size} k {k} saeseed {sae_seed}"
        )
        self.logger.log_stage_start(
            f"{run_name} train_tokens={train_data.size(0)} valid_tokens={valid_data.size(0)} "
            f"steps={self.sae_cfg.steps} batch={self.sae_cfg.batch_size}"
        )

        for step in range(1, self.sae_cfg.steps + 1):
            sae.train()
            idx = torch.randint(0, train_data.size(0), (self.sae_cfg.batch_size,), device=self.device)
            batch = train_data[idx]
            recon, _ = sae(batch)
            loss = F.mse_loss(recon, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            should_log = step == 1 or step == self.sae_cfg.steps or step % self.sae_cfg.eval_interval == 0
            should_progress = step % max(1, getattr(self.sae_cfg, "log_interval", 100)) == 0
            if should_progress and not should_log:
                elapsed = time.perf_counter() - start_time
                self.logger.write(
                    f"[progress] {run_name} step={step}/{self.sae_cfg.steps} "
                    f"loss={loss.item():.6f} elapsed={elapsed:.1f}s"
                )
            if should_log:
                sample = train_data[: min(train_data.size(0), valid_data.size(0), 8192)]
                train_eval = self.evaluate_sae(sae, sample)
                valid_eval = self.evaluate_sae(sae, valid_data)
                row = {
                    "step": step,
                    "train_mse": train_eval["validation_mse"],
                    "valid_mse": valid_eval["validation_mse"],
                    "explained_variance": valid_eval["explained_variance"],
                    "dead_feature_rate": valid_eval["dead_feature_rate"],
                }
                history.append(row)
                elapsed = time.perf_counter() - start_time
                self.logger.write(
                    f"[metric step={step}] {run_name}: "
                    f"loss={loss.item():.6f} train_mse={row['train_mse']:.6f} "
                    f"valid_mse={row['valid_mse']:.6f} ev={row['explained_variance']:.4f} "
                    f"dead={row['dead_feature_rate']:.4f} elapsed={elapsed:.1f}s"
                )

        elapsed = time.perf_counter() - start_time
        metrics = self.evaluate_sae(sae, valid_data)
        metrics["training_seconds"] = elapsed
        metrics["seconds_per_step"] = elapsed / max(self.sae_cfg.steps, 1)
        metrics["tokens_per_second"] = (
            self.sae_cfg.steps * self.sae_cfg.batch_size / elapsed if elapsed > 0 else None
        )
        metrics["final_train_mse"] = history[-1]["train_mse"] if history else None
        metrics["history"] = history

        ensure_dir(Path(PATH.ckpt_dir) / "saes")
        ckpt = Path(PATH.ckpt_dir) / "saes" / (
            f"{model_name}_modelseed{model_seed}_"
            f"saeseed{sae_seed}_layer{layer}_dict{dict_size}_k{k}.pt"
        )
        self.logger.write(f"[save] {run_name} checkpoint -> {ckpt}")
        tmp_ckpt = ckpt.with_suffix(ckpt.suffix + ".tmp")
        torch.save(
            {
                "sae": sae.state_dict(),
                "metrics": metrics,
                "normalization": {
                    "mean": stats["mean"].cpu(),
                    "std": stats["std"].cpu(),
                    "enabled": getattr(self.sae_cfg, "normalize_activations", True),
                },
            },
            tmp_ckpt,
        )
        os.replace(tmp_ckpt, ckpt)
        self.logger.log_stage_end(
            f"{run_name} valid_mse={metrics['validation_mse']:.6f} "
            f"ev={metrics['explained_variance']:.4f} dead={metrics['dead_feature_rate']:.4f}"
        )

        if getattr(self.sae_cfg, "save_individual_sae_curves", False):
            figure_prefix = (
                f"{model_name}_modelseed{model_seed}_"
                f"saeseed{sae_seed}_layer{layer}_dict{dict_size}_k{k}"
            )
            plot_sae_training_curves(
                history,
                Path(PATH.figure_dir) / f"{figure_prefix}_sae_mse.png",
                title=f"{model_name} seed {model_seed} L{layer} SAE MSE",
            )
            plot_sae_health_curves(
                history,
                Path(PATH.figure_dir) / f"{figure_prefix}_sae_health.png",
                title=f"{model_name} seed {model_seed} L{layer} SAE health",
            )

        return {
            "sae": sae,
            "meta": {
                "model_name": model_name,
                "model_seed": model_seed,
                "sae_seed": sae_seed,
                "layer": layer,
                "dict_size": dict_size,
                "k": k,
                "activation_site": self.sae_cfg.activation_site,
                "activation_normalization": getattr(self.sae_cfg, "normalize_activations", True),
                "model_checkpoint_rule": model_checkpoint_selection["selection_rule"],
                "model_checkpoint_step": model_checkpoint_selection["checkpoint_step"],
                "model_checkpoint_path": model_checkpoint_selection["checkpoint_path"],
                "model_tokens_seen": model_checkpoint_selection["tokens_seen"],
                "train_activation_tokens": train_acts.size(0),
                "valid_activation_tokens": valid_acts.size(0),
            },
            "metrics": metrics,
            "normalization_summary": {
                "mean_abs_mean": stats["mean"].abs().mean().item(),
                "std_mean": stats["std"].mean().item(),
                "std_min": stats["std"].min().item(),
                "std_max": stats["std"].max().item(),
            },
            "normalization": {
                "mean": stats["mean"],
                "std": stats["std"],
                "enabled": getattr(self.sae_cfg, "normalize_activations", True),
            },
            "checkpoint_path": str(ckpt),
        }

    def flatten_rows(self, serializable):
        rows = []
        for model_name, model_item in serializable.items():
            for model_seed, seed_item in model_item.items():
                for layer, layer_items in seed_item.items():
                    for item in layer_items:
                        metrics = item["metrics"]
                        rows.append(
                            {
                                "model_name": model_name,
                                "model_seed": model_seed,
                                "layer": layer,
                                "sae_seed": item["meta"]["sae_seed"],
                                "dict_size": item["meta"]["dict_size"],
                                "k": item["meta"]["k"],
                                "model_checkpoint_rule": item["meta"]["model_checkpoint_rule"],
                                "model_checkpoint_step": item["meta"]["model_checkpoint_step"],
                                "model_checkpoint_path": item["meta"]["model_checkpoint_path"],
                                "model_tokens_seen": item["meta"]["model_tokens_seen"],
                                "validation_mse": metrics["validation_mse"],
                                "explained_variance": metrics["explained_variance"],
                                "reconstruction_loss": metrics["reconstruction_loss"],
                                "normalized_reconstruction_mse": metrics["normalized_reconstruction_mse"],
                                "l0": metrics["l0"],
                                "average_active_features_per_token": metrics["average_active_features_per_token"],
                                "dead_feature_rate": metrics["dead_feature_rate"],
                                "active_feature_rate": metrics["active_feature_rate"],
                                "feature_reuse_rate": metrics["feature_reuse_rate"],
                                "top_feature_activation_frequency": metrics["top_feature_activation_frequency"],
                                "feature_frequency_entropy_normalized": metrics[
                                    "feature_frequency_entropy_normalized"
                                ],
                                "seconds_per_step": metrics["seconds_per_step"],
                                "tokens_per_second": metrics["tokens_per_second"],
                                "checkpoint_path": item["checkpoint_path"],
                            }
                        )
        return rows

    def summarize_rows(self, rows):
        metrics = [
            "validation_mse",
            "explained_variance",
            "reconstruction_loss",
            "normalized_reconstruction_mse",
            "l0",
            "average_active_features_per_token",
            "dead_feature_rate",
            "active_feature_rate",
            "feature_reuse_rate",
            "top_feature_activation_frequency",
            "feature_frequency_entropy_normalized",
        ]
        grouped = {}
        for row in rows:
            key = (row["model_name"], row["layer"])
            grouped.setdefault(key, {metric: [] for metric in metrics})
            for metric in metrics:
                grouped[key][metric].append(row[metric])
        summary_rows = []
        for (model_name, layer), values in grouped.items():
            row = {"model_name": model_name, "layer": layer}
            for metric, metric_values in values.items():
                stats = mean_std(metric_values)
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"] = stats["std"]
            summary_rows.append(row)
        return summary_rows

    def existing_sae_checkpoint_path(self, model_name, model_seed, layer, dict_size, k, sae_seed):
        return (
            Path(PATH.ckpt_dir)
            / "saes"
            / f"{model_name}_modelseed{model_seed}_saeseed{sae_seed}_layer{layer}_dict{dict_size}_k{k}.pt"
        )

    def load_existing_sae_item(
        self,
        model_name,
        model_seed,
        layer,
        dict_size,
        k,
        sae_seed,
        selected_checkpoint,
    ):
        ckpt_path = self.existing_sae_checkpoint_path(model_name, model_seed, layer, dict_size, k, sae_seed)
        if not valid_torch_checkpoint(ckpt_path):
            return None
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "metrics" not in ckpt or "normalization" not in ckpt:
            return None
        normalization = ckpt["normalization"]
        return {
            "meta": {
                "model_name": model_name,
                "model_seed": model_seed,
                "sae_seed": sae_seed,
                "layer": layer,
                "dict_size": dict_size,
                "k": k,
                "activation_site": getattr(self.sae_cfg, "activation_site", "residual_post_block"),
                "activation_normalization": getattr(self.sae_cfg, "normalize_activations", True),
                "model_checkpoint_rule": selected_checkpoint["selection_rule"],
                "model_checkpoint_step": selected_checkpoint["checkpoint_step"],
                "model_checkpoint_path": selected_checkpoint["checkpoint_path"],
                "model_tokens_seen": selected_checkpoint["tokens_seen"],
                "train_activation_tokens": getattr(self.sae_cfg, "max_activation_tokens", None),
                "valid_activation_tokens": getattr(self.sae_cfg, "max_validation_activation_tokens", None),
            },
            "metrics": ckpt["metrics"],
            "normalization_summary": {
                "mean_abs_mean": normalization["mean"].abs().mean().item(),
                "std_mean": normalization["std"].mean().item(),
                "std_min": normalization["std"].min().item(),
                "std_max": normalization["std"].max().item(),
            },
            "checkpoint_path": str(ckpt_path),
        }

    def plot_aggregate_curves(self, serializable):
        if not getattr(self.sae_cfg, "run_aggregate_sae_curves", True):
            return
        histories = {}
        for model_name, seed_items in serializable.items():
            for model_seed, layer_items in seed_items.items():
                for layer, items in layer_items.items():
                    for item in items:
                        meta = item["meta"]
                        history = item.get("metrics", {}).get("history", [])
                        if not history:
                            continue
                        label = (
                            f"{model_name} mseed {model_seed} L{layer} "
                            f"sae {meta['sae_seed']} dict {meta['dict_size']} k {meta['k']}"
                        )
                        histories[label] = history
        metric_specs = [
            ("train_mse", "SAE train MSE"),
            ("valid_mse", "SAE validation MSE"),
            ("explained_variance", "SAE explained variance"),
            ("dead_feature_rate", "SAE dead feature rate"),
        ]
        for metric_key, title in metric_specs:
            plot_metric_curves(
                histories,
                metric_key,
                Path(PATH.figure_dir) / f"phase4a_{metric_key}_all_models_layers_seeds.png",
                ylabel=metric_key,
                title=title,
            )

    def run(self):
        self.logger.log_stage_start(
            f"SAE stage models={list(self.train_res.keys())} layers={list(self.sae_cfg.layers)} "
            f"dict_sizes={list(self.sae_cfg.dictionary_sizes)} k={list(self.sae_cfg.topk_values)} "
            f"sae_seeds={list(self.sae_cfg.seeds)}"
        )
        if (
            getattr(self.sae_cfg, "skip_completed_stage", True)
            and manifest_is_current(self.stage_manifest_path(), self.stage_config(), self.stage_outputs())
        ):
            loaded = self.load_completed_sae_res()
            if loaded is not None:
                self.logger.log_stage_end("skip SAE stage: existing outputs match config")
                return loaded

        sae_res = {}
        serializable = {}
        for model_name, seeds in self.train_res.items():
            sae_res[model_name] = {}
            serializable[model_name] = {}
            for model_seed, train_item in seeds.items():
                model = train_item["model"].to(self.device)
                model.eval()
                sae_res[model_name][model_seed] = {}
                serializable[model_name][str(model_seed)] = {}
                checkpoint_selection = train_item["train_state"]["checkpoint_selection"]
                selected_checkpoint = self.selected_model_checkpoint(checkpoint_selection)
                self.load_model_checkpoint(model, selected_checkpoint)
                self.logger.log_stage_start(f"SAE model {model_name} seed {model_seed}")
                for layer in self.sae_cfg.layers:
                    existing_layer_res = []
                    missing_existing = False
                    for dict_size in self.sae_cfg.dictionary_sizes:
                        for k in self.sae_cfg.topk_values:
                            for sae_seed in self.sae_cfg.seeds:
                                item = self.load_existing_sae_item(
                                    model_name,
                                    model_seed,
                                    layer,
                                    dict_size,
                                    k,
                                    sae_seed,
                                    selected_checkpoint,
                                )
                                if item is None:
                                    missing_existing = True
                                else:
                                    existing_layer_res.append(item)
                    if not missing_existing and existing_layer_res:
                        self.logger.log_stage_end(
                            f"SAE reuse existing checkpoints {model_name} seed {model_seed} layer {layer}"
                        )
                        sae_res[model_name][model_seed][layer] = existing_layer_res
                        serializable[model_name][str(model_seed)][str(layer)] = [
                            {
                                "meta": item["meta"],
                                "metrics": item["metrics"],
                                "normalization_summary": item["normalization_summary"],
                                "checkpoint_path": item["checkpoint_path"],
                            }
                            for item in existing_layer_res
                        ]
                        continue
                    self.logger.log_stage_start(f"SAE prepare activations {model_name} seed {model_seed} layer {layer}")
                    train_acts_raw = self.collect_activations(
                        model,
                        layer,
                        "train",
                        self.sae_cfg.max_activation_tokens,
                        model_name=model_name,
                        model_seed=model_seed,
                    )
                    valid_acts_raw = self.collect_activations(
                        model,
                        layer,
                        "valid",
                        getattr(self.sae_cfg, "max_validation_activation_tokens", 16384),
                        model_name=model_name,
                        model_seed=model_seed,
                    )
                    stats = self.activation_stats(train_acts_raw)
                    train_acts = self.normalize(train_acts_raw, stats)
                    valid_acts = self.normalize(valid_acts_raw, stats)
                    self.logger.log_stage_end(
                        f"SAE prepare activations {model_name} seed {model_seed} layer {layer} "
                        f"train_tokens={train_acts.size(0)} valid_tokens={valid_acts.size(0)}"
                    )
                    layer_res = []
                    for dict_size in self.sae_cfg.dictionary_sizes:
                        for k in self.sae_cfg.topk_values:
                            for sae_seed in self.sae_cfg.seeds:
                                layer_res.append(
                                    self.train_one_sae(
                                        train_acts,
                                        valid_acts,
                                        model_name,
                                        model_seed,
                                        layer,
                                        dict_size,
                                        k,
                                        sae_seed,
                                        stats,
                                        selected_checkpoint,
                                    )
                                )
                    sae_res[model_name][model_seed][layer] = layer_res
                    serializable[model_name][str(model_seed)][str(layer)] = [
                        {
                            "meta": item["meta"],
                            "metrics": item["metrics"],
                            "normalization_summary": item["normalization_summary"],
                            "checkpoint_path": item["checkpoint_path"],
                        }
                        for item in layer_res
                    ]
                self.logger.log_stage_end(f"SAE model {model_name} seed {model_seed}")

        save_json(serializable, Path(PATH.raw_metrics_dir) / "sae_res.json")
        rows = self.flatten_rows(serializable)
        summary_rows = self.summarize_rows(rows)
        self.plot_aggregate_curves(serializable)
        write_csv(rows, Path(PATH.table_dir) / "phase4a_sae_runs.csv")
        write_csv(
            summary_rows,
            Path(PATH.table_dir) / "phase4a_sae_summary.csv",
        )
        save_json(
            {
                "phase": "4a",
                "design": {
                    "dictionary_size": self.sae_cfg.dictionary_sizes,
                    "k": self.sae_cfg.topk_values,
                    "layers": self.sae_cfg.layers,
                    "sae_seeds": self.sae_cfg.seeds,
                    "activation_site": self.sae_cfg.activation_site,
                    "normalize_activations": getattr(self.sae_cfg, "normalize_activations", True),
                    "dictionary_size_sensitivity": "deferred_to_phase4b",
                    "sparsity_sensitivity": "deferred_to_phase4c",
                },
                "summary_rows": summary_rows,
                "run_rows": rows,
            },
            Path(PATH.raw_metrics_dir) / "phase4a_summary.json",
        )
        save_manifest(self.stage_manifest_path(), "sae", self.stage_config(), self.stage_outputs())
        self.logger.log_stage_end("SAE stage")
        return sae_res
