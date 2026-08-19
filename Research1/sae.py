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
from visualize import (
    plot_metric_curves,
    plot_phase4a_summary_figures,
    plot_phase4b_stability_figures,
    plot_sae_health_curves,
    plot_sae_training_curves,
)


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
        outputs = [
            Path(PATH.raw_metrics_dir) / "sae_res.json",
            Path(PATH.raw_metrics_dir) / "phase4a_summary.json",
            Path(PATH.raw_metrics_dir) / "phase4a_sae_sweep_conclusions.json",
            Path(PATH.table_dir) / "phase4a_sae_sweep_summary.csv",
            Path(PATH.table_dir) / "phase4a_sae_sweep_conclusions.csv",
        ]
        if getattr(self.sae_cfg, "run_feature_stability", True):
            outputs.extend(
                [
                    Path(PATH.raw_metrics_dir) / "phase4b_feature_stability.json",
                    Path(PATH.table_dir) / "phase4b_feature_stability_pairs.csv",
                    Path(PATH.table_dir) / "phase4b_feature_stability_summary.csv",
                ]
            )
        return outputs

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

    def summarize_sweep_rows(self, rows):
        metrics = [
            "validation_mse",
            "explained_variance",
            "dead_feature_rate",
            "feature_reuse_rate",
            "top_feature_activation_frequency",
            "feature_frequency_entropy_normalized",
        ]
        grouped = {}
        for row in rows:
            key = (row["model_name"], row["layer"], row["dict_size"], row["k"])
            grouped.setdefault(key, {metric: [] for metric in metrics})
            for metric in metrics:
                value = row.get(metric)
                if value is not None and math.isfinite(value):
                    grouped[key][metric].append(value)
        summary_rows = []
        for (model_name, layer, dict_size, k), values in grouped.items():
            row = {"model_name": model_name, "layer": layer, "dict_size": dict_size, "k": k}
            for metric, metric_values in values.items():
                stats = mean_std(metric_values)
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"] = stats["std"]
            summary_rows.append(row)
        return summary_rows

    def sweep_conclusion_rows(self, sweep_summary_rows):
        primary_dict = int(getattr(self.sae_cfg, "primary_dictionary_size", self.sae_cfg.dictionary_sizes[0]))
        primary_k = int(getattr(self.sae_cfg, "primary_topk", self.sae_cfg.topk_values[0]))

        def ev(row):
            value = row.get("explained_variance_mean")
            return value if value is not None and math.isfinite(value) else None

        by_group = {}
        for row in sweep_summary_rows:
            key = (row["model_name"], int(row["layer"]))
            by_group.setdefault(key, []).append(row)

        rows = []
        for (model_name, layer), items in by_group.items():
            primary = next(
                (
                    item
                    for item in items
                    if int(item["dict_size"]) == primary_dict and int(item["k"]) == primary_k
                ),
                None,
            )
            dict_items = [item for item in items if int(item["k"]) == primary_k and ev(item) is not None]
            topk_items = [item for item in items if int(item["dict_size"]) == primary_dict and ev(item) is not None]
            primary_ev = ev(primary) if primary else None

            def sensitivity(axis_items, axis_name):
                if not axis_items:
                    return {}
                best = max(axis_items, key=lambda item: ev(item))
                worst = min(axis_items, key=lambda item: ev(item))
                return {
                    f"best_{axis_name}": best[axis_name],
                    f"best_{axis_name}_explained_variance": ev(best),
                    f"worst_{axis_name}": worst[axis_name],
                    f"worst_{axis_name}_explained_variance": ev(worst),
                    f"{axis_name}_explained_variance_range": ev(best) - ev(worst),
                    f"{axis_name}_gain_over_primary": (
                        ev(best) - primary_ev if primary_ev is not None else None
                    ),
                }

            row = {
                "model_name": model_name,
                "layer": layer,
                "primary_dict_size": primary_dict,
                "primary_k": primary_k,
                "primary_explained_variance": primary_ev,
                "primary_validation_mse": primary.get("validation_mse_mean") if primary else None,
                **sensitivity(dict_items, "dict_size"),
                **sensitivity(topk_items, "k"),
            }
            rows.append(row)

        aggregate = {}
        for row in rows:
            model_name = row["model_name"]
            aggregate.setdefault(
                model_name,
                {
                    "primary_explained_variance": [],
                    "dict_size_explained_variance_range": [],
                    "dict_size_gain_over_primary": [],
                    "k_explained_variance_range": [],
                    "k_gain_over_primary": [],
                },
            )
            for metric in aggregate[model_name]:
                value = row.get(metric)
                if value is not None and math.isfinite(value):
                    aggregate[model_name][metric].append(value)

        aggregate_rows = []
        for model_name, values in aggregate.items():
            row = {"model_name": model_name, "layer": "all"}
            for metric, metric_values in values.items():
                stats = mean_std(metric_values)
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"] = stats["std"]
            aggregate_rows.append(row)

        ranked = sorted(
            aggregate_rows,
            key=lambda item: (
                item.get("primary_explained_variance_mean")
                if item.get("primary_explained_variance_mean") is not None
                else float("-inf")
            ),
            reverse=True,
        )
        for rank, row in enumerate(ranked, start=1):
            row["primary_explained_variance_rank"] = rank

        return rows, ranked

    def sweep_conclusion_notes(self, aggregate_rows):
        if not aggregate_rows:
            return []
        notes = []
        by_primary = sorted(
            aggregate_rows,
            key=lambda row: row.get("primary_explained_variance_mean") or float("-inf"),
            reverse=True,
        )
        notes.append(
            "Primary SAE explained-variance ranking: "
            + " > ".join(row["model_name"] for row in by_primary)
        )
        by_dict_sensitivity = sorted(
            aggregate_rows,
            key=lambda row: row.get("dict_size_explained_variance_range_mean") or float("-inf"),
            reverse=True,
        )
        notes.append(
            "Dictionary-size sensitivity ranking: "
            + " > ".join(row["model_name"] for row in by_dict_sensitivity)
        )
        by_k_sensitivity = sorted(
            aggregate_rows,
            key=lambda row: row.get("k_explained_variance_range_mean") or float("-inf"),
            reverse=True,
        )
        notes.append(
            "Top-k sensitivity ranking: "
            + " > ".join(row["model_name"] for row in by_k_sensitivity)
        )
        return notes

    def configured_sae_runs(self):
        explicit = getattr(self.sae_cfg, "sweep_configs", None)
        runs = []
        if explicit:
            for item in explicit:
                runs.append(
                    {
                        "dict_size": int(item["dict_size"]),
                        "k": int(item["k"]),
                        "seeds": [int(seed) for seed in item.get("seeds", self.sae_cfg.seeds)],
                    }
                )
        else:
            for dict_size in self.sae_cfg.dictionary_sizes:
                for k in self.sae_cfg.topk_values:
                    runs.append(
                        {
                            "dict_size": int(dict_size),
                            "k": int(k),
                            "seeds": [int(seed) for seed in self.sae_cfg.seeds],
                        }
                    )

        seen = set()
        deduped = []
        for run in runs:
            for seed in run["seeds"]:
                key = (run["dict_size"], run["k"], seed)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append({"dict_size": run["dict_size"], "k": run["k"], "sae_seed": seed})
        return deduped

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

    def plot_summary_figures(self):
        if not getattr(self.sae_cfg, "run_aggregate_sae_curves", True):
            return []
        paths = plot_phase4a_summary_figures(PATH.raw_metrics_dir, PATH.figure_dir)
        if paths:
            self.logger.log_stage_end(f"Phase 4a summary figures: wrote {len(paths)} overview files")
        return paths

    def decoder_columns(self, sae):
        weight = sae.decoder.weight.detach().float().cpu()
        return F.normalize(weight.T, dim=1)

    def mutual_nearest_matches(self, sim):
        a_to_b = sim.argmax(dim=1)
        b_to_a = sim.argmax(dim=0)
        a_idx = torch.arange(sim.size(0))
        keep = b_to_a[a_to_b] == a_idx
        matched_a = a_idx[keep]
        matched_b = a_to_b[keep]
        scores = sim[matched_a, matched_b]
        return matched_a, matched_b, scores

    def matched_activation_correlations(self, acts_a, acts_b, matched_a, matched_b):
        if matched_a.numel() == 0:
            return torch.empty(0)
        xa = acts_a[:, matched_a].float()
        xb = acts_b[:, matched_b].float()
        xa = xa - xa.mean(dim=0, keepdim=True)
        xb = xb - xb.mean(dim=0, keepdim=True)
        denom = xa.square().sum(dim=0).sqrt() * xb.square().sum(dim=0).sqrt()
        return (xa * xb).sum(dim=0) / denom.clamp_min(1e-9)

    @torch.no_grad()
    def encode_for_stability(self, item, raw_acts):
        loaded = load_sae_item(item, self.device)
        sae = loaded["sae"]
        stats = loaded["normalization"]
        data = self.normalize(raw_acts, stats).to(self.device)
        features = sae.encode(data).float().cpu()
        return loaded, features

    def summarize_stability_pair(self, base_row, decoder_scores, activation_corrs):
        thresholds = list(getattr(self.sae_cfg, "stability_similarity_thresholds", [0.5, 0.7, 0.9]))
        row = {
            **base_row,
            "matched_feature_count": int(decoder_scores.numel()),
            "matched_feature_fraction": (
                decoder_scores.numel() / max(int(base_row["dict_size"]), 1)
            ),
            "decoder_cosine_mean": decoder_scores.mean().item() if decoder_scores.numel() else None,
            "decoder_cosine_median": decoder_scores.median().item() if decoder_scores.numel() else None,
            "decoder_cosine_p90": torch.quantile(decoder_scores, 0.9).item() if decoder_scores.numel() else None,
            "activation_correlation_mean": activation_corrs.mean().item() if activation_corrs.numel() else None,
            "activation_correlation_median": activation_corrs.median().item() if activation_corrs.numel() else None,
            "activation_correlation_p90": torch.quantile(activation_corrs, 0.9).item()
            if activation_corrs.numel()
            else None,
        }
        for threshold in thresholds:
            key = str(threshold).replace(".", "p")
            row[f"decoder_cosine_ge_{key}_fraction"] = (
                (decoder_scores >= threshold).float().mean().item() if decoder_scores.numel() else None
            )
            row[f"activation_correlation_ge_{key}_fraction"] = (
                (activation_corrs >= threshold).float().mean().item() if activation_corrs.numel() else None
            )
        return row

    def summarize_stability_rows(self, rows):
        metrics = [
            "matched_feature_count",
            "matched_feature_fraction",
            "decoder_cosine_mean",
            "decoder_cosine_median",
            "decoder_cosine_p90",
            "activation_correlation_mean",
            "activation_correlation_median",
            "activation_correlation_p90",
        ]
        for threshold in getattr(self.sae_cfg, "stability_similarity_thresholds", [0.5, 0.7, 0.9]):
            key = str(threshold).replace(".", "p")
            metrics.append(f"decoder_cosine_ge_{key}_fraction")
            metrics.append(f"activation_correlation_ge_{key}_fraction")
        grouped = {}
        for row in rows:
            key = (row["model_name"], row["model_seed"], row["layer"], row["dict_size"], row["k"])
            grouped.setdefault(key, {metric: [] for metric in metrics})
            for metric in metrics:
                value = row.get(metric)
                if value is not None and math.isfinite(value):
                    grouped[key][metric].append(value)
        summary = []
        for (model_name, model_seed, layer, dict_size, k), values in grouped.items():
            row = {
                "model_name": model_name,
                "model_seed": model_seed,
                "layer": layer,
                "dict_size": dict_size,
                "k": k,
            }
            for metric, metric_values in values.items():
                stats = mean_std(metric_values)
                row[f"{metric}_mean"] = stats["mean"]
                row[f"{metric}_std"] = stats["std"]
            summary.append(row)
        return summary

    def run_feature_stability(self, sae_res):
        if not getattr(self.sae_cfg, "run_feature_stability", True):
            return None
        self.logger.log_stage_start("Phase 4b SAE feature stability")
        target_dict_sizes = {int(x) for x in getattr(self.sae_cfg, "stability_dict_sizes", [])}
        target_topks = {int(x) for x in getattr(self.sae_cfg, "stability_topk_values", [])}
        pair_rows = []
        nested = {}

        for model_name, seeds in self.train_res.items():
            nested[model_name] = {}
            for model_seed, train_item in seeds.items():
                nested[model_name][str(model_seed)] = {}
                model = train_item["model"].to(self.device)
                model.eval()
                checkpoint_selection = train_item["train_state"]["checkpoint_selection"]
                selected_checkpoint = self.selected_model_checkpoint(checkpoint_selection)
                self.load_model_checkpoint(model, selected_checkpoint)
                for layer in self.sae_cfg.layers:
                    items = sae_res.get(model_name, {}).get(model_seed, {}).get(layer, [])
                    grouped = {}
                    for item in items:
                        meta = item.get("meta", {})
                        dict_size = int(meta.get("dict_size"))
                        k = int(meta.get("k"))
                        if target_dict_sizes and dict_size not in target_dict_sizes:
                            continue
                        if target_topks and k not in target_topks:
                            continue
                        grouped.setdefault((dict_size, k), []).append(item)
                    if not grouped:
                        continue
                    raw_acts = self.collect_activations(
                        model,
                        layer,
                        "valid",
                        getattr(self.sae_cfg, "stability_max_tokens", 4096),
                        model_name=model_name,
                        model_seed=model_seed,
                    )
                    for (dict_size, k), group_items in grouped.items():
                        if len(group_items) < 2:
                            continue
                        loaded = []
                        for item in sorted(group_items, key=lambda x: x["meta"]["sae_seed"]):
                            loaded_item, features = self.encode_for_stability(item, raw_acts)
                            loaded.append(
                                {
                                    "item": loaded_item,
                                    "features": features,
                                    "decoder": self.decoder_columns(loaded_item["sae"]),
                                }
                            )
                        for left_idx in range(len(loaded)):
                            for right_idx in range(left_idx + 1, len(loaded)):
                                left = loaded[left_idx]
                                right = loaded[right_idx]
                                sim = left["decoder"] @ right["decoder"].T
                                matched_a, matched_b, decoder_scores = self.mutual_nearest_matches(sim)
                                activation_corrs = self.matched_activation_correlations(
                                    left["features"],
                                    right["features"],
                                    matched_a,
                                    matched_b,
                                )
                                left_seed = left["item"]["meta"]["sae_seed"]
                                right_seed = right["item"]["meta"]["sae_seed"]
                                row = self.summarize_stability_pair(
                                    {
                                        "model_name": model_name,
                                        "model_seed": model_seed,
                                        "layer": layer,
                                        "dict_size": dict_size,
                                        "k": k,
                                        "sae_seed_a": left_seed,
                                        "sae_seed_b": right_seed,
                                    },
                                    decoder_scores,
                                    activation_corrs,
                                )
                                pair_rows.append(row)
                                nested[model_name][str(model_seed)].setdefault(str(layer), []).append(row)
        summary_rows = self.summarize_stability_rows(pair_rows)
        result = {
            "phase": "4b",
            "design": {
                "matching": "mutual_nearest_neighbor_on_decoder_cosine",
                "activation_correlation": "computed_on_matched_features_using_same_validation_residual_tokens",
                "stability_max_tokens": getattr(self.sae_cfg, "stability_max_tokens", 4096),
                "dict_sizes": sorted(target_dict_sizes) if target_dict_sizes else "all",
                "topk_values": sorted(target_topks) if target_topks else "all",
                "thresholds": list(getattr(self.sae_cfg, "stability_similarity_thresholds", [0.5, 0.7, 0.9])),
            },
            "summary_rows": summary_rows,
            "pair_rows": pair_rows,
            "nested": nested,
        }
        save_json(result, Path(PATH.raw_metrics_dir) / "phase4b_feature_stability.json")
        write_csv(pair_rows, Path(PATH.table_dir) / "phase4b_feature_stability_pairs.csv")
        write_csv(summary_rows, Path(PATH.table_dir) / "phase4b_feature_stability_summary.csv")
        paths = plot_phase4b_stability_figures(PATH.raw_metrics_dir, PATH.figure_dir)
        if paths:
            self.logger.log_stage_end(f"Phase 4b stability figures: wrote {len(paths)} overview files")
        self.logger.log_stage_end("Phase 4b SAE feature stability")
        return result

    def run(self):
        configured_runs = self.configured_sae_runs()
        self.logger.log_stage_start(
            f"SAE stage models={list(self.train_res.keys())} layers={list(self.sae_cfg.layers)} "
            f"configured_runs={configured_runs}"
        )
        if (
            getattr(self.sae_cfg, "skip_completed_stage", True)
            and manifest_is_current(self.stage_manifest_path(), self.stage_config(), self.stage_outputs())
        ):
            loaded = self.load_completed_sae_res()
            if loaded is not None:
                self.logger.log_stage_end("skip SAE stage: existing outputs match config")
                self.plot_summary_figures()
                plot_phase4b_stability_figures(PATH.raw_metrics_dir, PATH.figure_dir)
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
                    for run_spec in configured_runs:
                        item = self.load_existing_sae_item(
                            model_name,
                            model_seed,
                            layer,
                            run_spec["dict_size"],
                            run_spec["k"],
                            run_spec["sae_seed"],
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
                    for run_spec in configured_runs:
                        layer_res.append(
                            self.train_one_sae(
                                train_acts,
                                valid_acts,
                                model_name,
                                model_seed,
                                layer,
                                run_spec["dict_size"],
                                run_spec["k"],
                                run_spec["sae_seed"],
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
        sweep_summary_rows = self.summarize_sweep_rows(rows)
        sweep_conclusion_rows, sweep_conclusion_aggregate_rows = self.sweep_conclusion_rows(sweep_summary_rows)
        write_csv(rows, Path(PATH.table_dir) / "phase4a_sae_runs.csv")
        write_csv(
            summary_rows,
            Path(PATH.table_dir) / "phase4a_sae_summary.csv",
        )
        write_csv(
            sweep_summary_rows,
            Path(PATH.table_dir) / "phase4a_sae_sweep_summary.csv",
        )
        write_csv(
            sweep_conclusion_rows + sweep_conclusion_aggregate_rows,
            Path(PATH.table_dir) / "phase4a_sae_sweep_conclusions.csv",
        )
        save_json(
            {
                "phase": "4a",
                "design": {
                    "dictionary_size": self.sae_cfg.dictionary_sizes,
                    "k": self.sae_cfg.topk_values,
                    "configured_runs": configured_runs,
                    "layers": self.sae_cfg.layers,
                    "sae_seeds": self.sae_cfg.seeds,
                    "activation_site": self.sae_cfg.activation_site,
                    "normalize_activations": getattr(self.sae_cfg, "normalize_activations", True),
                    "dictionary_size_sensitivity": (
                        "available_when_multiple_dictionary_sizes_are_configured"
                    ),
                    "sparsity_sensitivity": "available_when_multiple_topk_values_are_configured",
                },
                "summary_rows": summary_rows,
                "sweep_summary_rows": sweep_summary_rows,
                "sweep_conclusion_rows": sweep_conclusion_rows,
                "sweep_conclusion_aggregate_rows": sweep_conclusion_aggregate_rows,
                "sweep_conclusion_notes": self.sweep_conclusion_notes(sweep_conclusion_aggregate_rows),
                "run_rows": rows,
            },
            Path(PATH.raw_metrics_dir) / "phase4a_summary.json",
        )
        save_json(
            {
                "phase": "4a_sweep_conclusions",
                "design": {
                    "primary_dictionary_size": getattr(self.sae_cfg, "primary_dictionary_size", None),
                    "primary_topk": getattr(self.sae_cfg, "primary_topk", None),
                    "purpose": (
                        "Summarize whether SAE conclusions are sensitive to dictionary size or top-k."
                    ),
                },
                "layer_rows": sweep_conclusion_rows,
                "aggregate_rows": sweep_conclusion_aggregate_rows,
                "notes": self.sweep_conclusion_notes(sweep_conclusion_aggregate_rows),
            },
            Path(PATH.raw_metrics_dir) / "phase4a_sae_sweep_conclusions.json",
        )
        self.run_feature_stability(sae_res)
        self.plot_summary_figures()
        save_manifest(self.stage_manifest_path(), "sae", self.stage_config(), self.stage_outputs())
        self.logger.log_stage_end("SAE stage")
        return sae_res
