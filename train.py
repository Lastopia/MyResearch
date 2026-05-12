import csv
import math
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from logger import ExperimentLogger
from metrics import (
    compute_attention_distance,
    compute_attention_entropy,
    compute_far_attention_mass,
    compute_head_singular_values,
    compute_local_attention_mass,
    compute_singular_values,
    compute_spectral_concentration,
    compute_toeplitz_deviation,
)
from para import PATH
from utils import (
    ensure_dir,
    get_device,
    load_json,
    manifest_is_current,
    perplexity,
    save_json,
    save_manifest,
    set_seed,
)
from model import GPTLikeTransformer
from visualize import (
    plot_attention_heatmap,
    plot_head_metric_bars,
    plot_loss_curve,
    plot_model_loss_curves,
    plot_singular_value_spectrum,
)


class Train:
    def __init__(self, train_cfg, model_res, data_res):
        self.train_cfg = train_cfg
        self.model_res = model_res
        self.data_res = data_res
        self.model_cfg = model_res["config"]
        self.seeds = train_cfg.seeds
        self.device = get_device(train_cfg.device)
        self.logger = ExperimentLogger()

    def loader(self, split, shuffle):
        return DataLoader(
            self.data_res[split],
            batch_size=self.train_cfg.batch_size,
            shuffle=shuffle,
            drop_last=True,
        )

    def optimizer(self, model):
        return AdamW(model.parameters(), lr=self.train_cfg.lr, weight_decay=self.train_cfg.weight_decay)

    def lr_scale(self, step):
        if step < self.train_cfg.warmup_steps:
            return max(step, 1) / max(self.train_cfg.warmup_steps, 1)
        return 1.0

    def autocast_context(self):
        enabled = self.device.type == "cuda" and self.train_cfg.precision in {"bf16", "fp16"}
        dtype = torch.bfloat16 if self.train_cfg.precision == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled)

    def stage_config(self):
        return {
            "train": self.train_cfg,
            "model": self.model_cfg,
            "data_meta": self.data_res.get("meta", {}),
        }

    def stage_outputs(self):
        return [
            Path(PATH.raw_metrics_dir) / "train_res.json",
            Path(PATH.raw_metrics_dir) / "phase2_summary.json",
            Path(PATH.raw_metrics_dir) / "phase2_checkpoint_comparison.json",
            Path(PATH.raw_metrics_dir) / "phase3_summary.json",
        ]

    def stage_manifest_path(self):
        return Path(PATH.raw_metrics_dir) / "train_manifest.json"

    def load_completed_train_res(self):
        serializable = load_json(Path(PATH.raw_metrics_dir) / "train_res.json")
        train_res = {}
        for model_name, seed_items in serializable.items():
            train_res[model_name] = {}
            for seed_text, item in seed_items.items():
                seed = int(seed_text)
                state = item["train_state"]
                checkpoint_path = state.get("checkpoint_path")
                if not checkpoint_path or not Path(checkpoint_path).exists():
                    return None
                model = GPTLikeTransformer(self.model_cfg, model_name).to(self.device)
                self.load_checkpoint(checkpoint_path, model)
                train_res[model_name][seed] = {
                    "model": model,
                    "train_state": state,
                    "analysis_res": item.get("analysis_res", {}),
                }
        return train_res

    @torch.no_grad()
    def validate(self, model):
        model.eval()
        losses = []
        for idx, (x, y) in enumerate(self.loader("valid", shuffle=False)):
            if idx >= max(1, self.train_cfg.analysis_batches):
                break
            x, y = x.to(self.device), y.to(self.device)
            with self.autocast_context():
                out = model(x, labels=y)
            losses.append(out["loss"].float().item())
        mean_loss = sum(losses) / len(losses)
        return {"valid_loss": mean_loss, "perplexity": perplexity(mean_loss)}

    @torch.no_grad()
    def attention_analysis(self, model, model_name=None, seed=None):
        model.eval()
        batches = []
        for idx, (x, _) in enumerate(self.loader("valid", shuffle=False)):
            if idx >= max(1, self.train_cfg.analysis_batches):
                break
            batches.append(x.to(self.device))

        layer_entropy, layer_distance, layer_spectral, layer_toeplitz = {}, {}, {}, {}
        head_metrics = {}
        spectra_by_head = {}
        local_windows = getattr(self.train_cfg, "local_attention_windows", [4, 16, 64])
        spectral_topks = getattr(self.train_cfg, "spectral_topk_values", [1, 4, 8])
        far_min_distance = max(1, int(self.model_cfg.seq_len * getattr(self.train_cfg, "long_range_fraction", 0.25)))
        representative_layers = self.representative_layers()
        representative_heads = getattr(self.train_cfg, "representative_heads", [0, 1, 2, 3])
        spectral_layers = set(getattr(self.train_cfg, "spectral_analysis_layers", representative_layers))
        spectral_heads = [
            head for head in getattr(self.train_cfg, "spectral_analysis_heads", representative_heads)
            if 0 <= head < self.model_cfg.n_heads
        ]
        first_out = None
        for x in batches:
            out = model(x, return_attention=True)
            if first_out is None:
                first_out = out
            for layer, attn in enumerate(out["attentions"]):
                head_metrics.setdefault(layer, self.empty_head_metric_accumulator())
                if self.train_cfg.run_attn_entropy:
                    entropy_per_head = compute_attention_entropy(attn).mean(dim=(0, 2))
                    layer_entropy.setdefault(layer, []).append(entropy_per_head.mean().item())
                    head_metrics[layer]["entropy"].append(entropy_per_head.detach().cpu())
                if self.train_cfg.run_attn_distance:
                    distance_per_head = compute_attention_distance(attn).mean(dim=(0, 2))
                    layer_distance.setdefault(layer, []).append(distance_per_head.mean().item())
                    head_metrics[layer]["distance"].append(distance_per_head.detach().cpu())
                    for window in local_windows:
                        local_per_head = compute_local_attention_mass(attn, window).mean(dim=(0, 2))
                        head_metrics[layer][f"local_mass_{window}"].append(local_per_head.detach().cpu())
                    far_per_head = compute_far_attention_mass(attn, far_min_distance).mean(dim=(0, 2))
                    head_metrics[layer]["far_mass"].append(far_per_head.detach().cpu())
            for layer, logits in enumerate(out["attention_logits"]):
                head_metrics.setdefault(layer, self.empty_head_metric_accumulator())
                if self.train_cfg.run_sv_distribution and layer in spectral_layers and spectral_heads:
                    selected_logits = logits[:, spectral_heads, :, :]
                    head_svals = compute_head_singular_values(selected_logits)
                    full_concentration = {}
                    for top_k in spectral_topks:
                        selected_concentration = compute_spectral_concentration(head_svals, top_k=top_k).mean(dim=0)
                        dense = torch.full((self.model_cfg.n_heads,), float("nan"))
                        for idx, head in enumerate(spectral_heads):
                            dense[head] = selected_concentration[idx].detach().cpu()
                        head_metrics[layer][f"spectral_concentration_top{top_k}"].append(dense)
                        full_concentration[top_k] = selected_concentration
                    spectra_by_head.setdefault(layer, []).append(
                        {
                            "heads": list(spectral_heads),
                            "svals": head_svals.detach().cpu(),
                        }
                    )
                    svals = compute_singular_values(logits)
                    layer_spectral.setdefault(layer, []).append(
                        compute_spectral_concentration(svals, top_k=max(spectral_topks)).mean().item()
                    )
                if self.train_cfg.run_toeplitz:
                    toeplitz = compute_toeplitz_deviation(logits).view(logits.size(0), logits.size(1)).mean(dim=0)
                    layer_toeplitz.setdefault(layer, []).append(toeplitz.mean().item())
                    head_metrics[layer]["toeplitz_deviation"].append(toeplitz.detach().cpu())

        def average(items):
            return {str(k): sum(v) / len(v) for k, v in items.items()}

        headwise = self.finalize_head_metrics(head_metrics)
        layer_groups = self.layer_group_summary(headwise)
        taxonomy = self.classify_heads(headwise)
        figure_paths = []
        if first_out is not None and model_name is not None and seed is not None:
            figure_paths = self.write_phase3_figures(
                first_out,
                spectra_by_head,
                headwise,
                model_name,
                seed,
                representative_layers,
                representative_heads,
            )
        phase3 = {
            "layer_wise": {
                "attn_entropy": average(layer_entropy),
                "attn_distance": average(layer_distance),
                f"spectral_concentration_top{max(spectral_topks)}": average(layer_spectral),
                "toeplitz_deviation": average(layer_toeplitz),
            },
            "layer_group_summary": layer_groups,
            "head_wise": headwise,
            "head_taxonomy": taxonomy,
            "figure_paths": figure_paths,
            "notes": {
                "content_matching_heads": (
                    "not_fully_classified_without_token_level_or_synthetic_content_matching_evidence"
                ),
                "head_index_alignment": "compare head distributions across models; same head index is not guaranteed semantically aligned",
                "spectral_analysis": {
                    "limited_to_layers": sorted(spectral_layers),
                    "limited_to_heads": spectral_heads,
                    "reason": "full per-layer per-head SVD at seq_len 1024 is expensive; entropy, distance, local mass, and Toeplitz remain full layer/head metrics",
                },
            },
        }
        self.write_phase3_tables(phase3, model_name, seed)
        return {
            **phase3["layer_wise"],
            "phase3": phase3,
        }

    def empty_head_metric_accumulator(self):
        metrics = {
            "entropy": [],
            "distance": [],
            "far_mass": [],
            "toeplitz_deviation": [],
        }
        for window in getattr(self.train_cfg, "local_attention_windows", [4, 16, 64]):
            metrics[f"local_mass_{window}"] = []
        for top_k in getattr(self.train_cfg, "spectral_topk_values", [1, 4, 8]):
            metrics[f"spectral_concentration_top{top_k}"] = []
        return metrics

    def finalize_head_metrics(self, head_metrics):
        final = {}
        for layer, metrics in head_metrics.items():
            layer_item = {}
            for name, values in metrics.items():
                if values:
                    stacked = torch.stack(values, dim=0).float()
                    means = stacked.mean(dim=0)
                    layer_item[name] = [
                        value.item() if torch.isfinite(value).item() else None
                        for value in means
                    ]
            final[str(layer)] = layer_item
        return final

    def representative_layers(self):
        configured = getattr(self.train_cfg, "representative_layers", None)
        if configured:
            return [layer for layer in configured if 0 <= layer < self.model_cfg.n_layers]
        if self.model_cfg.n_layers <= 3:
            return list(range(self.model_cfg.n_layers))
        return sorted({0, self.model_cfg.n_layers // 2, self.model_cfg.n_layers - 2})

    def layer_stage(self, layer):
        third = max(1, self.model_cfg.n_layers / 3)
        if layer < third:
            return "early"
        if layer < 2 * third:
            return "middle"
        return "late"

    def layer_group_summary(self, headwise):
        groups = {}
        for layer_key, metrics in headwise.items():
            stage = self.layer_stage(int(layer_key))
            groups.setdefault(stage, {})
            for metric, values in metrics.items():
                groups[stage].setdefault(metric, []).extend(values)
        return {
            stage: {metric: self.mean_std(values) for metric, values in metrics.items()}
            for stage, metrics in groups.items()
        }

    def classify_heads(self, headwise):
        rows = []
        local_key = f"local_mass_{min(getattr(self.train_cfg, 'local_attention_windows', [16]), key=lambda x: abs(x - 16))}"
        all_distance = []
        all_toeplitz = []
        all_local = []
        all_far = []
        for metrics in headwise.values():
            all_distance.extend(metrics.get("distance", []))
            all_toeplitz.extend(metrics.get("toeplitz_deviation", []))
            all_local.extend(metrics.get(local_key, []))
            all_far.extend(metrics.get("far_mass", []))

        def quantile(values, q):
            clean = sorted(value for value in values if value is not None and math.isfinite(value))
            if not clean:
                return None
            idx = min(len(clean) - 1, max(0, int(round((len(clean) - 1) * q))))
            return clean[idx]

        low_distance = quantile(all_distance, 0.25)
        high_distance = quantile(all_distance, 0.75)
        low_toeplitz = quantile(all_toeplitz, 0.25)
        high_local = quantile(all_local, 0.75)
        high_far = quantile(all_far, 0.75)
        counts = {
            "local_syntactic": 0,
            "long_range_dependency": 0,
            "position_sensitive": 0,
            "content_matching": 0,
            "unclassified": 0,
        }
        for layer_key, metrics in headwise.items():
            n_heads = len(metrics.get("distance", []))
            for head in range(n_heads):
                labels = []
                distance = metrics.get("distance", [None] * n_heads)[head]
                local = metrics.get(local_key, [None] * n_heads)[head]
                far = metrics.get("far_mass", [None] * n_heads)[head]
                toeplitz = metrics.get("toeplitz_deviation", [None] * n_heads)[head]
                if low_distance is not None and high_local is not None and distance <= low_distance and local >= high_local:
                    labels.append("local_syntactic")
                if high_distance is not None and high_far is not None and distance >= high_distance and far >= high_far:
                    labels.append("long_range_dependency")
                if low_toeplitz is not None and toeplitz <= low_toeplitz:
                    labels.append("position_sensitive")
                if not labels:
                    labels.append("unclassified")
                for label in labels:
                    counts[label] += 1
                rows.append(
                    {
                        "layer": int(layer_key),
                        "stage": self.layer_stage(int(layer_key)),
                        "head": head,
                        "labels": labels,
                        "distance": distance,
                        "local_mass": local,
                        "far_mass": far,
                        "toeplitz_deviation": toeplitz,
                    }
                )
        return {
            "thresholds": {
                "low_distance_q25": low_distance,
                "high_distance_q75": high_distance,
                "low_toeplitz_q25": low_toeplitz,
                "high_local_mass_q75": high_local,
                "high_far_mass_q75": high_far,
                "local_mass_key": local_key,
            },
            "counts": counts,
            "heads": rows,
        }

    def write_phase3_figures(
        self,
        first_out,
        spectra_by_head,
        headwise,
        model_name,
        seed,
        representative_layers,
        representative_heads,
    ):
        paths = []
        seq_cap = getattr(self.train_cfg, "max_heatmap_seq_len", 128)
        if getattr(self.train_cfg, "run_attention_heatmaps", True):
            for layer in representative_layers:
                if layer >= len(first_out["attentions"]):
                    continue
                attn = first_out["attentions"][layer][0].detach().cpu()
                for head in representative_heads:
                    if head >= attn.size(0):
                        continue
                    matrix = attn[head, :seq_cap, :seq_cap]
                    path = Path(PATH.figure_dir) / (
                        f"{model_name}_seed{seed}_"
                        f"layer{layer}_head{head}_attention_heatmap.png"
                    )
                    plot_attention_heatmap(matrix, path, title=f"{model_name} seed {seed} L{layer} H{head}")
                    paths.append(str(path))
        if getattr(self.train_cfg, "run_spectral_plots", True):
            for layer in representative_layers:
                spectra_items = spectra_by_head.get(layer)
                if not spectra_items:
                    continue
                heads = spectra_items[0]["heads"]
                spectra = torch.stack([item["svals"] for item in spectra_items], dim=0).mean(dim=0)[0]
                for head in representative_heads:
                    if head not in heads:
                        continue
                    head_idx = heads.index(head)
                    path = Path(PATH.figure_dir) / (
                        f"{model_name}_seed{seed}_"
                        f"layer{layer}_head{head}_singular_values.png"
                    )
                    plot_singular_value_spectrum(
                        spectra[head_idx],
                        path,
                        title=f"{model_name} seed {seed} L{layer} H{head}",
                    )
                    paths.append(str(path))
                layer_metrics = headwise.get(str(layer), {})
                for metric, ylabel in [("entropy", "entropy"), ("distance", "avg distance"), ("toeplitz_deviation", "toeplitz deviation")]:
                    if metric not in layer_metrics:
                        continue
                    path = Path(PATH.figure_dir) / (
                        f"{model_name}_seed{seed}_"
                        f"layer{layer}_{metric}_by_head.png"
                    )
                    plot_head_metric_bars(
                        layer_metrics[metric],
                        path,
                        ylabel=ylabel,
                        title=f"{model_name} seed {seed} L{layer} {metric}",
                    )
                    paths.append(str(path))
        return paths

    def write_phase3_tables(self, phase3, model_name, seed):
        if model_name is None or seed is None:
            return
        rows = []
        for layer_key, metrics in phase3["head_wise"].items():
            n_heads = max((len(values) for values in metrics.values()), default=0)
            for head in range(n_heads):
                row = {"model_name": model_name, "seed": seed, "layer": layer_key, "head": head}
                for metric, values in metrics.items():
                    if head < len(values):
                        row[metric] = values[head]
                rows.append(row)
        self.write_csv(rows, Path(PATH.table_dir) / f"{model_name}_seed{seed}_phase3_head_metrics.csv")

        taxonomy_rows = []
        for item in phase3["head_taxonomy"]["heads"]:
            taxonomy_rows.append(
                {
                    "model_name": model_name,
                    "seed": seed,
                    "layer": item["layer"],
                    "stage": item["stage"],
                    "head": item["head"],
                    "labels": "|".join(item["labels"]),
                    "distance": item["distance"],
                    "local_mass": item["local_mass"],
                    "far_mass": item["far_mass"],
                    "toeplitz_deviation": item["toeplitz_deviation"],
                }
            )
        self.write_csv(taxonomy_rows, Path(PATH.table_dir) / f"{model_name}_seed{seed}_phase3_head_taxonomy.csv")

    def save_checkpoint(
        self,
        model,
        model_name,
        seed,
        step,
        valid_loss=None,
        selection_rule=None,
        optimizer=None,
        history=None,
        train_losses=None,
        grad_norms=None,
        best_valid=None,
        divergence_count=0,
    ):
        ensure_dir(Path(PATH.ckpt_dir) / "models")
        path = Path(PATH.ckpt_dir) / "models" / (
            f"{model_name}_seed{seed}_step{step}.pt"
        )
        metadata = {
            "model_name": model_name,
            "seed": seed,
            "checkpoint_step": step,
            "tokens_seen": step * self.train_cfg.batch_size * self.model_cfg.seq_len,
            "valid_loss_at_checkpoint": valid_loss,
            "selection_rule": selection_rule,
        }
        payload = {
            "model": model.state_dict(),
            "metadata": metadata,
            "history": history or [],
            "train_losses": train_losses or [],
            "grad_norms": grad_norms or [],
            "best_valid": best_valid,
            "divergence_count": divergence_count,
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)
        return str(path)

    def latest_checkpoint_path(self, model_name, seed):
        ckpt_dir = Path(PATH.ckpt_dir) / "models"
        candidates = []
        for path in ckpt_dir.glob(f"{model_name}_seed{seed}_step*.pt"):
            stem = path.stem
            try:
                step = int(stem.rsplit("step", 1)[1])
            except (IndexError, ValueError):
                continue
            candidates.append((step, path))
        if not candidates:
            return None
        return str(max(candidates, key=lambda item: item[0])[1])

    def load_checkpoint(self, path, model, optimizer=None):
        ckpt = torch.load(path, map_location=self.device)
        model.load_state_dict(ckpt["model"])
        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        metadata = ckpt.get("metadata", {})
        return {
            "step": int(metadata.get("checkpoint_step", 0)),
            "history": ckpt.get("history", []),
            "train_losses": [tuple(item) for item in ckpt.get("train_losses", [])],
            "grad_norms": [tuple(item) for item in ckpt.get("grad_norms", [])],
            "best_valid": ckpt.get("best_valid", float("inf")),
            "divergence_count": ckpt.get("divergence_count", 0),
            "path": str(path),
        }

    def validation_candidates_from_history(self, model_name, seed, history):
        candidates = []
        for row in history:
            if "valid_loss" not in row:
                continue
            step = row["step"]
            path = Path(PATH.ckpt_dir) / "models" / f"{model_name}_seed{seed}_step{step}.pt"
            if path.exists():
                candidates.append(
                    self.checkpoint_metadata(
                        model_name,
                        seed,
                        step,
                        str(path),
                        row["valid_loss"],
                        "validation_candidate",
                    )
                )
        return candidates

    def checkpoint_metadata(self, model_name, seed, step, checkpoint_path, valid_loss, selection_rule):
        return {
            "model_name": model_name,
            "seed": seed,
            "checkpoint_step": step,
            "tokens_seen": step * self.train_cfg.batch_size * self.model_cfg.seq_len,
            "checkpoint_path": checkpoint_path,
            "valid_loss_at_checkpoint": valid_loss,
            "selection_rule": selection_rule,
        }

    def loss_threshold_step(self, losses):
        threshold = getattr(self.train_cfg, "loss_threshold", None)
        if threshold is None:
            return None
        for step, loss in losses:
            if loss <= threshold:
                return step
        return None

    def stability_metrics(self, train_losses, grad_norms, divergence_count, elapsed_seconds):
        loss_spike_threshold = getattr(self.train_cfg, "loss_spike_threshold", 0.5)
        loss_spikes = 0
        for (_, prev_loss), (_, loss) in zip(train_losses, train_losses[1:]):
            if loss - prev_loss > loss_spike_threshold:
                loss_spikes += 1

        grad_values = [value for _, value in grad_norms if math.isfinite(value)]
        loss_values = [value for _, value in train_losses if math.isfinite(value)]
        grad_mean = sum(grad_values) / len(grad_values) if grad_values else None
        loss_mean = sum(loss_values) / len(loss_values) if loss_values else None

        def variance(values, mean):
            if not values:
                return None
            return sum((value - mean) ** 2 for value in values) / len(values)

        tokens_seen = self.train_cfg.steps * self.train_cfg.batch_size * self.model_cfg.seq_len
        return {
            "grad_norm_mean": grad_mean,
            "grad_norm_variance": variance(grad_values, grad_mean) if grad_mean is not None else None,
            "train_loss_mean": loss_mean,
            "train_loss_variance": variance(loss_values, loss_mean) if loss_mean is not None else None,
            "loss_spike_count": loss_spikes,
            "divergence_count": divergence_count,
            "loss_threshold": getattr(self.train_cfg, "loss_threshold", None),
            "loss_threshold_step": self.loss_threshold_step(train_losses),
            "elapsed_seconds": elapsed_seconds,
            "seconds_per_step": elapsed_seconds / max(self.train_cfg.steps, 1),
            "tokens_seen": tokens_seen,
            "tokens_per_second": tokens_seen / elapsed_seconds if elapsed_seconds > 0 else None,
        }

    def length_sensitivity_report(self):
        return {
            "standard_context_validation": "covered_by_validate",
            "length_extrapolation": {
                "enabled": getattr(self.train_cfg, "run_length_extrapolation", False),
                "status": (
                    "not_run_requires_model_seq_len_and_data_blocks_for_longer_context"
                    if not getattr(self.train_cfg, "run_length_extrapolation", False)
                    else "configured_not_implemented"
                ),
                "requested_seq_lens": getattr(self.train_cfg, "extrapolation_seq_lens", []),
            },
            "position_synthetic_task": {
                "enabled": getattr(self.train_cfg, "run_position_synthetic_task", False),
                "status": (
                    "not_run"
                    if not getattr(self.train_cfg, "run_position_synthetic_task", False)
                    else "configured_not_implemented"
                ),
            },
        }

    def train_one_model(self, model_name, base_model, seed):
        set_seed(seed)
        model = GPTLikeTransformer(self.model_cfg, model_name).to(self.device)
        opt = self.optimizer(model)
        start_step = 0
        train_iter = iter(self.loader("train", shuffle=True))
        history = []
        train_losses = []
        grad_norms = []
        divergence_count = 0
        best_valid = float("inf")
        ckpt_path = None
        final_checkpoint = None
        validation_checkpoints = []
        latest_path = self.latest_checkpoint_path(model_name, seed)
        if getattr(self.train_cfg, "resume_from_checkpoint", True) and latest_path:
            resume_state = self.load_checkpoint(latest_path, model, opt)
            start_step = min(resume_state["step"], self.train_cfg.steps)
            history = resume_state["history"]
            train_losses = resume_state["train_losses"]
            grad_norms = resume_state["grad_norms"]
            best_valid = resume_state["best_valid"] if resume_state["best_valid"] is not None else float("inf")
            divergence_count = resume_state["divergence_count"]
            ckpt_path = resume_state["path"]
            validation_checkpoints = self.validation_candidates_from_history(model_name, seed, history)
            self.logger.log_stage_start(f"resume {model_name} seed {seed} from step {start_step}")
        start_time = time.perf_counter()

        for step in range(start_step + 1, self.train_cfg.steps + 1):
            model.train()
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(self.loader("train", shuffle=True))
                x, y = next(train_iter)
            x, y = x.to(self.device), y.to(self.device)
            scale = self.lr_scale(step)
            for group in opt.param_groups:
                group["lr"] = self.train_cfg.lr * scale
            opt.zero_grad(set_to_none=True)
            with self.autocast_context():
                out = model(x, labels=y)
            loss_value = out["loss"].float().item()
            if not math.isfinite(loss_value):
                divergence_count += 1
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss_value}")
            out["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.train_cfg.grad_clip)
            grad_norm_value = float(grad_norm)
            if not math.isfinite(grad_norm_value):
                divergence_count += 1
            opt.step()
            train_losses.append((step, loss_value))
            grad_norms.append((step, grad_norm_value))

            if step % self.train_cfg.log_interval == 0 or step == 1:
                row = {"step": step, "train_loss": loss_value, "grad_norm": grad_norm_value}
                history.append(row)
                self.logger.log_metric(f"{model_name}/train_loss", row["train_loss"], step)
            if step % self.train_cfg.eval_interval == 0 or step == self.train_cfg.steps:
                valid = self.validate(model)
                history.append({"step": step, **valid})
                best_valid = min(best_valid, valid["valid_loss"])
                self.logger.log_metric(f"{model_name}/valid_loss", valid["valid_loss"], step)
                if getattr(self.train_cfg, "save_eval_checkpoints", True):
                    ckpt_path = self.save_checkpoint(
                        model,
                        model_name,
                        seed,
                        step,
                        valid_loss=valid["valid_loss"],
                        selection_rule="validation_candidate",
                        optimizer=opt,
                        history=history,
                        train_losses=train_losses,
                        grad_norms=grad_norms,
                        best_valid=best_valid,
                        divergence_count=divergence_count,
                    )
                    validation_checkpoints.append(
                        self.checkpoint_metadata(
                            model_name,
                            seed,
                            step,
                            ckpt_path,
                            valid["valid_loss"],
                            "validation_candidate",
                        )
                    )
            if step % self.train_cfg.save_interval == 0 or step == self.train_cfg.steps:
                ckpt_path = self.save_checkpoint(
                    model,
                    model_name,
                    seed,
                    step,
                    selection_rule="periodic_or_final",
                    optimizer=opt,
                    history=history,
                    train_losses=train_losses,
                    grad_norms=grad_norms,
                    best_valid=best_valid,
                    divergence_count=divergence_count,
                )

        elapsed_seconds = time.perf_counter() - start_time
        final_valid = self.validate(model)
        ckpt_path = self.save_checkpoint(
            model,
            model_name,
            seed,
            self.train_cfg.steps,
            valid_loss=final_valid["valid_loss"],
            selection_rule="final_step",
            optimizer=opt,
            history=history,
            train_losses=train_losses,
            grad_norms=grad_norms,
            best_valid=best_valid,
            divergence_count=divergence_count,
        )
        final_checkpoint = self.checkpoint_metadata(
            model_name,
            seed,
            self.train_cfg.steps,
            ckpt_path,
            final_valid["valid_loss"],
            "final_step",
        )
        validation_checkpoints = [
            item for item in validation_checkpoints
            if item["checkpoint_step"] != self.train_cfg.steps
        ]
        validation_checkpoints.append(final_checkpoint)
        stability = self.stability_metrics(train_losses, grad_norms, divergence_count, elapsed_seconds)
        if self.train_cfg.run_loss_curve:
            plot_loss_curve(
                history,
                Path(PATH.figure_dir) / f"{model_name}_seed{seed}_loss.png",
            )
        return {
            "model": model,
            "train_state": {
                "final_step": self.train_cfg.steps,
                "best_valid_loss": best_valid,
                "final_valid_loss": final_valid["valid_loss"],
                "final_perplexity": final_valid["perplexity"],
                "checkpoint_path": ckpt_path,
                "checkpoint_step": self.train_cfg.steps,
                "tokens_seen": self.train_cfg.steps * self.train_cfg.batch_size * self.model_cfg.seq_len,
                "valid_loss_at_checkpoint": final_valid["valid_loss"],
                "selection_rule": "final_step",
                "checkpoint_selection": {
                    "primary": final_checkpoint,
                    "validation_candidates": validation_checkpoints,
                    "validation_loss_matched": None,
                    "protocol": {
                        "primary_checkpoint_rule": getattr(
                            self.train_cfg, "primary_checkpoint_rule", "final_step"
                        ),
                        "secondary_checkpoint_rule": getattr(
                            self.train_cfg, "secondary_checkpoint_rule", "validation_loss_matched"
                        ),
                        "validation_loss_match_target": getattr(
                            self.train_cfg, "validation_loss_match_target", None
                        ),
                    },
                },
                "history": history,
                "stability": stability,
                "efficiency": {
                    "elapsed_seconds": stability["elapsed_seconds"],
                    "seconds_per_step": stability["seconds_per_step"],
                    "tokens_per_second": stability["tokens_per_second"],
                },
                "length_sensitivity": self.length_sensitivity_report(),
            },
            "analysis_res": self.attention_analysis(model, model_name=model_name, seed=seed),
        }

    def mean_std(self, values):
        clean = [value for value in values if value is not None and math.isfinite(value)]
        if not clean:
            return {"mean": None, "std": None}
        mean = sum(clean) / len(clean)
        var = sum((value - mean) ** 2 for value in clean) / len(clean)
        return {"mean": mean, "std": math.sqrt(var)}

    def percentile(self, values, q):
        if not values:
            return None
        ordered = sorted(values)
        pos = (len(ordered) - 1) * q
        low = math.floor(pos)
        high = math.ceil(pos)
        if low == high:
            return ordered[low]
        weight = pos - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    def paired_difference_stats(self, per_seed_rows, baseline="rope", target="pope"):
        metrics = [
            "best_valid_loss",
            "final_valid_loss",
            "final_perplexity",
            "grad_norm_mean",
            "grad_norm_variance",
            "train_loss_variance",
            "loss_spike_count",
            "divergence_count",
            "seconds_per_step",
            "tokens_per_second",
            "matched_valid_loss",
        ]
        by_model_seed = {}
        for row in per_seed_rows:
            by_model_seed[(row["model_name"], row["seed"])] = row
        common_seeds = sorted(
            {
                seed for model_name, seed in by_model_seed
                if model_name == baseline and (target, seed) in by_model_seed
            }
        )
        rows = []
        rng = random.Random(0)
        for metric in metrics:
            differences = []
            for seed in common_seeds:
                base_value = by_model_seed[(baseline, seed)].get(metric)
                target_value = by_model_seed[(target, seed)].get(metric)
                if base_value is None or target_value is None:
                    continue
                if not (math.isfinite(base_value) and math.isfinite(target_value)):
                    continue
                differences.append(target_value - base_value)
            if not differences:
                rows.append(
                    {
                        "comparison": f"{target}_minus_{baseline}",
                        "metric": metric,
                        "n_pairs": 0,
                        "paired_difference_mean": None,
                        "paired_difference_std": None,
                        "cohen_dz": None,
                        "bootstrap_ci95_low": None,
                        "bootstrap_ci95_high": None,
                    }
                )
                continue
            mean = sum(differences) / len(differences)
            if len(differences) > 1:
                sample_var = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
                sample_std = math.sqrt(sample_var)
                cohen_dz = mean / sample_std if sample_std > 0 else None
            else:
                sample_std = None
                cohen_dz = None
            bootstrap_means = []
            for _ in range(10000):
                sample = [differences[rng.randrange(len(differences))] for _ in differences]
                bootstrap_means.append(sum(sample) / len(sample))
            rows.append(
                {
                    "comparison": f"{target}_minus_{baseline}",
                    "metric": metric,
                    "n_pairs": len(differences),
                    "paired_difference_mean": mean,
                    "paired_difference_std": sample_std,
                    "cohen_dz": cohen_dz,
                    "bootstrap_ci95_low": self.percentile(bootstrap_means, 0.025),
                    "bootstrap_ci95_high": self.percentile(bootstrap_means, 0.975),
                }
            )
        return rows

    def validation_loss_match_target(self, serializable):
        configured = getattr(self.train_cfg, "validation_loss_match_target", None)
        if configured is not None:
            return configured
        best_losses = []
        for seed_items in serializable.values():
            for item in seed_items.values():
                candidates = item["train_state"]["checkpoint_selection"]["validation_candidates"]
                losses = [
                    candidate["valid_loss_at_checkpoint"]
                    for candidate in candidates
                    if candidate.get("valid_loss_at_checkpoint") is not None
                ]
                if losses:
                    best_losses.append(min(losses))
        return max(best_losses) if best_losses else None

    def matched_validation_checkpoint(self, candidates, target):
        if target is None:
            return None
        valid_candidates = [
            candidate for candidate in candidates
            if candidate.get("valid_loss_at_checkpoint") is not None
        ]
        if not valid_candidates:
            return None
        selected = min(
            valid_candidates,
            key=lambda candidate: (
                abs(candidate["valid_loss_at_checkpoint"] - target),
                candidate["checkpoint_step"],
            ),
        )
        return {
            **selected,
            "selection_rule": "validation_loss_matched",
            "matched_target_valid_loss": target,
            "valid_loss_delta": selected["valid_loss_at_checkpoint"] - target,
        }

    def attach_checkpoint_selection(self, train_res, serializable):
        target = self.validation_loss_match_target(serializable)
        for model_name, seed_items in serializable.items():
            for seed, item in seed_items.items():
                state = item["train_state"]
                candidates = state["checkpoint_selection"]["validation_candidates"]
                matched = self.matched_validation_checkpoint(candidates, target)
                state["checkpoint_selection"]["validation_loss_matched"] = matched
                state["checkpoint_selection"]["protocol"]["validation_loss_match_target"] = target
                state["matched_checkpoint_path"] = matched["checkpoint_path"] if matched else None
                state["matched_valid_loss"] = matched["valid_loss_at_checkpoint"] if matched else None
                state["matched_checkpoint_step"] = matched["checkpoint_step"] if matched else None

                train_item = train_res[model_name][int(seed)]["train_state"]
                train_item["checkpoint_selection"] = state["checkpoint_selection"]
                train_item["matched_checkpoint_path"] = state["matched_checkpoint_path"]
                train_item["matched_valid_loss"] = state["matched_valid_loss"]
                train_item["matched_checkpoint_step"] = state["matched_checkpoint_step"]
        return target

    def summarize_results(self, serializable):
        summary = {}
        per_seed_rows = []
        for model_name, seed_items in serializable.items():
            metrics_by_name = {
                "best_valid_loss": [],
                "final_valid_loss": [],
                "final_perplexity": [],
                "grad_norm_mean": [],
                "grad_norm_variance": [],
                "train_loss_variance": [],
                "loss_spike_count": [],
                "divergence_count": [],
                "loss_threshold_step": [],
                "seconds_per_step": [],
                "tokens_per_second": [],
                "matched_valid_loss": [],
            }
            for seed, item in seed_items.items():
                state = item["train_state"]
                stability = state["stability"]
                row = {
                    "model_name": model_name,
                    "seed": seed,
                    "best_valid_loss": state["best_valid_loss"],
                    "final_valid_loss": state["final_valid_loss"],
                    "final_perplexity": state["final_perplexity"],
                    "grad_norm_mean": stability["grad_norm_mean"],
                    "grad_norm_variance": stability["grad_norm_variance"],
                    "train_loss_variance": stability["train_loss_variance"],
                    "loss_spike_count": stability["loss_spike_count"],
                    "divergence_count": stability["divergence_count"],
                    "loss_threshold_step": stability["loss_threshold_step"],
                    "seconds_per_step": stability["seconds_per_step"],
                    "tokens_per_second": stability["tokens_per_second"],
                    "checkpoint_step": state.get("checkpoint_step"),
                    "tokens_seen": state.get("tokens_seen"),
                    "checkpoint_path": state.get("checkpoint_path"),
                    "valid_loss_at_checkpoint": state.get("valid_loss_at_checkpoint"),
                    "selection_rule": state.get("selection_rule"),
                    "matched_checkpoint_step": state.get("matched_checkpoint_step"),
                    "matched_checkpoint_path": state.get("matched_checkpoint_path"),
                    "matched_valid_loss": state.get("matched_valid_loss"),
                }
                per_seed_rows.append(row)
                for key in metrics_by_name:
                    metrics_by_name[key].append(row[key])
            summary[model_name] = {key: self.mean_std(values) for key, values in metrics_by_name.items()}
        paired_stats = self.paired_difference_stats(per_seed_rows)
        return {"by_model": summary, "per_seed": per_seed_rows, "paired_stats": paired_stats}

    def checkpoint_comparison_rows(self, serializable):
        rows = []
        for model_name, seed_items in serializable.items():
            for seed, item in seed_items.items():
                state = item["train_state"]
                selection = state.get("checkpoint_selection", {})
                primary = selection.get("primary") or {}
                matched = selection.get("validation_loss_matched") or {}
                final_step = primary.get("checkpoint_step")
                matched_step = matched.get("checkpoint_step")
                final_tokens = primary.get("tokens_seen")
                matched_tokens = matched.get("tokens_seen")
                final_loss = primary.get("valid_loss_at_checkpoint")
                matched_loss = matched.get("valid_loss_at_checkpoint")
                rows.append(
                    {
                        "model_name": model_name,
                        "seed": seed,
                        "final_selection_rule": primary.get("selection_rule", "final_step"),
                        "final_checkpoint_step": final_step,
                        "final_tokens_seen": final_tokens,
                        "final_valid_loss": final_loss,
                        "final_checkpoint_path": primary.get("checkpoint_path"),
                        "matched_selection_rule": matched.get("selection_rule"),
                        "matched_target_valid_loss": matched.get("matched_target_valid_loss"),
                        "matched_checkpoint_step": matched_step,
                        "matched_tokens_seen": matched_tokens,
                        "matched_valid_loss": matched_loss,
                        "matched_valid_loss_delta_from_target": matched.get("valid_loss_delta"),
                        "matched_checkpoint_path": matched.get("checkpoint_path"),
                        "step_delta_matched_minus_final": (
                            matched_step - final_step
                            if matched_step is not None and final_step is not None
                            else None
                        ),
                        "tokens_seen_delta_matched_minus_final": (
                            matched_tokens - final_tokens
                            if matched_tokens is not None and final_tokens is not None
                            else None
                        ),
                        "valid_loss_delta_matched_minus_final": (
                            matched_loss - final_loss
                            if matched_loss is not None and final_loss is not None
                            else None
                        ),
                    }
                )
        return rows

    def write_csv(self, rows, path):
        ensure_dir(Path(path).parent)
        if not rows:
            return
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def write_summary_tables(self, summary):
        per_seed_path = Path(PATH.table_dir) / "phase2_per_seed.csv"
        self.write_csv(summary["per_seed"], per_seed_path)
        self.write_csv(summary.get("paired_stats", []), Path(PATH.table_dir) / "phase2_paired_stats.csv")
        self.write_csv(
            summary.get("checkpoint_comparison", []),
            Path(PATH.table_dir) / "phase2_checkpoint_comparison.csv",
        )
        rows = []
        for model_name, metrics in summary["by_model"].items():
            row = {"model_name": model_name}
            for metric_name, stats in metrics.items():
                row[f"{metric_name}_mean"] = stats["mean"]
                row[f"{metric_name}_std"] = stats["std"]
            rows.append(row)
        aggregate_path = Path(PATH.table_dir) / "phase2_summary.csv"
        self.write_csv(rows, aggregate_path)

    def plot_aggregate_curves(self, serializable):
        if not self.train_cfg.run_loss_curve:
            return
        for model_name, seed_items in serializable.items():
            seed_histories = {seed: item["train_state"]["history"] for seed, item in seed_items.items()}
            plot_model_loss_curves(
                seed_histories,
                Path(PATH.figure_dir) / f"{model_name}_loss_all_seeds.png",
            )

    def summarize_phase3(self, serializable):
        metric_names = ["attn_entropy", "attn_distance", "toeplitz_deviation"]
        spectral_key = f"spectral_concentration_top{max(getattr(self.train_cfg, 'spectral_topk_values', [8]))}"
        metric_names.append(spectral_key)
        by_model = {}
        layer_rows = []
        taxonomy_rows = []
        for model_name, seed_items in serializable.items():
            by_model.setdefault(model_name, {"layer_wise": {}, "stage_wise": {}, "taxonomy_counts": {}})
            for seed, item in seed_items.items():
                phase3 = item["analysis_res"].get("phase3", {})
                layer_wise = phase3.get("layer_wise", {})
                for metric in metric_names:
                    for layer, value in layer_wise.get(metric, {}).items():
                        if value is None or not math.isfinite(value):
                            continue
                        by_model[model_name]["layer_wise"].setdefault(metric, {}).setdefault(layer, []).append(value)
                        layer_rows.append(
                            {
                                "model_name": model_name,
                                "seed": seed,
                                "metric": metric,
                                "layer": layer,
                                "value": value,
                            }
                        )
                for stage, metrics in phase3.get("layer_group_summary", {}).items():
                    for metric, stats in metrics.items():
                        by_model[model_name]["stage_wise"].setdefault(stage, {}).setdefault(metric, []).append(
                            stats["mean"]
                        )
                counts = phase3.get("head_taxonomy", {}).get("counts", {})
                for label, count in counts.items():
                    by_model[model_name]["taxonomy_counts"].setdefault(label, []).append(count)
                    taxonomy_rows.append(
                        {
                            "model_name": model_name,
                            "seed": seed,
                            "label": label,
                            "count": count,
                        }
                    )

        summary = {}
        for model_name, model_item in by_model.items():
            summary[model_name] = {
                "layer_wise": {
                    metric: {layer: self.mean_std(values) for layer, values in layers.items()}
                    for metric, layers in model_item["layer_wise"].items()
                },
                "stage_wise": {
                    stage: {metric: self.mean_std(values) for metric, values in metrics.items()}
                    for stage, metrics in model_item["stage_wise"].items()
                },
                "taxonomy_counts": {
                    label: self.mean_std(values) for label, values in model_item["taxonomy_counts"].items()
                },
            }
        return {"by_model": summary, "layer_rows": layer_rows, "taxonomy_rows": taxonomy_rows}

    def write_phase3_summary_tables(self, phase3_summary):
        self.write_csv(
            phase3_summary["layer_rows"],
            Path(PATH.table_dir) / "phase3_layer_metrics.csv",
        )
        self.write_csv(
            phase3_summary["taxonomy_rows"],
            Path(PATH.table_dir) / "phase3_taxonomy_counts.csv",
        )

    def run(self):
        if (
            getattr(self.train_cfg, "skip_completed_runs", True)
            and manifest_is_current(self.stage_manifest_path(), self.stage_config(), self.stage_outputs())
        ):
            loaded = self.load_completed_train_res()
            if loaded is not None:
                self.logger.log_stage_start("skip train stage: existing outputs match config")
                return loaded

        train_res = {}
        serializable = {}
        for model_name, model in self.model_res["models"].items():
            train_res[model_name] = {}
            serializable[model_name] = {}
            for seed in self.seeds:
                self.logger.log_stage_start(f"train {model_name} seed {seed}")
                one = self.train_one_model(model_name, model, seed)
                train_res[model_name][seed] = one
                serializable[model_name][str(seed)] = {
                    "train_state": one["train_state"],
                    "analysis_res": one["analysis_res"],
                }
                self.logger.log_stage_end(f"train {model_name} seed {seed}")
        matched_target = self.attach_checkpoint_selection(train_res, serializable)
        save_json(serializable, Path(PATH.raw_metrics_dir) / "train_res.json")
        summary = self.summarize_results(serializable)
        checkpoint_comparison = self.checkpoint_comparison_rows(serializable)
        summary["checkpoint_comparison"] = checkpoint_comparison
        summary["checkpoint_protocol"] = {
            "primary_checkpoint_rule": getattr(self.train_cfg, "primary_checkpoint_rule", "final_step"),
            "secondary_checkpoint_rule": getattr(
                self.train_cfg, "secondary_checkpoint_rule", "validation_loss_matched"
            ),
            "validation_loss_match_target": matched_target,
            "downstream_default": "primary.final_step",
            "notes": (
                "Main comparisons use final checkpoints matched by training steps/tokens_seen; "
                "validation_loss_matched checkpoints are recorded for robustness analysis."
            ),
        }
        save_json(summary, Path(PATH.raw_metrics_dir) / "phase2_summary.json")
        save_json(
            {
                "checkpoint_protocol": summary["checkpoint_protocol"],
                "rows": checkpoint_comparison,
            },
            Path(PATH.raw_metrics_dir) / "phase2_checkpoint_comparison.json",
        )
        self.write_summary_tables(summary)
        self.plot_aggregate_curves(serializable)
        phase3_summary = self.summarize_phase3(serializable)
        save_json(phase3_summary, Path(PATH.raw_metrics_dir) / "phase3_summary.json")
        self.write_phase3_summary_tables(phase3_summary)
        save_manifest(self.stage_manifest_path(), "train", self.stage_config(), self.stage_outputs())
        return train_res
