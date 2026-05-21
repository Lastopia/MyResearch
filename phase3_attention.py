import math
from pathlib import Path

import torch

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
from utils import mean_std, write_csv
from visualize import (
    plot_attention_heatmap,
    plot_head_metric_bars,
    plot_singular_value_spectrum,
)


class Phase3AttentionAnalyzer:
    def __init__(self, train_cfg, model_cfg, valid_loader):
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg
        self.valid_loader = valid_loader

    @torch.no_grad()
    def run(self, model, model_name=None, seed=None):
        model.eval()
        device = next(model.parameters()).device
        batches = []
        for idx, (x, _) in enumerate(self.valid_loader()):
            if idx >= max(1, self.train_cfg.analysis_batches):
                break
            batches.append(x.to(device))

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
                    for top_k in spectral_topks:
                        selected_concentration = compute_spectral_concentration(head_svals, top_k=top_k).mean(dim=0)
                        dense = torch.full((self.model_cfg.n_heads,), float("nan"))
                        for idx, head in enumerate(spectral_heads):
                            dense[head] = selected_concentration[idx].detach().cpu()
                        head_metrics[layer][f"spectral_concentration_top{top_k}"].append(dense)
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

        headwise = self.finalize_head_metrics(head_metrics)
        taxonomy = self.classify_heads(headwise)
        figure_paths = []
        if first_out is not None and model_name is not None and seed is not None:
            figure_paths = self.write_figures(
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
                "attn_entropy": self.average(layer_entropy),
                "attn_distance": self.average(layer_distance),
                f"spectral_concentration_top{max(spectral_topks)}": self.average(layer_spectral),
                "toeplitz_deviation": self.average(layer_toeplitz),
            },
            "layer_group_summary": self.layer_group_summary(headwise),
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
        self.write_tables(phase3, model_name, seed)
        return {
            **phase3["layer_wise"],
            "phase3": phase3,
        }

    def average(self, items):
        return {str(k): sum(v) / len(v) for k, v in items.items()}

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
            stage: {metric: mean_std(values) for metric, values in metrics.items()}
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

        low_distance = self.quantile(all_distance, 0.25)
        high_distance = self.quantile(all_distance, 0.75)
        low_toeplitz = self.quantile(all_toeplitz, 0.25)
        high_local = self.quantile(all_local, 0.75)
        high_far = self.quantile(all_far, 0.75)
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

    def quantile(self, values, q):
        clean = sorted(value for value in values if value is not None and math.isfinite(value))
        if not clean:
            return None
        idx = min(len(clean) - 1, max(0, int(round((len(clean) - 1) * q))))
        return clean[idx]

    def write_figures(
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
                for metric, ylabel in [
                    ("entropy", "entropy"),
                    ("distance", "avg distance"),
                    ("toeplitz_deviation", "toeplitz deviation"),
                ]:
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

    def write_tables(self, phase3, model_name, seed):
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
        write_csv(rows, Path(PATH.table_dir) / f"{model_name}_seed{seed}_phase3_head_metrics.csv")

        taxonomy_rows = [
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
            for item in phase3["head_taxonomy"]["heads"]
        ]
        write_csv(taxonomy_rows, Path(PATH.table_dir) / f"{model_name}_seed{seed}_phase3_head_taxonomy.csv")
