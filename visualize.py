from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont

from utils import ensure_dir, load_json


def plot_loss_curve(history, save_path):
    ensure_dir(Path(save_path).parent)
    train = [(x["step"], x["train_loss"]) for x in history if "train_loss" in x]
    valid = [(x["step"], x["valid_loss"]) for x in history if "valid_loss" in x]
    plt.figure(figsize=(7, 4))
    if train:
        plt.plot([x for x, _ in train], [y for _, y in train], label="train")
    if valid:
        plt.plot([x for x, _ in valid], [y for _, y in valid], label="valid")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_model_loss_curves(seed_histories, save_path):
    ensure_dir(Path(save_path).parent)
    plt.figure(figsize=(8, 4.5))
    for seed, history in seed_histories.items():
        train = [(x["step"], x["train_loss"]) for x in history if "train_loss" in x]
        valid = [(x["step"], x["valid_loss"]) for x in history if "valid_loss" in x]
        if train:
            plt.plot([x for x, _ in train], [y for _, y in train], alpha=0.35, label=f"seed {seed} train")
        if valid:
            plt.plot([x for x, _ in valid], [y for _, y in valid], marker="o", label=f"seed {seed} valid")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_metric_curves(run_histories, metric_key, save_path, ylabel=None, title=None):
    ensure_dir(Path(save_path).parent)
    plt.figure(figsize=(9, 5))
    for label, history in run_histories.items():
        points = [(row["step"], row[metric_key]) for row in history if metric_key in row]
        if not points:
            continue
        plt.plot(
            [step for step, _ in points],
            [value for _, value in points],
            marker="o" if len(points) <= 20 else None,
            linewidth=1.5,
            alpha=0.8,
            label=label,
        )
    plt.xlabel("step")
    plt.ylabel(ylabel or metric_key)
    if title:
        plt.title(title)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_attention_heatmap(attn_matrix, save_path, title=None):
    ensure_dir(Path(save_path).parent)
    matrix = attn_matrix.detach().float().cpu() if isinstance(attn_matrix, torch.Tensor) else attn_matrix
    plt.figure(figsize=(5, 4.5))
    plt.imshow(matrix, aspect="auto", origin="upper", cmap="viridis")
    plt.colorbar(label="attention")
    plt.xlabel("key position")
    plt.ylabel("query position")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_singular_value_spectrum(svals, save_path, title=None):
    ensure_dir(Path(save_path).parent)
    values = svals.detach().float().cpu() if isinstance(svals, torch.Tensor) else svals
    plt.figure(figsize=(5, 3.5))
    plt.plot(range(1, len(values) + 1), values, marker="o", linewidth=1.5)
    plt.xlabel("rank")
    plt.ylabel("singular value")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_head_metric_bars(values, save_path, ylabel, title=None):
    ensure_dir(Path(save_path).parent)
    series = values.detach().float().cpu().tolist() if isinstance(values, torch.Tensor) else list(values)
    plt.figure(figsize=(6, 3.5))
    plt.bar(range(len(series)), series)
    plt.xlabel("head")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def _load_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _fit_image(image, size):
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def plot_existing_figure_grid(
    figure_dir,
    model_names,
    layers,
    filename_template,
    save_path,
    title,
    cell_size=(520, 420),
):
    """Combine per-model/layer figures into one layer x model overview."""
    figure_dir = Path(figure_dir)
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    model_names = list(model_names)
    layers = list(layers)
    if not model_names or not layers:
        return None

    left_w = 120
    top_h = 105
    gap = 16
    title_h = 50
    width = left_w + len(model_names) * cell_size[0] + (len(model_names) + 1) * gap
    height = title_h + top_h + len(layers) * cell_size[1] + (len(layers) + 1) * gap

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(32)
    label_font = _load_font(24)
    draw.text((gap, 12), title, fill=(20, 20, 20), font=title_font)

    for col, model_name in enumerate(model_names):
        x = left_w + gap + col * (cell_size[0] + gap)
        draw.text(
            (x + cell_size[0] // 2 - 40, title_h + 35),
            str(model_name).upper(),
            fill=(20, 20, 20),
            font=label_font,
        )

    created_any = False
    for row, layer in enumerate(layers):
        y = title_h + top_h + gap + row * (cell_size[1] + gap)
        draw.text((gap, y + cell_size[1] // 2 - 12), f"L{layer}", fill=(20, 20, 20), font=label_font)
        for col, model_name in enumerate(model_names):
            x = left_w + gap + col * (cell_size[0] + gap)
            draw.rectangle([x - 1, y - 1, x + cell_size[0] + 1, y + cell_size[1] + 1], outline=(220, 220, 220))
            path = figure_dir / filename_template.format(model=model_name, layer=layer)
            if not path.exists():
                draw.text((x + 40, y + cell_size[1] // 2), "missing", fill=(160, 0, 0), font=label_font)
                continue
            canvas.paste(_fit_image(Image.open(path), cell_size), (x, y))
            created_any = True

    if not created_any:
        return None
    canvas.save(save_path, quality=95)
    return str(save_path)


def plot_phase3_combined_figures(
    figure_dir,
    model_names,
    seeds,
    layers,
    representative_heads,
    spectral_heads=None,
    local_attention_windows=None,
):
    figure_dir = Path(figure_dir)
    source_dir = figure_dir / "detail" / "phase3"
    if not source_dir.exists():
        source_dir = figure_dir
    combined_dir = figure_dir / "summary" / "phase3"
    spectral_heads = list(spectral_heads or representative_heads)
    paths = []

    for seed in seeds:
        for head in representative_heads:
            path = plot_existing_figure_grid(
                figure_dir=source_dir,
                model_names=model_names,
                layers=layers,
                filename_template=f"{{model}}_seed{seed}_layer{{layer}}_head{head}_attention_heatmap.png",
                save_path=combined_dir / f"phase3_seed{seed}_attention_heatmap_head{head}_layers_x_models.png",
                title=f"Phase 3 Attention Heatmap - seed {seed} head {head}",
                cell_size=(520, 420),
            )
            if path:
                paths.append(path)

        metric_specs = [
            ("entropy", "Attention Entropy by Head", (520, 390)),
            ("distance", "Attention Distance by Head", (520, 390)),
            ("far_mass", "Far Attention Mass by Head", (520, 390)),
            ("attention_sink_mass", "Attention Sink Mass by Head", (520, 390)),
            ("toeplitz_deviation", "Toeplitz Deviation by Head", (520, 390)),
        ]
        for window in local_attention_windows or []:
            metric_specs.append((f"local_mass_{window}", f"Local Attention Mass@{window} by Head", (520, 390)))
        for metric, display, cell_size in metric_specs:
            path = plot_existing_figure_grid(
                figure_dir=source_dir,
                model_names=model_names,
                layers=layers,
                filename_template=f"{{model}}_seed{seed}_layer{{layer}}_{metric}_by_head.png",
                save_path=combined_dir / f"phase3_seed{seed}_{metric}_by_head_layers_x_models.png",
                title=f"Phase 3 {display} - seed {seed}",
                cell_size=cell_size,
            )
            if path:
                paths.append(path)

        for head in spectral_heads:
            path = plot_existing_figure_grid(
                figure_dir=source_dir,
                model_names=model_names,
                layers=layers,
                filename_template=f"{{model}}_seed{seed}_layer{{layer}}_head{head}_singular_values.png",
                save_path=combined_dir / f"phase3_seed{seed}_singular_values_head{head}_layers_x_models.png",
                title=f"Phase 3 Singular Values - seed {seed} head {head}",
                cell_size=(520, 390),
            )
            if path:
                paths.append(path)
    return paths


def _clean_points(history, metric_key):
    points = []
    for row in history:
        if metric_key not in row:
            continue
        value = row.get(metric_key)
        step = row.get("step")
        if value is None or step is None:
            continue
        points.append((step, value))
    return points


def _plot_lines(series, save_path, title, ylabel, xlabel="step"):
    ensure_dir(Path(save_path).parent)
    plt.figure(figsize=(10, 5.5))
    for label, points in series.items():
        if not points:
            continue
        plt.plot(
            [x for x, _ in points],
            [y for _, y in points],
            marker="o" if len(points) <= 20 else None,
            linewidth=1.6,
            alpha=0.85,
            label=label,
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    return str(save_path)


def _plot_grouped_layer_metric(rows, metric_key, save_path, title, ylabel):
    ensure_dir(Path(save_path).parent)
    models = list(dict.fromkeys(row["model_name"] for row in rows))
    layers = sorted({int(row["layer"]) for row in rows})
    by_model = {model: [] for model in models}
    for model in models:
        for layer in layers:
            values = [
                row.get(metric_key)
                for row in rows
                if row["model_name"] == model and int(row["layer"]) == layer and row.get(metric_key) is not None
            ]
            by_model[model].append(sum(values) / len(values) if values else None)
    plt.figure(figsize=(9, 5))
    for model, values in by_model.items():
        xs = [layer for layer, value in zip(layers, values) if value is not None]
        ys = [value for value in values if value is not None]
        if ys:
            plt.plot(xs, ys, marker="o", linewidth=2, label=model)
    plt.xlabel("layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    return str(save_path)


def plot_phase2_summary_figures(raw_metrics_dir, figure_dir):
    raw_metrics_dir = Path(raw_metrics_dir)
    summary_dir = Path(figure_dir) / "summary" / "phase2"
    train_res_path = raw_metrics_dir / "train_res.json"
    summary_path = raw_metrics_dir / "phase2_summary.json"
    paths = []
    if train_res_path.exists():
        train_res = load_json(train_res_path)
        for metric, title, ylabel in [
            ("train_loss", "Phase 2 Training Loss", "training loss"),
            ("valid_loss", "Phase 2 Validation Loss", "validation loss"),
        ]:
            series = {}
            for model_name, seed_items in train_res.items():
                for seed, item in seed_items.items():
                    history = item.get("train_state", {}).get("history", [])
                    series[f"{model_name} seed {seed}"] = _clean_points(history, metric)
            paths.append(
                _plot_lines(
                    series,
                    summary_dir / f"phase2_{metric}_all_models_seeds.png",
                    title=title,
                    ylabel=ylabel,
                )
            )
    if summary_path.exists():
        summary = load_json(summary_path)
        rows = []
        for model_name, metrics in summary.get("by_model", {}).items():
            row = {"model_name": model_name}
            for metric_name, stats in metrics.items():
                row[f"{metric_name}_mean"] = stats.get("mean")
            rows.append(row)
        metric_specs = [
            ("best_valid_loss_mean", "Best Validation Loss", "loss"),
            ("final_valid_loss_mean", "Final Validation Loss", "loss"),
            ("final_perplexity_mean", "Final Perplexity", "perplexity"),
            ("loss_spike_count_mean", "Loss Spike Count", "count"),
            ("final_generalization_gap_mean", "Final Generalization Gap", "loss gap"),
            ("best_final_valid_gap_mean", "Best-Final Validation Gap", "loss gap"),
            ("valid_loss_auc_mean", "Validation Loss AUC", "loss"),
            ("step_to_50pct_valid_improvement_mean", "Step to 50% Validation Improvement", "step"),
            ("step_to_90pct_valid_improvement_mean", "Step to 90% Validation Improvement", "step"),
        ]
        for metric_key, title, ylabel in metric_specs:
            values = [(row["model_name"], row.get(metric_key)) for row in rows if row.get(metric_key) is not None]
            if not values:
                continue
            ensure_dir(summary_dir)
            plt.figure(figsize=(7, 4.5))
            plt.bar([x for x, _ in values], [y for _, y in values])
            plt.ylabel(ylabel)
            plt.title(f"Phase 2 {title}")
            plt.tight_layout()
            path = summary_dir / f"phase2_{metric_key}.png"
            plt.savefig(path, dpi=180)
            plt.close()
            paths.append(str(path))
    return paths


def plot_phase4a_summary_figures(raw_metrics_dir, figure_dir):
    raw_metrics_dir = Path(raw_metrics_dir)
    summary_dir = Path(figure_dir) / "summary" / "phase4a"
    summary_path = raw_metrics_dir / "phase4a_summary.json"
    sae_res_path = raw_metrics_dir / "sae_res.json"
    paths = []
    if sae_res_path.exists():
        sae_res = load_json(sae_res_path)
        histories = {}
        for model_name, seed_items in sae_res.items():
            for model_seed, layer_items in seed_items.items():
                for layer, items in layer_items.items():
                    for item in items:
                        meta = item.get("meta", {})
                        label = f"{model_name} seed {model_seed} L{layer}"
                        dict_size = meta.get("dict_size")
                        if isinstance(dict_size, list) and len(dict_size) > 1:
                            label += f" dict {meta.get('dict_size')}"
                        history = item.get("metrics", {}).get("history", [])
                        if history:
                            histories[label] = history
        for metric, title, ylabel in [
            ("train_mse", "Phase 4a SAE Train MSE", "MSE"),
            ("valid_mse", "Phase 4a SAE Validation MSE", "MSE"),
            ("explained_variance", "Phase 4a SAE Explained Variance", "explained variance"),
            ("dead_feature_rate", "Phase 4a SAE Dead Feature Rate", "dead feature rate"),
        ]:
            series = {label: _clean_points(history, metric) for label, history in histories.items()}
            if any(series.values()):
                paths.append(_plot_lines(series, summary_dir / f"phase4a_{metric}_curves.png", title, ylabel))
    if summary_path.exists():
        summary = load_json(summary_path)
        rows = summary.get("summary_rows", [])
        for metric, title, ylabel in [
            ("validation_mse_mean", "Phase 4a Validation MSE by Layer", "validation MSE"),
            ("explained_variance_mean", "Phase 4a Explained Variance by Layer", "explained variance"),
            ("dead_feature_rate_mean", "Phase 4a Dead Feature Rate by Layer", "dead feature rate"),
            ("feature_reuse_rate_mean", "Phase 4a Feature Reuse Rate by Layer", "reuse rate"),
            ("top_feature_activation_frequency_mean", "Phase 4a Top Feature Activation Frequency by Layer", "frequency"),
            ("feature_frequency_entropy_normalized_mean", "Phase 4a Feature Frequency Entropy by Layer", "normalized entropy"),
        ]:
            if any(row.get(metric) is not None for row in rows):
                paths.append(_plot_grouped_layer_metric(rows, metric, summary_dir / f"phase4a_{metric}.png", title, ylabel))
    return paths


def plot_phase5_summary_figures(raw_metrics_dir, figure_dir):
    raw_metrics_dir = Path(raw_metrics_dir)
    summary_dir = Path(figure_dir) / "summary" / "phase5"
    summary_path = raw_metrics_dir / "phase5_summary.json"
    if not summary_path.exists():
        return []
    summary = load_json(summary_path)
    rows = summary.get("summary_rows", [])
    paths = []
    for metric, title, ylabel in [
        ("raw_position_r2_mean", "Phase 5 Raw Position R2", "R2"),
        ("raw_position_r2_baseline_margin_mean", "Phase 5 Raw Position R2 Baseline Margin", "R2 margin"),
        ("raw_position_bin_accuracy_mean", "Phase 5 Raw Position Bin Accuracy", "accuracy"),
        ("raw_position_bin_baseline_margin_mean", "Phase 5 Raw Position Bin Baseline Margin", "accuracy margin"),
        ("raw_segment_position_accuracy_mean", "Phase 5 Raw Segment Position Accuracy", "accuracy"),
        ("raw_segment_position_baseline_margin_mean", "Phase 5 Raw Segment Position Baseline Margin", "accuracy margin"),
        ("raw_token_category_accuracy_mean", "Phase 5 Raw Token Category Accuracy", "accuracy"),
        ("raw_token_category_baseline_margin_mean", "Phase 5 Raw Token Category Baseline Margin", "accuracy margin"),
        ("raw_token_frequency_bin_accuracy_mean", "Phase 5 Raw Token Frequency Bin Accuracy", "accuracy"),
        ("raw_token_frequency_bin_baseline_margin_mean", "Phase 5 Raw Token Frequency Baseline Margin", "accuracy margin"),
        ("raw_top_token_identity_accuracy_mean", "Phase 5 Raw Top Token Identity Accuracy", "accuracy"),
        ("raw_top_token_identity_baseline_margin_mean", "Phase 5 Raw Top Token Identity Baseline Margin", "accuracy margin"),
        ("sae_position_bin_accuracy_mean", "Phase 5 SAE Position Bin Accuracy", "accuracy"),
        ("sae_position_bin_baseline_margin_mean", "Phase 5 SAE Position Bin Baseline Margin", "accuracy margin"),
        ("sae_segment_position_accuracy_mean", "Phase 5 SAE Segment Position Accuracy", "accuracy"),
        ("sae_segment_position_baseline_margin_mean", "Phase 5 SAE Segment Position Baseline Margin", "accuracy margin"),
        ("sae_token_category_accuracy_mean", "Phase 5 SAE Token Category Accuracy", "accuracy"),
        ("sae_token_category_baseline_margin_mean", "Phase 5 SAE Token Category Baseline Margin", "accuracy margin"),
        ("sae_token_frequency_bin_accuracy_mean", "Phase 5 SAE Token Frequency Bin Accuracy", "accuracy"),
        ("sae_token_frequency_bin_baseline_margin_mean", "Phase 5 SAE Token Frequency Baseline Margin", "accuracy margin"),
        ("sae_top_token_identity_accuracy_mean", "Phase 5 SAE Top Token Identity Accuracy", "accuracy"),
        ("sae_top_token_identity_baseline_margin_mean", "Phase 5 SAE Top Token Identity Baseline Margin", "accuracy margin"),
        ("mixed_feature_ratio_mean", "Phase 5 Mixed Feature Ratio", "ratio"),
        ("content_only_ratio_mean", "Phase 5 Content-Only Feature Ratio", "ratio"),
        ("position_only_ratio_mean", "Phase 5 Position-Only Feature Ratio", "ratio"),
    ]:
        if any(row.get(metric) is not None for row in rows):
            paths.append(_plot_grouped_layer_metric(rows, metric, summary_dir / f"phase5_{metric}.png", title, ylabel))
    return paths


def plot_sae_training_curves(history, save_path, title=None):
    ensure_dir(Path(save_path).parent)
    steps = [row["step"] for row in history]
    plt.figure(figsize=(7, 4))
    if any("train_mse" in row for row in history):
        plt.plot(steps, [row.get("train_mse") for row in history], label="train mse")
    if any("valid_mse" in row for row in history):
        plt.plot(steps, [row.get("valid_mse") for row in history], label="valid mse")
    plt.xlabel("step")
    plt.ylabel("mse")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_sae_health_curves(history, save_path, title=None):
    ensure_dir(Path(save_path).parent)
    steps = [row["step"] for row in history]
    plt.figure(figsize=(7, 4))
    if any("explained_variance" in row for row in history):
        plt.plot(steps, [row.get("explained_variance") for row in history], label="explained variance")
    if any("dead_feature_rate" in row for row in history):
        plt.plot(steps, [row.get("dead_feature_rate") for row in history], label="dead feature rate")
    plt.xlabel("step")
    plt.ylabel("rate")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
