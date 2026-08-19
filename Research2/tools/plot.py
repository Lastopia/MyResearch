import json
from pathlib import Path

import matplotlib.pyplot as plt

from tools.io import ensure_dir


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def plot_loss_curves(out_dir, aliases, seeds=None):
    out_dir = Path(out_dir)
    eval_dir = out_dir / "eval"
    ensure_dir(eval_dir)
    plt.figure(figsize=(9, 5))
    plotted = False
    seeds = seeds or [None]
    for alias in aliases:
        for seed in seeds:
            suffix = "" if seed is None else f"seed{seed}"
            label = alias if seed is None else f"{alias} s{seed}"
            train_rows = read_jsonl(out_dir / "metrics" / f"[{alias}]{suffix}train.jsonl")
            valid_rows = read_jsonl(out_dir / "metrics" / f"[{alias}]{suffix}valid.jsonl")
            if not train_rows and seed is not None:
                train_rows = read_jsonl(out_dir / "metrics" / f"[{alias}]train.jsonl")
                valid_rows = read_jsonl(out_dir / "metrics" / f"[{alias}]valid.jsonl")
                label = alias
            if train_rows:
                plt.plot([r["step"] for r in train_rows], [r["train_loss"] for r in train_rows], label=f"{label} train")
                plotted = True
            if valid_rows:
                plt.plot([r["step"] for r in valid_rows], [r["valid_loss"] for r in valid_rows], linestyle="--", label=f"{label} valid")
                plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(eval_dir / "loss_curve.png", dpi=160)
    plt.close()


def plot_metric_bars(out_dir, rows, metric, filename):
    out_dir = Path(out_dir)
    eval_dir = out_dir / "eval"
    ensure_dir(eval_dir)
    eligible = [row for row in rows if metric in row and isinstance(row[metric], (int, float))]
    grouped = {}
    for row in eligible:
        parts = [str(row["model"])]
        if "model_seed" in row:
            parts.append(f"ms{row['model_seed']}")
        elif "seed" in row:
            parts.append(f"s{row['seed']}")
        if "layer" in row:
            parts.append(f"l{row['layer']}")
        if "expansion" in row:
            parts.append(f"e{row['expansion']}x")
        if "k" in row:
            parts.append(f"k{row['k']}")
        if "sae_seed" in row:
            parts.append(f"ss{row['sae_seed']}")
        grouped.setdefault(" ".join(parts), []).append(float(row[metric]))
    labels = list(grouped)
    values = [sum(grouped[label]) / len(grouped[label]) for label in labels]
    if not values:
        return
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(eval_dir / filename, dpi=160)
    plt.close()


def plot_metric_by_k(out_dir, rows, metric, filename):
    """Plot SAE sweeps as curves, keeping k as an explicit axis."""
    out_dir = Path(out_dir)
    eval_dir = out_dir / "eval"
    ensure_dir(eval_dir)
    eligible = [
        row for row in rows
        if isinstance(row.get("k"), (int, float))
        and isinstance(row.get(metric), (int, float))
    ]
    if not eligible:
        return
    groups = {}
    for row in eligible:
        key = (
            row.get("model"), row.get("model_seed"), row.get("layer"),
            row.get("expansion"), row.get("sae_seed"),
        )
        groups.setdefault(key, []).append(row)
    plt.figure(figsize=(8, 5))
    all_k = set()
    for key, group in groups.items():
        ordered = sorted(group, key=lambda row: float(row["k"]))
        x = [row["k"] for row in ordered]
        y = [row[metric] for row in ordered]
        all_k.update(x)
        model, model_seed, layer, expansion, sae_seed = key
        details = []
        if model_seed is not None:
            details.append(f"ms{model_seed}")
        if layer is not None:
            details.append(f"l{layer}")
        if expansion is not None:
            details.append(f"e{expansion}x")
        if sae_seed is not None:
            details.append(f"ss{sae_seed}")
        label = str(model) if not details else f"{model} {' '.join(details)}"
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel("SAE k (actual L0 for TopK SAE)")
    plt.ylabel(metric)
    plt.xticks(sorted(all_k))
    if len(groups) > 1:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(eval_dir / filename, dpi=160)
    plt.close()
