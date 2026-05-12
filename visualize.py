from pathlib import Path

import matplotlib.pyplot as plt
import torch

from utils import ensure_dir


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
