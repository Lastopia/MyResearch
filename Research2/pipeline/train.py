import math
import json
import re

import numpy as np
import torch

from model.factory import active_aliases, build_model, checkpoint_signature, experiment_cfg, is_pretrain_mode, resolve_experiment, validate_aliases
from tools.io import block_dir, checkpoint_dir, ensure_dir, output_dir, read_json, save_config, write_json
from tools.log import elapsed_seconds, event_line, now, stage_title, train_line
from tools.plot import plot_loss_curves
from tools.resource import configured_gpus, cuda_device, gpu_label, mem_usage, run_gpu_jobs, vram_usage


class BlockData:
    def __init__(self, path, n_blocks, block_size):
        self.n_blocks = n_blocks
        self.block_size = block_size
        if not path.exists():
            raise FileNotFoundError(f"missing block file: {path}. Run `run data` first.")
        meta_path = path.parent / "meta.json"
        dtype = read_json(meta_path).get("token_dtype", "uint16") if meta_path.exists() else "uint16"
        self.data = np.memmap(path, dtype=np.dtype(dtype), mode="r", shape=(n_blocks, block_size + 1))

    def batch(self, batch_size, device):
        idx = np.random.randint(0, self.n_blocks, size=batch_size)
        return self.batch_indices(idx, device)

    def batch_indices(self, indices, device):
        idx = np.asarray(indices, dtype=np.int64)
        rows = torch.from_numpy(np.asarray(self.data[idx], dtype=np.int64)).to(device)
        return rows[:, :-1], rows[:, 1:]

    def deterministic_batch(self, batch_size, device, seed, batch_index=0):
        rng = np.random.default_rng(int(seed) + int(batch_index) * 1000003)
        idx = rng.integers(0, self.n_blocks, size=int(batch_size), endpoint=False)
        return self.batch_indices(idx, device)


def lr_at(step, cfg):
    if cfg["lr_schedule"] == "constant":
        return cfg["lr"]
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    ratio = min(1.0, (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"]))
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg["min_lr"] + coeff * (cfg["lr"] - cfg["min_lr"])


def run_seeds(cfg):
    seed = cfg["run"].get("seed", [42])
    if isinstance(seed, list):
        return [int(s) for s in seed]
    return [int(seed)]


def checkpoint_path(ckpt_dir, alias, seed, name):
    return ckpt_dir / f"[{alias}]seed{seed}{name}.pt"


def load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def latest_checkpoint(ckpt_dir, alias, seed=None, prefer_best=True):
    if seed is not None and prefer_best:
        best = checkpoint_path(ckpt_dir, alias, seed, "best")
        if best.exists():
            return best
    seed_part = r"\d+" if seed is None else str(seed)
    pattern = re.compile(rf"^\[{re.escape(alias)}\]seed({seed_part})step(\d+)\.pt$")
    found = []
    for path in ckpt_dir.glob("*.pt"):
        match = pattern.match(path.name)
        if match:
            found.append((int(match.group(2)), path))
    if found:
        return max(found, default=(None, None))[1]
    if seed is None:
        old_pattern = re.compile(rf"^\[{re.escape(alias)}\]step(\d+)\.pt$")
        for path in ckpt_dir.glob("*.pt"):
            match = old_pattern.match(path.name)
            if match:
                found.append((int(match.group(1)), path))
    return max(found, default=(None, None))[1]


def checkpoint_step(path):
    state = load_checkpoint(path, map_location="cpu")
    if "step" in state:
        return int(state["step"])
    match = re.search(r"step(\d+)\.pt$", path.name)
    return 0 if match is None else int(match.group(1))


def checkpoint_compatible(path, cfg, alias):
    try:
        state = load_checkpoint(path, map_location="cpu")
        saved_alias = state.get("alias")
        if saved_alias is not None and saved_alias != alias:
            raise ValueError(f"checkpoint alias changed: {saved_alias} != {alias}")
        signature = state.get("signature")
        if signature is not None:
            if signature != checkpoint_signature(cfg, alias):
                raise ValueError("checkpoint signature changed")
            # The signature already contains the resolved model, variant, and
            # position encoding. Do not compare the raw saved cfg again: older
            # multi-position runs saved the global default position there.
            model = build_model(cfg, alias)
            model.load_state_dict(state["model"])
            return True
        if is_pretrain_mode(cfg):
            old_cfg = state.get("cfg", {})
            if old_cfg.get("run", {}).get("mode", "retrain") != "pretrain":
                raise ValueError("checkpoint mode changed")
            old_entry = old_cfg.get("premodel", {}).get("models", {}).get(alias, {})
            new_entry = cfg["premodel"]["models"][alias]
            for key in ("hf_id", "tokenizer_id", "revision"):
                if old_entry.get(key) is not None and old_entry.get(key) != new_entry.get(key):
                    raise ValueError(f"premodel {key} changed")
            if "model" not in state:
                raise ValueError("checkpoint has no model state")
            return True
        old_cfg = state.get("cfg", {})
        exp = resolve_experiment(cfg, alias)
        exp_cfg = experiment_cfg(cfg, alias)
        model_alias = exp["model_alias"]
        old_model_cfg = old_cfg.get("model")
        old_variant = old_cfg.get("models", {}).get(model_alias)
        if old_model_cfg is not None:
            # Legacy checkpoints stored the global default (usually RoPE)
            # even when the alias was a Cable experiment. The alias and the
            # state-dict parameter structure disambiguate the actual model.
            old_model_cfg = dict(old_model_cfg)
            old_model_cfg["position_encoding"] = exp["position_encoding"]
        if old_model_cfg is not None and old_model_cfg != exp_cfg["model"]:
            raise ValueError("model config changed")
        if old_variant is not None and old_variant != exp_cfg["models"][model_alias]:
            raise ValueError("model variant changed")
        model = build_model(cfg, alias)
        model.load_state_dict(state["model"])
        return True
    except Exception as exc:
        event_line("train", alias=alias, event="checkpoint incompatible", detail=f"{type(exc).__name__}: {exc}")
        return False


def move_optimizer_state(opt, device):
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def training_checkpoint_state(
    cfg, alias, seed, step, model, optimizer, scaler, elapsed_training_seconds,
    best_valid_ce_loss=None,
):
    """Build a checkpoint using the alias-resolved experiment configuration."""
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "cfg": experiment_cfg(cfg, alias),
        "signature": checkpoint_signature(cfg, alias),
        "step": int(step),
        "alias": alias,
        "seed": seed,
        "elapsed_training_seconds": max(0, int(elapsed_training_seconds)),
    }
    if best_valid_ce_loss is not None:
        state["best_valid_ce_loss"] = float(best_valid_ce_loss)
    return state


def accumulated_metric_time(path, through_step=None):
    """Recover cumulative time from old JSONL logs that reset on resume."""
    if not path.exists():
        return 0
    completed_sessions = 0
    last_session_time = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                row_step = int(row.get("step", 0))
                current = max(0, int(row.get("time", 0)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if through_step is not None and row_step > int(through_step):
                continue
            if current < last_session_time:
                completed_sessions += last_session_time
            last_session_time = current
    return completed_sessions + last_session_time


def checkpoint_elapsed_time(state, metrics_path=None, through_step=None):
    """Read cumulative training seconds with backward-compatible fallback."""
    stored = state.get("elapsed_training_seconds")
    if stored is not None:
        return max(0, int(stored))
    if metrics_path is not None:
        return accumulated_metric_time(metrics_path, through_step)
    return 0


def forward_training_loss(model, x, y):
    """Run the loss path without collecting diagnostic FFN reductions.

    CE models only need logits and loss during ordinary optimization.  Asking
    every block for L0/L1/density statistics on every step adds several large
    reductions, especially for sparse models.  L1-regularized experiments keep
    the old auxiliary-statistics path because those activations are part of the
    objective; the dedicated ``run ffn`` stage still collects diagnostics for
    every model.
    """
    loss_name = getattr(model, "loss_cfg", {}).get("name", "ce")
    if loss_name == "l1_act":
        _, loss, _, _, _ = model(x, y, return_aux=True)
        return loss
    _, loss = model(x, y)
    return loss


@torch.no_grad()
def eval_loss(model, valid_data, cfg, device, dtype):
    model.eval()
    losses = []
    ce_losses = []
    eval_seed = int(cfg.get("eval_seed", 424242))
    for batch_index in range(cfg["eval_batches"]):
        x, y = valid_data.deterministic_batch(
            cfg["batch_size"], device, eval_seed, batch_index,
        )
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            loss = forward_training_loss(model, x, y)
        aux = getattr(model, "last_aux", {})
        losses.append(loss.item())
        ce_losses.append(aux.get("ce_loss", loss).item())
    model.train()
    return {
        "valid_loss": sum(losses) / len(losses),
        "valid_ce_loss": sum(ce_losses) / len(ce_losses),
    }


def train_one(cfg, alias, seed, gpu_index):
    validate_aliases(cfg, [alias])
    if cfg["run"].get("num_threads"):
        torch.set_num_threads(cfg["run"]["num_threads"])
    out = output_dir(cfg)
    ckpt_dir = checkpoint_dir(cfg)
    metrics_dir = out / "metrics"
    ensure_dir(ckpt_dir)
    ensure_dir(metrics_dir)
    train_metrics = metrics_dir / f"[{alias}]seed{seed}train.jsonl"
    valid_metrics = metrics_dir / f"[{alias}]seed{seed}valid.jsonl"

    existing = latest_checkpoint(ckpt_dir, alias, seed, prefer_best=False)
    resume_state = None
    start_step = 0
    elapsed_before_resume = 0
    if existing and not cfg["train"]["retrain"]:
        if checkpoint_compatible(existing, cfg, alias):
            ckpt_step = checkpoint_step(existing)
            existing_state = load_checkpoint(existing, map_location="cpu")
            elapsed_before_resume = checkpoint_elapsed_time(existing_state, train_metrics, ckpt_step)
            if ckpt_step >= cfg["train"]["max_steps"]:
                event_line("train", gpu_label(gpu_index), alias, seed, "checkpoint exists", f"step={ckpt_step} >= max_steps={cfg['train']['max_steps']}")
                return {
                    "model": alias,
                    "seed": seed,
                    "skipped": True,
                    "checkpoint": str(existing),
                    "step": ckpt_step,
                    "time": elapsed_before_resume,
                }
            resume_state = existing_state
            start_step = ckpt_step
            event_line(
                "train", gpu_label(gpu_index), alias, seed, "resume",
                f"step={start_step} -> max_steps={cfg['train']['max_steps']} | elapsed={elapsed_before_resume}s",
            )
        else:
            event_line("train", gpu_label(gpu_index), alias, seed, "retrain", "checkpoint incompatible")

    if torch.cuda.is_available():
        device = torch.device(cuda_device(gpu_index))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    dtype = torch.float16 if cfg["run"]["dtype"] == "float16" else torch.bfloat16
    model = build_model(cfg, alias).to(device)
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and dtype == torch.float16))
    if resume_state is not None and "optimizer" in resume_state:
        opt.load_state_dict(resume_state["optimizer"])
        move_optimizer_state(opt, device)
    if resume_state is not None and "scaler" in resume_state:
        scaler.load_state_dict(resume_state["scaler"])

    blocks_path = block_dir(cfg, alias)
    train_data = BlockData(blocks_path / "train.bin", cfg["data"]["train_blocks"], cfg["data"]["block_size"])
    valid_data = BlockData(blocks_path / "valid.bin", cfg["data"]["valid_blocks"], cfg["data"]["block_size"])
    start = now()
    log_path = ckpt_dir / "train.log"
    if start_step == 0:
        for stale_metrics in (train_metrics, valid_metrics):
            if stale_metrics.exists():
                stale_metrics.unlink()
    last_log_seconds = elapsed_before_resume
    best_valid_ce = None
    best_step = None
    best_ckpt = checkpoint_path(ckpt_dir, alias, seed, "best")
    if best_ckpt.exists() and not cfg["train"]["retrain"]:
        try:
            best_state = load_checkpoint(best_ckpt, map_location="cpu")
            best_valid_ce = best_state.get("best_valid_ce_loss")
            best_step = best_state.get("step")
        except Exception:
            best_valid_ce = None
            best_step = None

    for step in range(start_step + 1, cfg["train"]["max_steps"] + 1):
        lr = lr_at(step, cfg["train"])
        for group in opt.param_groups:
            group["lr"] = lr
        accumulation_steps = max(1, int(cfg["train"].get("gradient_accumulation_steps", 1)))
        opt.zero_grad(set_to_none=True)
        accumulated_loss = torch.zeros((), device=device)
        accumulated_aux = {}
        for micro_step in range(accumulation_steps):
            x, y = train_data.batch(cfg["train"]["batch_size"], device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                micro_loss = forward_training_loss(model, x, y)
            micro_aux = getattr(model, "last_aux", {})
            if not torch.isfinite(micro_loss):
                raise RuntimeError(
                    f"non-finite train loss: model={alias}, step={step}, "
                    f"micro_step={micro_step}, loss={micro_loss.item()}"
                )
            accumulated_loss += micro_loss.detach()
            for key, value in micro_aux.items():
                accumulated_aux[key] = accumulated_aux.get(
                    key, torch.zeros((), device=value.device, dtype=value.dtype),
                ) + value.detach()
            scaler.scale(micro_loss / accumulation_steps).backward()
        loss = accumulated_loss / accumulation_steps
        aux = {key: value / accumulation_steps for key, value in accumulated_aux.items()}
        grad_clip = cfg["train"].get("grad_clip", 0.0)
        if grad_clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()

        valid = None
        if step % cfg["train"]["eval_interval"] == 0:
            valid = eval_loss(model, valid_data, cfg["train"], device, dtype)
            total_seconds = elapsed_before_resume + elapsed_seconds(start)
            with valid_metrics.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "model": alias,
                    "seed": seed,
                    "gpu": gpu_label(gpu_index),
                    "step": step,
                    "eval_seed": int(cfg["train"].get("eval_seed", 424242)),
                    "evaluation_batches": int(cfg["train"]["eval_batches"]),
                    "valid_loss": round(valid["valid_loss"], 6),
                    "valid_ce_loss": round(valid["valid_ce_loss"], 6),
                    "time": total_seconds,
                }, ensure_ascii=False) + "\n")
            if best_valid_ce is None or valid["valid_ce_loss"] < best_valid_ce:
                best_valid_ce = valid["valid_ce_loss"]
                best_step = step
                torch.save(
                    training_checkpoint_state(
                        cfg, alias, seed, step, model, opt, scaler, total_seconds,
                        best_valid_ce_loss=best_valid_ce,
                    ),
                    checkpoint_path(ckpt_dir, alias, seed, "best"),
                )
                event_line("train", gpu_label(gpu_index), alias, seed, f"best_step={step}", f"best_valid_ce_loss={best_valid_ce:.2f}")

        if step % cfg["train"]["log_interval"] == 0 or valid is not None:
            total_seconds = elapsed_before_resume + elapsed_seconds(start)
            delta_seconds = total_seconds - last_log_seconds
            last_log_seconds = total_seconds
            line = {
                "step": step,
                "train_loss": round(loss.item(), 6),
                "train_ce_loss": None if "ce_loss" not in aux else round(aux["ce_loss"].item(), 6),
                "aux_loss": None if "aux_loss" not in aux else round(aux["aux_loss"].item(), 6),
                "l1_act": None if "l1_act" not in aux else round(aux["l1_act"].item(), 6),
                "act_l0": None if "act_l0" not in aux else round(aux["act_l0"].item(), 6),
                "act_density": None if "act_density" not in aux else round(aux["act_density"].item(), 6),
                "group_utilization_std": None if "group_utilization_std" not in aux else round(aux["group_utilization_std"].item(), 6),
                "pre_sparse_l1": None if "pre_sparse_l1" not in aux else round(aux["pre_sparse_l1"].item(), 6),
                "valid_loss": None if valid is None else round(valid["valid_loss"], 6),
                "valid_ce_loss": None if valid is None else round(valid["valid_ce_loss"], 6),
                "time": total_seconds,
                "time_delta": delta_seconds,
                "model": alias,
                "seed": seed,
                "gpu": gpu_label(gpu_index),
                "lr": lr,
                "micro_batch_size": int(cfg["train"]["batch_size"]),
                "gradient_accumulation_steps": accumulation_steps,
                "tokens_per_step": int(cfg["train"]["batch_size"])
                * accumulation_steps * int(cfg["data"]["block_size"]),
            }
            valid_for_log = None if valid is None else valid["valid_ce_loss"]
            train_line("train", gpu_label(gpu_index), f"{alias} s{seed}", step, loss.item(), valid_for_log, delta_seconds, total_seconds, mem_usage(), vram_usage(device))
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            with train_metrics.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        if step % cfg["train"]["save_interval"] == 0:
            checkpoint_time = elapsed_before_resume + elapsed_seconds(start)
            torch.save(
                training_checkpoint_state(
                    cfg, alias, seed, step, model, opt, scaler, checkpoint_time,
                ),
                checkpoint_path(ckpt_dir, alias, seed, f"step{step}"),
            )

    final_path = checkpoint_path(ckpt_dir, alias, seed, f"step{cfg['train']['max_steps']}")
    if not final_path.exists():
        checkpoint_time = elapsed_before_resume + elapsed_seconds(start)
        torch.save(
            training_checkpoint_state(
                cfg, alias, seed, cfg["train"]["max_steps"], model, opt, scaler, checkpoint_time,
            ),
            final_path,
        )

    total_training_seconds = elapsed_before_resume + elapsed_seconds(start)
    write_json(metrics_dir / f"[{alias}]seed{seed}summary.json", {
        "model": alias,
        "seed": seed,
        "steps": cfg["train"]["max_steps"],
        "time": total_training_seconds,
        "last_train_loss": round(loss.item(), 6),
        "last_train_ce_loss": None if "ce_loss" not in getattr(model, "last_aux", {}) else round(model.last_aux["ce_loss"].item(), 6),
        "best_step": best_step,
        "best_valid_ce_loss": None if best_valid_ce is None else round(best_valid_ce, 6),
    })
    return {
        "model": alias,
        "seed": seed,
        "steps": cfg["train"]["max_steps"],
        "time": total_training_seconds,
        "last_train_loss": round(loss.item(), 6),
        "last_train_ce_loss": None if "ce_loss" not in getattr(model, "last_aux", {}) else round(model.last_aux["ce_loss"].item(), 6),
        "best_step": best_step,
        "best_valid_ce_loss": None if best_valid_ce is None else round(best_valid_ce, 6),
    }


def warn_block_size_mismatch(cfg):
    train_block = cfg["data"]["block_size"]
    if is_pretrain_mode(cfg):
        for alias in active_aliases(cfg):
            max_pos = cfg["premodel"]["models"][alias].get("max_position_embeddings")
            if max_pos and train_block > max_pos:
                print(f"[train] block_size exceeds premodel context | {alias} | block_size={train_block} | max_position_embeddings={max_pos}")
        return
    model_block = cfg["model"]["block_size"]
    if train_block != model_block:
        print(f"[train] block_size mismatch | train_block={train_block} | model_block={model_block}")


def run(cfg):
    stage_title("train")
    warn_block_size_mismatch(cfg)
    aliases = active_aliases(cfg)
    seeds = run_seeds(cfg)
    jobs = [(alias, seed) for seed in seeds for alias in aliases]
    validate_aliases(cfg, aliases)
    gpus = configured_gpus(cfg)
    metrics_dir = output_dir(cfg) / "metrics"
    ensure_dir(metrics_dir)
    failed = run_gpu_jobs(
        cfg,
        jobs,
        train_one,
        lambda job, gpu_index: (cfg, job[0], job[1], gpu_index),
        gpus,
        stage="train",
    )
    summaries = []
    for alias, seed in jobs:
        summary_path = metrics_dir / f"[{alias}]seed{seed}summary.json"
        if summary_path.exists():
            import json
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    write_json(metrics_dir / "summary.json", {"models": summaries})
    plot_loss_curves(output_dir(cfg), aliases, seeds)
    save_config(cfg)
    if failed:
        raise RuntimeError(f"train subprocess failed: {failed}")
