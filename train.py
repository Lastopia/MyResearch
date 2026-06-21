import math
import os
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from logger import ExperimentLogger
from para import PATH
from phase3_attention import Phase3AttentionAnalyzer
from utils import (
    ensure_dir,
    get_device,
    load_json,
    manifest_is_current,
    mean_std,
    perplexity,
    save_json,
    save_manifest,
    set_seed,
    valid_file,
    write_csv,
)
from model import GPTLikeTransformer
from visualize import (
    plot_loss_curve,
    plot_metric_curves,
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
        self.phase3 = Phase3AttentionAnalyzer(
            train_cfg,
            self.model_cfg,
            valid_loader=lambda: self.analysis_loader("valid", shuffle=False),
        )

    def loader(self, split, shuffle):
        return DataLoader(
            self.data_res[split],
            batch_size=self.train_cfg.batch_size,
            shuffle=shuffle,
            drop_last=True,
        )

    def analysis_loader(self, split, shuffle):
        return DataLoader(
            self.data_res[split],
            batch_size=getattr(self.train_cfg, "analysis_batch_size", self.train_cfg.batch_size),
            shuffle=shuffle,
            drop_last=True,
        )

    def optimizer(self, model):
        return AdamW(model.parameters(), lr=self.train_cfg.lr, weight_decay=self.train_cfg.weight_decay)

    def lr_scale(self, step):
        warmup_steps = max(getattr(self.train_cfg, "warmup_steps", 0), 0)
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps

        schedule = getattr(self.train_cfg, "lr_schedule", "constant")
        if schedule in {None, "constant"}:
            return 1.0
        if schedule != "cosine":
            raise ValueError(f"Unsupported lr_schedule: {schedule}")

        decay_steps = max(self.train_cfg.steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
        min_ratio = float(getattr(self.train_cfg, "min_lr_ratio", 0.0))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    def learning_rate(self, step):
        return self.train_cfg.lr * self.lr_scale(step)

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
        self.logger.log_stage_start("load completed train results")
        serializable = load_json(Path(PATH.raw_metrics_dir) / "train_res.json")
        train_res = {}
        for model_name, seed_items in serializable.items():
            train_res[model_name] = {}
            for seed_text, item in seed_items.items():
                seed = int(seed_text)
                state = item["train_state"]
                checkpoint_path = state.get("checkpoint_path")
                if not checkpoint_path or not valid_file(checkpoint_path):
                    return None
                self.logger.write(
                    f"[load] train checkpoint model={model_name} seed={seed} path={checkpoint_path}"
                )
                model = GPTLikeTransformer(self.model_cfg, model_name).to(self.device)
                try:
                    self.load_checkpoint(checkpoint_path, model)
                except Exception as exc:
                    self.logger.log_error(exc)
                    return None
                train_res[model_name][seed] = {
                    "model": model,
                    "train_state": state,
                    "analysis_res": item.get("analysis_res", {}),
                }
        self.logger.log_stage_end("load completed train results")
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

    def attention_analysis(self, model, model_name=None, seed=None):
        return self.phase3.run(model, model_name=model_name, seed=seed)

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
        name_suffix=None,
    ):
        ensure_dir(Path(PATH.ckpt_dir) / "models")
        filename = (
            f"{model_name}_seed{seed}_{name_suffix}.pt"
            if name_suffix
            else f"{model_name}_seed{seed}_step{step}.pt"
        )
        path = Path(PATH.ckpt_dir) / "models" / filename
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
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        return str(path)

    def remove_checkpoint_file(self, checkpoint_path):
        if not checkpoint_path:
            return
        path = Path(checkpoint_path)
        try:
            resolved = path.resolve()
            ckpt_root = (Path(PATH.ckpt_dir) / "models").resolve()
            if ckpt_root not in resolved.parents:
                return
            if path.exists():
                path.unlink()
        except OSError as exc:
            self.logger.log_error(exc)

    def prune_previous_best_checkpoint(self, validation_checkpoints, new_checkpoint_path):
        if not getattr(self.train_cfg, "keep_only_latest_best_checkpoint", False):
            return validation_checkpoints
        new_path = str(new_checkpoint_path)
        keep = []
        for item in validation_checkpoints:
            checkpoint_path = item.get("checkpoint_path")
            is_old_best = item.get("selection_rule") == "best_validation"
            if is_old_best:
                if checkpoint_path != new_path:
                    self.remove_checkpoint_file(checkpoint_path)
                continue
            keep.append(item)
        return keep

    def latest_checkpoint_path(self, model_name, seed):
        ckpt_dir = Path(PATH.ckpt_dir) / "models"
        candidates = []
        for path in ckpt_dir.glob(f"{model_name}_seed{seed}_step*.pt"):
            if not valid_file(path):
                continue
            stem = path.stem
            try:
                step = int(stem.rsplit("step", 1)[1])
            except (IndexError, ValueError):
                continue
            candidates.append((step, path))
        if not candidates:
            return None
        return str(max(candidates, key=lambda item: item[0])[1])

    def final_checkpoint_path(self, model_name, seed):
        path = Path(PATH.ckpt_dir) / "models" / f"{model_name}_seed{seed}_step{self.train_cfg.steps}.pt"
        return str(path) if valid_file(path) else None

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

    def checkpoint_protocol(self):
        return {
            "primary_checkpoint_rule": getattr(self.train_cfg, "primary_checkpoint_rule", "final_step"),
            "secondary_checkpoint_rule": getattr(
                self.train_cfg,
                "secondary_checkpoint_rule",
                "validation_loss_matched",
            ),
            "validation_loss_match_target": getattr(self.train_cfg, "validation_loss_match_target", None),
            "lr_schedule": getattr(self.train_cfg, "lr_schedule", "constant"),
            "base_lr": self.train_cfg.lr,
            "min_lr_ratio": getattr(self.train_cfg, "min_lr_ratio", None),
            "warmup_steps": getattr(self.train_cfg, "warmup_steps", 0),
        }

    def final_validation_row(self, model, history):
        final_valid = None
        for row in reversed(history):
            if row.get("step") == self.train_cfg.steps and "valid_loss" in row:
                final_valid = row
                break
        return final_valid or self.validate(model)

    def build_train_result(
        self,
        model,
        model_name,
        seed,
        checkpoint_path,
        final_valid,
        best_valid,
        validation_checkpoints,
        history,
        train_losses,
        grad_norms,
        divergence_count,
        elapsed_seconds,
        analysis_reason,
    ):
        if best_valid is None or not math.isfinite(best_valid):
            valid_losses = [row["valid_loss"] for row in history if "valid_loss" in row]
            best_valid = min(valid_losses) if valid_losses else final_valid["valid_loss"]

        final_checkpoint = self.checkpoint_metadata(
            model_name,
            seed,
            self.train_cfg.steps,
            checkpoint_path,
            final_valid["valid_loss"],
            "final_step",
        )
        validation_checkpoints = [
            item for item in validation_checkpoints
            if item["checkpoint_step"] != self.train_cfg.steps
        ]
        validation_checkpoints.append(final_checkpoint)
        best_checkpoint = self.best_validation_checkpoint(validation_checkpoints)
        stability = self.stability_metrics(train_losses, grad_norms, divergence_count, elapsed_seconds)
        if self.train_cfg.run_loss_curve and getattr(self.train_cfg, "save_individual_loss_curves", False):
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
                "checkpoint_path": checkpoint_path,
                "checkpoint_step": self.train_cfg.steps,
                "tokens_seen": self.train_cfg.steps * self.train_cfg.batch_size * self.model_cfg.seq_len,
                "valid_loss_at_checkpoint": final_valid["valid_loss"],
                "selection_rule": "final_step",
                "best_checkpoint_path": best_checkpoint["checkpoint_path"] if best_checkpoint else checkpoint_path,
                "best_checkpoint_step": best_checkpoint["checkpoint_step"] if best_checkpoint else self.train_cfg.steps,
                "checkpoint_selection": {
                    "primary": final_checkpoint,
                    "best_validation": best_checkpoint,
                    "validation_candidates": validation_checkpoints,
                    "validation_loss_matched": None,
                    "protocol": self.checkpoint_protocol(),
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
            "analysis_res": self.maybe_attention_analysis(
                model,
                model_name,
                seed,
                reason=analysis_reason,
            ),
        }

    def train_state_from_loaded_checkpoint(
        self,
        model,
        model_name,
        seed,
        loaded_state,
        elapsed_seconds=0.0,
    ):
        history = loaded_state["history"]
        best_valid = loaded_state["best_valid"]
        final_valid = self.final_validation_row(model, history)
        validation_checkpoints = self.validation_candidates_from_history(model_name, seed, history)
        return self.build_train_result(
            model=model,
            model_name=model_name,
            seed=seed,
            checkpoint_path=loaded_state["path"],
            final_valid=final_valid,
            best_valid=best_valid,
            validation_checkpoints=validation_checkpoints,
            history=history,
            train_losses=loaded_state["train_losses"],
            grad_norms=loaded_state["grad_norms"],
            divergence_count=loaded_state["divergence_count"],
            elapsed_seconds=elapsed_seconds,
            analysis_reason="final_checkpoint_skip",
        )

    def validation_candidates_from_history(self, model_name, seed, history):
        candidates = []
        for row in history:
            if "valid_loss" not in row:
                continue
            step = row["step"]
            path = Path(PATH.ckpt_dir) / "models" / f"{model_name}_seed{seed}_step{step}.pt"
            if valid_file(path):
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

    def best_validation_checkpoint(self, candidates):
        valid_candidates = [
            item for item in candidates
            if item.get("valid_loss_at_checkpoint") is not None
        ]
        if not valid_candidates:
            return None
        return min(valid_candidates, key=lambda item: item["valid_loss_at_checkpoint"])

    def upsert_checkpoint_candidate(self, candidates, candidate):
        return [
            item for item in candidates
            if item["checkpoint_step"] != candidate["checkpoint_step"]
        ] + [candidate]

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

    def maybe_attention_analysis(self, model, model_name, seed, reason):
        if not getattr(self.train_cfg, "run_phase3_analysis", True):
            self.logger.log_stage_end(f"Phase 3 skipped for {model_name} seed {seed}: run_phase3_analysis=False")
            return {}
        if reason == "final_checkpoint_skip" and not getattr(
            self.train_cfg, "run_phase3_on_final_checkpoint_skip", False
        ):
            self.logger.log_stage_end(
                f"Phase 3 skipped for {model_name} seed {seed}: final checkpoint already exists"
            )
            return {}
        self.logger.log_stage_start(f"Phase 3 attention analysis {model_name} seed {seed} reason={reason}")
        analysis = self.attention_analysis(model, model_name=model_name, seed=seed)
        self.logger.log_stage_end(f"Phase 3 attention analysis {model_name} seed {seed}")
        return analysis

    def train_one_model(self, model_name, base_model, seed):
        set_seed(seed)
        model = GPTLikeTransformer(self.model_cfg, model_name).to(self.device)
        opt = self.optimizer(model)
        final_path = self.final_checkpoint_path(model_name, seed)
        if getattr(self.train_cfg, "resume_from_checkpoint", True) and final_path:
            self.logger.log_stage_start(
                f"skip training {model_name} seed {seed}: final checkpoint exists at step {self.train_cfg.steps}"
            )
            loaded_state = self.load_checkpoint(final_path, model)
            return self.train_state_from_loaded_checkpoint(
                model,
                model_name,
                seed,
                loaded_state,
                elapsed_seconds=0.0,
            )
        if getattr(self.train_cfg, "require_final_checkpoints_for_phase3", False):
            raise FileNotFoundError(
                f"Final checkpoint required for Phase 3 rerun but not found: "
                f"model={model_name} seed={seed} step={self.train_cfg.steps}"
            )
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
        latest_valid = None

        for step in range(start_step + 1, self.train_cfg.steps + 1):
            model.train()
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(self.loader("train", shuffle=True))
                x, y = next(train_iter)
            x, y = x.to(self.device), y.to(self.device)
            current_lr = self.learning_rate(step)
            for group in opt.param_groups:
                group["lr"] = current_lr
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
                row = {
                    "step": step,
                    "train_loss": loss_value,
                    "grad_norm": grad_norm_value,
                    "lr": current_lr,
                }
                history.append(row)
                self.logger.log_metric(f"{model_name}/train_loss", row["train_loss"], step)
            if step % self.train_cfg.eval_interval == 0 or step == self.train_cfg.steps:
                valid = self.validate(model)
                history.append({"step": step, **valid})
                improved = valid["valid_loss"] <= best_valid
                best_valid = min(best_valid, valid["valid_loss"])
                self.logger.log_metric(f"{model_name}/valid_loss", valid["valid_loss"], step)
                save_best = getattr(self.train_cfg, "save_best_checkpoint", True) and improved
                save_eval = getattr(self.train_cfg, "save_eval_checkpoints", True)
                validation_selection_rule = "best_validation" if improved else "validation_candidate"
                latest_valid = {
                    "step": step,
                    **valid,
                    "selection_rule": validation_selection_rule,
                }
                if save_eval or save_best:
                    best_name_suffix = (
                        "best"
                        if validation_selection_rule == "best_validation"
                        and getattr(self.train_cfg, "keep_only_latest_best_checkpoint", False)
                        else None
                    )
                    ckpt_path = self.save_checkpoint(
                        model,
                        model_name,
                        seed,
                        step,
                        valid_loss=valid["valid_loss"],
                        selection_rule=validation_selection_rule,
                        optimizer=None,
                        history=history,
                        train_losses=train_losses,
                        grad_norms=grad_norms,
                        best_valid=best_valid,
                        divergence_count=divergence_count,
                        name_suffix=best_name_suffix,
                    )
                    if validation_selection_rule == "best_validation":
                        validation_checkpoints = self.prune_previous_best_checkpoint(
                            validation_checkpoints,
                            ckpt_path,
                        )
                    validation_checkpoints = self.upsert_checkpoint_candidate(
                        validation_checkpoints,
                        self.checkpoint_metadata(
                            model_name,
                            seed,
                            step,
                            ckpt_path,
                            valid["valid_loss"],
                            validation_selection_rule,
                        ),
                    )
            if step % self.train_cfg.save_interval == 0 or step == self.train_cfg.steps:
                should_save_optimizer = bool(getattr(self.train_cfg, "save_optimizer_checkpoints", True))
                checkpoint_valid_loss = (
                    latest_valid["valid_loss"]
                    if latest_valid is not None and latest_valid["step"] == step
                    else None
                )
                checkpoint_selection_rule = (
                    "periodic_or_final"
                )
                ckpt_path = self.save_checkpoint(
                    model,
                    model_name,
                    seed,
                    step,
                    valid_loss=checkpoint_valid_loss,
                    selection_rule=checkpoint_selection_rule,
                    optimizer=opt if should_save_optimizer else None,
                    history=history,
                    train_losses=train_losses,
                    grad_norms=grad_norms,
                    best_valid=best_valid,
                    divergence_count=divergence_count,
                )
                has_validation_candidate = any(
                    item["checkpoint_step"] == step
                    for item in validation_checkpoints
                )
                if checkpoint_valid_loss is not None and not has_validation_candidate:
                    validation_checkpoints = self.upsert_checkpoint_candidate(
                        validation_checkpoints,
                        self.checkpoint_metadata(
                            model_name,
                            seed,
                            step,
                            ckpt_path,
                            checkpoint_valid_loss,
                            checkpoint_selection_rule,
                        ),
                    )

        elapsed_seconds = time.perf_counter() - start_time
        final_valid = self.validate(model)
        best_valid = min(best_valid, final_valid["valid_loss"])
        ckpt_path = self.save_checkpoint(
            model,
            model_name,
            seed,
            self.train_cfg.steps,
            valid_loss=final_valid["valid_loss"],
            selection_rule="final_step",
            optimizer=opt if getattr(self.train_cfg, "save_final_optimizer", True) else None,
            history=history,
            train_losses=train_losses,
            grad_norms=grad_norms,
            best_valid=best_valid,
            divergence_count=divergence_count,
        )
        return self.build_train_result(
            model=model,
            model_name=model_name,
            seed=seed,
            checkpoint_path=ckpt_path,
            final_valid=final_valid,
            best_valid=best_valid,
            validation_checkpoints=validation_checkpoints,
            history=history,
            train_losses=train_losses,
            grad_norms=grad_norms,
            divergence_count=divergence_count,
            elapsed_seconds=elapsed_seconds,
            analysis_reason="trained_to_final",
        )

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

    def paired_difference_stats(self, per_seed_rows, baseline=None, targets=None):
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
        model_names = list(dict.fromkeys(row["model_name"] for row in per_seed_rows))
        baseline = baseline or getattr(self.model_cfg, "baseline_model_name", model_names[0] if model_names else "rope")
        targets = list(targets or [name for name in model_names if name != baseline])
        comparisons = [(baseline, target) for target in targets]
        for pope_alibi_name in ("pope_alibi", "pop1_alibi"):
            if "pope" in model_names and pope_alibi_name in model_names:
                comparison = ("pope", pope_alibi_name)
                if comparison not in comparisons:
                    comparisons.append(comparison)
        rows = []
        rng = random.Random(0)
        for comparison_baseline, target in comparisons:
            common_seeds = sorted(
                {
                    seed for model_name, seed in by_model_seed
                    if model_name == comparison_baseline and (target, seed) in by_model_seed
                }
            )
            for metric in metrics:
                differences = []
                for seed in common_seeds:
                    base_value = by_model_seed[(comparison_baseline, seed)].get(metric)
                    target_value = by_model_seed[(target, seed)].get(metric)
                    if base_value is None or target_value is None:
                        continue
                    if not (math.isfinite(base_value) and math.isfinite(target_value)):
                        continue
                    differences.append(target_value - base_value)
                if not differences:
                    rows.append(
                        {
                            "comparison": f"{target}_minus_{comparison_baseline}",
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
                        "comparison": f"{target}_minus_{comparison_baseline}",
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
                best = state["checkpoint_selection"].get("best_validation")
                if best is None:
                    best = self.best_validation_checkpoint(candidates)
                matched = self.matched_validation_checkpoint(candidates, target)
                state["checkpoint_selection"]["best_validation"] = best
                state["checkpoint_selection"]["validation_loss_matched"] = matched
                state["checkpoint_selection"]["protocol"]["validation_loss_match_target"] = target
                state["best_checkpoint_path"] = best["checkpoint_path"] if best else state.get("checkpoint_path")
                state["best_checkpoint_step"] = best["checkpoint_step"] if best else state.get("checkpoint_step")
                state["matched_checkpoint_path"] = matched["checkpoint_path"] if matched else None
                state["matched_valid_loss"] = matched["valid_loss_at_checkpoint"] if matched else None
                state["matched_checkpoint_step"] = matched["checkpoint_step"] if matched else None

                train_item = train_res[model_name][int(seed)]["train_state"]
                train_item["checkpoint_selection"] = state["checkpoint_selection"]
                train_item["best_checkpoint_path"] = state["best_checkpoint_path"]
                train_item["best_checkpoint_step"] = state["best_checkpoint_step"]
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
                    "best_checkpoint_step": state.get("best_checkpoint_step"),
                    "best_checkpoint_path": state.get("best_checkpoint_path"),
                    "matched_checkpoint_step": state.get("matched_checkpoint_step"),
                    "matched_checkpoint_path": state.get("matched_checkpoint_path"),
                    "matched_valid_loss": state.get("matched_valid_loss"),
                }
                per_seed_rows.append(row)
                for key in metrics_by_name:
                    metrics_by_name[key].append(row[key])
            summary[model_name] = {key: mean_std(values) for key, values in metrics_by_name.items()}
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

    def write_summary_tables(self, summary):
        per_seed_path = Path(PATH.table_dir) / "phase2_per_seed.csv"
        write_csv(summary["per_seed"], per_seed_path)
        write_csv(summary.get("paired_stats", []), Path(PATH.table_dir) / "phase2_paired_stats.csv")
        write_csv(
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
        write_csv(rows, aggregate_path)

    def plot_aggregate_curves(self, serializable):
        if not self.train_cfg.run_loss_curve:
            return
        histories = {}
        for model_name, seed_items in serializable.items():
            for seed, item in seed_items.items():
                histories[f"{model_name} seed {seed}"] = item["train_state"]["history"]
        plot_metric_curves(
            histories,
            "train_loss",
            Path(PATH.figure_dir) / "phase2_train_loss_all_models_seeds.png",
            ylabel="training loss",
            title="Phase 2 training loss",
        )
        plot_metric_curves(
            histories,
            "valid_loss",
            Path(PATH.figure_dir) / "phase2_valid_loss_all_models_seeds.png",
            ylabel="validation loss",
            title="Phase 2 validation loss",
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
                    metric: {layer: mean_std(values) for layer, values in layers.items()}
                    for metric, layers in model_item["layer_wise"].items()
                },
                "stage_wise": {
                    stage: {metric: mean_std(values) for metric, values in metrics.items()}
                    for stage, metrics in model_item["stage_wise"].items()
                },
                "taxonomy_counts": {
                    label: mean_std(values) for label, values in model_item["taxonomy_counts"].items()
                },
            }
        return {"by_model": summary, "layer_rows": layer_rows, "taxonomy_rows": taxonomy_rows}

    def write_phase3_summary_tables(self, phase3_summary):
        write_csv(
            phase3_summary["layer_rows"],
            Path(PATH.table_dir) / "phase3_layer_metrics.csv",
        )
        write_csv(
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
