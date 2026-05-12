import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from para import PATH
from utils import (
    ensure_dir,
    get_device,
    load_json,
    manifest_is_current,
    save_json,
    save_manifest,
    set_seed,
)


class Evaluate:
    def __init__(self, eval_cfg, train_res, sae_res, data_res):
        self.eval_cfg = eval_cfg
        self.train_res = train_res
        self.sae_res = sae_res
        self.data_res = data_res
        self.device = get_device(getattr(eval_cfg, "device", "cuda"))
        self.token_category_cache = {}

    def loader(self, split, batch_size):
        return torch.utils.data.DataLoader(
            self.data_res[split],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    def stage_config(self):
        sae_summary = []
        for model_name, seeds in self.sae_res.items():
            for model_seed, layer_items in seeds.items():
                for layer, items in layer_items.items():
                    for item in items:
                        sae_summary.append(
                            {
                                "model_name": model_name,
                                "model_seed": model_seed,
                                "layer": layer,
                                "checkpoint_path": item.get("checkpoint_path"),
                                "meta": item.get("meta", {}),
                            }
                        )
        return {
            "eval": self.eval_cfg,
            "sae_inputs": sae_summary,
            "data_meta": self.data_res.get("meta", {}),
        }

    def stage_outputs(self):
        return [
            Path(PATH.raw_metrics_dir) / "eval_res.json",
            Path(PATH.raw_metrics_dir) / "phase5_summary.json",
            Path(PATH.table_dir) / "phase5_disentanglement_runs.csv",
            Path(PATH.table_dir) / "phase5_disentanglement_summary.csv",
            Path(PATH.table_dir) / "phase5_feature_scores.csv",
        ]

    def stage_manifest_path(self):
        return Path(PATH.raw_metrics_dir) / "eval_manifest.json"

    @torch.no_grad()
    def collect_raw(self, model, layer, split, max_tokens):
        model.eval()
        acts, token_ids, positions = [], [], []
        seen = 0
        for x, _ in self.loader(split, batch_size=max(1, getattr(self.eval_cfg, "probe_batch_size", 512) // 128)):
            x = x.to(self.device)
            out = model(x, capture_layers=[layer])
            flat = out["activations"][layer].reshape(-1, out["activations"][layer].size(-1)).float().cpu()
            ids = x.reshape(-1).detach().cpu()
            pos = torch.arange(x.size(1)).repeat(x.size(0))
            take = min(flat.size(0), max_tokens - seen)
            if take > 0:
                acts.append(flat[:take])
                token_ids.append(ids[:take])
                positions.append(pos[:take])
                seen += take
            if seen >= max_tokens:
                break
        if not acts:
            raise RuntimeError(f"No eval activations collected for split={split}, layer={layer}")
        return {
            "acts": torch.cat(acts, dim=0),
            "token_ids": torch.cat(token_ids, dim=0),
            "positions": torch.cat(positions, dim=0),
        }

    def normalize_with_stats(self, acts, stats):
        if not stats.get("enabled", True):
            return acts
        return (acts - stats["mean"].cpu()) / stats["std"].cpu().clamp_min(1e-6)

    @torch.no_grad()
    def encode_sae(self, sae, raw_acts, stats):
        sae.eval()
        normalized = self.normalize_with_stats(raw_acts, stats).to(self.device)
        features = sae.encode(normalized).float().cpu()
        return features

    def token_to_text(self, token_id):
        token_id = int(token_id)
        if token_id in self.token_category_cache:
            return self.token_category_cache[token_id]
        tokenizer = self.data_res.get("tokenizer")
        if hasattr(tokenizer, "decode"):
            text = tokenizer.decode([token_id])
        else:
            inv = {v: k for k, v in getattr(tokenizer, "stoi", {}).items()}
            text = inv.get(token_id, "")
        self.token_category_cache[token_id] = text
        return text

    def token_categories(self, token_ids):
        labels = []
        for token_id in token_ids.tolist():
            text = self.token_to_text(token_id)
            if text.isspace():
                label = 3
            elif any(ch.isalpha() for ch in text):
                label = 0
            elif any(ch.isdigit() for ch in text):
                label = 1
            elif text and all(not ch.isalnum() and not ch.isspace() for ch in text):
                label = 2
            else:
                label = 4
            labels.append(label)
        return torch.tensor(labels, dtype=torch.long)

    def frequency_bins(self, train_ids, target_ids):
        counts = {}
        for token_id in train_ids.tolist():
            counts[int(token_id)] = counts.get(int(token_id), 0) + 1
        train_counts = torch.tensor([counts[int(token_id)] for token_id in train_ids.tolist()], dtype=torch.float)
        if train_counts.numel() < 3:
            q1 = q2 = 1.0
        else:
            q1, q2 = torch.quantile(train_counts, torch.tensor([1 / 3, 2 / 3])).tolist()
        bins = []
        for token_id in target_ids.tolist():
            count = counts.get(int(token_id), 0)
            if count <= q1:
                bins.append(0)
            elif count <= q2:
                bins.append(1)
            else:
                bins.append(2)
        return torch.tensor(bins, dtype=torch.long)

    def targets(self, train_ids, valid_ids, train_pos, valid_pos, seq_len):
        bins = getattr(self.eval_cfg, "position_bins", 16)

        def position_bin(pos):
            return torch.clamp((pos.float() / max(seq_len, 1) * bins).long(), 0, bins - 1)

        def segment(pos):
            frac = pos.float() / max(seq_len - 1, 1)
            return torch.where(frac < 1 / 3, 0, torch.where(frac < 2 / 3, 1, 2)).long()

        return {
            "train": {
                "position_bin": position_bin(train_pos),
                "normalized_position": train_pos.float() / max(seq_len - 1, 1),
                "segment_position": segment(train_pos),
                "token_category": self.token_categories(train_ids),
                "token_frequency_bin": self.frequency_bins(train_ids, train_ids),
            },
            "valid": {
                "position_bin": position_bin(valid_pos),
                "normalized_position": valid_pos.float() / max(seq_len - 1, 1),
                "segment_position": segment(valid_pos),
                "token_category": self.token_categories(valid_ids),
                "token_frequency_bin": self.frequency_bins(train_ids, valid_ids),
            },
        }

    def standardize(self, train_x, valid_x):
        mean = train_x.mean(dim=0, keepdim=True)
        std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        return (train_x - mean) / std, (valid_x - mean) / std

    def train_regression_probe(self, train_x, train_y, valid_x, valid_y):
        set_seed(0)
        train_x, valid_x = self.standardize(train_x, valid_x)
        model = nn.Linear(train_x.size(1), 1).to(self.device)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=getattr(self.eval_cfg, "probe_lr", 1e-2),
            weight_decay=getattr(self.eval_cfg, "probe_weight_decay", 1e-4),
        )
        train_x, train_y = train_x.to(self.device), train_y.float().to(self.device)
        valid_x, valid_y = valid_x.to(self.device), valid_y.float().to(self.device)
        batch_size = min(getattr(self.eval_cfg, "probe_batch_size", 512), train_x.size(0))
        for _ in range(getattr(self.eval_cfg, "probe_steps", 200)):
            idx = torch.randint(0, train_x.size(0), (batch_size,), device=self.device)
            pred = model(train_x[idx]).squeeze(-1)
            loss = F.mse_loss(pred, train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = model(valid_x).squeeze(-1)
            ss_res = (valid_y - pred).square().sum()
            ss_tot = (valid_y - valid_y.mean()).square().sum().clamp_min(1e-9)
            return {"r2": (1.0 - ss_res / ss_tot).item(), "mse": F.mse_loss(pred, valid_y).item()}

    def macro_f1(self, pred, target, num_classes):
        scores = []
        for cls in range(num_classes):
            tp = ((pred == cls) & (target == cls)).sum().item()
            fp = ((pred == cls) & (target != cls)).sum().item()
            fn = ((pred != cls) & (target == cls)).sum().item()
            denom = 2 * tp + fp + fn
            scores.append(0.0 if denom == 0 else (2 * tp) / denom)
        return sum(scores) / len(scores)

    def train_classification_probe(self, train_x, train_y, valid_x, valid_y):
        set_seed(0)
        train_x, valid_x = self.standardize(train_x, valid_x)
        classes = torch.unique(torch.cat([train_y, valid_y])).tolist()
        class_map = {int(cls): idx for idx, cls in enumerate(classes)}
        train_y = torch.tensor([class_map[int(y)] for y in train_y.tolist()], dtype=torch.long)
        valid_y = torch.tensor([class_map[int(y)] for y in valid_y.tolist()], dtype=torch.long)
        num_classes = len(classes)
        if num_classes < 2:
            return {"accuracy": 1.0, "macro_f1": 1.0, "num_classes": num_classes}
        model = nn.Linear(train_x.size(1), num_classes).to(self.device)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=getattr(self.eval_cfg, "probe_lr", 1e-2),
            weight_decay=getattr(self.eval_cfg, "probe_weight_decay", 1e-4),
        )
        train_x, train_y = train_x.to(self.device), train_y.to(self.device)
        valid_x, valid_y = valid_x.to(self.device), valid_y.to(self.device)
        batch_size = min(getattr(self.eval_cfg, "probe_batch_size", 512), train_x.size(0))
        for _ in range(getattr(self.eval_cfg, "probe_steps", 200)):
            idx = torch.randint(0, train_x.size(0), (batch_size,), device=self.device)
            loss = F.cross_entropy(model(train_x[idx]), train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = model(valid_x).argmax(dim=-1)
            acc = (pred == valid_y).float().mean().item()
            f1 = self.macro_f1(pred.cpu(), valid_y.cpu(), num_classes)
            return {"accuracy": acc, "macro_f1": f1, "num_classes": num_classes}

    def representation_probes(self, train_x, valid_x, targets):
        return {
            "normalized_position": self.train_regression_probe(
                train_x, targets["train"]["normalized_position"], valid_x, targets["valid"]["normalized_position"]
            ),
            "position_bin": self.train_classification_probe(
                train_x, targets["train"]["position_bin"], valid_x, targets["valid"]["position_bin"]
            ),
            "segment_position": self.train_classification_probe(
                train_x, targets["train"]["segment_position"], valid_x, targets["valid"]["segment_position"]
            ),
            "token_category": self.train_classification_probe(
                train_x, targets["train"]["token_category"], valid_x, targets["valid"]["token_category"]
            ),
            "token_frequency_bin": self.train_classification_probe(
                train_x, targets["train"]["token_frequency_bin"], valid_x, targets["valid"]["token_frequency_bin"]
            ),
        }

    def discretize_features(self, features):
        bins = getattr(self.eval_cfg, "feature_activation_bins", 10)
        quantiles = torch.linspace(0, 1, bins + 1)[1:-1]
        thresholds = torch.quantile(features.float(), quantiles, dim=0)
        out = torch.zeros_like(features, dtype=torch.long)
        for idx in range(thresholds.size(0)):
            out += (features > thresholds[idx]).long()
        return out

    def normalized_mi(self, feature_bins, target, num_feature_bins, num_target_bins):
        target_np = target.cpu().numpy().astype(np.int64)
        rows = []
        for feat in range(feature_bins.size(1)):
            feat_np = feature_bins[:, feat].cpu().numpy().astype(np.int64)
            joint = np.zeros((num_feature_bins, num_target_bins), dtype=np.float64)
            np.add.at(joint, (feat_np, target_np), 1)
            joint /= max(joint.sum(), 1.0)
            px = joint.sum(axis=1, keepdims=True)
            py = joint.sum(axis=0, keepdims=True)
            expected = px @ py
            mask = joint > 0
            mi = float((joint[mask] * np.log(joint[mask] / np.clip(expected[mask], 1e-12, None))).sum())
            hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
            rows.append(0.0 if hy <= 1e-12 else mi / hy)
        return torch.tensor(rows, dtype=torch.float)

    def abs_corr_with_position(self, features, pos):
        x = features.float()
        y = pos.float()
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean()
        denom = x.square().sum(dim=0).sqrt() * y.square().sum().sqrt().clamp_min(1e-9)
        return ((x * y[:, None]).sum(dim=0) / denom.clamp_min(1e-9)).abs()

    def feature_feature_correlation_summary(self, features):
        sample = min(getattr(self.eval_cfg, "correlation_feature_sample", 512), features.size(1))
        if sample < 2:
            return {"mean_abs": None, "max_abs": None}
        x = features[:, :sample].float()
        x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        corr = (x.T @ x) / max(x.size(0) - 1, 1)
        mask = ~torch.eye(sample, dtype=torch.bool)
        vals = corr[mask].abs()
        return {"mean_abs": vals.mean().item(), "max_abs": vals.max().item()}

    def feature_level_scores(self, features, targets):
        dead = (features != 0).any(dim=0) == 0
        feature_bins = self.discretize_features(features)
        num_bins = getattr(self.eval_cfg, "feature_activation_bins", 10)
        position_corr = self.abs_corr_with_position(features, targets["valid"]["normalized_position"])
        position_mi = self.normalized_mi(
            feature_bins, targets["valid"]["position_bin"], num_bins, getattr(self.eval_cfg, "position_bins", 16)
        )
        content_category_mi = self.normalized_mi(feature_bins, targets["valid"]["token_category"], num_bins, 5)
        content_freq_mi = self.normalized_mi(feature_bins, targets["valid"]["token_frequency_bin"], num_bins, 3)
        position_score = torch.maximum(position_corr, position_mi)
        content_score = torch.maximum(content_category_mi, content_freq_mi)
        active_mask = ~dead
        q = getattr(self.eval_cfg, "selectivity_quantile", 0.9)
        content_threshold = torch.quantile(content_score[active_mask], q).item() if active_mask.any() else float("inf")
        position_threshold = torch.quantile(position_score[active_mask], q).item() if active_mask.any() else float("inf")
        labels = []
        counts = {"content_only": 0, "position_only": 0, "mixed": 0, "low_selectivity": 0, "dead": 0}
        for idx in range(features.size(1)):
            if dead[idx]:
                label = "dead"
            elif content_score[idx] >= content_threshold and position_score[idx] >= position_threshold:
                label = "mixed"
            elif content_score[idx] >= content_threshold:
                label = "content_only"
            elif position_score[idx] >= position_threshold:
                label = "position_only"
            else:
                label = "low_selectivity"
            labels.append(label)
            counts[label] += 1
        total = max(features.size(1), 1)
        corr = np.corrcoef(content_score.numpy(), position_score.numpy())[0, 1]
        corr = None if not np.isfinite(corr) else float(corr)
        return {
            "summary": {
                "mixed_feature_ratio": counts["mixed"] / total,
                "content_only_ratio": counts["content_only"] / total,
                "position_only_ratio": counts["position_only"] / total,
                "low_selectivity_ratio": counts["low_selectivity"] / total,
                "dead_ratio": counts["dead"] / total,
                "content_position_score_correlation": corr,
                "mean_content_score": content_score.mean().item(),
                "mean_position_score": position_score.mean().item(),
                "top_content_score": content_score.max().item(),
                "top_position_score": position_score.max().item(),
                "content_threshold": content_threshold,
                "position_threshold": position_threshold,
                **self.feature_feature_correlation_summary(features),
            },
            "feature_rows": [
                {
                    "feature": idx,
                    "content_score": content_score[idx].item(),
                    "position_score": position_score[idx].item(),
                    "position_corr": position_corr[idx].item(),
                    "position_mi": position_mi[idx].item(),
                    "content_category_mi": content_category_mi[idx].item(),
                    "content_frequency_mi": content_freq_mi[idx].item(),
                    "label": labels[idx],
                }
                for idx in range(features.size(1))
            ],
        }

    def write_csv(self, rows, path):
        ensure_dir(Path(path).parent)
        if not rows:
            return
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def run_one(self, model_name, model_seed, layer, train_item, sae_items):
        model = train_item["model"].to(self.device)
        train_raw = self.collect_raw(model, layer, "train", getattr(self.eval_cfg, "max_probe_train_tokens", 8192))
        valid_raw = self.collect_raw(model, layer, "valid", getattr(self.eval_cfg, "max_probe_valid_tokens", 4096))
        targets = self.targets(
            train_raw["token_ids"],
            valid_raw["token_ids"],
            train_raw["positions"],
            valid_raw["positions"],
            train_raw["positions"].max().item() + 1,
        )
        raw_probe = self.representation_probes(train_raw["acts"], valid_raw["acts"], targets)
        rows, feature_rows = [], []
        for sae_item in sae_items:
            sae = sae_item["sae"].to(self.device)
            stats = sae_item["normalization"]
            train_features = self.encode_sae(sae, train_raw["acts"], stats)
            valid_features = self.encode_sae(sae, valid_raw["acts"], stats)
            sae_probe = self.representation_probes(train_features, valid_features, targets)
            feature_scores = self.feature_level_scores(valid_features, targets)
            meta = sae_item["meta"]
            row = {
                "model_name": model_name,
                "model_seed": model_seed,
                "layer": layer,
                "sae_seed": meta["sae_seed"],
                "dict_size": meta["dict_size"],
                "k": meta["k"],
                "model_checkpoint_rule": meta.get("model_checkpoint_rule"),
                "model_checkpoint_step": meta.get("model_checkpoint_step"),
                "model_checkpoint_path": meta.get("model_checkpoint_path"),
                "model_tokens_seen": meta.get("model_tokens_seen"),
                "raw_position_r2": raw_probe["normalized_position"]["r2"],
                "raw_position_bin_accuracy": raw_probe["position_bin"]["accuracy"],
                "raw_token_category_accuracy": raw_probe["token_category"]["accuracy"],
                "sae_position_r2": sae_probe["normalized_position"]["r2"],
                "sae_position_bin_accuracy": sae_probe["position_bin"]["accuracy"],
                "sae_token_category_accuracy": sae_probe["token_category"]["accuracy"],
                **feature_scores["summary"],
            }
            rows.append(row)
            for feature_row in feature_scores["feature_rows"]:
                feature_rows.append({**{k: row[k] for k in ["model_name", "model_seed", "layer", "sae_seed"]}, **feature_row})
        return {
            "raw_probe": raw_probe,
            "rows": rows,
            "feature_rows": feature_rows,
        }

    def summarize(self, rows):
        metrics = [
            "raw_position_r2",
            "raw_position_bin_accuracy",
            "raw_token_category_accuracy",
            "sae_position_r2",
            "sae_position_bin_accuracy",
            "sae_token_category_accuracy",
            "mixed_feature_ratio",
            "content_only_ratio",
            "position_only_ratio",
            "content_position_score_correlation",
            "mean_content_score",
            "mean_position_score",
        ]
        grouped = {}
        for row in rows:
            key = (row["model_name"], row["layer"])
            grouped.setdefault(key, {metric: [] for metric in metrics})
            for metric in metrics:
                value = row.get(metric)
                if value is not None and math.isfinite(value):
                    grouped[key][metric].append(value)
        summary = []
        for (model_name, layer), values in grouped.items():
            row = {"model_name": model_name, "layer": layer}
            for metric, metric_values in values.items():
                if metric_values:
                    mean = sum(metric_values) / len(metric_values)
                    var = sum((x - mean) ** 2 for x in metric_values) / len(metric_values)
                    row[f"{metric}_mean"] = mean
                    row[f"{metric}_std"] = math.sqrt(var)
                else:
                    row[f"{metric}_mean"] = None
                    row[f"{metric}_std"] = None
            summary.append(row)
        return summary

    def run(self):
        if (
            getattr(self.eval_cfg, "skip_completed_stage", True)
            and manifest_is_current(self.stage_manifest_path(), self.stage_config(), self.stage_outputs())
        ):
            return load_json(Path(PATH.raw_metrics_dir) / "eval_res.json")

        all_rows, all_feature_rows, nested = [], [], {}
        layers = getattr(self.eval_cfg, "layers", [2, 6, 10])
        for model_name, seeds in self.train_res.items():
            nested[model_name] = {}
            for model_seed, train_item in seeds.items():
                nested[model_name][str(model_seed)] = {}
                for layer in layers:
                    if layer not in self.sae_res.get(model_name, {}).get(model_seed, {}):
                        continue
                    one = self.run_one(
                        model_name,
                        model_seed,
                        layer,
                        train_item,
                        self.sae_res[model_name][model_seed][layer],
                    )
                    nested[model_name][str(model_seed)][str(layer)] = {
                        "raw_probe": one["raw_probe"],
                        "run_rows": one["rows"],
                    }
                    all_rows.extend(one["rows"])
                    all_feature_rows.extend(one["feature_rows"])
        summary = self.summarize(all_rows)
        eval_res = {
            "phase": "5",
            "design": {
                "max_probe_train_tokens": getattr(self.eval_cfg, "max_probe_train_tokens", 8192),
                "max_probe_valid_tokens": getattr(self.eval_cfg, "max_probe_valid_tokens", 4096),
                "position_bins": getattr(self.eval_cfg, "position_bins", 16),
                "feature_activation_bins": getattr(self.eval_cfg, "feature_activation_bins", 10),
                "selectivity_quantile": getattr(self.eval_cfg, "selectivity_quantile", 0.9),
                "top_token_identity": "deferred",
                "pos_tag_probe": "deferred",
                "permutation_baseline": "deferred",
            },
            "summary_rows": summary,
            "run_rows": all_rows,
            "nested": nested,
        }
        save_json(eval_res, Path(PATH.raw_metrics_dir) / "eval_res.json")
        save_json(eval_res, Path(PATH.raw_metrics_dir) / "phase5_summary.json")
        self.write_csv(all_rows, Path(PATH.table_dir) / "phase5_disentanglement_runs.csv")
        self.write_csv(summary, Path(PATH.table_dir) / "phase5_disentanglement_summary.csv")
        self.write_csv(all_feature_rows, Path(PATH.table_dir) / "phase5_feature_scores.csv")
        save_manifest(self.stage_manifest_path(), "eval", self.stage_config(), self.stage_outputs())
        return eval_res
