import csv
import math
from pathlib import Path

import torch

from interpret_prompting import FeatureInterpreter
from interpret_reports import InterpretationReporter
from para import PATH
from sae import load_sae_item
from utils import ensure_dir, load_json, manifest_is_current, save_json, save_manifest, write_csv


class InterpretSAE:
    def __init__(self, interp_cfg, train_res, sae_res, data_res, eval_res=None):
        self.interp_cfg = interp_cfg
        self.train_res = train_res or {}
        self.sae_res = sae_res or {}
        self.data_res = data_res or {}
        self.eval_res = eval_res or {}
        self.feature_scores = self.load_feature_scores()
        self.interpreter = FeatureInterpreter(interp_cfg)
        self.reporter = InterpretationReporter()

    def load_feature_scores(self):
        path = Path(PATH.table_dir) / "phase5_feature_scores.csv"
        if not path.exists():
            return {}
        rows = {}
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key = (
                    row.get("model_name"),
                    str(row.get("model_seed")),
                    str(row.get("layer")),
                    str(row.get("sae_seed")),
                )
                rows.setdefault(key, []).append(row)
        return rows

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
            "interpret": self.interp_cfg,
            "sae_inputs": sae_summary,
            "eval_phase": self.eval_res.get("phase"),
        }

    def stage_outputs(self):
        return [
            Path(PATH.raw_metrics_dir) / "phase6_interpretation_summary.json",
            Path(PATH.raw_metrics_dir) / "phase6_prompts.json",
            Path(PATH.raw_metrics_dir) / "phase6_run_records.json",
            Path(PATH.table_dir) / "phase6_interpretation_scores.csv",
            Path(PATH.table_dir) / "phase6_interpretation_summary.csv",
            Path(PATH.report_dir) / "phase6_feature_cases.md",
        ]

    def stage_manifest_path(self):
        return Path(PATH.raw_metrics_dir) / "interpret_manifest.json"

    def tokenizer_decode(self, ids):
        tokenizer = self.data_res.get("tokenizer")
        if hasattr(tokenizer, "decode"):
            return tokenizer.decode([int(x) for x in ids])
        inv = {v: k for k, v in getattr(tokenizer, "stoi", {}).items()}
        return "".join(inv.get(int(x), "") for x in ids)

    def read_csv(self, path):
        path = Path(path)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def load_existing_prompt_rows(self):
        path = Path(PATH.raw_metrics_dir) / "phase6_prompts.json"
        if not path.exists():
            return []
        try:
            return load_json(path).get("prompts", [])
        except Exception:
            return []

    def load_existing_run_records(self):
        path = Path(PATH.raw_metrics_dir) / "phase6_run_records.json"
        if not path.exists():
            return []
        try:
            return load_json(path).get("records", [])
        except Exception:
            return []

    def load_existing_score_rows(self):
        rows = self.read_csv(Path(PATH.table_dir) / "phase6_interpretation_scores.csv")
        numeric_float = {"content_score", "position_score", "quality_score", "confidence_score"}
        numeric_int = {
            "model_seed",
            "layer",
            "feature",
            "model_checkpoint_step",
            "model_tokens_seen",
            "interpretability_score",
            "specificity_score",
            "coverage_score",
            "false_positive_risk",
        }
        for row in rows:
            for key in numeric_float:
                if row.get(key) not in {None, ""}:
                    row[key] = float(row[key])
            for key in numeric_int:
                if row.get(key) not in {None, ""}:
                    row[key] = int(float(row[key]))
            if row.get("dry_run") in {"True", "False"}:
                row["dry_run"] = row["dry_run"] == "True"
            if not row.get("run_type"):
                row["run_type"] = "dry_run" if row.get("dry_run") else "openai_run"
            if not row.get("llm_model"):
                row["llm_model"] = getattr(self.interp_cfg, "model", "gpt-4o-mini")
            if not row.get("provider"):
                row["provider"] = getattr(self.interp_cfg, "provider", "openai")
        return rows

    def score_row_key(self, row):
        return (
            str(row.get("model_name")),
            str(row.get("model_seed")),
            str(row.get("layer")),
            str(row.get("feature")),
            str(row.get("phase5_label")),
            str(row.get("run_type") or ("dry_run" if row.get("dry_run") else "openai_run")),
        )

    def feature_candidates(self, model_name, model_seed, layer, sae_seed, dict_size):
        key = (model_name, str(model_seed), str(layer), str(sae_seed))
        label_order = list(getattr(self.interp_cfg, "feature_labels", ["content_only", "position_only", "mixed"]))
        max_total = getattr(self.interp_cfg, "max_features_per_model_layer", 30)
        rows = self.feature_scores.get(key, [])
        selected = []
        if rows:
            per_label = max(1, math.ceil(max_total / max(len(label_order), 1)))
            for label in label_order:
                label_rows = [row for row in rows if row.get("label") == label]
                label_rows.sort(
                    key=lambda row: max(
                        float(row.get("content_score") or 0.0),
                        float(row.get("position_score") or 0.0),
                    ),
                    reverse=True,
                )
                selected.extend(label_rows[:per_label])
            selected = selected[:max_total]
            return [
                {
                    "feature": int(row["feature"]),
                    "phase5_label": row.get("label", "unknown"),
                    "content_score": float(row.get("content_score") or 0.0),
                    "position_score": float(row.get("position_score") or 0.0),
                }
                for row in selected
            ]
        return [
            {
                "feature": idx,
                "phase5_label": "unknown",
                "content_score": None,
                "position_score": None,
            }
            for idx in range(min(max_total, dict_size))
        ]

    @torch.no_grad()
    def collect_contexts(self, model, sae_item, layer, feature_id):
        model.eval()
        device = next(model.parameters()).device
        sae_item = load_sae_item(sae_item, device)
        sae = sae_item["sae"]
        sae.eval()
        stats = sae_item["normalization"]
        max_contexts = getattr(self.interp_cfg, "max_contexts_per_feature", 12)
        window = getattr(self.interp_cfg, "context_window", 8)
        threshold = getattr(self.interp_cfg, "active_threshold", 0.0)
        candidates = []
        loader = torch.utils.data.DataLoader(self.data_res["valid"], batch_size=1, shuffle=False)
        for x, _ in loader:
            x = x.to(device)
            out = model(x, capture_layers=[layer])
            raw = out["activations"][layer][0].float().cpu()
            normalized = raw
            if stats.get("enabled", True):
                normalized = (raw - stats["mean"].cpu()) / stats["std"].cpu().clamp_min(1e-6)
            features = sae.encode(normalized.to(device)).float().cpu()
            vals = features[:, feature_id]
            active_positions = torch.nonzero(vals > threshold, as_tuple=False).flatten()
            for pos in active_positions.tolist():
                left = max(0, pos - window)
                right = min(x.size(1), pos + window + 1)
                token_ids = x[0, left:right].detach().cpu().tolist()
                token_text = self.tokenizer_decode([int(x[0, pos].item())])
                context_text = self.tokenizer_decode(token_ids)
                candidates.append(
                    {
                        "activated_token": token_text,
                        "activation": float(vals[pos].item()),
                        "position": int(pos),
                        "context": context_text,
                    }
                )
            if len(candidates) >= max_contexts * 8:
                break
        candidates.sort(key=lambda row: row["activation"], reverse=True)
        return candidates[:max_contexts]

    @torch.no_grad()
    def collect_contexts_for_features(self, model, sae_item, layer, feature_ids):
        model.eval()
        device = next(model.parameters()).device
        sae_item = load_sae_item(sae_item, device)
        sae = sae_item["sae"]
        sae.eval()
        stats = sae_item["normalization"]
        max_contexts = getattr(self.interp_cfg, "max_contexts_per_feature", 12)
        window = getattr(self.interp_cfg, "context_window", 8)
        threshold = getattr(self.interp_cfg, "active_threshold", 0.0)
        keep_per_feature = max_contexts * 8
        candidates = {int(feature_id): [] for feature_id in feature_ids}
        loader = torch.utils.data.DataLoader(self.data_res["valid"], batch_size=1, shuffle=False)
        feature_ids = [int(feature_id) for feature_id in feature_ids]
        for x, _ in loader:
            x = x.to(device)
            out = model(x, capture_layers=[layer])
            raw = out["activations"][layer][0].float().cpu()
            normalized = raw
            if stats.get("enabled", True):
                normalized = (raw - stats["mean"].cpu()) / stats["std"].cpu().clamp_min(1e-6)
            features = sae.encode(normalized.to(device)).float().cpu()
            for feature_id in feature_ids:
                vals = features[:, feature_id]
                active_positions = torch.nonzero(vals > threshold, as_tuple=False).flatten()
                for pos in active_positions.tolist():
                    left = max(0, pos - window)
                    right = min(x.size(1), pos + window + 1)
                    token_ids = x[0, left:right].detach().cpu().tolist()
                    token_text = self.tokenizer_decode([int(x[0, pos].item())])
                    context_text = self.tokenizer_decode(token_ids)
                    candidates[feature_id].append(
                        {
                            "activated_token": token_text,
                            "activation": float(vals[pos].item()),
                            "position": int(pos),
                            "context": context_text,
                        }
                    )
                if len(candidates[feature_id]) > keep_per_feature:
                    candidates[feature_id].sort(key=lambda row: row["activation"], reverse=True)
                    candidates[feature_id] = candidates[feature_id][:keep_per_feature]
            if all(len(items) >= keep_per_feature for items in candidates.values()):
                break
        for feature_id, items in candidates.items():
            items.sort(key=lambda row: row["activation"], reverse=True)
            candidates[feature_id] = items[:max_contexts]
        return candidates

    def run(self):
        if (
            getattr(self.interp_cfg, "skip_completed_stage", True)
            and manifest_is_current(self.stage_manifest_path(), self.stage_config(), self.stage_outputs())
        ):
            return load_json(Path(PATH.raw_metrics_dir) / "phase6_interpretation_summary.json")

        rows = self.load_existing_score_rows()
        prompts = self.load_existing_prompt_rows()
        run_records = self.load_existing_run_records()
        items_by_id = {row["blinded_feature_id"]: None for row in rows if row.get("blinded_feature_id")}
        scored_keys = {self.score_row_key(row) for row in rows}
        existing_ids = []
        for row in rows:
            blinded = row.get("blinded_feature_id", "")
            if blinded.startswith("Feature-"):
                try:
                    existing_ids.append(int(blinded.split("-", 1)[1]))
                except ValueError:
                    pass
        next_feature_id = max(existing_ids, default=0) + 1
        dry_run = getattr(self.interp_cfg, "dry_run", True)
        layers = getattr(self.interp_cfg, "layers", [2, 6, 10])
        model_seeds = set(str(seed) for seed in getattr(self.interp_cfg, "model_seeds", [42]))
        for model_name, seed_items in self.sae_res.items():
            for model_seed, layer_items in seed_items.items():
                if str(model_seed) not in model_seeds:
                    continue
                model = self.train_res[model_name][model_seed]["model"]
                for layer in layers:
                    if layer not in layer_items:
                        continue
                    for sae_item in layer_items[layer]:
                        meta = sae_item["meta"]
                        candidates = self.feature_candidates(
                            model_name,
                            model_seed,
                            layer,
                            meta["sae_seed"],
                            meta["dict_size"],
                        )
                        context_map = self.collect_contexts_for_features(
                            model,
                            sae_item,
                            layer,
                            [candidate["feature"] for candidate in candidates],
                        )
                        for candidate in candidates:
                            key = (
                                str(model_name),
                                str(model_seed),
                                str(layer),
                                str(candidate["feature"]),
                                str(candidate["phase5_label"]),
                            )
                            if key in scored_keys:
                                continue
                            contexts = context_map.get(candidate["feature"], [])
                            if len(contexts) < getattr(self.interp_cfg, "min_active_contexts", 4):
                                continue
                            blinded = f"Feature-{next_feature_id:05d}"
                            next_feature_id += 1
                            item = {
                                "blinded_feature_id": blinded,
                                "model_name": model_name,
                                "model_seed": model_seed,
                                "layer": layer,
                                "feature": candidate["feature"],
                                "phase5_label": candidate["phase5_label"],
                                "content_score": candidate["content_score"],
                                "position_score": candidate["position_score"],
                                "contexts": contexts,
                            }
                            response = self.interpreter.score_item(item)
                            quality = self.interpreter.quality_score(response)
                            run_type = "dry_run" if dry_run else "openai_run"
                            provider = getattr(self.interp_cfg, "provider", "openai")
                            llm_model = getattr(self.interp_cfg, "model", "gpt-4o-mini")
                            row = {
                                "blinded_feature_id": blinded,
                                "model_name": model_name,
                                "model_seed": model_seed,
                                "layer": layer,
                                "feature": candidate["feature"],
                                "phase5_label": candidate["phase5_label"],
                                "content_score": candidate["content_score"],
                                "position_score": candidate["position_score"],
                                "model_checkpoint_rule": meta.get("model_checkpoint_rule"),
                                "model_checkpoint_step": meta.get("model_checkpoint_step"),
                                "model_checkpoint_path": meta.get("model_checkpoint_path"),
                                "model_tokens_seen": meta.get("model_tokens_seen"),
                                "feature_type": response.get("feature_type"),
                                "interpretability_score": int(response.get("interpretability_score", 0)),
                                "specificity_score": int(response.get("specificity_score", 0)),
                                "coverage_score": int(response.get("coverage_score", 0)),
                                "false_positive_risk": int(response.get("false_positive_risk", 0)),
                                "confidence_score": float(response.get("confidence_score", 0.0)),
                                "quality_score": quality,
                                "short_explanation": response.get("short_explanation", ""),
                                "evidence_summary": response.get("evidence_summary", ""),
                                "provider": provider,
                                "llm_model": llm_model,
                                "run_type": run_type,
                                "dry_run": dry_run,
                            }
                            rows.append(row)
                            scored_keys.add(self.score_row_key(row))
                            run_records.append(
                                {
                                    "blinded_feature_id": blinded,
                                    "model_name": model_name,
                                    "model_seed": model_seed,
                                    "layer": layer,
                                    "feature": candidate["feature"],
                                    "phase5_label": candidate["phase5_label"],
                                    "provider": provider,
                                    "llm_model": llm_model,
                                    "run_type": run_type,
                                    "dry_run": dry_run,
                                    "prompt": item["prompt"],
                                    "parsed_response": {
                                        key: value
                                        for key, value in response.items()
                                        if key not in {"raw_api_response"}
                                    },
                                    "raw_response": response.get("raw_response"),
                                    "raw_api_response": response.get("raw_api_response"),
                                    "confidence_score": float(response.get("confidence_score", 0.0)),
                                    "false_positive_risk": int(response.get("false_positive_risk", 0)),
                                }
                            )
                            prompts.append(
                                {
                                    "blinded_feature_id": blinded,
                                    "provider": provider,
                                    "llm_model": llm_model,
                                    "run_type": run_type,
                                    "dry_run": dry_run,
                                    "prompt": item["prompt"],
                                }
                            )
                            items_by_id[blinded] = item
        current_run_type = "dry_run" if dry_run else "openai_run"
        current_rows = [row for row in rows if row.get("run_type") == current_run_type]
        summary = self.reporter.summarize(current_rows) if current_rows else []
        figure_paths = self.reporter.plot_quality_summary(summary)
        result = {
            "phase": "6",
            "design": {
                "dry_run": dry_run,
                "provider": getattr(self.interp_cfg, "provider", "openai"),
                "model": getattr(self.interp_cfg, "model", "gpt-4o-mini"),
                "max_features_per_model_layer": getattr(self.interp_cfg, "max_features_per_model_layer", 30),
                "max_contexts_per_feature": getattr(self.interp_cfg, "max_contexts_per_feature", 12),
                "optional_deferred": [
                    "human_calibration",
                    "repeated_explanations",
                    "top_vs_random_context_validation",
                    "all_seed_interpretation",
                ],
            },
            "summary_rows": summary,
            "score_rows": current_rows,
            "all_score_rows_path": str(Path(PATH.table_dir) / "phase6_interpretation_scores.csv"),
            "figure_paths": figure_paths,
        }
        save_json(result, Path(PATH.raw_metrics_dir) / "phase6_interpretation_summary.json")
        save_json({"prompts": prompts}, Path(PATH.raw_metrics_dir) / "phase6_prompts.json")
        save_json({"records": run_records}, Path(PATH.raw_metrics_dir) / "phase6_run_records.json")
        write_csv(rows, Path(PATH.table_dir) / "phase6_interpretation_scores.csv")
        write_csv(summary, Path(PATH.table_dir) / "phase6_interpretation_summary.csv")
        case_path = Path(PATH.report_dir) / "phase6_feature_cases.md"
        ensure_dir(case_path.parent)
        with open(case_path, "w", encoding="utf-8") as f:
            f.write(
                self.reporter.case_markdown(current_rows, items_by_id)
                if current_rows
                else "# Phase 6 Feature Interpretation Cases\n\nNo cases selected.\n"
            )
        save_manifest(self.stage_manifest_path(), "interpret", self.stage_config(), self.stage_outputs())
        return result
