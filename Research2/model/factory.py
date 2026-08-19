import copy

from .model import Transformer
from .premodel import PretrainedCausalLM, premodel_entry, premodel_aliases


def run_mode(cfg):
    return cfg.get("run", {}).get("mode", "retrain")


def is_pretrain_mode(cfg):
    return run_mode(cfg) == "pretrain"


def active_aliases(cfg):
    if is_pretrain_mode(cfg):
        return premodel_aliases(cfg)
    return [exp["alias"] for exp in active_experiments(cfg)]


def base_model_aliases(cfg):
    aliases = cfg["run"].get("models", [])
    return aliases if isinstance(aliases, list) else [aliases]


def position_encodings(cfg):
    values = cfg["run"].get("position_encodings", [cfg["model"].get("position_encoding", "rope")])
    return values if isinstance(values, list) else [values]


def default_position_encoding(cfg):
    return cfg["model"].get("position_encoding", "rope")


def experiment_alias(cfg, model_alias, position_encoding):
    positions = position_encodings(cfg)
    default = default_position_encoding(cfg)
    if len(positions) == 1 and position_encoding == default:
        return model_alias
    return f"{model_alias}_{position_encoding}"


def active_experiments(cfg):
    experiments = []
    for model_alias in base_model_aliases(cfg):
        for pos in position_encodings(cfg):
            experiments.append({
                "alias": experiment_alias(cfg, model_alias, pos),
                "model_alias": model_alias,
                "position_encoding": pos,
            })
    return experiments


def resolve_experiment(cfg, alias):
    if is_pretrain_mode(cfg):
        return {"alias": alias, "model_alias": alias, "position_encoding": None}
    for exp in active_experiments(cfg):
        if exp["alias"] == alias:
            return exp
    if alias in cfg["models"]:
        return {
            "alias": alias,
            "model_alias": alias,
            "position_encoding": default_position_encoding(cfg),
        }
    raise KeyError(alias)


def experiment_cfg(cfg, alias):
    if is_pretrain_mode(cfg):
        return cfg
    exp = resolve_experiment(cfg, alias)
    out = copy.deepcopy(cfg)
    out["model"]["position_encoding"] = exp["position_encoding"]
    return out


def validate_aliases(cfg, aliases):
    if is_pretrain_mode(cfg):
        known = cfg["premodel"]["models"]
        missing = [alias for alias in aliases if alias not in known]
    else:
        known = cfg["models"]
        missing = []
        for alias in aliases:
            try:
                model_alias = resolve_experiment(cfg, alias)["model_alias"]
            except KeyError:
                missing.append(alias)
                continue
            if model_alias not in known:
                missing.append(model_alias)
    if missing:
        section = "premodel.models" if is_pretrain_mode(cfg) else "models"
        raise KeyError(f"unknown aliases in run.models for run.mode={run_mode(cfg)}: {missing} (expected keys in {section})")


def build_model(cfg, alias):
    if is_pretrain_mode(cfg):
        return PretrainedCausalLM(cfg, alias)
    exp = resolve_experiment(cfg, alias)
    exp_cfg = experiment_cfg(cfg, alias)
    return Transformer(exp_cfg["model"], exp_cfg["models"][exp["model_alias"]])


def checkpoint_signature(cfg, alias):
    if is_pretrain_mode(cfg):
        entry = premodel_entry(cfg, alias)
        return {
            "mode": "pretrain",
            "hf_id": entry["hf_id"],
            "revision": entry.get("revision"),
            "tokenizer_id": entry.get("tokenizer_id"),
        }
    exp = resolve_experiment(cfg, alias)
    exp_cfg = experiment_cfg(cfg, alias)
    return {
        "mode": "retrain",
        "model": exp_cfg["model"],
        "variant": exp_cfg["models"][exp["model_alias"]],
        "model_alias": exp["model_alias"],
        "position_encoding": exp["position_encoding"],
    }


def hidden_size(model, cfg):
    return getattr(model, "hidden_size", cfg["model"]["d_model"])
