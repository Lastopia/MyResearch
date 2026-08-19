import json
from pathlib import Path


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def output_dir(cfg):
    return Path("output") / cfg["run"]["task"]


def checkpoint_dir(cfg):
    return Path("checkpoints") / cfg["run"]["task"]


def is_pretrain_mode(cfg):
    return cfg.get("run", {}).get("mode", "retrain") == "pretrain"


def data_blocks_alias(cfg, alias=None):
    data_cfg = cfg["data"]
    if is_pretrain_mode(cfg) and cfg.get("premodel", {}).get("auto_blocks_alias", True):
        aliases = cfg["run"].get("models") or [cfg["premodel"]["default"]]
        if isinstance(aliases, str):
            aliases = [aliases]
        alias = alias or aliases[0]
        return f"{data_cfg['corpus']}_{alias}_b{data_cfg['block_size']}_train{data_cfg['train_blocks']}"
    return data_cfg["blocks_alias"]


def data_tokenized_alias(cfg, alias=None):
    data_cfg = cfg["data"]
    if is_pretrain_mode(cfg):
        aliases = cfg["run"].get("models") or [cfg["premodel"]["default"]]
        if isinstance(aliases, str):
            aliases = [aliases]
        alias = alias or aliases[0]
        return f"{data_cfg['corpus']}_{alias}"
    return data_cfg["tokenized_alias"]


def data_tokenizer_alias(cfg, alias=None):
    if is_pretrain_mode(cfg):
        aliases = cfg["run"].get("models") or [cfg["premodel"]["default"]]
        if isinstance(aliases, str):
            aliases = [aliases]
        return alias or aliases[0]
    return cfg["data"]["tokenizer_alias"]


def block_dir(cfg, alias=None):
    return Path("data") / "blocks" / data_blocks_alias(cfg, alias)


def save_config(cfg):
    write_json(output_dir(cfg) / "config.json", cfg)
