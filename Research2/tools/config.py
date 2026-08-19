import copy
import json


SECTIONS = ["run", "premodel", "model", "models", "data", "train", "benchmark", "sae", "interpretability", "attn", "ffn", "eval"]


def cfg_dict(module):
    return {name: copy.deepcopy(getattr(module, name)) for name in SECTIONS}


def parse_like(old, value):
    if isinstance(old, bool):
        return value.lower() in ("true", "1", "yes", "y")
    if isinstance(old, int) and not isinstance(old, bool):
        return int(value)
    if isinstance(old, float):
        return float(value)
    if isinstance(old, list):
        values = [v for v in value.split(",") if v]
        if not old:
            return values
        return [parse_like(old[0], v) for v in values]
    return value


def set_value(target, key, value, allow_new=False):
    if "." in key:
        head, rest = key.split(".", 1)
        if head not in target:
            raise KeyError(key)
        set_value(target[head], rest, value, allow_new=allow_new)
        return
    if key not in target and not allow_new:
        raise KeyError(key)
    if key in target:
        target[key] = parse_like(target[key], value)
    else:
        target[key] = parse_scalar(value)


def parse_scalar(value):
    v = value.lower()
    if v in ("true", "false"):
        return v == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def print_cfg(cfg, key):
    if key == "all":
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return
    if key not in cfg:
        print(f"unknown cfg section: {key}")
        return
    print(json.dumps(cfg[key], ensure_ascii=False, indent=2))


def serializable_cfg(cfg):
    return copy.deepcopy(cfg)
