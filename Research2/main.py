import importlib
import json
from pathlib import Path
import shutil
import shlex

import cfg as default_cfg
from tools.config import cfg_dict, print_cfg, set_value
from tools.io import block_dir, checkpoint_dir, data_tokenized_alias, data_tokenizer_alias, output_dir
from tools.resource import allocated_cpu_count, auto_num_threads, check_gpu, parallel_slots, resolve_jobs_per_gpu


STAGES = ["data", "benchmark", "train", "sae", "interpret", "attn", "ffn", "eval"]


def prompt(cfg):
    return f"[{cfg['run']['task']}] > "


def show_help():
    print(f"stage: {STAGES}")
    print("run [stage]")
    print("cfg [stage]")
    print("set [stage] [parameters]")
    print("set run mode=pretrain models=pythia410m")
    print("set run models=base,sp_both position_encodings=rope,alibi,cable")
    print("cfg premodel")
    print("clear [stage ...]")
    print("check gpu")
    print("plan")
    print("exit")


def run_stage(stage, cfg):
    if stage == "all":
        for name in STAGES:
            run_stage(name, cfg)
        return
    if stage not in STAGES:
        print(f"unknown stage: {stage}")
        return
    mod = importlib.import_module(f"pipeline.{stage}")
    mod.run(cfg)


def refresh_resources(cfg):
    available = check_gpu()
    n_gpu = len(available)
    n_cpu = allocated_cpu_count()
    jobs_per_gpu = resolve_jobs_per_gpu(cfg, available)
    cfg["run"]["gpus"] = available
    cfg["run"]["active_jobs_per_gpu"] = jobs_per_gpu
    cfg["run"]["num_threads"] = auto_num_threads(n_gpu, jobs_per_gpu)
    labels = ",".join(gpu_label for gpu_label in (f"GPU{i + 1}" for i in available)) or "CPU"
    requested_jobs = cfg["run"].get("jobs_per_gpu", 1)
    jobs_label = f"auto<={jobs_per_gpu}" if isinstance(requested_jobs, str) and requested_jobs.lower() == "auto" else str(jobs_per_gpu)
    print(f"[run] gpus={labels} | n_gpu={n_gpu} | jobs_per_gpu={jobs_label} | slots<={parallel_slots(available, jobs_per_gpu)} | n_cpu={n_cpu} | num_threads={cfg['run']['num_threads']}")


def handle_set(parts, cfg):
    if len(parts) < 3:
        print("usage: set section key=value ...")
        return
    section = parts[1]
    if section == "models":
        if len(parts) < 4:
            print("usage: set models alias key=value ...")
            return
        alias = parts[2]
        if alias not in cfg["models"]:
            cfg["models"][alias] = {"attn": {"name": "std"}, "ffn": {"name": "std"}, "loss": {"name": "ce"}}
        target = cfg["models"][alias]
        kvs = parts[3:]
    else:
        if section not in cfg:
            print(f"unknown cfg section: {section}")
            return
        target = cfg[section]
        kvs = parts[2:]
    for item in kvs:
        if "=" not in item:
            print(f"bad set item: {item}")
            return
        key, value = item.split("=", 1)
        try:
            set_value(target, key, value, allow_new=(section == "models"))
        except KeyError:
            print(f"unknown key: {key}")
            return


def remove_path(path):
    path = Path(path)
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"[clear] removed {path}")


def remove_glob(root, pattern):
    root = Path(root)
    if not root.exists():
        return
    for path in root.glob(pattern):
        remove_path(path)


def clear_stage(stage, cfg):
    out = output_dir(cfg)
    metrics = out / "metrics"
    eval_dir = out / "eval"

    if stage == "data":
        data_cfg = cfg["data"]
        remove_path(Path("data") / "raw" / data_cfg["raw_alias"])
        aliases = cfg["run"].get("models", [None]) if cfg["run"].get("mode", "retrain") == "pretrain" else [None]
        if isinstance(aliases, str):
            aliases = [aliases]
        for alias in aliases:
            for path in [
                Path("data") / "tokenizer" / data_tokenizer_alias(cfg, alias),
                Path("data") / "tokenized" / data_tokenized_alias(cfg, alias),
                block_dir(cfg, alias),
            ]:
                remove_path(path)
    elif stage == "train":
        remove_path(checkpoint_dir(cfg))
        remove_glob(metrics, "*train.jsonl")
        remove_glob(metrics, "*valid.jsonl")
        remove_path(metrics / "summary.json")
        if metrics.exists():
            for path in metrics.glob("*.json"):
                if path.name.startswith("[") and "seed" in path.name and path.name.endswith("summary.json"):
                    remove_path(path)
        remove_path(eval_dir / "loss_curve.png")
    elif stage == "sae":
        remove_path(out / "sae")
        remove_path(checkpoint_dir(cfg) / "sae")
        remove_glob(metrics, "*sae*")
        for name in [
            "sae_valid_loss.png",
            "loss_recovered.png",
            "active.png",
            "dead_feature_rate.png",
            "rare_feature_rate.png",
            "decoder_duplication_proxy.png",
            "feature_ablation_loss_delta_mean.png",
            "feature_ablation_loss_delta_max.png",
            "feature_ablation_coverage_mean.png",
        ]:
            remove_path(eval_dir / name)
    elif stage == "interpret":
        remove_path(out / "interpretability")
        remove_glob(metrics, "*interpret*")
        remove_glob(metrics, "*stability*")
        remove_path(metrics / "compute_plan.json")
        remove_glob(eval_dir, "*interpret*")
    elif stage == "attn":
        remove_glob(metrics, "*attn*")
        remove_glob(eval_dir, "*attn*")
    elif stage == "ffn":
        remove_glob(metrics, "*ffn*")
        remove_glob(eval_dir, "*ffn*")
    elif stage == "benchmark":
        remove_path(metrics / "benchmark_summary.json")
    elif stage == "eval":
        remove_path(metrics / "final_summary.json")
        remove_path(metrics / "final_summary.jsonl")
        remove_glob(eval_dir, "summary_*.png")
    else:
        print(f"unknown stage: {stage}")
        return
    print(f"[clear] {stage} done")


def handle_clear(parts, cfg):
    if len(parts) == 1:
        print("clear must specify stage name: data/benchmark/train/sae/interpret/attn/ffn/eval")
        return
    targets = parts[1:]
    for stage in targets:
        clear_stage(stage, cfg)


def main():
    cfg = cfg_dict(default_cfg)
    show_help()
    while True:
        try:
            line = input(prompt(cfg)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        parts = shlex.split(line)
        cmd = parts[0].lower()
        if cmd == "exit":
            break
        if cmd == "help":
            show_help()
        elif cmd == "cfg":
            key = parts[1] if len(parts) > 1 else "all"
            print_cfg(cfg, key)
        elif cmd == "set":
            handle_set(parts, cfg)
        elif cmd == "clear":
            handle_clear(parts, cfg)
        elif cmd == "run":
            refresh_resources(cfg)
            stages = parts[1:] if len(parts) > 1 else []
            for stage in stages:
                run_stage(stage, cfg)
        elif cmd == "check" and len(parts) > 1 and parts[1] == "gpu":
            refresh_resources(cfg)
        elif cmd == "plan":
            from model.factory import active_aliases
            from pipeline.interpret import experiment_counts, selected_interpret_specs
            from pipeline.train import run_seeds

            counts = experiment_counts(cfg, active_aliases(cfg), run_seeds(cfg), selected_interpret_specs(cfg))
            print(json.dumps(counts, ensure_ascii=False, indent=2))
        else:
            print(f"unknown command: {line}")


if __name__ == "__main__":
    main()
