import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import torch

from data import GenerateData
from eval import Evaluate
from interpret import InterpretSAE
from logger import ExperimentLogger
from model import SelfTransformer
from para import (DATA,EVAL,INTERP,MODEL,PATH,SAE,SECRETS,TRAIN,REMARK)
from sae import SelfSAE
from train import Train
from utils import (ensure_dirs,load_json,manifest_is_current,namespace_to_dict,runtime_environment_info,save_json,save_manifest)


# Pipeline switches. Turning a stage off does not fabricate its output:
# downstream stages must be able to load the corresponding saved result.
PIPELINE = SimpleNamespace(
    run_data=True,
    run_model=True,
    run_train=True,
    run_attention=True,
    # Attention is a separate post-train stage. Direct "run attention" requires
    # completed train checkpoints; "run all" runs train before attention.
    run_sae=True,
    run_eval=True,
    run_interpret=False,
    # Optional training scheduler. It shards TRAIN by model_name x seed and
    # launches one shard per selected GPU.
    use_train_scheduler=False,
    train_scheduler_gpu_ids=["0", "1"],
    train_scheduler_max_parallel=None,
    task_name="default",
)

def prepare_dirs():
    ensure_dirs(
        [
            PATH.cache_dir,
            Path(PATH.cache_dir) / "tokens",
            Path(PATH.cache_dir) / "activations",
            Path(PATH.cache_dir) / "sae_features",
            Path(PATH.ckpt_dir) / "models",
            Path(PATH.ckpt_dir) / "saes",
            PATH.figure_dir,
            PATH.table_dir,
            PATH.log_dir,
            PATH.report_dir,
            PATH.raw_metrics_dir,
        ]
    )


def set_task_paths(task_name):
    PIPELINE.task_name = task_name
    PATH.output_dir = f"./output/{task_name}"
    PATH.ckpt_dir = f"./ckpt/{task_name}"
    PATH.figure_dir = f"./output/{task_name}/figures"
    PATH.table_dir = f"./output/{task_name}/tables"
    PATH.log_dir = f"./output/{task_name}/logs"
    PATH.report_dir = f"./output/{task_name}/reports"
    PATH.raw_metrics_dir = f"./output/{task_name}/raw_metrics"
    prepare_dirs()


def task_status():
    return {
        "task_name": PIPELINE.task_name,
        "paths": namespace_to_dict(PATH),
        "exists": {
            "output_dir": Path(PATH.output_dir).exists(),
            "ckpt_dir": Path(PATH.ckpt_dir).exists(),
            "raw_metrics_dir": Path(PATH.raw_metrics_dir).exists(),
        },
        "shared_cache_dir": PATH.cache_dir,
        "shared_hf_cache_dir": DATA.hf_cache_dir,
    }


def print_task_status():
    print(json.dumps(task_status(), indent=2, ensure_ascii=False))


def print_current_task():
    print(f"[task] current: {PIPELINE.task_name}")


def paths_for_task(task_name):
    output_dir = Path("./output") / task_name
    ckpt_dir = Path("./ckpt") / task_name
    return {
        "output_dir": output_dir,
        "ckpt_dir": ckpt_dir,
        "figure_dir": output_dir / "figures",
        "table_dir": output_dir / "tables",
        "raw_metrics_dir": output_dir / "raw_metrics",
    }


def maybe_prepare_current_task(task_name):
    if task_name == PIPELINE.task_name:
        prepare_dirs()

def remove_path(path):
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def final_model_checkpoints_exist():
    model_dir = Path(PATH.ckpt_dir) / "models"
    for model_name in MODEL.model_names:
        for seed in TRAIN.seeds:
            path = model_dir / f"{model_name}_seed{seed}_step{TRAIN.steps}.pt"
            if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
                return False
    return True


def data_cache_exists():
    if not getattr(DATA, "use_cache", True):
        return False
    runner = GenerateData(DATA)
    path = runner.cache_path
    return path.exists() and path.is_file() and path.stat().st_size > 0


def stage_done(stage):
    raw = Path(PATH.raw_metrics_dir)
    checks = {
        "data": lambda: data_cache_exists(),
        "train": lambda: all(
            (raw / name).exists()
            for name in [
                "train_res.json",
                "phase2_summary.json",
                "phase2_checkpoint_comparison.json",
                "train_manifest.json",
            ]
        )
        and final_model_checkpoints_exist(),
        "sae": lambda: all(
            (raw / name).exists()
            for name in ["sae_res.json", "phase4a_summary.json", "sae_manifest.json"]
        ),
        "eval": lambda: all(
            (raw / name).exists()
            for name in ["eval_res.json", "phase5_summary.json", "eval_manifest.json"]
        ),
        "attention": lambda: attention_outputs_exist(),
        "phase3": lambda: attention_outputs_exist(),
        "interpret": lambda: all(
            (raw / name).exists()
            for name in [
                "phase6_interpretation_summary.json",
                "phase6_prompts.json",
                "phase6_run_records.json",
                "interpret_manifest.json",
            ]
        ),
    }
    return checks[stage]()


def stage_config(stage):
    configs = {
        "data": {"data": DATA},
        "model": {"model": MODEL},
        "train": {"train": TRAIN},
        "attention": {
            "train_attention": {
                "analysis_batches": TRAIN.analysis_batches,
                "analysis_batch_size": TRAIN.analysis_batch_size,
                "local_attention_windows": TRAIN.local_attention_windows,
                "long_range_fraction": TRAIN.long_range_fraction,
                "spectral_topk_values": TRAIN.spectral_topk_values,
                "spectral_analysis_layers": TRAIN.spectral_analysis_layers,
                "spectral_analysis_heads": TRAIN.spectral_analysis_heads,
                "representative_layers": TRAIN.representative_layers,
                "representative_heads": TRAIN.representative_heads,
                "max_heatmap_seq_len": TRAIN.max_heatmap_seq_len,
                "run_phase3_analysis": TRAIN.run_phase3_analysis,
                "run_attention_heatmaps": TRAIN.run_attention_heatmaps,
                "run_spectral_plots": TRAIN.run_spectral_plots,
                "run_attn_entropy": TRAIN.run_attn_entropy,
                "run_attn_distance": TRAIN.run_attn_distance,
                "run_sv_distribution": TRAIN.run_sv_distribution,
                "run_toeplitz": TRAIN.run_toeplitz,
            }
        },
        "phase3": {
            "train_attention": {
                "analysis_batches": TRAIN.analysis_batches,
                "analysis_batch_size": TRAIN.analysis_batch_size,
                "local_attention_windows": TRAIN.local_attention_windows,
                "long_range_fraction": TRAIN.long_range_fraction,
                "spectral_topk_values": TRAIN.spectral_topk_values,
                "spectral_analysis_layers": TRAIN.spectral_analysis_layers,
                "spectral_analysis_heads": TRAIN.spectral_analysis_heads,
                "representative_layers": TRAIN.representative_layers,
                "representative_heads": TRAIN.representative_heads,
                "max_heatmap_seq_len": TRAIN.max_heatmap_seq_len,
                "run_phase3_analysis": TRAIN.run_phase3_analysis,
                "run_attention_heatmaps": TRAIN.run_attention_heatmaps,
                "run_spectral_plots": TRAIN.run_spectral_plots,
                "run_attn_entropy": TRAIN.run_attn_entropy,
                "run_attn_distance": TRAIN.run_attn_distance,
                "run_sv_distribution": TRAIN.run_sv_distribution,
                "run_toeplitz": TRAIN.run_toeplitz,
            }
        },
        "sae": {"sae": SAE},
        "eval": {"eval": EVAL},
        "interpret": {"interpret": INTERP},
        "task": task_status,
        "all": experiment_config,
    }
    item = configs[stage]
    return item() if callable(item) else namespace_to_dict(item)


def print_stage_config(stage):
    print_json = json.dumps(stage_config(stage), indent=2, ensure_ascii=False)
    print(print_json)


def cfg_namespace(stage):
    namespaces = {
        "data": DATA,
        "model": MODEL,
        "train": TRAIN,
        "attention": TRAIN,
        "phase3": TRAIN,
        "sae": SAE,
        "eval": EVAL,
        "interpret": INTERP,
        "pipeline": PIPELINE,
    }
    return namespaces[stage]


def parse_cfg_value(text, current_value):
    if isinstance(current_value, bool):
        lowered = text.lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"Expected boolean value, got {text}")
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(text)
    if isinstance(current_value, float):
        return float(text)
    if isinstance(current_value, str):
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                return text
        return parsed if isinstance(parsed, str) else text
    if isinstance(current_value, list):
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception as literal_exc:
                raise ValueError(f"Expected list value, got {text}") from literal_exc
        if not isinstance(parsed, list):
            raise ValueError(f"Expected list value, got {text}")
        return parsed
    if isinstance(current_value, tuple):
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception as literal_exc:
                raise ValueError(f"Expected list/tuple value, got {text}") from literal_exc
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"Expected list/tuple value, got {text}")
        return tuple(parsed)
    if isinstance(current_value, dict):
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception as literal_exc:
                raise ValueError(f"Expected object value, got {text}") from literal_exc
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected object value, got {text}")
        return parsed
    if current_value is None:
        lowered = text.lower()
        if lowered in {"none", "null"}:
            return None
        try:
            return json.loads(text)
        except Exception:
            try:
                return ast.literal_eval(text)
            except Exception:
                return text
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return text


def split_cfg_assignments(text):
    tokens = []
    current = []
    quote = None
    escape = False
    bracket_depth = 0
    pairs = {"[": "]", "{": "}", "(": ")"}
    closers = set(pairs.values())

    for char in text:
        if escape:
            current.append(char)
            escape = False
            continue
        if char == "\\":
            current.append(char)
            escape = True
            continue
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            current.append(char)
            quote = char
            continue
        if char in pairs:
            bracket_depth += 1
            current.append(char)
            continue
        if char in closers:
            bracket_depth = max(0, bracket_depth - 1)
            current.append(char)
            continue
        if char.isspace() and bracket_depth == 0:
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(char)

    if quote:
        raise ValueError(f"Unclosed quote: {quote}")
    if bracket_depth:
        raise ValueError("Unclosed bracket in config assignments.")
    if current:
        tokens.append("".join(current))
    return tokens


def set_stage_config(stage, assignments_text):
    cfg = cfg_namespace(stage)
    tokens = split_cfg_assignments(assignments_text)

    for token in tokens:
        key, value_text = token.split("=", 1)
        key = key.strip()
        current_value = getattr(cfg, key)
        setattr(cfg, key, parse_cfg_value(value_text, current_value))
    print("done")


def nvidia_smi_status():
    query = "index,name,driver_version,memory.total,memory.free,memory.used,compute_cap"
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found on PATH", "gpu_count": 0, "gpus": []}
    except subprocess.CalledProcessError as exc:
        return {
            "available": False,
            "error": (exc.stderr or exc.stdout or str(exc)).strip(),
            "gpu_count": 0,
            "gpus": [],
        }

    gpus = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        index, name, driver, total, free, used, compute_cap = parts[:7]
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "driver_version": driver,
                "compute_capability": compute_cap,
                "total_memory_gb": round(float(total) / 1024, 3),
                "free_memory_gb": round(float(free) / 1024, 3),
                "used_memory_gb": round(float(used) / 1024, 3),
            }
        )
    return {"available": True, "error": None, "gpu_count": len(gpus), "gpus": gpus}


def torch_cuda_status():
    status = {
        "available": False,
        "gpu_count": 0,
        "cuda_version": torch.version.cuda,
        "error": None,
        "gpus": [],
    }
    try:
        status["available"] = torch.cuda.is_available()
        status["gpu_count"] = torch.cuda.device_count() if status["available"] else 0
    except Exception as exc:
        status["error"] = str(exc)
        return status

    if not status["available"]:
        return status

    for idx in range(status["gpu_count"]):
        props = torch.cuda.get_device_properties(idx)
        free_bytes = None
        total_visible_bytes = None
        try:
            with torch.cuda.device(idx):
                free_bytes, total_visible_bytes = torch.cuda.mem_get_info()
        except Exception as exc:
            status["error"] = str(exc)

        total_bytes = int(props.total_memory)
        used_bytes = (
            int(total_visible_bytes - free_bytes)
            if free_bytes is not None and total_visible_bytes is not None
            else None
        )
        status["gpus"].append(
            {
                "index": idx,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count_sm": props.multi_processor_count,
                "total_memory_gb": round(total_bytes / (1024**3), 3),
                "free_memory_gb": round(int(free_bytes) / (1024**3), 3) if free_bytes is not None else None,
                "used_memory_gb": round(used_bytes / (1024**3), 3) if used_bytes is not None else None,
                "torch_current_process_allocated_gb": round(torch.cuda.memory_allocated(idx) / (1024**3), 3),
                "torch_current_process_reserved_gb": round(torch.cuda.memory_reserved(idx) / (1024**3), 3),
            }
        )
    return status


def use_gpu_count(requested_count):
    smi = nvidia_smi_status()
    torch_status = torch_cuda_status()
    physical_count = smi["gpu_count"]
    torch_count = torch_status["gpu_count"]
    if physical_count and physical_count != requested_count:
        raise RuntimeError(
            f"Requested {requested_count} GPU(s), but nvidia-smi detects {physical_count}. "
            "Exact match is required."
        )
    if torch_count != requested_count:
        if smi["available"]:
            raise RuntimeError(
                f"Requested {requested_count} GPU(s). nvidia-smi detects {physical_count} physical GPU(s), "
                f"but PyTorch detects {torch_count} usable CUDA GPU(s). "
                f"torch.version.cuda={torch_status['cuda_version']}. "
                "This usually means the NVIDIA driver is too old for this PyTorch CUDA build, "
                "or the current Python/conda environment cannot access CUDA."
            )
        raise RuntimeError(
            f"Requested {requested_count} GPU(s), but nvidia-smi is unavailable ({smi['error']}) "
            f"and PyTorch detects {torch_count} usable CUDA GPU(s)."
        )
    gpu_ids = [str(idx) for idx in range(requested_count)]
    TRAIN.device = "cuda"
    PIPELINE.train_scheduler_gpu_ids = gpu_ids
    PIPELINE.train_scheduler_max_parallel = requested_count
    PIPELINE.use_train_scheduler = requested_count > 1
    print(
        "done "
        f"detected_gpus={torch_count} "
        f"use_train_scheduler={PIPELINE.use_train_scheduler} "
        f"train_scheduler_gpu_ids={PIPELINE.train_scheduler_gpu_ids}"
    )


def gpu_status():
    smi = nvidia_smi_status()
    torch_status = torch_cuda_status()
    return {
        "nvidia_smi": smi,
        "torch_cuda": torch_status,
        "diagnosis": (
            "nvidia-smi sees GPU(s), but PyTorch cannot use CUDA. Fix driver/PyTorch CUDA compatibility."
            if smi["gpu_count"] > 0 and torch_status["gpu_count"] == 0
            else None
        ),
        "configured_train_device": TRAIN.device,
        "use_train_scheduler": PIPELINE.use_train_scheduler,
        "train_scheduler_gpu_ids": PIPELINE.train_scheduler_gpu_ids,
        "train_scheduler_max_parallel": PIPELINE.train_scheduler_max_parallel,
    }


def print_gpu_status():
    print(json.dumps(gpu_status(), indent=2, ensure_ascii=False))


def attention_outputs_exist():
    raw = Path(PATH.raw_metrics_dir)
    required = ["attention_res.json", "phase3_summary.json", "attention_manifest.json"]
    if not all((raw / name).exists() for name in required):
        return False
    try:
        attention_res = load_json(raw / "attention_res.json")
    except Exception:
        return False
    for model_name in MODEL.model_names:
        seed_items = attention_res.get(model_name, {})
        for seed in TRAIN.seeds:
            item = seed_items.get(str(seed), {})
            if "phase3" not in item.get("analysis_res", {}):
                return False
    return True


def attention_stage_config(train_runner, data_res, train_res):
    train_checkpoints = {}
    for model_name, seed_items in train_res.items():
        train_checkpoints[model_name] = {}
        for seed, item in seed_items.items():
            state = item["train_state"]
            train_checkpoints[model_name][str(seed)] = {
                "checkpoint_path": state.get("checkpoint_path"),
                "checkpoint_step": state.get("checkpoint_step"),
                "tokens_seen": state.get("tokens_seen"),
                "valid_loss_at_checkpoint": state.get("valid_loss_at_checkpoint"),
            }
    return {
        "attention": stage_config("attention"),
        "model": train_runner.model_cfg,
        "data_meta": data_res.get("meta", {}),
        "train_checkpoints": train_checkpoints,
    }


def attention_stage_outputs():
    return [
        Path(PATH.raw_metrics_dir) / "attention_res.json",
        Path(PATH.raw_metrics_dir) / "phase3_summary.json",
        Path(PATH.raw_metrics_dir) / "attention_manifest.json",
        Path(PATH.table_dir) / "phase3_layer_metrics.csv",
        Path(PATH.table_dir) / "phase3_taxonomy_counts.csv",
    ]


def attention_manifest_path():
    return Path(PATH.raw_metrics_dir) / "attention_manifest.json"


def run_data_stage():
    prepare_dirs()
    logger = ExperimentLogger()
    save_experiment_metadata(logger)
    return load_or_run_data(logger)


def run_train_stage():
    prepare_dirs()
    logger = ExperimentLogger()
    save_experiment_metadata(logger)
    data_res = load_or_run_data(logger)
    model_res = load_or_run_model(logger)
    return load_or_run_train(model_res, data_res, logger)


def run_sae_stage():
    prepare_dirs()
    logger = ExperimentLogger()
    save_experiment_metadata(logger)
    data_res = load_or_run_data(logger)
    model_res = load_or_run_model(logger)
    train_res = load_or_run_train(model_res, data_res, logger)
    return load_or_run_sae(train_res, data_res, logger)


def run_eval_stage():
    prepare_dirs()
    logger = ExperimentLogger()
    save_experiment_metadata(logger)
    data_res = load_or_run_data(logger)
    model_res = load_or_run_model(logger)
    train_res = load_or_run_train(model_res, data_res, logger)
    sae_res = load_or_run_sae(train_res, data_res, logger)
    return load_or_run_eval(train_res, sae_res, data_res, logger)


def run_attention_stage():
    prepare_dirs()
    logger = ExperimentLogger()
    save_experiment_metadata(logger)
    data_res = load_or_run_data(logger)
    model_res = load_or_run_model(logger)
    train_runner = Train(TRAIN, model_res, data_res)
    train_res = train_runner.load_completed_train_res()
    if train_res is None:
        raise RuntimeError("Attention analysis requires completed train results/checkpoints. Run 'run train' first.")
    return load_or_run_attention(train_res, data_res, model_res, logger)


def load_or_run_attention(train_res, data_res, model_res, logger):
    train_runner = Train(TRAIN, model_res, data_res)
    stage_config = attention_stage_config(train_runner, data_res, train_res)
    if manifest_is_current(attention_manifest_path(), stage_config, attention_stage_outputs()):
        logger.log_stage_start("skip attention stage: existing outputs match config")
        train_runner.plot_phase3_combined_figures()
        return load_json(Path(PATH.raw_metrics_dir) / "attention_res.json")

    attention_res = {}
    for model_name, seed_items in train_res.items():
        attention_res[model_name] = {}
        for seed, item in seed_items.items():
            logger.log_stage_start(f"Attention stage: model={model_name} seed={seed}")
            analysis_res = train_runner.attention_analysis(
                item["model"],
                model_name=model_name,
                seed=seed,
            )
            attention_res[model_name][str(seed)] = {"analysis_res": analysis_res}
            logger.log_stage_end(f"Attention stage: model={model_name} seed={seed}")

    save_json(attention_res, Path(PATH.raw_metrics_dir) / "attention_res.json")
    phase3_summary = train_runner.summarize_phase3(attention_res)
    save_json(phase3_summary, Path(PATH.raw_metrics_dir) / "phase3_summary.json")
    train_runner.write_phase3_summary_tables(phase3_summary)
    train_runner.plot_phase3_combined_figures()
    save_manifest(
        attention_manifest_path(),
        "attention",
        stage_config,
        attention_stage_outputs(),
    )
    return attention_res


def run_interpret_stage():
    prepare_dirs()
    logger = ExperimentLogger()
    save_experiment_metadata(logger)
    data_res = load_or_run_data(logger)
    model_res = load_or_run_model(logger)
    train_res = load_or_run_train(model_res, data_res, logger)
    sae_res = load_or_run_sae(train_res, data_res, logger)
    eval_res = load_or_run_eval(train_res, sae_res, data_res, logger)
    logger.log_stage_start(
        f"Interpret stage: provider={INTERP.provider} model={INTERP.model} "
        f"dry_run={INTERP.dry_run} layers={list(INTERP.layers)}"
    )
    result = InterpretSAE(INTERP, train_res, sae_res, data_res, eval_res).run()
    log_result(logger, "Interpret stage", result)
    return result


def clear_output(task_name):
    paths = paths_for_task(task_name)
    remove_path(paths["output_dir"])
    maybe_prepare_current_task(task_name)


def clear_models(task_name):
    paths = paths_for_task(task_name)
    remove_path(paths["ckpt_dir"] / "models")
    for name in [
        "train_res.json",
        "phase2_summary.json",
        "phase2_checkpoint_comparison.json",
        "train_manifest.json",
    ]:
        remove_path(paths["raw_metrics_dir"] / name)
    clear_attention(task_name)
    maybe_prepare_current_task(task_name)


def clear_attention(task_name):
    paths = paths_for_task(task_name)
    for name in ["attention_res.json", "phase3_summary.json", "attention_manifest.json"]:
        remove_path(paths["raw_metrics_dir"] / name)
    for name in ["phase3_layer_metrics.csv", "phase3_taxonomy_counts.csv"]:
        remove_path(paths["table_dir"] / name)
    remove_path(paths["figure_dir"] / "detail" / "phase3")
    maybe_prepare_current_task(task_name)


def clear_sae(task_name):
    paths = paths_for_task(task_name)
    remove_path(paths["ckpt_dir"] / "saes")
    for name in ["sae_res.json", "phase4a_summary.json", "sae_manifest.json"]:
        remove_path(paths["raw_metrics_dir"] / name)
    maybe_prepare_current_task(task_name)


def print_console_help():
    print("Commands:")
    print("  run train | run attention | run sae | run eval | run interpret | run all")
    print("  check data result | check train result | check attention result | check sae result")
    print("  check eval result | check interpret result")
    print("  check data cfg | check model cfg | check train cfg | check attention cfg")
    print("  check sae cfg | check eval cfg | check interpret cfg | check task cfg | check all cfg")
    print("  set task <task_name>")
    print("  set train cfg key=value [key=value ...]")
    print("  set data/model/attention/sae/eval/interpret/pipeline cfg key=value [key=value ...]")
    print("  use gpu <count>")
    print("  check gpu")
    print("  clear <task_name> output | clear <task_name> models | clear <task_name> attention | clear <task_name> sae")
    print("  help | exit")


def handle_console_command(command):
    raw_command = command.strip()
    command = raw_command.lower()
    if not command:
        return True
    if command in {"exit", "quit", "q"}:
        return False
    if command == "help":
        print_console_help()
        return True
    if command in {"checkgpu", "check gpu"}:
        print_gpu_status()
        return True

    if command == "set task" or command.startswith("set task "):
        parts = raw_command.split(maxsplit=2)
        set_task_paths(parts[2].strip())
        print_task_status()
        return True

    run_commands = {
        "run train": run_train_stage,
        "run attention": run_attention_stage,
        "run phase3": run_attention_stage,
        "run sae": run_sae_stage,
        "run eval": run_eval_stage,
        "run interpret": run_interpret_stage,
        "run all": run,
    }
    if command in run_commands:
        run_commands[command]()
        print("done")
        return True

    check_prefix = "check "
    if command.startswith(check_prefix):
        parts = command[len(check_prefix) :].split()
        stage, target = parts
        if target == "result":
            print(str(stage_done(stage)).lower())
        elif target == "cfg":
            print_stage_config(stage)
        return True

    set_prefix = "set "
    if command.startswith(set_prefix):
        parts = raw_command.split(maxsplit=3)
        set_stage_config(parts[1].lower(), parts[3])
        return True

    use_gpu_prefix = "use gpu "
    if command.startswith(use_gpu_prefix):
        parts = command.split()
        use_gpu_count(int(parts[2]))
        return True

    if command.startswith("clear "):
        parts = raw_command.split()
        task_name = parts[1]
        target = parts[2].lower()
        clear_commands = {
            "output": clear_output,
            "models": clear_models,
            "attention": clear_attention,
            "sae": clear_sae,
        }
        clear_commands[target](task_name)
        print("done")
        return True

    print(f"Unknown command: {command}")
    print_console_help()
    return True


def interactive_console():
    prepare_dirs()
    print(REMARK)
    print("Interactive experiment console. Type 'help' for commands, 'exit' to quit.")
    print_current_task()
    while True:
        try:
            command = input(f"main.py[{PIPELINE.task_name}]> ")
        except EOFError:
            print()
            break
        try:
            keep_running = handle_console_command(command)
        except Exception as exc:
            print(f"error: {exc}")
            keep_running = True
        if not keep_running:
            break


def experiment_config():
    return {
        "path": namespace_to_dict(PATH),
        "data": namespace_to_dict(DATA),
        "model": namespace_to_dict(MODEL),
        "train": namespace_to_dict(TRAIN),
        "sae": namespace_to_dict(SAE),
        "eval": namespace_to_dict(EVAL),
        "interpretation": namespace_to_dict(INTERP),
        "main": {
            "run_data": PIPELINE.run_data,
            "run_model": PIPELINE.run_model,
            "run_train": PIPELINE.run_train,
            "run_attention": PIPELINE.run_attention,
            "run_sae": PIPELINE.run_sae,
            "run_eval": PIPELINE.run_eval,
            "run_interpret": PIPELINE.run_interpret,
            "use_train_scheduler": PIPELINE.use_train_scheduler,
            "train_scheduler_gpu_ids": PIPELINE.train_scheduler_gpu_ids,
            "train_scheduler_max_parallel": PIPELINE.train_scheduler_max_parallel,
            "task_name": PIPELINE.task_name,
        },
        "secrets": {
            "hf_token_set": bool(getattr(SECRETS, "hf_token", None)),
            "openai_api_key_set": bool(getattr(SECRETS, "openai_api_key", None)),
        },
    }

def experiment_manifest():
    # This manifest is the reproducibility record for a run. Stage-level
    # manifests use smaller config snapshots to decide whether cached outputs
    # are still valid.
    config = experiment_config()
    return {
        "task_name": PIPELINE.task_name,
        "config_snapshot": config,
        "random_seeds": {
            "data_seed": getattr(DATA, "seed", None),
            "train_seeds": list(getattr(TRAIN, "seeds", [])),
            "sae_seeds": list(getattr(SAE, "seeds", [])),
            "interpret_model_seeds": list(getattr(INTERP, "model_seeds", [])),
        },
        "environment": runtime_environment_info(),
    }

def set_shard_output_dirs(shard_root):
    # Shards share DATA/MODEL/TRAIN config, but write metrics/logs into isolated
    # folders so parallel runs do not overwrite each other.
    shard_root = Path(shard_root)
    PATH.figure_dir = str(shard_root / "figures")
    PATH.table_dir = str(shard_root / "tables")
    PATH.log_dir = str(shard_root / "logs")
    PATH.report_dir = str(shard_root / "reports")
    PATH.raw_metrics_dir = str(shard_root / "raw_metrics")

def run_train_shard(model_name, seed, device, shard_root):
    MODEL.model_names = [model_name]
    TRAIN.seeds = [int(seed)]
    TRAIN.device = device
    set_shard_output_dirs(shard_root)
    prepare_dirs()
    save_json(experiment_config(), Path(PATH.raw_metrics_dir) / "experiment_config.json")
    save_json(experiment_manifest(), Path(PATH.raw_metrics_dir) / "experiment_manifest.json")
    data_res = GenerateData(DATA).run()
    model_res = SelfTransformer(MODEL).run()
    Train(TRAIN, model_res, data_res).run()

def train_shard_commands():
    shard_root = Path(PATH.output_dir) / "train_shards"
    commands = []
    for model_name in MODEL.model_names:
        for seed in TRAIN.seeds:
            shard_dir = shard_root / f"{model_name}_seed{seed}"
            commands.append(
                {
                    "model_name": model_name,
                    "seed": seed,
                    "shard_dir": shard_dir,
                    "cmd": [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--train-shard",
                        "--model-name",
                        model_name,
                        "--seed",
                        str(seed),
                        "--device",
                        "cuda:0",
                        "--shard-root",
                        str(shard_dir),
                        "--task-name",
                        PIPELINE.task_name,
                    ],
                }
            )
    return commands

def launch_train_scheduler():
    gpu_ids = PIPELINE.train_scheduler_gpu_ids or ["0"]
    max_parallel = PIPELINE.train_scheduler_max_parallel or len(gpu_ids)
    max_parallel = max(1, min(max_parallel, len(gpu_ids)))
    queue = train_shard_commands()
    running = []
    completed = []

    while queue or running:
        while queue and len(running) < max_parallel:
            item = queue.pop(0)
            gpu_id = gpu_ids[(len(completed) + len(running)) % len(gpu_ids)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            ensure_dirs([item["shard_dir"]])
            process = subprocess.Popen(item["cmd"], env=env)
            running.append({**item, "gpu_id": gpu_id, "process": process})

        still_running = []
        for item in running:
            code = item["process"].poll()
            if code is None:
                still_running.append(item)
            elif code == 0:
                completed.append(item)
            else:
                raise RuntimeError(
                    f"Train shard failed: model={item['model_name']} seed={item['seed']} gpu={item['gpu_id']} exit={code}"
                )
        running = still_running
        if queue or running:
            time.sleep(5)
    return completed

def aggregate_train_shards(model_res, data_res, shard_items):
    # After scheduled training, merge per-shard train_res.json files back into
    # the same structure returned by a normal Train.run().
    serializable = {}
    for item in shard_items:
        shard_train_path = Path(item["shard_dir"]) / "raw_metrics" / "train_res.json"
        shard_res = load_json(shard_train_path)
        for model_name, seed_items in shard_res.items():
            serializable.setdefault(model_name, {}).update(seed_items)

    save_json(serializable, Path(PATH.raw_metrics_dir) / "train_res.json")
    train_runner = Train(TRAIN, model_res, data_res)
    train_res = train_runner.load_completed_train_res()
    matched_target = train_runner.attach_checkpoint_selection(train_res, serializable)
    save_json(serializable, Path(PATH.raw_metrics_dir) / "train_res.json")

    summary = train_runner.summarize_results(serializable)
    checkpoint_comparison = train_runner.checkpoint_comparison_rows(serializable)
    summary["checkpoint_comparison"] = checkpoint_comparison
    summary["checkpoint_protocol"] = {
        "primary_checkpoint_rule": getattr(TRAIN, "primary_checkpoint_rule", "final_step"),
        "secondary_checkpoint_rule": getattr(TRAIN, "secondary_checkpoint_rule", "validation_loss_matched"),
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
    train_runner.write_summary_tables(summary)
    train_runner.plot_phase2_summary_figures()
    save_manifest(train_runner.stage_manifest_path(), "train", train_runner.stage_config(), train_runner.stage_outputs())
    return train_res

def scheduled_train(model_res, data_res):
    train_runner = Train(TRAIN, model_res, data_res)
    if manifest_is_current(train_runner.stage_manifest_path(), train_runner.stage_config(), train_runner.stage_outputs()):
        maybe_loaded = train_runner.load_completed_train_res()
        if maybe_loaded is not None:
            return maybe_loaded
    shard_items = launch_train_scheduler()
    return aggregate_train_shards(model_res, data_res, shard_items)

def log_result(logger, stage_name, result):
    if result is None:
        logger.log_stage_end(f"{stage_name}: no result")
    elif isinstance(result, dict):
        logger.log_stage_end(f"{stage_name}: result keys={list(result.keys())}")
    else:
        logger.log_stage_end(f"{stage_name}: result type={type(result).__name__}")


def save_experiment_metadata(logger):
    logger.log_stage_start("Manifest stage: writing experiment_config and experiment_manifest")
    save_json(experiment_config(), Path(PATH.raw_metrics_dir) / "experiment_config.json")
    save_json(experiment_manifest(), Path(PATH.raw_metrics_dir) / "experiment_manifest.json")
    logger.log_stage_end("Manifest stage")


def stage_needs(*flags):
    return any(flags)


def load_or_run_data(logger):
    logger.log_stage_start(
        f"Data stage: dataset={DATA.dataset} tokenizer={DATA.tokenizer} "
        f"seq_len={DATA.seq_len} train_blocks={DATA.train_blocks} valid_blocks={DATA.valid_blocks}"
    )
    result = GenerateData(DATA).run()
    meta = result.get("meta", {})
    logger.log_stage_end(
        f"Data stage: train_blocks={meta.get('num_train_blocks')} "
        f"valid_blocks={meta.get('num_valid_blocks')} vocab_size={meta.get('vocab_size')}"
    )
    return result

def load_or_run_model(logger):
    logger.log_stage_start(
        f"Model stage: models={list(MODEL.model_names)} layers={MODEL.n_layers} "
        f"d_model={MODEL.d_model} heads={MODEL.n_heads} seq_len={MODEL.seq_len}"
    )
    result = SelfTransformer(MODEL).run()
    logger.log_stage_end(
        f"Model stage: built={list(result.get('models', {}).keys())} "
        f"checks={list(result.get('checks', {}).keys())}"
    )
    return result

def load_or_run_train(model_res, data_res, logger):
    train_runner = Train(TRAIN, model_res, data_res)
    if PIPELINE.run_train and PIPELINE.use_train_scheduler:
        logger.log_stage_start(
            f"Train stage: scheduled run on GPUs={PIPELINE.train_scheduler_gpu_ids} "
            f"models={list(MODEL.model_names)} seeds={list(TRAIN.seeds)} steps={TRAIN.steps}"
        )
        result = scheduled_train(model_res, data_res)
        log_result(logger, "Train stage scheduled", result)
        return result
    if PIPELINE.run_train:
        logger.log_stage_start(
            f"Train stage: run models={list(MODEL.model_names)} seeds={list(TRAIN.seeds)} "
            f"steps={TRAIN.steps} batch={TRAIN.batch_size} device={TRAIN.device}"
        )
        result = train_runner.run()
        log_result(logger, "Train stage", result)
        return result
    logger.log_stage_start("Train stage: disabled, loading existing train results")
    loaded = train_runner.load_completed_train_res()
    if loaded is None:
        raise RuntimeError("Train stage is disabled, but existing train_res/checkpoints could not be loaded.")
    log_result(logger, "Train stage loaded", loaded)
    return loaded

def load_or_run_sae(train_res, data_res, logger):
    sae_runner = SelfSAE(SAE, train_res, data_res)
    if PIPELINE.run_sae:
        logger.log_stage_start(
            f"SAE stage: run layers={list(SAE.layers)} dict_sizes={list(SAE.dictionary_sizes)} "
            f"k={list(SAE.topk_values)} seeds={list(SAE.seeds)} steps={SAE.steps}"
        )
        result = sae_runner.run()
        log_result(logger, "SAE stage", result)
        return result
    logger.log_stage_start("SAE stage: disabled, loading existing SAE results")
    loaded = sae_runner.load_completed_sae_res()
    if loaded is None:
        raise RuntimeError("SAE stage is disabled, but existing sae_res/checkpoints could not be loaded.")
    log_result(logger, "SAE stage loaded", loaded)
    return loaded

def load_or_run_eval(train_res, sae_res, data_res, logger):
    eval_path = Path(PATH.raw_metrics_dir) / "eval_res.json"
    if PIPELINE.run_eval:
        logger.log_stage_start(
            f"Eval stage: run layers={list(EVAL.layers)} probe_steps={EVAL.probe_steps} "
            f"probe_train_tokens={EVAL.max_probe_train_tokens}"
        )
        result = Evaluate(EVAL, train_res, sae_res, data_res).run()
        log_result(logger, "Eval stage", result)
        return result
    logger.log_stage_start("Eval stage: disabled, trying to load existing eval results")
    if eval_path.exists():
        result = load_json(eval_path)
        log_result(logger, "Eval stage loaded eval_res", result)
        return result
    phase5_path = Path(PATH.raw_metrics_dir) / "phase5_summary.json"
    if phase5_path.exists():
        result = load_json(phase5_path)
        log_result(logger, "Eval stage loaded phase5_summary", result)
        return result
    logger.log_stage_end("Eval stage: no existing eval output found")
    return None

def run():
    # The main pipeline intentionally passes only config objects and prior stage
    # result dictionaries. Paths are imported by each stage from para.py.
    prepare_dirs()
    logger = ExperimentLogger()
    logger.log_stage_start(
        f"Main pipeline: task={PIPELINE.task_name} "
        f"flags=data:{PIPELINE.run_data} model:{PIPELINE.run_model} train:{PIPELINE.run_train} "
        f"attention:{PIPELINE.run_attention} sae:{PIPELINE.run_sae} "
        f"eval:{PIPELINE.run_eval} interpret:{PIPELINE.run_interpret}"
    )
    save_experiment_metadata(logger)

    needs_data = stage_needs(
        PIPELINE.run_data,
        PIPELINE.run_train,
        PIPELINE.run_attention,
        PIPELINE.run_sae,
        PIPELINE.run_eval,
        PIPELINE.run_interpret,
    )
    needs_model = stage_needs(
        PIPELINE.run_model,
        PIPELINE.run_train,
        PIPELINE.run_attention,
        PIPELINE.run_sae,
        PIPELINE.run_eval,
        PIPELINE.run_interpret,
    )
    needs_train = stage_needs(
        PIPELINE.run_train,
        PIPELINE.run_attention,
        PIPELINE.run_sae,
        PIPELINE.run_eval,
        PIPELINE.run_interpret,
    )
    needs_sae = stage_needs(PIPELINE.run_sae, PIPELINE.run_eval, PIPELINE.run_interpret)
    needs_eval = stage_needs(PIPELINE.run_eval, PIPELINE.run_interpret)

    data_res = load_or_run_data(logger) if needs_data else None
    model_res = load_or_run_model(logger) if needs_model else None
    train_res = load_or_run_train(model_res, data_res, logger) if needs_train else None
    attention_res = (
        load_or_run_attention(train_res, data_res, model_res, logger)
        if PIPELINE.run_attention and train_res is not None
        else None
    )
    sae_res = load_or_run_sae(train_res, data_res, logger) if needs_sae else None
    eval_res = load_or_run_eval(train_res, sae_res, data_res, logger) if needs_eval else None
    if PIPELINE.run_interpret:
        logger.log_stage_start(
            f"Interpret stage: provider={INTERP.provider} model={INTERP.model} "
            f"dry_run={INTERP.dry_run} layers={list(INTERP.layers)}"
        )
        interp_res = InterpretSAE(INTERP, train_res, sae_res, data_res, eval_res).run()
        log_result(logger, "Interpret stage", interp_res)
    else:
        interp_res = None
        logger.log_stage_end("Interpret stage: skipped")

    main_res = {
        "data_meta": data_res["meta"] if data_res else None,
        "model_meta": model_res["meta"] if model_res else None,
        "model_checks": model_res["checks"] if model_res else None,
        "parameter_audit": model_res["parameter_audit"] if model_res else None,
        "has_train_res": train_res is not None,
        "has_attention_res": attention_res is not None,
        "has_sae_res": sae_res is not None,
        "eval_res": eval_res,
        "interp_res": interp_res,
    }
    logger.log_stage_start("Main result stage: writing main_res.json")
    save_json(main_res, Path(PATH.raw_metrics_dir) / "main_res.json")
    logger.log_stage_end("Main result stage")
    logger.log_stage_end("Main pipeline")
    return main_res

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--train-shard", action="store_true")
    parser.add_argument("--model-name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-root")
    parser.add_argument("--task-name")
    parser.add_argument("--use-train-scheduler", action="store_true")
    parser.add_argument("--train-gpus")
    parser.add_argument("--train-max-parallel", type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.task_name:
        set_task_paths(args.task_name)
    if args.train_shard:
        print(REMARK)
        run_train_shard(args.model_name, args.seed, args.device, args.shard_root)
    else:
        if args.use_train_scheduler:
            PIPELINE.use_train_scheduler = True
        if args.train_gpus:
            PIPELINE.train_scheduler_gpu_ids = [
                item.strip() for item in args.train_gpus.split(",") if item.strip()
            ]
        if args.train_max_parallel:
            PIPELINE.train_scheduler_max_parallel = args.train_max_parallel
        if args.run_once:
            print(REMARK)
            run()
        else:
            interactive_console()
