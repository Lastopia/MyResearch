import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from data import GenerateData
from eval import Evaluate
from interpret import InterpretSAE
from logger import ExperimentLogger
from model import SelfTransformer
from para import DATA, EVAL, INTERP, MODEL, PATH, SAE, SECRETS, SMOKE_TEST, TRAIN, REMARK
from sae import SelfSAE
from train import Train
from utils import (
    ensure_dirs,
    load_json,
    manifest_is_current,
    namespace_to_dict,
    runtime_environment_info,
    save_json,
    save_manifest,
)


# Pipeline switches. Turning a stage off does not fabricate its output:
# downstream stages must be able to load the corresponding saved result.
RUN_DATA = True
RUN_MODEL = True
RUN_TRAIN = True
RUN_SAE = True
RUN_EVAL = True
RUN_INTERPRET = True
EXPERIMENT_NAME = "v0.2"

# Optional training scheduler. It shards TRAIN by model_name x seed and launches
# one shard per selected GPU. Leave disabled for a single H100 or simple runs.
USE_TRAIN_SCHEDULER = False
TRAIN_SCHEDULER_GPU_IDS = ["0", "1"]
TRAIN_SCHEDULER_MAX_PARALLEL = None

if SMOKE_TEST:
    RUN_SAE = False
    RUN_EVAL = False
    RUN_INTERPRET = False
    EXPERIMENT_NAME = "smoke_test"

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
            "run_data": RUN_DATA,
            "run_model": RUN_MODEL,
            "run_train": RUN_TRAIN,
            "run_sae": RUN_SAE,
            "run_eval": RUN_EVAL,
            "run_interpret": RUN_INTERPRET,
            "experiment_name": EXPERIMENT_NAME,
            "use_train_scheduler": USE_TRAIN_SCHEDULER,
            "train_scheduler_gpu_ids": TRAIN_SCHEDULER_GPU_IDS,
            "train_scheduler_max_parallel": TRAIN_SCHEDULER_MAX_PARALLEL,
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
        "experiment_name": EXPERIMENT_NAME,
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
                    ],
                }
            )
    return commands

def launch_train_scheduler():
    gpu_ids = TRAIN_SCHEDULER_GPU_IDS or ["0"]
    max_parallel = TRAIN_SCHEDULER_MAX_PARALLEL or len(gpu_ids)
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
    train_runner.plot_aggregate_curves(serializable)
    phase3_summary = train_runner.summarize_phase3(serializable)
    save_json(phase3_summary, Path(PATH.raw_metrics_dir) / "phase3_summary.json")
    train_runner.write_phase3_summary_tables(phase3_summary)
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
    if RUN_TRAIN and USE_TRAIN_SCHEDULER:
        logger.log_stage_start(
            f"Train stage: scheduled run on GPUs={TRAIN_SCHEDULER_GPU_IDS} "
            f"models={list(MODEL.model_names)} seeds={list(TRAIN.seeds)} steps={TRAIN.steps}"
        )
        result = scheduled_train(model_res, data_res)
        log_result(logger, "Train stage scheduled", result)
        return result
    if RUN_TRAIN:
        logger.log_stage_start(
            f"Train stage: run models={list(MODEL.model_names)} seeds={list(TRAIN.seeds)} "
            f"steps={TRAIN.steps} batch={TRAIN.batch_size} device={TRAIN.device}"
        )
        result = train_runner.run()
        log_result(logger, "Train stage", result)
        return result
    logger.log_stage_start("Train stage: RUN_TRAIN=False, loading existing train results")
    loaded = train_runner.load_completed_train_res()
    if loaded is None:
        raise RuntimeError("RUN_TRAIN is False, but existing train_res/checkpoints could not be loaded.")
    log_result(logger, "Train stage loaded", loaded)
    return loaded

def load_or_run_sae(train_res, data_res, logger):
    sae_runner = SelfSAE(SAE, train_res, data_res)
    if RUN_SAE:
        logger.log_stage_start(
            f"SAE stage: run layers={list(SAE.layers)} dict_sizes={list(SAE.dictionary_sizes)} "
            f"k={list(SAE.topk_values)} seeds={list(SAE.seeds)} steps={SAE.steps}"
        )
        result = sae_runner.run()
        log_result(logger, "SAE stage", result)
        return result
    logger.log_stage_start("SAE stage: RUN_SAE=False, loading existing SAE results")
    loaded = sae_runner.load_completed_sae_res()
    if loaded is None:
        raise RuntimeError("RUN_SAE is False, but existing sae_res/checkpoints could not be loaded.")
    log_result(logger, "SAE stage loaded", loaded)
    return loaded

def load_or_run_eval(train_res, sae_res, data_res, logger):
    eval_path = Path(PATH.raw_metrics_dir) / "eval_res.json"
    if RUN_EVAL:
        logger.log_stage_start(
            f"Eval stage: run layers={list(EVAL.layers)} probe_steps={EVAL.probe_steps} "
            f"probe_train_tokens={EVAL.max_probe_train_tokens}"
        )
        result = Evaluate(EVAL, train_res, sae_res, data_res).run()
        log_result(logger, "Eval stage", result)
        return result
    logger.log_stage_start("Eval stage: RUN_EVAL=False, trying to load existing eval results")
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
        f"Main pipeline: experiment={EXPERIMENT_NAME} "
        f"flags=data:{RUN_DATA} model:{RUN_MODEL} train:{RUN_TRAIN} "
        f"sae:{RUN_SAE} eval:{RUN_EVAL} interpret:{RUN_INTERPRET}"
    )
    save_experiment_metadata(logger)

    needs_data = stage_needs(RUN_DATA, RUN_TRAIN, RUN_SAE, RUN_EVAL, RUN_INTERPRET)
    needs_model = stage_needs(RUN_MODEL, RUN_TRAIN, RUN_SAE, RUN_EVAL, RUN_INTERPRET)
    needs_train = stage_needs(RUN_TRAIN, RUN_SAE, RUN_EVAL, RUN_INTERPRET)
    needs_sae = stage_needs(RUN_SAE, RUN_EVAL, RUN_INTERPRET)
    needs_eval = stage_needs(RUN_EVAL, RUN_INTERPRET)

    data_res = load_or_run_data(logger) if needs_data else None
    model_res = load_or_run_model(logger) if needs_model else None
    train_res = load_or_run_train(model_res, data_res, logger) if needs_train else None
    sae_res = load_or_run_sae(train_res, data_res, logger) if needs_sae else None
    eval_res = load_or_run_eval(train_res, sae_res, data_res, logger) if needs_eval else None
    if RUN_INTERPRET:
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
    parser.add_argument("--train-shard", action="store_true")
    parser.add_argument("--model-name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-root")
    parser.add_argument("--use-train-scheduler", action="store_true")
    parser.add_argument("--train-gpus")
    parser.add_argument("--train-max-parallel", type=int)
    return parser.parse_args()

if __name__ == "__main__":
    print(REMARK)
    args = parse_args()
    if args.train_shard:
        run_train_shard(args.model_name, args.seed, args.device, args.shard_root)
    else:
        if args.use_train_scheduler:
            USE_TRAIN_SCHEDULER = True
        if args.train_gpus:
            TRAIN_SCHEDULER_GPU_IDS = [item.strip() for item in args.train_gpus.split(",") if item.strip()]
        if args.train_max_parallel:
            TRAIN_SCHEDULER_MAX_PARALLEL = args.train_max_parallel
        run()
