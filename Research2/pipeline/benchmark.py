import gc
import math
import statistics
import time

import torch

from model.factory import active_aliases, build_model, resolve_experiment, validate_aliases
from tools.io import ensure_dir, output_dir, save_config, write_json
from tools.log import stage_title
from tools.resource import configured_gpus, cuda_device, gpu_label, job_gpu


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _training_block_step(block, x, device, dtype):
    block.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda",
    ):
        output = block(x)
        loss = output.float().square().mean()
    loss.backward()


def _logical_compute(cfg, alias, seq_len):
    experiment = resolve_experiment(cfg, alias)
    variant = cfg["models"][experiment["model_alias"]]
    ffn_cfg = variant["ffn"]
    attn_cfg = variant["attn"]
    if ffn_cfg["name"] in ("fast_sp", "fast_spark"):
        # Fused Spark GLU: a d->2w up projection plus w->d down,
        # compared with the standard d->d_ff->d pair.
        width = int(ffn_cfg.get("active_width", ffn_cfg.get("expert_width")))
        ffn_fraction = 1.5 * width / int(cfg["model"]["d_ff"])
    else:
        # The original SP reference produces sparse activations but executes
        # both full FFN projections, so its arithmetic fraction is still one.
        ffn_fraction = 1.0

    if attn_cfg["name"] in ("fast_sp", "fast_spark"):
        window = int(attn_cfg.get("window_size", attn_cfg.get("k", 256)))
        logical_pairs = sum(min(index + 1, window) for index in range(seq_len))
        causal_pairs = seq_len * (seq_len + 1) // 2
        chunk_size = int(attn_cfg.get("chunk_size", 512))
        fallback_pair_bound = 0
        for query_start in range(0, seq_len, chunk_size):
            query_end = min(seq_len, query_start + chunk_size)
            key_start = max(0, query_start - window + 1)
            fallback_pair_bound += (query_end - query_start) * (query_end - key_start)
        flex_block_size = int(attn_cfg.get("flex_block_size", 128))
        block_count = math.ceil(seq_len / flex_block_size)
        nonempty_blocks = 0
        for query_block in range(block_count):
            query_start = query_block * flex_block_size
            query_end = min(seq_len, query_start + flex_block_size)
            first_key = max(0, query_start - window + 1)
            first_key_block = first_key // flex_block_size
            last_key_block = (query_end - 1) // flex_block_size
            nonempty_blocks += last_key_block - first_key_block + 1
        attention_fraction = logical_pairs / max(1, causal_pairs)
        flex_block_density = nonempty_blocks / max(1, block_count * block_count)
        fallback_fraction = fallback_pair_bound / max(1, seq_len * seq_len)
    else:
        attention_fraction = 1.0
        flex_block_density = 1.0
        fallback_fraction = 1.0
    return {
        "ffn_executed_fraction": round(ffn_fraction, 6),
        "logical_attention_fraction_of_causal": round(attention_fraction, 6),
        "attention_flex_block_density": round(flex_block_density, 6),
        "attention_sdpa_fallback_pair_fraction": round(fallback_fraction, 6),
    }


def benchmark_one(cfg, alias, device, dtype, context_length=None):
    bench_cfg = cfg["benchmark"]
    batch_size = int(bench_cfg.get("batch_size", 1))
    seq_len = int(
        context_length
        if context_length is not None
        else bench_cfg.get("context_length", cfg["model"]["block_size"])
    )
    warmup_steps = int(bench_cfg.get("warmup_steps", 3))
    measure_steps = int(bench_cfg.get("measure_steps", 10))
    repeats = int(bench_cfg.get("repeats", 3))
    layer_index = int(bench_cfg.get("layer", 1)) - 1
    torch.manual_seed(int(bench_cfg.get("seed", 2027)))

    model = build_model(cfg, alias).to(device)
    if not hasattr(model, "blocks"):
        raise TypeError("run benchmark currently targets locally defined Transformer blocks")
    if not 0 <= layer_index < len(model.blocks):
        raise ValueError(f"benchmark.layer must be in [1, {len(model.blocks)}]")
    block = model.blocks[layer_index].train()
    x = torch.randn(
        batch_size,
        seq_len,
        int(cfg["model"]["d_model"]),
        device=device,
    )

    for _ in range(warmup_steps):
        _training_block_step(block, x, device, dtype)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(measure_steps):
            _training_block_step(block, x, device, dtype)
        _synchronize(device)
        samples.append(
            (time.perf_counter() - start) * 1000.0 / max(1, measure_steps)
        )
    milliseconds = statistics.median(samples)
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else None
    )
    row = {
        "model": alias,
        "model_alias": resolve_experiment(cfg, alias)["model_alias"],
        "position_encoding": resolve_experiment(cfg, alias)["position_encoding"],
        "device": str(device),
        "gpu": gpu_label(device.index or 0) if device.type == "cuda" else "CPU",
        "batch_size": batch_size,
        "context_length": seq_len,
        "tokens_per_measurement": batch_size * seq_len,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "repeats": repeats,
        "block_forward_backward_samples_ms": [round(value, 4) for value in samples],
        "block_forward_backward_ms": round(milliseconds, 4),
        "block_tokens_per_second": round(batch_size * seq_len / (milliseconds / 1000.0), 2),
        "peak_memory_mib": None if peak_memory is None else round(peak_memory, 2),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "block_parameters": sum(parameter.numel() for parameter in block.parameters()),
        **_logical_compute(cfg, alias, seq_len),
    }
    del block, model, x
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def _add_relative_metrics(cfg, rows):
    by_variant = {}
    for row in rows:
        if "block_forward_backward_ms" not in row:
            continue
        experiment = resolve_experiment(cfg, row["model"])
        by_variant[
            (
                experiment["model_alias"],
                experiment["position_encoding"],
                row.get("context_length"),
            )
        ] = row
    for row in rows:
        milliseconds = row.get("block_forward_backward_ms")
        if milliseconds is None:
            continue
        experiment = resolve_experiment(cfg, row["model"])
        position_encoding = experiment["position_encoding"]
        context_length = row.get("context_length")
        base = by_variant.get(("base", position_encoding, context_length))
        reference = by_variant.get(("sp_both", position_encoding, context_length))
        if base:
            base_ms = base["block_forward_backward_ms"]
            row["time_over_base_percent"] = round((milliseconds / base_ms - 1.0) * 100.0, 2)
        if reference:
            reference_ms = reference["block_forward_backward_ms"]
            row["speedup_vs_original_sp"] = round(reference_ms / milliseconds, 4)
            row["time_reduction_vs_original_sp_percent"] = round(
                (1.0 - milliseconds / reference_ms) * 100.0, 2,
            )
    return rows


def run(cfg):
    stage_title("benchmark")
    aliases = active_aliases(cfg)
    validate_aliases(cfg, aliases)
    gpus = configured_gpus(cfg)
    gpu_index = job_gpu(gpus, 0)
    if torch.cuda.is_available():
        device = torch.device(cuda_device(gpu_index))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    dtype = torch.float16 if cfg["run"]["dtype"] == "float16" else torch.bfloat16
    context_lengths = cfg["benchmark"].get("context_lengths")
    if context_lengths is None:
        context_lengths = [cfg["benchmark"].get(
            "context_length", cfg["model"]["block_size"],
        )]
    elif not isinstance(context_lengths, list):
        context_lengths = [context_lengths]

    rows = []
    for context_length in context_lengths:
        for alias in aliases:
            try:
                row = benchmark_one(
                    cfg, alias, device, dtype, context_length=context_length,
                )
                rows.append(row)
                print(
                    f"[benchmark] T={int(context_length)} | {alias} | "
                    f"{row['block_forward_backward_ms']:.2f} ms | "
                    f"{row['block_tokens_per_second']:.0f} tok/s | "
                    f"peak={row['peak_memory_mib']} MiB",
                    flush=True,
                )
            except RuntimeError as error:
                rows.append({
                    "model": alias,
                    "context_length": int(context_length),
                    "error": str(error),
                })
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(
                    f"[benchmark] T={int(context_length)} | {alias} | ERROR | {error}",
                    flush=True,
                )
    rows = _add_relative_metrics(cfg, rows)
    target = output_dir(cfg) / "metrics"
    ensure_dir(target)
    write_json(target / "benchmark_summary.json", rows)
    save_config(cfg)
    print(f"[benchmark] saved {target / 'benchmark_summary.json'}", flush=True)
    return rows
