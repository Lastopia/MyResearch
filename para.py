import os
from types import SimpleNamespace

REMARK = "PoPE research pipeline v0.9"

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

PATH = SimpleNamespace(
    # Stage code imports PATH directly, so paths do not need to be threaded
    # through every run() call.
    cache_dir="./cache",
    ckpt_dir="./ckpt/default",
    output_dir="./output/default",
    figure_dir="./output/default/figures",
    table_dir="./output/default/tables",
    log_dir="./output/default/logs",
    report_dir="./output/default/reports",
    raw_metrics_dir="./output/default/raw_metrics",
)

SECRETS = SimpleNamespace(
    hf_token=os.getenv("HF_TOKEN"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

DATA = SimpleNamespace(
    # Online local-cache mode: HuggingFace reuses ./cache/dataset and downloads
    # only missing files.
    dataset="Skylion007/openwebtext",
    dataset_config=None,
    text_key="text",
    tokenizer="gpt2",
    hf_cache_dir="./cache/dataset",
    local_files_only=False,
    seq_len=1024,
    train_blocks=20000,
    valid_blocks=2000,
    streaming=False,
    use_cache=True,
    seed=42,
)

MODEL = SimpleNamespace(
    # GPT-2 small shape trained from scratch.
    model_names=["rope", "pope", "alibi"],
    baseline_model_name="rope",
    run_model_checks=True,
    max_parameter_ratio_delta=0.01,
    n_layers=12,
    d_model=768,
    n_heads=12,
    d_ff=3072,
    dropout=0.1,
    vocab_size=50257,
    seq_len=1024,
    rope_base=10000.0,
    pope_base=10000.0,
    pope_type="origin",
    tie_embeddings=True,
)

TRAIN = SimpleNamespace(
    # Phase 2 training settings plus Phase 3 attention-analysis limits.
    # Phase 3 is memory-heavy: return_attention=True materializes attention
    # weights and raw logits for every layer as batch x heads x seq_len x seq_len.
    # With seq_len=1024, batch_size=8, n_layers=12 this can trigger the Linux
    # OOM killer during attention analysis; nvidia-smi may look empty afterward
    # because the process has already been killed and released GPU memory.
    # Lower analysis_batches/batch_size/seq_len, or disable the expensive
    # run_sv_distribution/run_toeplitz/heatmap paths if Phase 3 is killed.
    # representative_* keeps Phase 3 bounded by limiting figure/SVD targets.
    seeds=[41,42,43],
    steps=30000,
    batch_size=8,
    lr=6e-5,
    lr_schedule="cosine",
    min_lr_ratio=0.1,
    weight_decay=0.1,
    warmup_steps=500,
    grad_clip=1.0,
    precision="bf16",
    device="cuda",
    log_interval=100,
    eval_interval=5000,
    save_interval=10000,
    analysis_batches=8,
    analysis_batch_size=2,
    loss_spike_threshold=0.5,
    loss_threshold=None,
    primary_checkpoint_rule="final_step",
    secondary_checkpoint_rule="validation_loss_matched",
    validation_loss_match_target=None,
    save_eval_checkpoints=False,
    save_best_checkpoint=True,
    keep_only_latest_best_checkpoint=True,
    save_optimizer_checkpoints=False,
    save_final_optimizer=True,
    skip_completed_runs=True,
    resume_from_checkpoint=True,
    require_final_checkpoints_for_phase3=False,
    run_phase3_analysis=True,
    run_phase3_on_final_checkpoint_skip=True,
    run_length_extrapolation=False,
    extrapolation_seq_lens=[2048],
    run_position_synthetic_task=False,
    run_loss_curve=True,
    save_individual_loss_curves=False,
    local_attention_windows=[4, 16, 64],
    long_range_fraction=0.25,
    spectral_topk_values=[1, 4, 8],
    spectral_analysis_layers=[1, 6, 10],
    spectral_analysis_heads=[0, 1],
    representative_layers=[1, 6, 10],
    representative_heads=[0, 1, 2, 3],
    max_heatmap_seq_len=128,
    run_attention_heatmaps=True,
    run_spectral_plots=True,
    run_attn_entropy=True,
    run_attn_distance=True,
    run_sv_distribution=True,
    run_toeplitz=True,
)

SAE = SimpleNamespace(
    # Phase 4a only: fixed SAE capacity/sparsity for the main matched comparison.
    # Dictionary-size and k sweeps are intentionally left as later extensions.
    sae_type="topk",
    activation_site="residual_post_block",
    layers=[2, 6, 10],
    dictionary_sizes=[4096],
    topk_values=[64],
    device="cuda",
    activation_batch_size=4,
    batch_size=2048,
    lr=2e-3,
    steps=2000,
    seeds=[42],
    max_activation_tokens=65536,
    max_validation_activation_tokens=16384,
    normalize_activations=True,
    skip_completed_stage=True,
    save_individual_sae_curves=False,
    run_aggregate_sae_curves=True,
    log_interval=100,
    eval_interval=200,
    dead_feature_threshold=0.0,
    feature_reuse_frequency_threshold=0.001,
)

EVAL = SimpleNamespace(
    # Phase 5 is deliberately bounded: small probes and sampled features first,
    # enough to compare trends without turning it into another full training job.
    run_r2_tok=True,
    run_r2_pos=True,
    run_mutual_information=True,
    run_correlation=True,
    run_feature_selectivity=True,
    layers=[2, 6, 10],
    max_probe_train_tokens=8192,
    max_probe_valid_tokens=4096,
    probe_steps=200,
    probe_batch_size=512,
    probe_lr=1e-2,
    probe_weight_decay=1e-4,
    position_bins=16,
    feature_activation_bins=10,
    selectivity_quantile=0.9,
    correlation_feature_sample=512,
    run_top_token_identity=True,
    skip_completed_stage=True,
)

INTERP = SimpleNamespace(
    # Real interpretation run. Requires OPENAI_API_KEY in .env.
    provider="openai",
    model="gpt-4o-mini",
    dry_run=False,
    layers=[2, 6, 10],
    model_seeds=[42],
    max_features_per_model_layer=20,
    max_contexts_per_feature=12,
    min_active_contexts=4,
    context_window=8,
    feature_labels=["content_only", "position_only", "mixed", "low_selectivity"],
    active_threshold=0.0,
    max_tokens=700,
    temperature=0.0,
    skip_completed_stage=False,
)
