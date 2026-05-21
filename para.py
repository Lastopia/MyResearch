import os
from types import SimpleNamespace

REMARK = "PoPE research pipeline v0.9"

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

# Set True only for local pipeline checks. Smoke test changes model/data/training
# sizes near the bottom of this file and should not be used for conclusions.
SMOKE_TEST = False

PATH = SimpleNamespace(
    # Stage code imports PATH directly, so paths do not need to be threaded
    # through every run() call.
    cache_dir="./cache",
    ckpt_dir="./ckpt",
    output_dir="./output",
    figure_dir="./output/figures",
    table_dir="./output/tables",
    log_dir="./output/logs",
    report_dir="./output/reports",
    raw_metrics_dir="./output/raw_metrics",
)

SECRETS = SimpleNamespace(
    hf_token=os.getenv("HF_TOKEN"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

DATA = SimpleNamespace(
    # Online local-cache mode: HuggingFace reuses ./cache/dataset and downloads
    # only missing files. Use "tiny" for an offline smoke test.
    dataset="Skylion007/openwebtext",
    dataset_config=None,
    text_key="text",
    tokenizer="gpt2",
    hf_cache_dir="./cache/dataset",
    local_files_only=False,
    seq_len=1024,
    train_blocks=50000,
    valid_blocks=5000,
    streaming=False,
    use_cache=True,
    seed=42,
)

MODEL = SimpleNamespace(
    # GPT-2 small shape trained from scratch. "std" is fixed sinusoidal PE;
    # "rope"/"pope" rotate Q/K.
    model_names=["std", "rope", "pope"],
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
    pope_type="modify",
    tie_embeddings=True,
)

TRAIN = SimpleNamespace(
    # Phase 2 training settings plus Phase 3 attention-analysis limits.
    # representative_* keeps Phase 3 affordable on two A100s.
    seeds=[42],
    steps=100000,
    batch_size=12,
    lr=6e-5,
    weight_decay=0.1,
    warmup_steps=1000,
    grad_clip=1.0,
    precision="bf16",
    device="cuda",
    log_interval=100,
    eval_interval=10000,
    save_interval=10000,
    analysis_batches=1,
    loss_spike_threshold=0.5,
    loss_threshold=None,
    primary_checkpoint_rule="final_step",
    secondary_checkpoint_rule="validation_loss_matched",
    validation_loss_match_target=None,
    save_eval_checkpoints=False,
    save_optimizer_checkpoints=False,
    save_final_optimizer=True,
    skip_completed_runs=True,
    resume_from_checkpoint=True,
    run_phase3_analysis=True,
    run_phase3_on_final_checkpoint_skip=False,
    run_length_extrapolation=False,
    extrapolation_seq_lens=[2048],
    run_position_synthetic_task=False,
    run_loss_curve=True,
    local_attention_windows=[4, 16, 64],
    long_range_fraction=0.25,
    spectral_topk_values=[1, 4, 8],
    spectral_analysis_layers=[1, 6, 10],
    spectral_analysis_heads=[0, 1, 2, 3],
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
    run_top_token_identity=False,
    skip_completed_stage=True,
)

INTERP = SimpleNamespace(
    # Phase 6 defaults to dry-run so prompt/record formats can be checked before
    # spending OpenAI calls. Set dry_run=False only for selected final samples.
    provider="openai",
    model="gpt-4o-mini",
    dry_run=True,
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
    skip_completed_stage=True,
)


def set_output_tree(root):
    PATH.output_dir = root
    PATH.cache_dir = f"{root}/cache"
    PATH.ckpt_dir = f"{root}/ckpt"
    PATH.figure_dir = f"{root}/figures"
    PATH.table_dir = f"{root}/tables"
    PATH.log_dir = f"{root}/logs"
    PATH.report_dir = f"{root}/reports"
    PATH.raw_metrics_dir = f"{root}/raw_metrics"


def apply_smoke_test_config(root="./output/smoke_test", include_downstream=True):
    global SMOKE_TEST
    SMOKE_TEST = True
    set_output_tree(root)

    DATA.dataset = "tiny"
    DATA.seq_len = 64
    DATA.train_blocks = 32
    DATA.valid_blocks = 8
    DATA.use_cache = False
    DATA.hf_cache_dir = f"{root}/cache/dataset"

    MODEL.model_names = ["std", "rope", "pope"]
    MODEL.n_layers = 2
    MODEL.d_model = 128
    MODEL.n_heads = 4
    MODEL.d_ff = 512
    MODEL.vocab_size = 128
    MODEL.seq_len = 64
    
    TRAIN.seeds = [42]
    TRAIN.steps = 5
    TRAIN.batch_size = 2
    TRAIN.device = "cpu"
    TRAIN.precision = "fp32"
    TRAIN.log_interval = 1
    TRAIN.eval_interval = 5
    TRAIN.save_interval = 5
    TRAIN.analysis_batches = 1
    TRAIN.save_eval_checkpoints = False
    TRAIN.save_optimizer_checkpoints = False
    TRAIN.save_final_optimizer = False
    TRAIN.representative_layers = [0, 1]
    TRAIN.spectral_analysis_layers = [0, 1]
    TRAIN.representative_heads = [0, 1]
    TRAIN.spectral_analysis_heads = [0, 1]
    TRAIN.max_heatmap_seq_len = 32

    SAE.layers = [0, 1]
    SAE.dictionary_sizes = [256]
    SAE.topk_values = [16]
    SAE.device = "cpu"
    SAE.activation_batch_size = 2
    SAE.batch_size = 256
    SAE.steps = 20
    SAE.seeds = [42]
    SAE.max_activation_tokens = 2048
    SAE.max_validation_activation_tokens = 512
    SAE.log_interval = 10
    SAE.eval_interval = 10

    EVAL.layers = [0, 1]
    EVAL.max_probe_train_tokens = 512
    EVAL.max_probe_valid_tokens = 256
    EVAL.probe_steps = 5
    EVAL.probe_batch_size = 64
    EVAL.correlation_feature_sample = 64

    INTERP.layers = [0, 1]
    INTERP.model_seeds = [42]
    INTERP.max_features_per_model_layer = 4
    INTERP.max_contexts_per_feature = 4
    INTERP.min_active_contexts = 1
    INTERP.context_window = 4
    INTERP.dry_run = True
    return include_downstream


if SMOKE_TEST:
    apply_smoke_test_config(include_downstream=False)
