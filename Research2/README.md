# MR2

本项目用于比较 Transformer 中 `attn`、`ffn` 和 `loss` 机制变体对模型表示空间和 SAE 分析结果的影响。当前默认设置为 `OWT + GPT-2 tokenizer + RoPE + test 小模型`。

## 启动

```bash
python main.py
```

进入控制台后，提示符会显示当前 task：

```text
[default] >
```

修改 task：

```text
set run task=Task2
```

## 常用命令

```text
cfg all
cfg run
cfg model
cfg models
cfg data
cfg train
cfg sae
cfg attn
cfg ffn
cfg eval

set run task=stage1_full seed=42

check gpu
run data
run train
run sae
run attn
run ffn
run eval
run all
exit
```

`run all` 的顺序是：

```text
data -> train -> sae -> attn -> ffn -> eval
```

## 训练模式

`run.mode` 控制模型来源：

- `retrain`：默认模式，使用本项目自建 `Transformer`，从随机初始化开始训练。
- `pretrain`：预训练模式，使用 `premodel.models` 中配置的 HuggingFace 预训练模型，并在其基础上继续训练、训练 SAE 和做评估。

从零训练示例：

```text
set run mode=retrain models=base,adatre02attn task=Task2
run data train sae attn ffn eval
```

Pythia 开发与机制验证示例：

```text
set run mode=pretrain models=pythia410m task=pythia410m_dev
run data train sae attn ffn eval
```

主实验 Gemma-2-2B Base 示例：

```text
set run mode=pretrain models=gemma2_2b task=gemma2_2b_main
run data train sae attn ffn eval
```

规模验证 Gemma-2-9B Base 示例：

```text
set run mode=pretrain models=gemma2_9b task=gemma2_9b_scale
run data train sae attn ffn eval
```

预训练模式下，`run.models` 指向 `premodel.models` 里的预训练模型别名。当前内置别名包括：

```text
pythia410m
pythia1b
gemma2_2b
gemma2_9b
```

可查看预训练模型配置：

```text
cfg premodel
```

位置编码也支持列表批量实验。下面会自动展开为 `base_rope/base_alibi/base_cable/sp_both_rope/sp_both_alibi/sp_both_cable` 这类实验别名，并分别保存 checkpoint 和 metrics：

```text
set run mode=retrain models=base,sp_both position_encodings=rope,alibi,cable task=pos_grid
run data train sae attn ffn eval
```

`jobs_per_gpu` 控制每张 GPU 同时跑几个独立实验任务，默认 `auto` 会根据 GPU 显存和当前小模型规模保守估计。它用于提高整组消融实验的资源利用率，不会加快单个模型自己的单步训练速度。

```text
set run jobs_per_gpu=auto
set run jobs_per_gpu=2
```

## 首次服务器测试

第一次上服务器建议先跑小参数 smoke test，确认 OWT、GPU、checkpoint、SAE 和图表生成链路都能通。

```text
set run task=smoke models=base
set data train_blocks=100 valid_blocks=20
set train max_steps=10 log_interval=1 eval_interval=5 save_interval=10 eval_batches=1
set sae max_steps=10 log_interval=1
set attn eval_batches=1
run data train sae attn ffn eval
```

`run sae` 默认会删除当前 `model_alias + sae_alias` 对应的旧 SAE 目录并重新训练。若确实想复用旧 SAE，可显式设置：

```text
set sae retrain=false
```

## Task2 matrix

```text
set run task=Task2 models=base,adatre02attn,sparse12ffn,structured25ffn,groupmix2ffn,groupmix4ffn seed=42
run data train sae attn ffn eval
```

并行方式是“多个模型变体并行跑”。若 GPU 数不少于模型数，所有模型可同时跑；若 GPU 数少于模型数，代码会按 GPU 数自动分批。

建议资源：

```text
8 GPU + 64 CPU cores: 一次跑完整矩阵
4 GPU + 32 CPU cores: 分两批左右跑完整矩阵
2 GPU + 16 CPU cores: 分三批左右跑，仍然可用
```

## 输出位置

```text
output/
    {task}/
        config.json
        sae/
        metrics/
        eval/
        samples/
checkpoints/
    {task}/
        [model_alias]seed{seed}best.pt
        [model_alias]seed{seed}step{step}.pt
        sae/
```

checkpoint 命名格式：

```text
[model_alias]seed{seed}best.pt
[model_alias]seed{seed}step{step}.pt
```

`seed` 是列表配置。单 seed 可写 `seed=42`，多 seed 可写 `seed=42,43,44`。

主要图表会保存在：

```text
output/{task}/eval/
```

包括 loss curve、SAE reconstruction 指标图和 attn 指标图。

## 数据

默认数据配置：

```text
corpus = OWT
tokenizer = gpt2
block_size = 256
blocks_alias = OWT_gpt2_b256_train100k
train_blocks = 100000
valid_blocks = 5000
max_steps = 10000
```

`run data` 会先完整下载/cache `Skylion007/openwebtext`，再按当前 seed shuffle 并生成 blocks，不使用 fake data。
