# 命令使用手册

```text
check gpu
use gpu 2
set task task1

check data result
check train result
check attention result
check sae result
check eval result
check interpret result

check data cfg
check model cfg
check train cfg
check attention cfg
check sae cfg
check eval cfg
check interpret cfg
check task cfg
check all cfg

run train
run attention
run sae
run eval
run interpret
run all

clear task1 output
clear task1 models
clear task1 attention
clear task1 sae
```

Task folders:

```text
set task task1
check task cfg
```

After `set task task1`, task-specific outputs are separated into:

```text
output/task1/
ckpt/task1/
```

`cache/` stays shared across tasks, so the same token/data cache is not duplicated.

Switching back is just:

```text
set task task1
```

Then `check ... result` and `run ...` will use that task's existing files.

Clear commands always require an explicit task name:

```text
clear task1 output
clear task1 models
clear task1 attention
clear task1 sae
```

`clear task1 output` removes only `output/task1/`. `clear task1 models` removes model checkpoints under `ckpt/task1/models/` and train/attention results under `output/task1/`. `clear task1 sae` removes SAE checkpoints under `ckpt/task1/saes/` and SAE results under `output/task1/`. Shared `cache/` is never cleared by task clear commands.

Copy-paste preset: lightweight formal run:

```text
set data cfg train_blocks=20000 valid_blocks=2000
set train cfg steps=30000 batch_size=8 warmup_steps=500 eval_interval=5000 save_interval=10000
set attention cfg analysis_batches=4 analysis_batch_size=2 run_sv_distribution=true run_toeplitz=false
set sae cfg steps=1200 max_activation_tokens=32768 max_validation_activation_tokens=8192
set eval cfg max_probe_train_tokens=4096 max_probe_valid_tokens=2048 probe_steps=100
check all cfg
```

Copy-paste preset: fast trend run:

```text
set data cfg train_blocks=10000 valid_blocks=1000
set train cfg steps=20000 batch_size=8 warmup_steps=300 eval_interval=5000 save_interval=10000
set attention cfg analysis_batches=2 analysis_batch_size=1 run_sv_distribution=false run_toeplitz=false
set sae cfg steps=800 max_activation_tokens=16384 max_validation_activation_tokens=4096
set eval cfg max_probe_train_tokens=2048 max_probe_valid_tokens=1024 probe_steps=80
check all cfg
```

Recommended stage order:

```text
run train
check train result
run attention
check attention result
run sae
check sae result
run eval
check eval result
```

`set <stage> cfg key=value ...` only changes mentioned keys. Unknown keys raise an error, and the whole line is applied atomically.

# 常用指令大全
```bash
git fetch --all && git reset --hard origin/main && git clean -fd
```
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
```bash
rm -f *_step4000.pt *_step8000.pt *_step6000.pt *_step10000.pt
```
```bash
python main.py --use-train-scheduler --train-gpus 0,1
```

# RoPE / PoPE SAE Research

本项目用于比较不同 positional encoding 对语言模型训练、attention 机制、SAE 分解、disentanglement 和 LLM-assisted interpretability 的影响。

当前默认只比较三个位置编码：

- `std`：standard fixed sinusoidal positional encoding
- `rope`：RoPE
- `pope`：PoPE
- `alibi`：ALiBi-style linear relative-distance bias in attention logits
- `pope_alibi`：PoPE + ALiBi-style explicit relative routing bias

实验计划和阶段定义见 [Experiment Report.md](./Experiment%20Report.md)。

## 1. 环境准备

安装依赖：

```bash
pip install -r requirements.txt
```

如果使用 OpenWebText 或 Phase 6 的 OpenAI 自动解释，在项目根目录创建 `.env`：

```env
HF_TOKEN=hf_your_token
OPENAI_API_KEY=sk_your_key
```

说明：

- `HF_TOKEN` 用于访问 HuggingFace 数据集。
- `OPENAI_API_KEY` 只在 `INTERP.dry_run = False` 时需要。
- `.env` 已被 `.gitignore` 忽略，不应上传。

默认数据缓存位置是：

```text
cache/dataset/
```

代码会把 HuggingFace 的 `HF_HOME`、`HF_HUB_CACHE`、`HF_XET_CACHE` 和 `HF_DATASETS_CACHE` 指向这个目录。有网运行时会优先读取已有缓存，缺少的文件会自动下载补齐。

Tokenizer 会额外保存到稳定目录：

```text
cache/dataset/tokenizers/gpt2/
```

后续运行会先检查这个目录；如果完整就直接读取，如果不存在或不完整，就重新从 HuggingFace 下载并保存。

## 2. 配置入口

主要配置都在 [para.py](./para.py)。

最常改的配置：

- `SMOKE_TEST`：是否启用本地快速测试。
- `DATA`：数据集、tokenizer、序列长度、训练/验证 block 数。
- `MODEL`：模型结构和 positional encoding 列表。
- `TRAIN`：训练步数、batch size、学习率、checkpoint 和 Phase 3 attention 分析设置。
- `SAE`：Phase 4a 的 SAE 层数、字典大小、Top-K、训练步数。
- `EVAL`：Phase 5 probe / disentanglement 设置。
- `INTERP`：Phase 6 自动解释设置，默认 `dry_run=True`，不会调用 OpenAI。

默认主实验模型列表：

```python
MODEL.model_names = ["std", "rope", "pope", "alibi", "pope_alibi"]
TRAIN.seeds = [41, 42, 43]
```

## 3. 快速测试

如果只是检查 pipeline 能不能跑通，把 [para.py](./para.py) 中：

```python
SMOKE_TEST = True
```

然后运行：

```bash
python main.py
```

Smoke test 会使用 tiny synthetic data、小模型、CPU、极少训练步数，并跳过 SAE / Eval / Interpret。它只用于验证代码路径，不用于实验结论。

## 4. 正式运行

确认 [para.py](./para.py) 中：

```python
SMOKE_TEST = False
```

然后运行完整 pipeline：

```bash
python main.py
```

第一次运行会下载 GPT-2 tokenizer 和 OpenWebText 需要的文件到 `cache/dataset/`。如果下载中断，重新运行即可继续复用已下载部分。

默认阶段顺序：

1. `data.py`：加载/缓存数据。
2. `model.py`：构建 `std / rope / pope / alibi / pope_alibi` 模型。
3. `train.py`：Phase 2 训练评估 + Phase 3 attention 机制分析。
4. `sae.py`：Phase 4a TopK-SAE。
5. `eval.py`：Phase 5 disentanglement benchmark。
6. `interpret.py`：Phase 6 SAE feature 自动解释。

## 5. 只跑部分阶段

在 [main.py](./main.py) 顶部修改：

```python
RUN_DATA = True
RUN_MODEL = True
RUN_TRAIN = True
RUN_SAE = True
RUN_EVAL = True
RUN_INTERPRET = True
```

如果某个阶段设为 `False`，下游阶段会尝试读取已有结果；如果找不到必要文件，会直接报错，避免静默产生不完整实验。

常见用法：

```python
# 已经训练完成，只重新跑 SAE / Eval / Interpret
RUN_TRAIN = False
RUN_SAE = True
RUN_EVAL = True
RUN_INTERPRET = True
```

```python
# 只做 Phase 2/3
RUN_TRAIN = True
RUN_SAE = False
RUN_EVAL = False
RUN_INTERPRET = False
```

## 6. 多 GPU 训练调度

如果有两张 A100，可以把 `model_name × seed` 拆成独立训练任务：

```bash
python main.py --use-train-scheduler --train-gpus 0,1
```

如果只有单张 H100，可以不用 scheduler，直接：

```bash
python main.py
```

也可以限制并行数：

```bash
python main.py --use-train-scheduler --train-gpus 0,1 --train-max-parallel 2
```

注意：当前默认是 3 个 positional encoding × 3 seeds = 9 个训练 run。

## 7. 恢复与跳过

代码已经支持两类恢复：

- 训练 checkpoint resume：中断后从最近 checkpoint 继续。
- 阶段级 skip：如果输出已存在且配置 manifest 未变化，则跳过已完成阶段。

相关配置：

```python
TRAIN.skip_completed_runs = True
TRAIN.resume_from_checkpoint = True
SAE.skip_completed_stage = True
EVAL.skip_completed_stage = True
INTERP.skip_completed_stage = True
```

如果你修改了配置，manifest hash 会变化，对应阶段会重新运行。

## 8. 输出位置

主要输出目录：

- `cache/`：数据和 token cache。
- `ckpt/models/`：语言模型 checkpoint。
- `ckpt/saes/`：SAE checkpoint。
- `output/raw_metrics/`：JSON 结果、manifest、prompt 和中间指标。
- `output/tables/`：CSV 表格。
- `output/figures/`：loss curve、attention heatmap、SAE 和解释性图。
- `output/reports/`：Phase 6 案例报告。
- `output/logs/`：运行日志。

重点结果文件：

- `phase2_summary.json/csv`：训练和语言建模结果。
- `phase2_paired_stats.csv`：PoPE vs RoPE paired difference、effect size、bootstrap CI。
- `phase2_checkpoint_comparison.json/csv`：final checkpoint 与 validation-loss-matched checkpoint 对比。
- `phase3_layer_metrics.csv`：attention entropy、distance、spectral、Toeplitz 指标。
- `phase4a_sae_summary.csv`：SAE 重构、稀疏性和 feature health。
- `phase5_disentanglement_summary.csv`：content/position disentanglement 结果。
- `phase6_interpretation_scores.csv`：LLM-assisted feature 解释评分。
- `phase6_run_records.json`：Phase 6 dry-run/OpenAI-run 记录。

## 9. Phase 6 OpenAI 调用

默认不会调用 OpenAI：

```python
INTERP.dry_run = True
```

如果要真正调用 OpenAI：

```python
INTERP.dry_run = False
INTERP.model = "gpt-4o-mini"
```

并确保 `.env` 中有：

```env
OPENAI_API_KEY=sk_your_key
```

每次 Phase 6 都会保存：

- dry-run / OpenAI-run 标记
- 使用模型
- prompt
- response
- confidence
- false_positive_risk

## 10. 代码结构约定

当前代码遵循两个核心约定：

- [para.py](./para.py) 只放配置集合。
- 每个阶段文件封装一个入口类，`run()` 返回结果字典。

阶段之间只传递必要的结果字典。例如：

- `Train(TRAIN, model_res, data_res).run()`
- `SelfSAE(SAE, train_res, data_res).run()`
- `Evaluate(EVAL, train_res, sae_res, data_res).run()`
- `InterpretSAE(INTERP, train_res, sae_res, data_res, eval_res).run()`

路径配置由各文件直接从 `para.py` 读取，不作为额外参数传入。

## 11. 当前实验建议

在两张 A100 条件下，建议先跑：

- Phase 2/3：完整 `std / rope / pope / alibi / pope_alibi × 3 seeds`
- Phase 4a：代表层 `[2, 6, 10]`，固定 dictionary size 和 Top-K
- Phase 5：限流 probe 和 feature-level 指标
- Phase 6：先 `dry_run=True`，确认 prompt 和样例格式，再少量 OpenAI-run

暂时不建议一开始做完整 dictionary size sweep 或大规模 sparsity sweep。
