# 0. 代办事项
## P0：必须补齐
- [x] 明确 checkpoint 选择协议：主实验使用相同 training steps / tokens seen 的 final checkpoint；补充实验使用 validation-loss-matched checkpoint。
- [x] 在 Phase 2 训练结果中记录 checkpoint metadata：tokens_seen、checkpoint_step、checkpoint_path、valid_loss_at_checkpoint、selection_rule。
- [x] 让 Phase 4/5/6 显式读取同一套 checkpoint selection rule，保证 SAE、disentanglement、interpretation 比较对象一致。
- [x] 增加阶段恢复与自动跳过机制：如果某阶段输出已存在且配置未变化，则跳过已完成 run。
- [x] 增加训练 checkpoint resume 能力：中断后可从最近 checkpoint 继续训练。
- [x] 增加可选的两张 A100 的简单任务调度：按 model_name × seed 拆分任务，每张 GPU 分配一个独立 run。但实际上也可能会用单张H100
- [x] 增加实验 manifest：保存配置快照、Python / PyTorch / CUDA 版本、GPU 名称、时间戳、随机种子和代码版本信息。
- [x] 增加统计检验：报告 PoPE vs RoPE 的 paired difference、effect size、95% CI，并用 bootstrap 作为小 seed 数下的补充。

## P1：重要补强
- [x] 补充 validation-loss-matched 分析：为每个 model × seed 选择验证损失最接近共享目标值的 checkpoint，并与 final checkpoint 结果并列表。
- [x] 增加 SAE / Eval / Interpret 的恢复能力：允许从已有 train_res、sae_res、phase5 tables 或 phase6 prompts 继续后续阶段。
- [x] 明确 Phase 6 的 OpenAI-run 记录：保存 dry-run / OpenAI-run 标记、使用模型、prompt、response、confidence 和 false_positive_risk。

## P2：论文增强项
- [ ] 加入 dictionary size sensitivity analysis，用于验证 PoPE 是否降低对大字典的依赖。
- [ ] 加入 sparsity sensitivity analysis，用于验证 PoPE 优势是否能跨不同 TopK 稀疏度成立。
- [ ] 加入一个额外 positional encoding baseline：暂不考虑，当前只保留 std / RoPE / PoPE，其中 std 即 Sinusoidal PE。
- [ ] 为后续 steering 实验加入副作用量化指标。

## P3：可选扩展
- [ ] 为 Phase 6 增加少量 manual qualitative audit 样例：由单人抽查自动解释结果，只用于发现明显错误和失败模式，不作为正式 human calibration。
- [ ] 扩展到更大或更多样的数据集。
- [ ] 分析更多 layers 或更大模型。
- [ ] 探索 hybrid 或 per-head positional encoding。
- [ ] 将想法扩展到 multimodal models。
# 1. 研究目标

本项目研究的问题是：**位置编码的设计是否会影响 Transformer 表示对 Sparse Autoencoder（SAE）分解的友好程度**。

核心假设是：如果一种位置编码能够在结构上更好地区分内容信息（what）和位置信息（where），那么模型内部表示中的纠缠程度可能会下降，从而让 SAE 更容易学习到稀疏、单义、稳定、且具备因果干预价值的特征。

本研究以 **RoPE** 和 **PoPE** 作为主要比较对象。RoPE 是当前大语言模型中非常主流的位置编码方式，具有较好的训练稳定性和长度外推能力；PoPE 则通过极坐标形式，将内容主要放在 magnitude，将位置主要放在 phase，从结构上显式区分 what 和 where。

为了让实验结论更可靠，后续实验中应尽可能加入额外的位置编码 baseline，例如 ALiBi 或 Sinusoidal PE。这样可以避免结论只停留在 “PoPE 和 RoPE 不同”，而是进一步支持 “位置编码设计会系统性影响 SAE 可解释性” 这一更强的研究命题。

# 2. 核心研究问题

**RQ1：模型性能稳定性**
	PoPE 是否能够在语言模型训练和下游任务中保持与 RoPE 相近，甚至更好的性能？
	这个问题关注的是：更强的 what-where 解耦是否会损害模型训练效率、收敛稳定性、perplexity 或任务表现。

**RQ2：内容-位置解耦效果**
	PoPE 是否真的能让模型表示中的内容信息和位置信息更加分离？
	这个问题用于验证：PoPE 在几何结构上的解耦设计，是否会在训练后的 Transformer 表示中产生可测量的解耦效果。

**RQ3：SAE 可解释性提升**
	PoPE 是否能让 SAE 学到更稀疏、更稳定、更单义的特征？
	这是本研究最核心的可解释性问题。这里不能只看 SAE reconstruction error 是否更低，还需要进一步验证 SAE 学到的 latent 是否更独立、更容易被解释，并且是否在后续 steering 或因果干预中更有用。

**RQ4：对 Attention 机制结构的影响**
	不同位置编码如何影响 attention logits、attention pattern、谱结构，以及近似 Toeplitz 性质？
	这个问题用于连接位置编码的几何设计和 attention 机制内部的可观测变化。它是理论论证和实验结果之间的重要桥梁。

# 3. 实验设计
## Phase 0 : 总览

```text
Phase 1：构建受控 Transformer 变体
Phase 2：训练并评估语言模型性能
Phase 3：分析 attention logits、谱结构和 Toeplitz 性质
Phase 4：在 matched activations 上训练统一架构的 SAE
Phase 5：运行 disentanglement probes
Phase 6：使用 LLM 解释 SAE latents，并加入人类标注校准
Phase 7：测试 steering 与因果干预效果
Phase 8：通过多 seed、更多数据集和敏感性分析增强可靠性
Phase 9：将新位置编码设计作为 exploratory future work
```

改进后的重点是：项目不再只依赖 “PoPE 的 SAE reconstruction loss 更低” 这一单点证据，而是建立一条更完整的证据链，从位置编码设计，到 attention 机制变化，到表示解耦，到 SAE feature quality，再到实际 steering utility。
## Phase 1：模型构建
### 简介
构建一个受控的 Transformer 实验平台，使位置编码成为主要实验变量。从而使所有模型变体需要尽量保持一致：
- 相同 tokenizer
- 相同数据预处理方式
- 相同层数
- 相同 hidden size
- 相同 attention head 数量
- 相同 FFN 维度
- 相同 dropout
- 相同 optimizer
- 相同训练 schedule
- 相同初始化策略，尤其是在 matched seed 下
理想情况下，模型之间唯一主要区别应是位置编码方式。

需要构建的位置编码方式目前是：
实验组：
- RoPE
- PoPE
对照组：
- Sinusoidal PE

### 期望产出
- 可切换的位置编码实现
- 每个模型变体的参数量统计
- 确认不同模型参数规模基本一致，少量参数差异可以容忍
- 检查 attention mask、sequence length、position operation 是否正确
- 保存可复现实验配置

## Phase 2：训练与语言模型性能评估
### 简介
评估 PoPE 是否能够保持稳定训练，并取得与 RoPE 相近或更好的语言建模性能。
- Dataset：OpenWebText
- Tokenizer：GPT-2
- Sequence length：1024
- Training blocks：固定数量训练块
- Validation blocks：固定数量验证块
所有位置编码条件都应使用完全一致的训练设置：
- 相同训练步数
- 相同 batch size
- 相同 learning rate
- 相同 warmup
- 相同 precision
- 相同保存间隔
- 相同验证间隔
主实验应包含 3 个 seeds: `41, 42, 43` 和三个位置编码 : `std, rope, pope`，其中 `std` 即 Sinusoidal PE。
评估指标如下：
1. 训练稳定性：
	- Training loss curve
	- Validation loss curve
	- Gradient norm
	- Gradient variance
	- Loss spike 或 divergence 次数
	- 达到某一 loss threshold 所需步数

2. 语言建模性能：
	- Validation perplexity
	- Final validation loss
	- Best validation loss
	- 训练效率

3. 长度和位置敏感性：
	- 标准长度下的 validation performance
	- 更长 context 下的 extrapolation performance，如果可行
	- 位置敏感 synthetic task，如果后续能构建

### 期望产出
- 每种位置编码的 loss curve
- 多 seed 下的 mean 和 standard deviation
- Perplexity 对比表
- 训练稳定性分析
- 明确回答 PoPE 是否损害、保持或提升模型性能

## Phase 3：Attention Pattern 与机制分析
### 简介
分析不同位置编码如何改变 attention logits 和 attention 行为，并直接回答 RQ4：**不同位置编码是否会系统性改变 attention 的位置结构、内容混合方式和 head-level 功能分工。**

本阶段关注的是机制层面的可观测变化，而不是直接宣称可解释性提升。谱结构、Toeplitz deviation 和 head taxonomy 都应作为机制证据，与 Phase 2 的语言模型性能、后续 SAE 指标共同解释。
### 基础指标
#### 1. Attention Entropy
将每个 query token 对所有 key token 的 attention weight 视为概率分布，计算 Shannon entropy。
解释：
- Entropy 越低，表示 attention 越集中。
- Entropy 越高，表示 attention 越分散。
- 应同时报告 layer-wise mean 和 head-wise distribution。
#### 2. Average Attention Distance
计算 attention-weighted token distance。
解释：
- 数值越大，说明模型更倾向于关注远距离 token。
- 数值越小，说明模型更倾向于局部 attention。
- 应同时报告每层平均值、每个 head 的平均值，以及不同位置编码之间的差异。
#### 3. Local Window Mass
计算 attention 落在 query token 附近窗口内的比例，例如距离 <= 4、<= 16、<= 64 的 attention mass。
目的：
- 辅助区分 local heads 和 long-range heads。
- 避免只依赖 average distance，因为 average distance 可能被少数远距离 attention 拉高。
#### 4. Attention Heatmap
可视化不同 layer 和 head 的 attention matrix。
要求：
- 使用相同 validation samples 比较 std / rope / pope。
- 每种位置编码至少展示 early / middle / late 各一层。
- 每层展示多个代表性 heads，而不是只展示平均 attention。
- heatmap 用于定性说明，核心结论应以数值指标为准。
### Attention Logits 谱结构指标
#### 5. Singular Value Distribution
在 softmax 之前，对 attention logits matrix 计算奇异值分布。
目的：
- 比较不同位置编码是否导致 logits 结构更尖锐、更平滑或更低秩。
- 应分别报告 selected layers 和 selected heads 的 singular value spectra。
注意：
- 该指标只能说明 logits 结构复杂度和能量分布变化。
- 不能单独解释为“可解释性更好”。
#### 6. Spectral Concentration
计算 top-k singular values 捕获了多少整体 logit energy，例如 top-1、top-4、top-8。
目的：
- 判断 positional encoding 是否导致 attention logits 出现更强低秩化或能量集中。
- 应报告 layer-wise 和 head-wise 的均值与方差。
#### 7. Approximate Toeplitz Deviation
如果 attention logits 主要依赖相对位置，那么 logits matrix 应具有近似 Toeplitz 结构。即相同 relative-position diagonal 上，logit 值应更相似。
定义：
`Toeplitz deviation = 相同 relative-position diagonal 上 logits 方差的平均值`
解释：
- Toeplitz deviation 越低，说明相对位置规律越强。
- 低 Toeplitz deviation 不一定表示更好，也可能说明模型过度依赖位置。
- 需要和 entropy、distance、local mass、validation loss 一起解释。
### 分层分析
对 12 层 GPT-2 small 结构，按如下方式划分：
- Early layers：Layer 0-3
- Middle layers：Layer 4-7
- Late layers：Layer 8-11
期望观察：
- Early layers 可能保留更多位置结构。
- Middle layers 可能出现更强的 content-position interaction。
- Late layers 可能更偏向语义抽象和任务相关混合。
如果后续扩展到不同层数模型，则改用相对划分：
- bottom third
- middle third
- top third
每个阶段至少选择一层作为代表层，例如：
`Early: Layer 1 Middle: Layer 6 Late: Layer 10`
同时保留所有层的 layer-wise summary，避免只依赖代表层。
### Head-Wise 行为分析
不同 attention heads 可能承担不同功能。本阶段需要从“描述性观察”升级为“可操作分类”。
#### Head Taxonomy Operationalization
使用以下启发式标准标注 head 类型。一个 head 可以同时属于多个类别。
#### 1. Local Syntactic Head
判定依据：
- Average attention distance 较低。
- Local window mass 较高，例如距离 <= 16 的 attention mass 高。
- Heatmap 呈现明显 diagonal 或 near-diagonal pattern。
解释：
- 该类 head 可能更偏向局部句法或短程依赖。
#### 2. Long-Range Dependency Head
判定依据：
- Average attention distance 较高。
- Local window mass 较低。
- 远距离 attention mass 较高，例如距离 > 128 或 > seq_len / 4 的 attention mass 高。
解释：
- 该类 head 可能参与长程依赖或跨段信息整合。
#### 3. Position-Sensitive Head
判定依据：
- Toeplitz deviation 较低。
- Attention 或 logits 对 relative distance 呈现稳定结构。
- 不同 validation samples 上 relative-position profile 方差较低。
解释：
- 该类 head 可能主要编码相对位置规律。
- 对 RoPE / PoPE 的对比尤其关键。
#### 4. Content-Matching Head
仅靠 attention matrix 不足以可靠判断 content matching，因此需要额外 token-level 或 synthetic evidence。
判定依据：
- 在 repeated-token、copied-token、entity-matching 或 key-value retrieval synthetic examples 上，head 对匹配 token 的 attention mass 显著高。
- 或在真实文本中，对相同 token / 重复实体 / 引用对象的 attention mass 显著高于随机 baseline。
解释：
- 该类 head 可能承担内容匹配、复制、实体回指或检索功能。
- 如果当前阶段暂不实现 synthetic task，应明确标记为 “not fully classified”。
### 控制与比较原则
为了让不同位置编码之间的比较公平，需要满足：
- 使用相同模型规模。
- 使用相同 tokenizer。
- 使用相同 validation samples。
- 使用相同 sequence length。
- 使用相同 random seeds。
- 使用相同 checkpoint step。
- 比较同一 layer、同一 head index 时，需要谨慎，因为不同模型中的 head index 不一定语义对齐。
- 因此除了 head index 对比，还应比较 head-level distribution，例如所有 heads 的 entropy/distance/Toeplitz 分布。
### 期望产出
- Attention entropy 对比表。
- Average attention distance 对比表。
- Local window mass 对比表。
- Attention heatmaps。
- Singular value spectra 图。
- Spectral concentration 表格。
- Toeplitz deviation 表格。
- Layer-wise analysis。
- Head-wise metrics table。
- Head taxonomy summary。
- 对 RQ4 的直接回答。
### RQ4 回答模板
最终应回答：
1. std / rope / pope 是否产生不同的 attention concentration？
2. std / rope / pope 是否改变 attention 的平均距离和局部性？
3. std / rope / pope 是否改变 attention logits 的谱结构？
4. std / rope / pope 是否改变相对位置结构强度，即 Toeplitz-like behavior？
5. PoPE 是否相比 RoPE 减少、增强或重分配了 position-sensitive heads？
6. PoPE 是否产生更多 content-position separation 的证据？
7. 这些机制变化是否在不显著损害 Phase 2 validation performance 的情况下出现？
## Phase 4：SAE 训练与核心指标
### 简介
在 matched std / RoPE / PoPE 模型表示上训练统一架构的 TopK-SAE，测试 PoPE 表示是否更容易被稀疏分解。
本阶段的核心问题是：
`在相同模型规模、相同训练数据、相同 activation site、相同 SAE 架构和相同 SAE 超参数下， PoPE 是否相比 RoPE/std 更容易被 SAE 重构，并产生更健康、更稀疏的特征表示？`
所有模型变体都应使用相同 activation site：
`Residual stream after each Transformer block`
至少从 early / middle / late layers 中各选一层：
`layers = [2, 6, 10]`

主 Phase 4 不做大规模 dictionary size sweep，也不做完整 k sweep。先固定一组合理 SAE 超参数，完成 matched SAE 对比。

推荐主实验配置：
```python
models = [std, rope, pope] 
model_seeds = [41, 42, 43] 
layers = [2, 6, 10] 
sae_type = TopK-SAE 
dictionary_size = 4096 
k = 64 
sae_seed = 42 
activation_site = residual_post_block 
activation_normalization = per-layer mean/std normalization 
max_activation_tokens = 65536 或 131072 
sae_batch_size = 2048 
sae_lr = 2e-3 
sae_steps = 2000
```

总 SAE runs：
`3 models × 3 model seeds × 3 layers × 1 dict size × 1 k × 1 SAE seed = 27 SAE runs`
这个规模适合两张 A100，也足够作为第一版核心结论。

所有 SAE runs 必须保持一致：
- 相同 SAE 架构：TopK-SAE
- 相同 dictionary size
- 相同 k
- 相同 learning rate
- 相同 batch size
- 相同 training steps
- 相同 optimizer
- 相同 activation normalization
- 相同 validation sampling protocol
- 相同 SAE random seed
- 相同 activation site
- 相同 layer selection
- 相同 model checkpoint step

为了公平比较不同位置编码，需要固定 activation 采样方式。
要求：
- 使用相同 validation 或 held-out token blocks。
- 每个模型、每个 layer 采样相同数量 token。
- 默认不把全部 activation 长期缓存到硬盘，优先使用 streaming / on-the-fly collection。
- 如果缓存 activation，应使用 bf16 或 fp16，并记录 token 数、layer、model、seed。
- SAE 训练前对每个 layer 的 activation 做 normalization。
推荐 normalization：
`x_norm = (x - mean) / std`
其中 mean/std 在该 model seed + layer 的 activation sample 上估计。
注意：
- normalization 统计量必须保存。
- validation MSE 和 explained variance 应在相同 normalization 空间中计算。
- 如果需要报告原始空间 reconstruction，可作为附加指标。

### 核心 SAE 指标
#### 1. Reconstruction Quality
用于衡量 SAE 是否更容易重构模型表示。
指标：
- Validation MSE
- Reconstruction loss
- Explained variance
- Normalized reconstruction MSE
解释：
- MSE 越低越好。
- Explained variance 越高越好。
- 如果 PoPE 在相同 sparsity 和 dictionary size 下有更低 MSE / 更高 explained variance，说明它可能降低 SAE decomposition difficulty。
#### 2. Sparsity
用于确认不同模型之间 SAE 稀疏程度可比较。
指标：
- L0 norm
- Average active features per token
- Sparsity distribution
注意：
- TopK-SAE 下 average active features per token 应接近 k。
- 因此 L0 更多是 sanity check，不应作为主要优势证据。
- 更重要的是在相同 k 下比较 reconstruction quality 和 feature health。
#### 3. Feature Health
用于判断 SAE 是否训练出了可用字典，而不是大量死特征或少数特征垄断。
指标：
- Dead feature rate
- Feature activation frequency
- Feature reuse rate
- Feature density distribution
- Top feature activation frequency
- Feature frequency entropy
解释：
- Dead feature rate 越低越好。
- Feature activation frequency 分布不能过度集中。
- 如果少数 feature 被过度复用，说明字典利用不健康。
- 如果 PoPE 在相同 reconstruction 下有更低 dead feature rate 或更均衡 feature usage，可以作为更容易分解的证据。

#### 4. Training Dynamics
用于判断 SAE 训练是否稳定。
指标：
- SAE train loss curve
- SAE validation MSE curve
- Dead feature rate over training
- Explained variance over training
- Reconstruction-sparsity trade-off summary
注意：
- “Monosemanticity growth” 暂不作为 Phase 4 主指标。
- Monosemanticity 更适合放到 Phase 5/6，结合 probes 或 feature interpretation 再分析。
### 期望产出
主 Phase 4 应产出：
- 每个 SAE run 的 raw metrics JSON
- 每个模型 × seed × layer 的 SAE 指标表
- 多 seed mean / std 汇总表
- Layer-wise SAE 对比
- Reconstruction quality 对比
- Feature health 对比
- SAE train/validation curves
- Dead feature evolution curves
- 对 Phase 4 核心问题的回答：
	`在固定 dictionary size 和 k 下，PoPE 是否比 RoPE/std 更容易被 TopK-SAE 分解？`

### Phase 4b：Dictionary Size Sensitivity（可选鲁棒性分析）
Dictionary size sensitivity 不放入 Phase 4 主实验，作为后续 robustness check。
目的：
`验证 PoPE 的优势是否依赖 dictionary size。`
推荐轻量设置：
```python
models = [rope, pope] 
model_seed = 42 
layer = 6 
dictionary_size = [1024, 2048, 4096, 8192] 
k = 64 
sae_seed = 42`
```
总 runs：
`2 models × 1 seed × 1 layer × 4 dict sizes × 1 k × 1 SAE seed = 8 SAE runs`
如果资源允许，可扩展到：
`layers = [2, 6, 10]`
但不建议一开始全量展开。
期望产出：
- Dictionary size vs validation MSE 曲线
- Dictionary size vs explained variance 曲线
- Dictionary size vs dead feature rate 曲线
- 判断 PoPE 是否在较小 dictionary 下仍保持竞争力
### Phase 4c：Sparsity k Sensitivity（可选鲁棒性分析）
Sparsity sensitivity 同样作为后续 robustness check。
目的：
`验证 PoPE 的优势是否只出现在某个特定 sparsity regime。`
推荐轻量设置：
```python
models = [rope, pope] 
model_seed = 42 
layer = 6 
dictionary_size = 4096 
k_values = [16, 32, 64, 128] 
sae_seed = 42`
```
总 runs：
`2 models × 1 seed × 1 layer × 1 dict size × 4 k values × 1 SAE seed = 8 SAE runs`
期望产出：
- k vs validation MSE 曲线
- k vs explained variance 曲线
- k vs dead feature rate 曲线
- Reconstruction-sparsity trade-off curve
### 资源策略
为了适配两张 A100：
主实验优先级：
1. 固定 dictionary_size=4096, k=64，完成 27 个主 SAE runs 
2. 只保存 SAE checkpoint、metrics、normalization stats 
3. 不长期保存全量 activation，除非需要复查 
4. activation 使用 on-the-fly collection 或小规模 bf16 cache
暂缓：
5. 完整 dictionary size sweep 
6. 完整 k sweep 
7. 多 SAE seed sweep 
8. Monosemanticity growth`
### 结论判定标准
如果在主实验中观察到：
- PoPE validation MSE 更低
- PoPE explained variance 更高
- PoPE dead feature rate 不高于 RoPE/std
- PoPE feature usage 更均衡
- PoPE 在 early/middle/late layers 中至少多数层表现稳定
- Phase 2 中 PoPE validation performance 没有明显损害
则可以初步支持：
`PoPE 在固定 SAE 容量和固定 sparsity 下，降低了 SAE decomposition difficulty。`
如果 PoPE 只在某一层或某个 seed 上更好，则应表述为：
`PoPE 改变了 SAE decomposition behavior，但优势尚不稳定，需要 Phase 4b/4c 鲁棒性分析。`
## Phase 5：Disentanglement Benchmark
### 目标
定量测试 `std / RoPE / PoPE` 的 residual activations 和 SAE features 是否更好地区分 content information 与 position information。本阶段直接回答 RQ2，并为 Phase 4 的 SAE 结果提供机制解释。

核心判断不是“位置信息是否消失”，而是：

```text
position information 是否仍然存在，
同时 content 和 position 是否更少混在同一组 SAE features 中。
```

### 输入与控制
和 Phase 4 保持一致：

- Activation site：`Residual stream after each Transformer block`
- Layers：`[2, 6, 10]`
- Models：`std / rope / pope`
- Model seeds：`[41, 42, 43]`
- SAE config：使用 Phase 4a 的固定 TopK-SAE 配置
- Token samples、train/validation split、feature binning、probe regularization 必须一致

本阶段同时比较两种表示：

1. Raw residual activations：判断模型内部表示本身是否更可分离。
2. SAE feature activations：判断 SAE 分解后的 sparse features 是否更少 content-position 混合。

这样可以区分“模型表示本身更解耦”和“SAE 更容易分解出解耦 features”这两种情况。

### Probe Targets
主实验只使用轻量、稳定、可复现的 targets。

Position targets：

- Absolute position bin：将 sequence position 分成 `16` 或 `32` 个 bin。
- Normalized position：`position / (seq_len - 1)`。
- Segment position：`beginning / middle / end`。

Content targets：

- Token category：`alphabetic / numeric / punctuation / whitespace / other`。
- Token frequency bin：`rare / medium / frequent`。
- Top-N token identity 只作为 optional analysis，默认不做 full GPT-2 vocab token identity probe。

决定依据：full token identity probe 代价高、长尾噪声大；POS 和 syntactic role 需要额外标注器，暂不作为主实验目标。

### Probe 方法
Representation-level 使用简单 probe：

- Linear regression：预测 normalized position。
- Logistic regression / linear classifier：预测 position bin、segment position、token category、frequency bin。
- 使用 L2 regularization，所有模型共享相同 train/validation split。

指标：

- Position bin accuracy / macro F1
- Normalized position R2
- Segment position accuracy
- Token category accuracy / macro F1
- Token frequency bin accuracy

这些指标用于确认 content 和 position information 是否仍然可解码；它们不直接等价于 feature-level disentanglement。

### Feature-Level Disentanglement
对每个 SAE feature 计算两个分数：

```text
content_score(feature)
position_score(feature)
```

Position score 使用：

- Feature activation 与 normalized position 的 absolute correlation。
- Feature activation bin 与 position bin 的 normalized mutual information。

Content score 使用：

- Feature activation bin 与 token category 的 normalized mutual information。
- Feature activation bin 与 token frequency bin 的 normalized mutual information。

Feature activation 先按 quantile discretization 分成 `10` 个 bins，再计算离散 mutual information。MI 作为辅助证据；核心结论需要同时看 probe、correlation 和 feature selectivity。

### Feature Selectivity
每个 feature 标注为：

- `content-only`
- `position-only`
- `mixed`
- `low-selectivity`
- `inactive / dead`

主实验使用 quantile threshold：

```text
content_threshold = top 10% content_score
position_threshold = top 10% position_score
```

判定规则：

```text
dead feature -> inactive/dead
content >= threshold and position >= threshold -> mixed
content >= threshold -> content-only
position >= threshold -> position-only
otherwise -> low-selectivity
```

关键指标：

- Mixed-feature ratio
- Content-only feature ratio
- Position-only feature ratio
- Low-selectivity feature ratio
- Dead / inactive feature ratio
- Content-position score correlation
- Mean / top content_score
- Mean / top position_score

关键预期：

```text
PoPE 应降低 mixed-feature ratio，
同时不显著降低 content-only 和 position-only 信息承载能力。
```

### 相关性与互信息
报告：

- Feature-feature correlation mean / max / distribution
- SAE feature activation 与 normalized position 的 correlation
- Content score 与 position score 的 correlation
- Feature activation bin 与 position bin 的 normalized MI
- Feature activation bin 与 token category / frequency bin 的 normalized MI

如果 PoPE 降低 content-position score correlation 和 mixed-feature ratio，同时 representation-level probes 仍能解码 content 与 position，则说明 PoPE 更可能是在组织信息，而不是删除信息。

### 可选扩展
以下内容不作为 Phase 5 主实验要求：

- POS tag probe：需要 spaCy / Stanza，存在额外依赖和标注噪声。
- Local syntactic role：成本更高，后续再做。
- Top-N token identity probe：只在 top frequent tokens 上做，不做 full vocab probe。
- Permutation baseline：后续可用于替代 quantile threshold，增强阈值稳健性。

### 阶段产出
- Representation-level probe scores
- Raw residual activation vs SAE feature activation 对比
- Position decoding scores
- Content decoding scores
- Feature-level content_score / position_score 表格
- Feature selectivity 标签表
- Mixed-feature ratio
- Content-position score correlation
- Correlation 和 mutual information 汇总表
- Layer-wise disentanglement analysis
- 对 RQ2 的直接回答

### RQ2 回答标准
最终回答以下问题：

1. `std / RoPE / PoPE` 是否都保留 position information？
2. `std / RoPE / PoPE` 是否都保留 content information？
3. PoPE 的 SAE features 是否有更低 mixed-feature ratio？
4. PoPE 是否降低 content_score 与 position_score 的 feature-level correlation？
5. PoPE 是否在不丢失 position information 的情况下，让 position information 更干净地组织？
6. 这些 disentanglement 改善是否与 Phase 4 的 SAE reconstruction / feature health 改善一致？

如果结果显示 PoPE 的 position/content probes 仍然有效，同时 mixed-feature ratio 和 content-position score correlation 低于 RoPE/std，则可以支持：

```text
PoPE 没有简单删除位置信息，而是让 content 与 position 在 SAE feature 层面更少纠缠。
```

## Phase 6：LLM 辅助 SAE Latent 解释

### 目标

使用 RouteSAE 风格的 LLM-assisted interpretation pipeline，评估 `std / RoPE / PoPE` 的 SAE features 是否更容易被解释。本阶段作为 RQ3 的辅助证据，不单独作为最终证明。

核心问题：

```text
PoPE 的 SAE features 是否更容易得到清晰、一致、低混合度的自然语言解释？
```

### 资源约束与主实验范围

考虑当前实验条件为两张 A100，本阶段不解释所有 SAE features，只解释筛选后的 active/high-value features。

主实验推荐：

- Models：`std / rope / pope`
- Layers：`[2, 6, 10]`
- Model seeds：优先 `seed=42`；资源允许再扩展到 `[41, 42, 43]`
- 每个 model × layer 采样 `30-50` 个 features
- 每个 feature 收集 `8-20` 个 top activating contexts
- 每个 feature 主实验只调用 LLM 解释一次
- 使用 blinded feature IDs，不向 LLM 暴露 feature 来自 RoPE 还是 PoPE

决定依据：LLM 调用成本与上下文长度是主要限制；解释所有 dictionary features 不必要，也会引入大量 dead / low-selectivity feature 噪声。

### Feature 采样策略

不随机解释所有 features，而是从 Phase 4 和 Phase 5 的结果中筛选。

保留条件：

- 非 dead feature
- 至少有 `4` 个以上 active contexts
- activation frequency 不过低
- 排除极端高频、过度泛化的 features

采样时应尽量平衡：

- `content-only`
- `position-only`
- `mixed`
- `low-selectivity`
- early / middle / late layers

注意：不同 SAE 的 feature index 不具备语义对齐关系，因此不比较 “RoPE feature 102 vs PoPE feature 102”。比较对象应是同一采样规则下的 feature 分布。

### Context 收集

对每个 feature，收集 top activating contexts。每个 context 应包含：

- activated token
- activation value
- token position
- surrounding tokens
- feature rank within this token，如果可得

为了判断 false-positive behavior，可额外采样少量 low-activation 或 random contexts，但主实验可以先只保存，不强制纳入 LLM 评分。

### LLM 输出格式

对每个 feature，使用固定 prompt，让 LLM 输出结构化结果：

```text
feature_type: content / position / mixed / low-level / undiscernible
interpretability_score: 1-5
specificity_score: 1-5
coverage_score: 1-5
false_positive_risk: 1-5
short_explanation: ...
evidence_summary: ...
```

指标解释：

- `feature_type`：LLM 判断该 feature 主要表示什么类型的信息。`content` 表示语义或词汇内容，`position` 表示位置或结构模式，`mixed` 表示同时混合 content 与 position，`low-level` 表示标点、空格、格式等低层模式，`undiscernible` 表示无法稳定解释。
- `interpretability_score`：整体可解释性分数。`5` 表示模式清晰且样例一致，`1` 表示无法看出稳定模式。
- `specificity_score`：解释是否具体。高分表示解释能指出明确 token/context pattern，低分表示解释过于泛泛。
- `coverage_score`：解释能覆盖多少 high-activation examples。高分表示大多数 top contexts 都符合解释。
- `false_positive_risk`：解释是否过宽。高分表示该解释可能错误预测很多无关 context 也会激活，因此该指标越低越好。
- `short_explanation`：一句到数句自然语言解释。
- `evidence_summary`：简要说明哪些 top contexts 支持该解释。

主分数可以使用：

```text
quality_score = mean(interpretability_score, specificity_score, coverage_score) - false_positive_risk_penalty
```

其中 false-positive penalty 可以先简单设为：

```text
false_positive_risk_penalty = 0.25 * (false_positive_risk - 1)
```

### 数值化比较

报告以下 summary metrics：

- Mean interpretability score：平均整体可解释性，越高表示 features 越容易解释。
- Mean specificity score：平均解释具体性，越高表示解释越不泛泛。
- Mean coverage score：平均覆盖率，越高表示解释能覆盖更多 top activating contexts。
- Mean false-positive risk：平均误报风险，越低越好。
- Mean quality score：综合解释质量分数，越高越好。
- Undiscernible ratio：被判为无法解释的 feature 比例，越低越好。
- Mixed explanation ratio：被判为 mixed 的 feature 比例，越低表示 content-position 混合更少。
- Content / position / low-level type distribution：不同类型 feature 的比例，用于判断模型是否只是改变了 feature 类型分布。

这些指标按 model、layer、feature type 汇总，并报告 mean / std。

### 可视化产出

建议输出：

- `std / RoPE / PoPE` 的 mean quality score bar chart
- Interpretability score distribution histogram
- Undiscernible ratio by model
- Feature type distribution stacked bar chart
- Layer-wise quality score heatmap
- Phase 5 mixed-feature ratio vs Phase 6 quality score scatter plot

这些图用于展示数值解释结果，而不是只给少量定性案例。

### 案例展示格式

每个模型至少展示若干高质量和低质量案例。推荐格式：

```text
Model: PoPE
Layer: 6
Blinded feature id: Feature A-1842
Feature type: position
Interpretability score: 5
Specificity score: 5
Coverage score: 4
False-positive risk: 1

Explanation:
Activates on tokens near paragraph openings or after sentence boundaries.

Top activating contexts:
1. "... The result was surprising. [The] model ..."
   activated token: "The"
   activation: 18.4
   position: 128

2. "... In this section, [we] describe ..."
   activated token: "we"
   activation: 17.9
   position: 132

3. "... However, [this] does not ..."
   activated token: "this"
   activation: 16.8
   position: 140
```

低质量或 mixed feature 也应展示，例如：

```text
Model: RoPE
Layer: 6
Blinded feature id: Feature B-0912
Feature type: mixed
Interpretability score: 2

Explanation:
The feature appears to activate on both punctuation-adjacent tokens and mid-sequence positions, without a single stable semantic pattern.
```

### 避免评价偏差

- LLM evaluator 不应知道 feature 来自 `std / RoPE / PoPE`。
- Feature ID 使用 blinded names，例如 `Feature A-1842`。
- Prompt、context 数量、context window、采样规则必须一致。
- 不比较相同 feature index；只比较同一采样规则下的分布。
- 主实验使用低 temperature 或 deterministic setting。

### 可选扩展

以下内容不作为两张 A100 条件下的主实验要求：

- Human-labeled calibration subset：可抽样 `30-50` 个 features，由人类标注 feature type 和解释质量，用于校准 LLM score。
- Repeated LLM explanations：只对 calibration subset 或争议 features 重复 `3` 次，不对全部 features 重复。
- Explanation validation：给 LLM top contexts 与 random contexts，让它判断哪些 context 应激活，再计算 explanation precision / recall / F1。
- 多模型全 seed 解释：主实验先用 `seed=42`，资源和预算允许再扩展到 `[41, 42, 43]`。

### 阶段产出

- LLM interpretation score table
- Feature type distribution table
- Quality score summary by model and layer
- Undiscernible ratio comparison
- Mixed explanation ratio comparison
- 高质量和低质量 feature 案例 Markdown
- 可视化图表
- Optional human-LLM agreement report
- 对 RQ3 的辅助回答

### RQ3 回答标准

如果 PoPE 相比 RoPE/std 满足：

- mean quality score 更高
- undiscernible ratio 更低
- mixed explanation ratio 更低
- 高分案例更多且低分案例更少
- Phase 4 reconstruction / feature health 不差
- Phase 5 mixed-feature ratio 更低

则可以支持：

```text
PoPE 的 SAE features 不仅在数值指标上更健康，也更容易形成清晰、一致的自然语言解释。
```

## Phase 7：Steering 与因果干预

### 目标

测试更干净的 SAE features 是否能产生更精准、副作用更少的 steering intervention。

这一阶段用于评估可解释性提升是否真的具有实际价值。

### Steering Vector 构造

从筛选后的 SAE features 中构造 steering vectors。

Feature selection criteria：

- 高 monosemanticity
- 高 activation specificity
- 低 mixed-feature score
- 跨 seed 稳定激活
- 有清晰自然语言解释

### 干预类型

可选干预目标：

- Style control
- Topic control
- Sentiment control
- Safety-related refusal behavior
- Factuality-related behavior
- Syntactic or formatting behavior

当前项目建议先从相对可控的目标开始，例如 style、sentiment、topic，再扩展到 safety-related steering。

### 主要 Steering 指标

Intervention success：

- 目标行为是否增强？

Specificity：

- 干预是否只影响目标行为？

Side effects：

- Perplexity 是否恶化？
- Output fluency 是否下降？
- 无关行为是否改变？

Dose-response：

- 随着 steering strength 增大，目标行为如何变化？

Reversibility：

- 移除 steering vector 后，模型行为是否恢复正常？

### 副作用量化指标

Claude 的反馈中提到副作用指标不足，这一点非常重要。

推荐加入：

- Neutral validation text 上的 perplexity change
- 无关 classifier scores 的变化
- Output length 的变化
- Repetition rate 的变化
- Toxicity 或 safety classifier score 的变化，如果相关
- Human 或 LLM 对 fluency 和 coherence 的 preference comparison

### RoPE vs. PoPE 对比

关键假设：

```text
PoPE-derived SAE features 应该能产生更精准、副作用更少的 steering。
```

需要报告：

- Steering success rate
- Side-effect magnitude
- Success-side-effect trade-off curve
- Feature interpretability 与 steering precision 的相关性

### 阶段产出

- Steering 实验协议
- Steering success metrics
- Side-effect metrics
- RoPE vs. PoPE steering 对比
- 成功和失败干预案例

## Phase 8：可靠性与扩展实验

### 目标

增强实验结果的可靠性和外部有效性。

这一阶段用于确认观察到的现象不是小数据集、单 seed 或某个 SAE 设置造成的偶然结果。

### 多 Seed 扩展

最重要的实验至少应在 3 个 seeds 下运行：

- Model training seeds
- SAE training seeds
- Probe training seeds

推荐报告：

- Mean
- Standard deviation
- Confidence interval，如果可行
- Effect size 或 statistical significance

### 数据集扩展

先使用 WikiText-103 完成受控实验。

之后可扩展到：

- OpenWebText
- C4 subset
- The Pile subset
- Domain-specific text，如果有具体研究需求

目标是测试 what-where decoupling 在更复杂、更开放语境下是否仍然有效。

### 层数和规模扩展

如果算力允许：

- 分析更多 layers。
- 训练稍大模型。
- 比较模型规模增大后 PoPE 的优势是否保持。

重要问题：

```text
随着模型 capacity 增加，PoPE 的 SAE advantage 是增强、减弱，还是保持不变？
```

### 阶段产出

- Multi-seed reliability report
- Dataset generalization report
- Layer-wise robustness report
- Scaling discussion

## Phase 9：新位置编码设计

### 目标

基于前面实验结果，探索新的 positional encoding 设计。

这一阶段建议定位为 future work 或 exploratory work，不应作为主论文必须完成的核心实验。否则研究 scope 会过大。

### 可能方向

Hybrid coupling：

- 在 RoPE-style coupling 和 PoPE-style decoupling 之间插值。
- 测试 partial decoupling 是否能同时保留性能和提升 SAE 可解释性。

Per-head positional encoding：

- 一部分 attention heads 使用 RoPE。
- 一部分 attention heads 使用 PoPE。
- 让不同 heads 专门处理 position-sensitive 或 content-sensitive 功能。

Layer-dependent positional encoding：

- 早期层使用更强 content-position separation。
- 后期层允许更灵活的 content-position interaction。

Multimodal extension：

- 测试 what-where decoupling 是否有助于 vision-language models 中 language-side interpretability。
- 需要注意，视觉表示通常比语言表示更依赖空间结构和内容的紧密绑定。

### 阶段产出

- Conceptual design proposal
- 小规模 pilot experiment，如果可行
- Thesis 或 paper 中的 future work section

# 4. 代码结构设计

## 总体设计目标

本项目的代码结构应服务于一个核心目标：用统一、可复现、可扩展的 pipeline，比较不同 positional encoding 对模型训练、attention 机制、SAE 可解释性和 disentanglement 指标的影响。

代码设计应遵循以下原则：

- 所有实验参数集中存放在 `para.py`。
- 所有入口类都接收统一的 `cfg` 参数。
- 每个主要模块都提供 `.run()` 方法。
- 每个模块的输出统一命名为 `xx_res`，例如 `data_res`、`model_res`、`train_res`。
- 模块之间只通过清晰的结果对象传递信息，避免互相读取内部状态。
- 所有中间结果、checkpoint、图表、表格都保存到固定目录。
- 支持多模型、多 seed、多 layer、多 SAE 配置的扩展。

推荐主流程：

```text
para.py
  ↓
main.py
  ↓
GenerateData
  ↓
SelfTransformer
  ↓
Train
  ↓
SelfSAE
  ↓
Evaluate
  ↓
output/
```

## 目录结构

推荐目录结构如下：

```text
MyResearch/
│
├── para.py
├── main.py
│
├── model.py
├── data.py
├── train.py
├── sae.py
├── eval.py
│
├── utils.py
├── metrics.py
├── visualize.py
├── logger.py
│
├── cache/
│   ├── data/
│   ├── tokens/
│   ├── activations/
│   └── sae_features/
│
├── ckpt/
│   ├── models/
│   └── saes/
│
├── output/
│   ├── figures/
│   ├── tables/
│   ├── logs/
│   ├── reports/
│   └── raw_metrics/
│
└── README.md
```

核心文件包括：

- `para.py`：集中管理所有配置。
- `data.py`：负责数据下载、清洗、tokenization、切分和缓存。
- `model.py`：负责构建不同 positional encoding 的 Transformer 模型。
- `train.py`：负责模型训练、训练行为记录和 attention 机制分析。
- `sae.py`：负责 activation 收集、SAE 训练和 SAE feature 缓存。
- `eval.py`：负责 disentanglement、相关性、互信息和可解释性评估。
- `main.py`：总入口，负责调度完整实验流程。

辅助文件包括：

- `utils.py`：通用工具函数。
- `metrics.py`：通用指标函数。
- `visualize.py`：画图函数。
- `logger.py`：日志与实验记录工具。

## para.py：参数配置中心

#### 描述

`para.py` 负责集中存放所有实验配置。所有配置建议使用 `SimpleNamespace` 表示，并最终统一打包成一个总配置对象 `cfg`。

该文件只负责定义参数，不应包含训练、建模、数据处理或评估逻辑。

建议配置分组：

- `PATH`：路径配置。
- `DATA`：数据集和 tokenizer 配置。
- `MODEL`：模型结构和 positional encoding 配置。
- `TRAIN`：训练、验证、checkpoint 和 attention 分析配置。
- `SAE`：SAE 架构、activation site、dictionary size 和 sparsity 配置。
- `EVAL`：disentanglement、MI、correlation 和解释性评估配置。
- `MAIN`：控制主流程是否运行各个阶段。

#### 代码示例

```python
from types import SimpleNamespace

PATH = SimpleNamespace(
    cache_dir="./cache",
    ckpt_dir="./ckpt",
    output_dir="./output",
    figure_dir="./output/figures",
    table_dir="./output/tables",
    log_dir="./output/logs",
    report_dir="./output/reports",
)

DATA = SimpleNamespace(
    dataset="wikitext",
    config="wikitext-103-raw-v1",
    key="text",
    tokenizer="gpt2",
    seq_len=1024,
    train_blocks=1500,
    valid_blocks=300,
    use_cache=True,
    seed=42,
)

MODEL = SimpleNamespace(
    model_names=["rope", "pope"],
    n_layers=12,
    d_embed=512,
    n_heads=8,
    d_ff=2048,
    dropout=0.1,
    vocab_size=50257,
    seq_len=1024,
    rope_base=10000,
    pope_base=10000,
)

TRAIN = SimpleNamespace(
    seeds=[42, 43, 44],
    steps=70000,
    batch_size=24,
    lr=3e-5,
    warmup_steps=2000,
    data_type="bfloat16",
    device="cuda",
    log_interval=200,
    eval_interval=1000,
    save_interval=10000,
    analysis_interval=5000,
    run_loss_curve=True,
    run_attn_entropy=True,
    run_attn_distance=True,
    run_sv_distribution=True,
    run_toeplitz=True,
)

SAE = SimpleNamespace(
    sae_type="topk",
    activation_site="residual_post_block",
    layers=[2, 6, 10],
    dictionary_sizes=[2048, 4096],
    topk_values=[32, 64],
    batch_size=2048,
    lr=2e-3,
    epochs=400,
    seeds=[42, 43, 44],
)

EVAL = SimpleNamespace(
    run_r2_tok=True,
    run_r2_pos=True,
    run_mutual_information=True,
    run_correlation=True,
    run_feature_selectivity=True,
)

MAIN = SimpleNamespace(
    run_data=True,
    run_model=True,
    run_train=True,
    run_sae=True,
    run_eval=True,
    experiment_name="rope_vs_pope_sae",
)

cfg = SimpleNamespace(
    path=PATH,
    data=DATA,
    model=MODEL,
    train=TRAIN,
    sae=SAE,
    eval=EVAL,
    main=MAIN,
)
```

#### 输出示例

`para.py` 不需要单独输出结果对象。它的作用是提供全局配置对象 `cfg`，其他模块直接 import 使用即可。

```python
from para import cfg
```

## model.py：模型工厂类

#### 描述

`model.py` 的入口类是 `SelfTransformer`。它负责根据 `cfg` 构建不同 positional encoding 的 Transformer 模型。

它只负责模型构建，不负责训练、数据处理、SAE 或评估。

建议支持的模型变体：

- RoPE Transformer
- PoPE Transformer
- std Transformer（Sinusoidal PE）

此外，`model.py` 内部可以放一些通用 Transformer 组件，例如 embedding、attention、feed-forward、decoder layer、Transformer block 和 positional encoding modules。

#### 代码示例

```python
class SelfTransformer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.models = {}

    def build_rope_model(self):
        ...

    def build_pope_model(self):
        ...

    def build_std_model(self):
        ...

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters())

    def build_all(self):
        for model_name in self.cfg.model.model_names:
            if model_name == "rope":
                self.models["rope"] = self.build_rope_model()
            elif model_name == "pope":
                self.models["pope"] = self.build_pope_model()
            elif model_name == "std":
                self.models["std"] = self.build_std_model()
            else:
                raise ValueError(f"Unknown model name: {model_name}")

    def run(self):
        self.build_all()
        model_res = {
            "models": self.models,
            "meta": {
                name: {"num_parameters": self.count_parameters(model)}
                for name, model in self.models.items()
            },
        }
        return model_res
```

#### 输出示例

`model.py` 的输出统一命名为 `model_res`。

```python
model_res = {
    "models": {
        "rope": rope_model,
        "pope": pope_model,
    },
    "meta": {
        "rope": {"num_parameters": 89343057},
        "pope": {"num_parameters": 89346129},
    },
}
```

## data.py：数据生成与预处理类

#### 描述

`data.py` 的入口类是 `GenerateData`。它负责根据 `cfg` 下载、读取、清洗、tokenize、切分和缓存数据。

它不负责模型构建、训练、SAE 或评估。

主要职责包括：加载原始数据集、清洗文本、tokenization、构建 token blocks、切分 train / valid、保存和读取缓存、构建 metadata。

#### 代码示例

```python
class GenerateData:
    def __init__(self, cfg):
        self.cfg = cfg

    def load_raw_dataset(self):
        ...

    def clean_text(self, raw_dataset):
        ...

    def tokenize(self, texts):
        ...

    def build_blocks(self, tokens):
        ...

    def split_train_valid(self, blocks):
        ...

    def build_metadata(self, data_res):
        ...

    def save_cache(self, data_res):
        ...

    def load_cache(self):
        ...

    def validate_data_res(self, data_res):
        ...

    def run(self):
        if self.cfg.data.use_cache:
            data_res = self.load_cache()
            if data_res is not None:
                self.validate_data_res(data_res)
                return data_res

        raw_dataset = self.load_raw_dataset()
        texts = self.clean_text(raw_dataset)
        tokens = self.tokenize(texts)
        blocks = self.build_blocks(tokens)
        data_res = self.split_train_valid(blocks)
        data_res["meta"] = self.build_metadata(data_res)
        self.validate_data_res(data_res)

        if self.cfg.data.use_cache:
            self.save_cache(data_res)

        return data_res
```

#### 输出示例

`data.py` 的输出统一命名为 `data_res`。

```python
data_res = {
    "train": train_dataset,
    "valid": valid_dataset,
    "tokenizer": tokenizer,
    "meta": {
        "dataset": "wikitext",
        "config": "wikitext-103-raw-v1",
        "tokenizer": "gpt2",
        "seq_len": 1024,
        "vocab_size": 50257,
        "num_train_blocks": 1500,
        "num_valid_blocks": 300,
    },
}
```

## train.py：训练与 Attention 机制分析类

#### 描述

`train.py` 的入口类是 `Train`。它负责模型训练、验证、checkpoint 保存，以及训练过程中的机制分析。

`Train` 可以包含 attention entropy、attention distance、singular value distribution、Toeplitz deviation 等分析函数，因为这些指标依赖训练好的模型或训练过程中的模型状态。

它不负责 SAE 训练，也不负责最终 disentanglement 评估。

#### 代码示例

```python
class Train:
    def __init__(self, cfg, model_res, data_res, seeds):
        self.cfg = cfg
        self.model_res = model_res
        self.data_res = data_res
        self.seeds = seeds

    def set_seed(self, seed):
        ...

    def build_optimizer(self, model):
        ...

    def build_scheduler(self, optimizer):
        ...

    def train_one_model(self, model_name, model, seed):
        ...

    def validate(self, model):
        ...

    def save_checkpoint(self, model, model_name, seed, step):
        ...

    def LossCurve(self, train_state):
        ...

    def AttnEntropy(self, model):
        ...

    def AttnDistance(self, model):
        ...

    def SVDistribution(self, model):
        ...

    def Toeplitz(self, model):
        ...

    def run(self):
        train_res = {}

        for model_name, model in self.model_res["models"].items():
            train_res[model_name] = {}

            for seed in self.seeds:
                self.set_seed(seed)
                train_state = self.train_one_model(model_name, model, seed)
                analysis_res = {}

                if self.cfg.train.run_loss_curve:
                    analysis_res["loss_curve"] = self.LossCurve(train_state)
                if self.cfg.train.run_attn_entropy:
                    analysis_res["attn_entropy"] = self.AttnEntropy(model)
                if self.cfg.train.run_attn_distance:
                    analysis_res["attn_distance"] = self.AttnDistance(model)
                if self.cfg.train.run_sv_distribution:
                    analysis_res["sv_distribution"] = self.SVDistribution(model)
                if self.cfg.train.run_toeplitz:
                    analysis_res["toeplitz"] = self.Toeplitz(model)

                train_res[model_name][seed] = {
                    "model": model,
                    "train_state": train_state,
                    "analysis_res": analysis_res,
                }

        return train_res
```

#### 输出示例

`train.py` 的输出统一命名为 `train_res`。

```python
train_res = {
    "rope": {
        42: {
            "model": trained_rope_model_seed42,
            "train_state": {
                "final_step": 70000,
                "best_valid_loss": 2.91,
                "final_valid_loss": 2.95,
                "final_perplexity": 19.1,
            },
            "analysis_res": {
                "loss_curve": loss_curve_res,
                "attn_entropy": attn_entropy_res,
                "attn_distance": attn_distance_res,
                "sv_distribution": sv_distribution_res,
                "toeplitz": toeplitz_res,
            },
        },
    },
}
```

## sae.py：SAE 训练与特征缓存类

#### 描述

`sae.py` 的入口类是 `SelfSAE`。它负责在训练好的模型 activation 上训练 SAE，并输出 SAE 指标、SAE checkpoint 和 feature activations。

主实验建议统一使用 TopK-SAE。Vanilla SAE 可以作为 pilot experiment，但不建议和 TopK-SAE 的主实验结果混合解释。

#### 代码示例

```python
class SelfSAE:
    def __init__(self, cfg, train_res, data_res):
        self.cfg = cfg
        self.train_res = train_res
        self.data_res = data_res

    def collect_activations(self, model, layer, data):
        ...

    def collect_or_load_activations(self, model_name, seed, layer):
        ...

    def build_topk_sae(self, input_dim, dict_size, k):
        ...

    def train_one_sae(self, sae, activations, model_name, seed, layer, dict_size, k):
        ...

    def evaluate_sae(self, sae, activations):
        ...

    def run(self):
        sae_res = {}

        for model_name in self.train_res:
            sae_res[model_name] = {}
            for seed in self.cfg.sae.seeds:
                sae_res[model_name][seed] = {}
                for layer in self.cfg.sae.layers:
                    activations = self.collect_or_load_activations(model_name, seed, layer)
                    layer_res = []
                    for dict_size in self.cfg.sae.dictionary_sizes:
                        for k in self.cfg.sae.topk_values:
                            sae = self.build_topk_sae(activations.shape[-1], dict_size, k)
                            one_sae_res = self.train_one_sae(
                                sae, activations, model_name, seed, layer, dict_size, k
                            )
                            layer_res.append(one_sae_res)
                    sae_res[model_name][seed][layer] = layer_res

        return sae_res
```

#### 输出示例

`sae.py` 的输出统一命名为 `sae_res`。

```python
sae_res = {
    "rope": {
        42: {
            6: [
                {
                    "sae": sae_model,
                    "meta": {
                        "model_name": "rope",
                        "seed": 42,
                        "layer": 6,
                        "dict_size": 4096,
                        "k": 64,
                    },
                    "metrics": {
                        "val_mse": 0.0261,
                        "explained_variance": 0.81,
                        "l0": 64,
                        "dead_feature_rate": 0.08,
                    },
                }
            ]
        }
    },
}
```

## eval.py：可解释性与解耦评估类

#### 描述

`eval.py` 的入口类是 `Evaluate`。它负责整合 `train_res`、`sae_res` 和 `data_res`，计算 disentanglement 与 interpretability 指标。

它不负责模型训练，也不负责 SAE 训练。

必须包含的指标包括 `R2_tok`、`R2_pos`、`MI` 和 `Correlation`。推荐补充 `FeatureSelectivity`、`MixedFeatureRatio`、`ProbeAccuracy`、`MonosemanticityScore`、`HumanLLMAgreement`、`SteeringSuccess` 和 `SteeringSideEffect`。

#### 代码示例

```python
class Evaluate:
    def __init__(self, cfg, train_res, sae_res, data_res):
        self.cfg = cfg
        self.train_res = train_res
        self.sae_res = sae_res
        self.data_res = data_res

    def R2_tok(self):
        ...

    def R2_pos(self):
        ...

    def MI(self):
        ...

    def Correlation(self):
        ...

    def FeatureSelectivity(self):
        ...

    def MixedFeatureRatio(self):
        ...

    def save_results(self, eval_res):
        ...

    def run(self):
        eval_res = {}

        if self.cfg.eval.run_r2_tok:
            eval_res["r2_tok"] = self.R2_tok()
        if self.cfg.eval.run_r2_pos:
            eval_res["r2_pos"] = self.R2_pos()
        if self.cfg.eval.run_mutual_information:
            eval_res["mi"] = self.MI()
        if self.cfg.eval.run_correlation:
            eval_res["correlation"] = self.Correlation()
        if self.cfg.eval.run_feature_selectivity:
            eval_res["feature_selectivity"] = self.FeatureSelectivity()

        self.save_results(eval_res)
        return eval_res
```

#### 输出示例

`eval.py` 的输出统一命名为 `eval_res`。

```python
eval_res = {
    "r2_tok": r2_tok_res,
    "r2_pos": r2_pos_res,
    "mi": mi_res,
    "correlation": correlation_res,
    "feature_selectivity": feature_selectivity_res,
    "mixed_feature_ratio": mixed_feature_ratio_res,
}
```

## main.py：总入口 Pipeline

#### 描述

`main.py` 是整个项目的总入口。它负责调用所有入口类，并将不同阶段串成一个完整实验流程。

核心思想保持不变：`main.py` 不写具体训练细节，只负责调度。

#### 代码示例

```python
from data import GenerateData
from model import SelfTransformer
from train import Train
from sae import SelfSAE
from eval import Evaluate
from para import cfg


class MainPipeline:
    def __init__(self, cfg):
        self.cfg = cfg

    def prepare_dirs(self):
        ...

    def run(self):
        self.prepare_dirs()

        data_res = None
        model_res = None
        train_res = None
        sae_res = None
        eval_res = None

        if self.cfg.main.run_data:
            data_res = GenerateData(self.cfg).run()

        if self.cfg.main.run_model:
            model_res = SelfTransformer(self.cfg).run()

        if self.cfg.main.run_train:
            train_res = Train(self.cfg, model_res, data_res, self.cfg.train.seeds).run()

        if self.cfg.main.run_sae:
            sae_res = SelfSAE(self.cfg, train_res, data_res).run()

        if self.cfg.main.run_eval:
            eval_res = Evaluate(self.cfg, train_res, sae_res, data_res).run()

        main_res = {
            "data_res": data_res,
            "model_res": model_res,
            "train_res": train_res,
            "sae_res": sae_res,
            "eval_res": eval_res,
        }
        return main_res


if __name__ == "__main__":
    pipeline = MainPipeline(cfg)
    main_res = pipeline.run()
```

#### 输出示例

`main.py` 的输出统一命名为 `main_res`。

```python
main_res = {
    "data_res": data_res,
    "model_res": model_res,
    "train_res": train_res,
    "sae_res": sae_res,
    "eval_res": eval_res,
}
```

## utils.py：通用工具

#### 描述

`utils.py` 存放真正通用的工具函数，不应包含具体训练、SAE 或评估逻辑。

#### 代码示例

```python
def set_seed(seed):
    ...

def ensure_dir(path):
    ...

def save_json(obj, path):
    ...

def load_json(path):
    ...

def count_parameters(model):
    ...

def format_experiment_name(cfg):
    ...
```

#### 输出示例

工具函数通常不产生统一模块级输出，但如果需要记录结果，可以使用 `utils_res`。

```python
utils_res = {
    "created_dirs": ["./cache", "./ckpt", "./output"],
    "experiment_name": "rope_vs_pope_sae",
}
```

## metrics.py：通用指标函数

#### 描述

`metrics.py` 存放可复用指标函数，避免 `train.py` 和 `eval.py` 变得过于臃肿。

#### 代码示例

```python
def compute_attention_entropy(attn_weights):
    ...

def compute_attention_distance(attn_weights):
    ...

def compute_singular_values(logits):
    ...

def compute_toeplitz_deviation(logits):
    ...

def compute_r2_score(x, y):
    ...

def compute_mutual_information(x, y):
    ...

def compute_correlation(x, y):
    ...

def compute_dead_feature_rate(feature_acts):
    ...

def compute_l0(feature_acts):
    ...
```

#### 输出示例

单个指标函数输出也应使用 `xx_res` 命名。

```python
attn_entropy_res = {
    "layer_mean": {...},
    "head_mean": {...},
}

toeplitz_res = {
    "layer_mean": {...},
    "head_mean": {...},
}
```

## visualize.py：画图函数

#### 描述

`visualize.py` 负责所有图片生成逻辑。所有画图函数都应接收明确输入，并保存到 `output/figures/`。

#### 代码示例

```python
def plot_loss_curve(loss_curve_res, save_path):
    ...

def plot_attention_entropy(attn_entropy_res, save_path):
    ...

def plot_attention_distance(attn_distance_res, save_path):
    ...

def plot_sv_distribution(sv_distribution_res, save_path):
    ...

def plot_toeplitz_deviation(toeplitz_res, save_path):
    ...

def plot_sae_metrics(sae_res, save_path):
    ...

def plot_r2_comparison(eval_res, save_path):
    ...
```

#### 输出示例

```python
figure_res = {
    "loss_curve": "./output/figures/train/loss_curve_rope_seed42.png",
    "attn_entropy": "./output/figures/train/attn_entropy_rope_seed42.png",
    "toeplitz": "./output/figures/train/toeplitz_rope_seed42.png",
}
```

## logger.py：实验记录工具

#### 描述

`logger.py` 负责实验日志和配置快照，帮助后续复现实验。

#### 代码示例

```python
class ExperimentLogger:
    def __init__(self, cfg):
        self.cfg = cfg

    def log_config(self):
        ...

    def log_stage_start(self, stage_name):
        ...

    def log_stage_end(self, stage_name):
        ...

    def log_metric(self, name, value, step=None):
        ...

    def log_error(self, error):
        ...
```

#### 输出示例

```python
logger_res = {
    "log_path": "./output/logs/rope_vs_pope_sae.log",
    "config_snapshot_path": "./output/reports/rope_vs_pope_sae_config_snapshot.md",
}
```

## cache：缓存文件夹

`cache/` 存放可以复用的中间结果，避免重复计算。

推荐结构：

```text
cache/
├── data/
├── tokens/
├── activations/
└── sae_features/
```

命名建议：

```text
cache/activations/{model_name}_seed{seed}_layer{layer}_{activation_site}.pt
cache/sae_features/{model_name}_seed{seed}_layer{layer}_dict{dict_size}_k{k}.pt
```

## ckpt：模型与 SAE 权重

`ckpt/` 存放训练过程中可恢复的模型权重和 SAE 权重。

推荐结构：

```text
ckpt/
├── models/
└── saes/
```

命名建议：

```text
ckpt/models/{experiment_name}_{model_name}_seed{seed}_step{step}.pt
ckpt/saes/{experiment_name}_{model_name}_seed{seed}_layer{layer}_dict{dict_size}_k{k}.pt
```

## output：图表、指标与报告

`output/` 保存所有最终可读的实验结果。

推荐结构：

```text
output/
├── figures/
├── tables/
├── logs/
├── reports/
└── raw_metrics/
```

内容建议：

- `figures/`：loss curve、attention heatmap、singular value spectrum、Toeplitz plot、SAE metric plot。
- `tables/`：perplexity table、SAE metrics table、R2 table、MI table、correlation table。
- `logs/`：训练日志和 pipeline 日志。
- `reports/`：自动生成的实验总结。
- `raw_metrics/`：`.json`、`.csv` 或 `.pt` 格式的原始指标。

## 推荐数据流

```text
para.py
  ↓
main.py
  ↓
GenerateData(cfg).run()
  → data_res
  ↓
SelfTransformer(cfg).run()
  → model_res
  ↓
Train(cfg, model_res, data_res, seeds).run()
  → train_res
  ↓
SelfSAE(cfg, train_res, data_res).run()
  → sae_res
  ↓
Evaluate(cfg, train_res, sae_res, data_res).run()
  → eval_res
  ↓
main_res
  ↓
output/
```

## 推荐命名规范

模型名建议统一小写：

```text
rope
pope
std
nope
```

实验名建议包含主比较对象、数据集、模型规模和日期：

```text
{main_comparison}_{dataset}_{model_size}_{date}
```

示例：

```text
rope_vs_pope_wikitext_90m_20260508
```

结果变量命名统一采用：

```text
data_res
model_res
train_res
sae_res
eval_res
main_res
```

文件名尽量包含 experiment name、model name、seed、layer、dict size、k value 和 step。

## 最小可运行版本

如果先做最小可运行版本，建议只实现：

```text
para.py
data.py
model.py
train.py
main.py
```

最小 pipeline：

```text
1. 加载配置
2. 生成 WikiText 数据
3. 构建 RoPE 和 PoPE 模型
4. 训练一个 seed
5. 输出 loss curve 和 validation perplexity
```

之后再逐步加入：

```text
6. 多 seed
7. Attention entropy / distance
8. SVDistribution / Toeplitz
9. SAE training
10. eval.py
11. LLM-assisted interpretation
12. Steering
```

## 最终建议

你的原始结构已经很清楚，尤其是每个核心阶段都有一个入口类，并统一使用 `.run()` 作为外部接口。接下来最重要的是把命名和输出结构固定下来。

最终结构可以概括为：

```text
para.py       负责配置，提供 cfg
data.py       负责数据，输出 data_res
model.py      负责建模，输出 model_res
train.py      负责训练和 attention 机制分析，输出 train_res
sae.py        负责 SAE 训练与 feature 缓存，输出 sae_res
eval.py       负责 disentanglement 和 interpretability 评估，输出 eval_res
main.py       负责调度所有模块，输出 main_res
utils.py      负责通用工具
metrics.py    负责通用指标
visualize.py  负责画图
logger.py     负责日志
```

这样设计后，代码会比较适合从 toy experiment 平滑扩展到 multi-seed、multi-baseline、multi-layer 的正式实验。

# 5. 成本与规模分析
## 5.1 OpenWebText 与 GPT-2 small 的数据规模
如果以 GPT-2 small 级别预训练作为参照，常见的 OpenWebText 量级约为：
- documents: 约 8 million
- raw text: 约 38GB-42GB
- GPT-2 tokenizer 后 token 数: 约 8B-9B tokens
因此，真正接近 GPT-2 small 风格预训练的数据规模，不是当前 pilot 设置中的百万级 tokens，而是至少十亿级 tokens，理想情况下接近完整 OpenWebText 的 8B-9B tokens。
## 5.2 9B tokens 实验的本地空间需求

9B tokens 指的是 tokenized 后的训练量，并不直接等于本地磁盘空间。磁盘占用取决于是否缓存 HuggingFace dataset、是否保存 tokenized blocks、以及 token 使用什么 dtype 存储。

以 OpenWebText 为例，粗略空间估计如下：

```text
HuggingFace 原始/展开缓存: 约 40GB-55GB
9B tokens, int64 token cache: 约 72GB
9B tokens, int32 token cache: 约 36GB
9B tokens, uint16 token cache: 约 18GB
```

如果同时保留 HuggingFace dataset cache 和 tokenized cache，完整 9B tokens 实验通常需要：

```text
80GB-150GB+ 仅用于数据
```

如果再考虑模型 checkpoint、SAE activation cache、SAE checkpoint、attention 指标和日志，建议云服务器磁盘至少：

```text
最低: 200GB
更舒服: 500GB
```

尤其当实验包含：

```text
std / rope / pope × 3 seeds
```

时，checkpoint 和 activation cache 会迅速膨胀。因此不建议长期保存所有中间 activation。更合理的做法是：

```text
1. 原始 OpenWebText 尽量 streaming 读取。
2. tokenized 数据可保存为 uint16 .bin 或 memory-mapped 格式。
3. SAE activation 按 layer / seed 分批生成，用完后只保留必要样本或统计。
4. checkpoint 只保留 best 和 final，避免频繁保存全部权重。
```

## 5.3 实验说服力分档

对于本研究，实验说服力主要来自：

```text
控制变量是否严格
是否有多个 positional encoding baseline
是否有多个 random seeds
是否有 attention / disentanglement / SAE / steering 的完整证据链
```

因此，不建议一开始直接追求 9B tokens。更推荐分阶段推进。

### Pilot 级别

```text
10M tokens
std / rope / pope
1 seed
小模型或 GPT-2 small shape
```

用途：

```text
验证 pipeline
检查 loss curve
检查 attention metrics 是否正常
检查 PoPE 是否能稳定训练
```

该级别不能支持强结论，但适合快速排错。

### 最低有说服力版本

```text
100M-300M tokens
std / rope / pope
3 seeds
GPT-2 small shape 或稍小模型
layers = [2, 6, 10]
TopK-SAE
```

该级别可以支持较初步但有意义的结论：如果 PoPE 在多个 seed 下稳定表现出更低 mixed-feature ratio、更好 SAE feature health 或更强 disentanglement，那么可以说趋势不只是 toy setting 的偶然现象。

### 推荐主实验版本

```text
500M-1B tokens
std / rope / pope
3 seeds
GPT-2 small shape
attention + Toeplitz + SAE + disentanglement + steering
```

这是当前项目最推荐的正式目标。原因是 1B tokens 对 117M 参数模型已经具备较强说服力，同时成本仍明显低于完整 9B tokens。更重要的是，这个规模下仍有可能把完整证据链跑全。

推荐将实验主线设为：

```text
Pilot: 10M tokens
Main: 500M tokens
Robustness: 1B tokens
```

对应到 `seq_len = 1024` 时，大致需要：

```text
10M tokens   -> train_blocks ≈ 9,750
100M tokens  -> train_blocks ≈ 97,500
500M tokens  -> train_blocks ≈ 488,000
1B tokens    -> train_blocks ≈ 976,000
9B tokens    -> train_blocks ≈ 8,780,000
```

### 高成本完整版本

```text
3B-9B tokens
std / rope / pope
3 seeds
GPT-2 small shape
完整 SAE / steering / robustness 分析
```

该版本最强，但成本非常高。因为完整主比较至少包含：

```text
3 positional encodings × 3 seeds = 9 个 GPT-2 small 训练
```

再加上 SAE 训练、activation cache 和 steering 评估，整体成本会远高于单个 GPT-2 small 预训练。因此，除非算力和时间非常充足，否则不建议作为第一阶段目标。

## 5.4 推荐结论

本项目更应该优先保证：

```text
multi-seed > single huge run
完整指标链 > 单一 reconstruction loss
500M-1B tokens 的稳定比较 > 9B tokens 的单 seed 比较
```

换句话说，如果资源有限，更推荐：

```text
500M tokens × std/rope/pope × 3 seeds
```

而不是：

```text
9B tokens × std/rope/pope × 1 seed
```

因为本研究的关键结论依赖稳定差异，而不是一次大规模训练中的偶然结果。
