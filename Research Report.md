# 建议论文结构

### Introduction

说明当前 mechanistic interpretability 通常把模型架构视为固定背景，但本研究认为位置编码本身会影响模型表示是否容易被解释。

### Background

介绍：

- SAE interpretability
- Superposition 和 monosemanticity
- RoPE 与 positional-content coupling
- PoPE 与 polar-coordinate decoupling

### Hypothesis

明确提出主假设：

```text
通过位置编码设计降低 what-where entanglement，
可以在保持语言模型性能的同时提升 SAE decomposition quality。
```

### Methods

描述：

- Model variants
- Dataset
- Training protocol
- SAE training protocol
- Disentanglement probes
- Attention spectral analysis
- LLM 与 human evaluation
- Steering experiments

### Results

按研究问题组织结果：

- RQ1：模型性能
- RQ2：解耦效果
- RQ3：SAE 可解释性
- RQ4：attention 机制变化
- Steering utility

### Discussion

讨论：

- PoPE 是否提升可解释性
- PoPE 是否损害模型性能
- 更好的 SAE metrics 是否对应更好的人类可解释性
- 更好的可解释性是否带来更精准的 steering
- Decoupling 和 model expressivity 之间的 trade-off

### Limitations

包括：

- 模型规模较小
- 数据集多样性有限
- 评价部分依赖 LLM judge
- 算力限制
- Toy model 与 production LLM 之间可能存在差异

### Future Work

包括：

- Hybrid positional encoding
- Per-head positional encoding
- 更大模型 scaling
- Multimodal extension
- 更严格的 causal intervention benchmark

## 15. 最终可以支撑的 Claim 层级

最后写作时需要避免过度宣称。比较稳妥的做法是分层建立 claim。

### 弱 Claim

PoPE 相比 RoPE 改变了 SAE reconstruction 和 sparsity metrics。

### 中等 Claim

PoPE 产生了更容易区分 content 与 position 的表示，因此降低了 SAE decomposition difficulty。

### 强 Claim

位置编码等架构选择可以与 SAE 等可解释性工具共同设计，从而得到既保持性能、又更容易分析和干预的模型。

当前项目最适合努力支撑中等 Claim，并将强 Claim 作为更大的研究愿景。


