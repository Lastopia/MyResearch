# MyResearch

本仓库统一管理四个相互独立、可分别运行的研究实验。每个实验保留自己的入口、配置、依赖与研究文档；请从对应目录启动，以避免同名模块和相对路径互相影响。

| 目录 | 研究主题 | 使用说明 |
| --- | --- | --- |
| [`Research1/`](Research1/) | RoPE / PoPE 与 SAE 分析 | [`Research1/README.md`](Research1/README.md) |
| [`Research2/`](Research2/) | Transformer attention、FFN 与 loss 机制变体 | [`Research2/README.md`](Research2/README.md) |
| [`Research3/`](Research3/) | RoPE、ALiBi、CABLE 与 RA-CABLE 对照实验 | [`Research3/实验设计.md`](Research3/实验设计.md) |
| [`Research4/`](Research4/) | FFN Concept Subspace Bus V2 | [`Research4/README.md`](Research4/README.md) |

## 仓库约定

- 四个实验分别维护自己的 `requirements.txt`；建议为每个实验使用独立虚拟环境。
- 数据集、下载缓存、checkpoint、模型权重、日志及 `output*` 结果目录均由根目录 `.gitignore` 统一排除。
- 研究设计、理论、结果分析等 Markdown 文档属于仓库内容，会正常被 Git 跟踪。
- 大型实验产物应存放在仓库外部或专用制品存储中，不直接提交到 Git。

根目录还包含汇总材料：[`Research Report.md`](Research%20Report.md) 与 `Research Presentation.pptx`。
