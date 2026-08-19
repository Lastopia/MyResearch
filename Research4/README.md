# RE3：FFN Concept Subspace Bus V2

本项目检验一个收缩后的假设：能否从完整 FFN 中间激活中提取少量并列概念槽，让它们经独立 causal attention 通信，并通过可干预的小残差子空间参与预测，同时基本保持语言性能。

这里“独立”指国家/颜色序列及其 K/V 状态不互相读取；为降低小 kernel 开销，当前主版本共享 QKV 参数并合批计算。

## 一键运行

```bash
pip install -r requirements.txt
python main.py --size small
python main.py --size medium
python main.py --size large
```

| size | 用途 | runs | 结论效力 |
|---|---|---:|---|
| `small` | CPU smoke test | 1 | 仅验证代码路径 |
| `medium` | 三模型、三 seed 公平确认 | 9 | 正式检验 V2 相对 Projector 的稳定性 |
| `large` | 最小论文级确认矩阵 | 27 | 三 seed 正式结果 |

只查看计划：

```bash
python main.py --size medium --dry-run
```

继续最终受控复验并复用 `output4` 中已经完成的 seed 11：

```bash
python main.py --size medium --output-root output4
```

该命令会校验现有 seed 11 的配置哈希与源码指纹，跳过三个已完成运行，
只调度 seed 22、33 的 Standard、Concept Projector 和 Concept Bus V2，
完成后自动重建逐 seed 汇总、均值、标准差、置信区间及配对比较。

## V2 结构

```text
标准 causal attention（保留）
        ↓
完整 FFN 激活 U: d_model → d_ff
   ├─ private_down → 普通 FFN 更新的前 93.75% 残差坐标
   └─ 两个低秩投影 → 国家槽 / 颜色槽
                         ↓
                 独立 causal attention
                 （无输入残差旁路）
                         ↓
        独立 sigmoid 概念 → 后 6.25% 概念写回坐标
```

`concept_projector` 与 V2 参数量、投影深度及 attention 算子完全相同，但 attention 只允许每个 token 读取自身；V2 才允许读取当前及历史 token。因而两者的主要受控变量就是“是否跨 token 通信”。训练采用尺度归一化 smooth-max，使任务、概念和反事实三项目标中最弱的一项获得更大梯度；验证损失始终只计算任务交叉熵。

## 实验矩阵

| 预设 | 阶段 | 方法 | seed | runs |
|---|---|---|---|---:|
| medium | DualTag 公平确认 | standard / projector / V2 | 11、22、33 | 9 |
| large | DualTag | standard / matched / projector / V2 | 11、22、33 | 12 |
| large | CLUTRR | standard / V2 | 11、22、33 | 6 |
| large | FineWeb-Edu LM | standard / matched / V2 | 11、22、33 | 9 |

## 正式 LM 理论成本

512-token、约 124M 参数配置：

| 方法 | 参数量 | 理论 MAC |
|---|---:|---:|
| standard | 123,551,232 | 48.318B |
| parameter_matched | 123,809,280 | 48.451B |
| concept_projector | 123,814,860 | 48.453B |
| concept_bus_v2 | 123,814,860 | 48.529B |

V2 相对 standard：参数约 `+0.21%`，理论 MAC 约 `+0.44%`。真实训练时间与显存必须以服务器日志为准。

medium 的九个运行固定使用同一生成数据、各 seed 内一致的数据顺序、
`micro-batch=64`、`gradient accumulation=2` 和完整 20,000 条验证集。
显存不够时任务进入下一 wave，禁止静默缩小 micro-batch。每 500 步按纯任务
验证损失更新 `best.pt`，最终测试和因果审计均加载最佳验证权重，而非最后一步权重。

## 存储与验证

```text
RE3/
├─ data/       # 下载或生成的数据
├─ ckpt/       # 每约 120 分钟的恢复点、final 训练状态与 best 验证权重
└─ output/     # 日志、final.json、CSV、SVG、HTML/JS
```

```bash
pytest -q
```

`medium/large` 启动前会执行真实 CUDA 前向与反向探针；`medium` 还会在每张
可用 GPU 上按 `batch=64、length=256` 跑一次 V2 前向、反向和 AdamW step。
GPU 不可用或真实训练形状发生 OOM 时，长训练不会启动。
