<!-- 模板说明：本文件保存可直接复制到 main.py 控制台的 set ... 参数预设。每个预设 = 用途说明 + 命令块。key 必须与 cfg.py 中的字典严格匹配；新预设追加新节，废弃的预设注明废弃原因后保留，避免重复试错。 -->

# cfg 预设

## 筛选阶段参数（小档位）

用于快速筛选 {{机制变体}}，优先降低训练成本，方便多组对照实验。

```text
set model {{规模参数，如 n_layer=6 n_head=6 d_model=384 d_ff=1536 block_size=256}}
set data {{数据别名与用量，如 blocks_alias=XX_b256_train100k train_blocks=100000 valid_blocks=5000}}
set train {{训练参数，如 max_steps=10000 lr=1e-4 min_lr=1e-5 grad_clip=1.0}}
```

## 筛选阶段机制配置

<!-- 命名规则：别名 = 机制简写 + 关键超参，如 top16attn / groupmix4ffn / l1act1e4。 -->

```text
set run task={{task 名}} models=base,{{机制别名 1}},{{机制别名 2}} seed=42
set models base {{机制字段}}.name=std ...
set models {{机制别名 1}} {{机制字段}}.name={{机制名}} {{参数}}={{值}} ...
set models {{机制别名 2}} ...
```

## 复验阶段参数（大档位）

用于对筛选阶段胜出的机制做更大规模复验。

```text
set model {{规模参数，如 n_layer=12 n_head=12 d_model=768 ...}}
```

## 复验阶段机制配置

```text
set run task={{task 名}} models=base,{{胜出机制别名}} seed=42
set models {{胜出机制别名}} ...
```

## 评估阶段参数

```text
set {{评估阶段字典}} {{评估设置，如 hook_layer=-1 expansion=8 k=32}}
```

## 废弃预设

<!-- 格式：预设名 + 废弃原因 + 保留条件，防止未来重复试错。 -->

- {{预设名}}：{{废弃原因，如排序成本过高 / 学习效果下降明显}}。保留条件：{{如出现低成本的近似实现可重启}}
