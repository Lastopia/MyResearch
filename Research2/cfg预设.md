# cfg 预设

## 小模型筛选参数

用于快速筛选 `attn`、`ffn` 和 `loss` 机制变体，优先降低训练成本，方便多组对照实验。

```text
set model n_layer=6 n_head=6 d_model=384 d_ff=1536 block_size=256 position_encoding=rope
set data blocks_alias=OWT_gpt2_b256_train100k train_blocks=100000 valid_blocks=5000
set train max_steps=10000 lr=1e-4 min_lr=1e-5 grad_clip=1.0
set sae retrain=true
```

## 小模型机制配置

```text
set run task=stage1_full models=base,top16attn,top32attn,top64attn,top128attn,groupmix4ffn,groupmix8ffn,sparse12ffn,sparse25ffn,sparse50ffn,l1act1e4,l1act3e4 seed=42
set models base attn.name=std ffn.name=std loss.name=ce
set models top16attn attn.name=topk attn.k=16 ffn.name=std loss.name=ce
set models top32attn attn.name=topk attn.k=32 ffn.name=std loss.name=ce
set models top64attn attn.name=topk attn.k=64 ffn.name=std loss.name=ce
set models top128attn attn.name=topk attn.k=128 ffn.name=std loss.name=ce
set models groupmix4ffn attn.name=std ffn.name=groupmix ffn.groups=4 ffn.mix_ratio=0.125 ffn.mix_alpha=0.25 loss.name=ce
set models groupmix8ffn attn.name=std ffn.name=groupmix ffn.groups=8 ffn.mix_ratio=0.125 ffn.mix_alpha=0.25 loss.name=ce
set models sparse12ffn attn.name=std ffn.name=sparse ffn.mode=topk ffn.k=192 loss.name=ce
set models sparse25ffn attn.name=std ffn.name=sparse ffn.mode=topk ffn.k=384 loss.name=ce
set models sparse50ffn attn.name=std ffn.name=sparse ffn.mode=topk ffn.k=768 loss.name=ce
set models l1act1e4 attn.name=std ffn.name=std loss.name=l1_act loss.lambda=1e-4
set models l1act3e4 attn.name=std ffn.name=std loss.name=l1_act loss.lambda=3e-4
```

## GPT-2 Small 级别参数

用于对第一阶段筛选出的有效机制进行更接近 GPT-2 small 规模的复验。

```text
set model n_layer=12 n_head=12 d_model=768 d_ff=3072 block_size=512 position_encoding=rope
```

## GPT-2 Small 级别机制配置

```text
set models base_gpt2 attn.name=std ffn.name=std loss.name=ce
set models top32attn_gpt2 attn.name=topk attn.k=32 ffn.name=std loss.name=ce
set models groupmixffn_gpt2 attn.name=std ffn.name=groupmix ffn.groups=4 ffn.mix_ratio=0.125 ffn.mix_alpha=0.25 loss.name=ce
set models sparseffn_gpt2 attn.name=std ffn.name=sparse ffn.mode=topk ffn.k=768 loss.name=ce
set models l1act_gpt2 attn.name=std ffn.name=std loss.name=l1_act loss.lambda=1e-4
set models top64attn_gpt2 attn.name=topk attn.k=64 ffn.name=std loss.name=ce
```
