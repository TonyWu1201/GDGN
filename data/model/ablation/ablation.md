# Phase 6 消融实验

> data/model/ablation 目录存放 GDGN 调参重训 + 消融实验的所有产物。详细规划见 [guidance/Phase6消融实验规划.md](../../guidance/Phase6消融实验规划.md)。

## 目录结构

```
data/model/ablation/
├── README.md                         # 本文件 (目录说明 + 执行指南)
├── configs/                          # 各消融变体的 JSON 配置
│   ├── 00_original.json              # A0 对照 (Phase 4 原配置, 无需重训)
│   ├── 01_main.json                  # A1 主推荐: lr=1e-4 + use_main_drug_emb=True
│   ├── 02_lr_only.json               # A2 单变量: 仅 lr_encoder=1e-4
│   ├── 03_freeze_main.json           # A3 freeze_encoder=True + use_main_drug_emb=True
│   ├── 04_no_pretrain.json           # A4 随机初始化 (无预训练) + lr=1e-4
│   ├── 05_arch.json                  # A5 架构改进: n_query_tokens=4 + predictor_hidden=512
│   └── 06_heads8.json                # A6 (可选): num_heads=8 + 与 A1 其余一致
├── 01_main/                          # A1 训练 + 评估产物
│   ├── best_model.pt
│   ├── last_model.pt
│   ├── train_log.json
│   ├── trained_config.json
│   ├── final_eval_report.txt          # paired split 指标 (train/val/test + 7 无 DTI)
│   └── generalization_report.{txt,json}  # LODO / LOCO 真泛化指标
├── 02_lr_only/
├── 03_freeze_main/
├── 04_no_pretrain/
├── 05_arch/
├── 06_heads8/                        # (可选)
└── summary.md                         # 跑完后填入跨变体对比表
```

A0 不重训,直接复用 `data/model/gdgn/best_model.pt` (Phase 4 产物) 作对照。

## 变体与假设

| ID | 名称 | 关键参数变化 | 假设 |
|:---:|---|---|---|
| A0 | original | (Phase 4 已训) | baseline, 作为对照 |
| A1 | main | `lr_encoder=1e-4` + `use_main_drug_emb=True` | 综合最优: 大 lr 让 encoder 真正微调到 IC50 + 修复 M2 死分支 |
| A2 | lr_only | `lr_encoder=1e-4` (单变量) | 验证 lr 是不是核心瓶颈 (隔离 use_main_drug_emb 影响) |
| A3 | freeze_main | `freeze_encoder=True` + `use_main_drug_emb=True` | fixed feature extractor + 死分支消除: 测预训练本身是否够用 |
| A4 | no_pretrain | `pretrain_ckpt=null` + `lr_encoder=1e-4` | 测图架构本身 vs Phase 2 预训练的相对贡献 |
| A5 | arch | `n_query_tokens=4` + `predictor_hidden=512` + lr=1e-4 + main_drug | 解 cross-attn 单 query 信息瓶颈 + 增容量 |
| A6 | heads8 (可选) | `num_heads=8` + 其余与 A1 一致 | 更多 attention 头是否能更好捕捉 gene×drug 异质关系 |

## 执行流程 (每个变体三步)

> 4 卡 DDP, batch_size=64 → per_gpu_batch=16 (与 Phase 4 原配置一致, 不易 OOM)

### Step 1 — 训练 (~10-15h / 变体)

```bash
# Linux 服务器 4 卡 DDP
torchrun --nproc_per_node=4 program/model/train_gdgn.py \
    --model gdgn --config data/model/ablation/configs/01_main.json
```

```powershell
# Windows 本机 (.venv 已配好, AGENTS.md 提示不要用 uv run 避免 torch 重下载;
# Windows 不支持 NCCL DDP, 单卡跑也可, batch_size=64 单卡会 OOM:
# 改 config.json 里 batch_size=16 单卡跑, 时间约 40h/变体)
.venv\Scripts\python.exe program/model/train_gdgn.py `
    --model gdgn --config data/model/ablation/configs/01_main.json `
    --batch_size 16
```

### Step 2 — Paired split 评估 (post_eval, ~1-2 min)

```bash
# 服务器
uv run python program/model/train_gdgn.py --model gdgn --post_eval \
    --ckpt data/model/ablation/01_main/best_model.pt \
    --output_dir data/model/ablation/01_main
```

### Step 3 — 泛化测试 LODO/LOCO (~1-5 min)

```bash
uv run python program/model/eval_generalization.py \
    --model gdgn --ckpt data/model/ablation/01_main/best_model.pt \
    --output_dir data/model/ablation/01_main --batch_size 32
```

## 批量执行脚本 (Linux 服务器, 4 卡 DDP)

把 5 个核心变体 (A1-A5) 顺序跑完 (~50-75h ≈ 2-3 天):

```bash
# ablation_run.sh
#!/bin/bash
set -e
VARIANTS=("01_main" "02_lr_only" "03_freeze_main" "04_no_pretrain" "05_arch")
for v in "${VARIANTS[@]}"; do
    echo "========================================"
    echo "[$(date)] START $v"
    echo "========================================"
    uv run torchrun --nproc_per_node=4 program/model/train_gdgn.py \
        --model gdgn --config data/model/ablation/configs/${v}.json
    uv run python program/model/train_gdgn.py --model gdgn --post_eval \
        --ckpt data/model/ablation/${v}/best_model.pt \
        --output_dir data/model/ablation/${v}
    uv run python program/model/eval_generalization.py \
        --model gdgn --ckpt data/model/ablation/${v}/best_model.pt \
        --output_dir data/model/ablation/${v}
    echo "[$(date)] DONE $v"
done
echo "All ablation variants finished."
```

可选第 6 个变体 A6 heads8 单独跑:

```bash
uv run torchrun --nproc_per_node=4 program/model/train_gdgn.py \
    --model gdgn --config data/model/ablation/configs/06_heads8.json
# ... 同样 post_eval + generalization
```

## 时间估算 (4 卡 DDP)

| 变体 | 训练 (~10-15h) | post_eval (~1min) | generalization (~5min) | 合计 |
|---|---:|---:|---:|---:|
| A1-A5 (5 个核心) | 50-75h | 5-10min | 25-30min | ~55-80h (2-3 天) |
| + A6 (可选第 6 个) | 11-12h | 1-2min | 5min | ~12h (额外半天) |

A5 arch 因参数量增大 (~2.9M vs A0 1.77M), 训练时间略长 (~12-15h)。

## 优先级分组

- **必跑 (诊断核心)**: A1 main, A2 lr_only, A4 no_pretrain
- **推荐 (架构验证)**: A3 freeze_main, A5 arch
- **可选 (扩展性)**: A6 heads8

资源紧张时只跑必跑 3 个 (~30-45h ≈ 1.5 天)。

## 产出对照表 (summary.md, 跑完后回填)

| ID | val PCC | test PCC | LODO mean | LODO median | LOCO mean | 结论 |
|:---:|---:|---:|---:|---:|---:|---|
| A0 | 0.9083 | 0.9055 | 0.6505 | 0.6675 | 0.9124 | (Phase 4 已测) |
| A1 | - | - | - | - | - |  |
| A2 | - | - | - | - | - |  |
| A3 | - | - | - | - | - |  |
| A4 | - | - | - | - | - |  |
| A5 | - | - | - | - | - |  |
| A6 | - | - | - | - | - | (可选) |
| Baseline | 0.9387 | 0.9378 | 0.8154 | 0.8200 | 0.9584 | (Phase 4 已测, target) |

**关键 Benchmarks** (判断消融是否成功):
- LODO mean PCC > 0.8 (追平 baseline) = 成功
- LODO mean PCC > A0 (0.65) + 0.05 = 显著改进
- LOCO mean PCC > baseline (0.9584) = 全面超越