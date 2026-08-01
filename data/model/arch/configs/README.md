# Phase 4.2 架构改造变体配置 (arch/configs/)

> data/model/arch/configs/ 目录存放 Phase 4.2 架构改造变体的 JSON 配置。
> 详细规划见 [guidance/Phase4.2架构改造规划.md](../../../guidance/Phase4.2架构改造规划.md)。
> Phase 6 消融对照见 [data/model/ablation/summary.md](../../ablation/summary.md)。

## 本批变体 (本批实施: B1 / B2 / B3 / B4)

`feature/phase4.2-arch-bypass-cell-crossattn` 分支已实施 plan §4 完整规定的
`cell_bypass_mode` (none/residual/cat) + `learnable_alpha` 两参数,覆盖 B1/B2/B3。
B4 dual_cell 于 2026-08-01 按 plan §5.4 + §10.4 补充实施 (gdgn_model.dual_cell +
dpredictor.extra_cell_dim,见下方 B4 行)。B5 / B6 仍暂缓。

| 文件 | ID | 路线 | `cell_bypass_mode` | `dual_cell` | `n_query_tokens` | `predictor_hidden` | head_params 预估 | 备注 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---:|---|
| `B1_bypass_residual.json` | B1 | L1 主推荐 | `residual` | false | 1 | 256 | ~510K | bypass 主路径 + cross-attn 残差调制, 与 A4 做 head-to-head |
| `B2_bypass_arch.json` | B2 | L1 + 扩容 | `residual` | false | 4 | 512 | ~1.46M | 在 B1 已稳基础上扩容 (n_q=4 + hidden=512), 与 A5 arch 对照 |
| `B3_bypass_cat.json` | B3 | L2 主推荐 | `cat` | false | 1 | 256 | ~576K | 双通道显式并行 cell_repr=cat([bypass,attended]), 不需 modulation 强度调参 |
| `B4_dual_encoder.json` | B4 | L3 兜底 | `residual` | **true** | 1 | 256 | ~577K (+ flatten enc ~17.5M) | GDGN 图路径 ‖ baseline 原版 SimpleCellEncoder flatten 路径 (drug-agnostic), cell=512, fusion=704 |

**共同点**(承自 A4 no_pretrain,基于 Phase 6 已证"Phase 2 预训练有害"):
- `pretrain_ckpt=null`(全新 init,不加载 Phase 2 best_encoder.pt)
- `lr_encoder=1e-4`(从零训需大 lr,与 A4 一致)
- `use_main_drug_emb=false`(clean 对照,变体不混 main_drug 信号)
- `batch_size=64`(4 卡 DDP,per-gpu=16)
- `freeze_encoder=false`, `seed=42`, `max_epochs=50`, `early_stopping_patience=10`
- `grad_clip=1.0`, `weight_decay=0.0`, `num_heads=4`, `aux_loss_weight=0.0`

## 暂缓变体 (待 B1/B3/B4 结果消化后决定是否实施)

| ID | 路线 | 原计划参数 | 状态 | 原因 |
|---|:---:|---|:---:|---|
| B5 | L1 补充 | B1 + cross-attn dropout=0.5 | 暂缓 | 需暴露 `nn.MultiheadAttention(dropout=...)`;留作"抑制 cross-attn 主导"的补充对照 |
| B6 | L1 低风险 | B1 + gate 由 drug_emb MLP 出,`cell_repr = bypass*gate + attended*(1-gate)` | 暂缓 | gate MLP 设计 plan §5.6 仅作草案 |

> B4 dual_cell 已于 2026-08-01 实施 (按 plan §5.4/§10.4, 因为 B1/B3 判定
> bypass 思路无效, §8.4 触发模式 "B1 ≈ B3 < A4 + 0.02 → 转向 B4")。
> 实施内容: `gdgn_model.py` `dual_cell` 参数 (内嵌 baseline 原版
> `SimpleCellEncoder`, ~17.5M) + `dpredictor.py` `extra_cell_dim` (fusion 追加
> flatten 通道) + `train_gdgn.py` `--dual_cell` override + `B4_dual_encoder.json`。
>
> ⚠️ **参数量修正**: plan §4.4 预估 flatten ~2.3M 有误, 原样复制
> `SimpleCellEncoder` 实际 ~17.5M (首层 Linear(33834, 512) = 17.3M),
> B4 总参数 ~19.3M。训练瓶颈仍在 GDGN per-cell GNN, 耗时与 B1 相当。

## 执行流程 (每个变体三步)

> 4 卡 DDP, batch_size=64 → per_gpu_batch=16(与 Phase 6 ablation 一致, 不易 OOM)

### Step 1 — 训练 (~10-15h / 变体)

```bash
# Linux 服务器 4 卡 DDP,以 B1 为例
v=B1_bypass_residual
torchrun --nproc_per_node=4 program/model/train_gdgn.py \
    --model gdgn --config data/model/arch/configs/${v}.json
```

```powershell
# Windows 本机单卡 (不支持 NCCL DDP; 单卡 batch_size=16, ~40h/变体)
.venv\Scripts\python.exe program/model/train_gdgn.py `
    --model gdgn --config data/model/arch/configs/B1_bypass_residual.json `
    --batch_size 16
```

### Step 2 — Paired split 评估 (post_eval, ~2 min)

```bash
uv run python program/model/train_gdgn.py --model gdgn --post_eval \
    --ckpt data/model/arch/B1_bypass_residual/best_model.pt \
    --output_dir data/model/arch/B1_bypass_residual
```

### Step 3 — 泛化测试 LODO/LOCO (~5-10 min)

```bash
uv run python program/model/eval_generalization.py \
    --model gdgn --ckpt data/model/arch/B1_bypass_residual/best_model.pt \
    --output_dir data/model/arch/B1_bypass_residual --batch_size 32
```

## 批量执行脚本 (Linux 服务器, 4 卡 DDP)

```bash
# arch_run.sh
#!/bin/bash
set -e
# 必跑(诊断核心): B1 + B3
VARIANTS_MUST=("B1_bypass_residual" "B3_bypass_cat")
for v in "${VARIANTS_MUST[@]}"; do
    echo "========================================"
    echo "[$(date)] START $v"
    echo "========================================"
    torchrun --nproc_per_node=4 program/model/train_gdgn.py \
        --model gdgn --config data/model/arch/configs/${v}.json
    uv run python program/model/train_gdgn.py --model gdgn --post_eval \
        --ckpt data/model/arch/${v}/best_model.pt \
        --output_dir data/model/arch/${v}
    uv run python program/model/eval_generalization.py \
        --model gdgn --ckpt data/model/arch/${v}/best_model.pt \
        --output_dir data/model/arch/${v} --batch_size 32
    echo "[$(date)] DONE $v"
done

# 推荐跑(B1/B3 >= A4 + 0.05 后再跑): B2
echo "[$(date)] START B2_bypass_arch"
torchrun --nproc_per_node=4 program/model/train_gdgn.py \
    --model gdgn --config data/model/arch/configs/B2_bypass_arch.json
uv run python program/model/train_gdgn.py --model gdgn --post_eval \
    --ckpt data/model/arch/B2_bypass_arch/best_model.pt \
    --output_dir data/model/arch/B2_bypass_arch
uv run python program/model/eval_generalization.py \
    --model gdgn --ckpt data/model/arch/B2_bypass_arch/best_model.pt \
    --output_dir data/model/arch/B2_bypass_arch --batch_size 32
echo "[$(date)] DONE B2_bypass_arch"

# B4 dual encoder (B1/B3 判定 bypass 思路无效后按 §8.4 触发; L3 兜底):
# 训练: ~12-15h; post_eval + LODO/LOCO 同三步流程
echo "[$(date)] START B4_dual_encoder"
torchrun --nproc_per_node=4 program/model/train_gdgn.py \
    --model gdgn --config data/model/arch/configs/B4_dual_encoder.json
uv run python program/model/train_gdgn.py --model gdgn --post_eval \
    --ckpt data/model/arch/B4_dual_encoder/best_model.pt \
    --output_dir data/model/arch/B4_dual_encoder
uv run python program/model/eval_generalization.py \
    --model gdgn --ckpt data/model/arch/B4_dual_encoder/best_model.pt \
    --output_dir data/model/arch/B4_dual_encoder --batch_size 32
echo "[$(date)] DONE B4_dual_encoder"

echo "Phase 4.2 arch variants (B1+B3+B2+B4) finished."
# nohup bash arch_run.sh > nohup.arch.log 2>&1 &
```

## 时间估算 (4 卡 DDP, plan §7.1)

| 变体 | 训练 (~10-15h) | post_eval (~2min) | generalization (~10min) | 合计 | 优先级 |
|---|---|---:|---:|---:|:---:|
| B1 bypass_residual | ~12h | ~2min | ~10min | ~12h | **必跑(诊断核心)** |
| B3 bypass_cat | ~12h | ~2min | ~10min | ~12h | **必跑(诊断核心)** |
| B2 bypass_arch(参数 +40%) | ~14h | ~2min | ~10min | ~14h | 推荐(已跑 B1/B3 后判定未触发) |
| B4 dual_encoder(参数 ~19.3M) | ~15h | ~2min | ~10min | ~15h | **B1/B3 未达标后触发, L3 兜底** |

**必跑 2 个** ~24h ≈ 1 天即可判定 B1/B3 哪条路线更有潜力。
**+ B2 推荐** ~38h ≈ 1.5 天验证扩容收益。

## 评估策略与判定标准 (plan §8.1-§8.4)

### 三个口径同时看(与 Phase 6 一致)

| 口径 | 衡量 | 来自 |
|---|---|---|
| **paired test PCC** | overall fit | `final_eval_report.txt` Table 1 |
| **LODO per-fold mean PCC** | drug-side 泛化(关键) | `generalization_report.txt` LODO summary |
| **LOCO per-fold mean PCC** | cell-side 泛化 | `generalization_report.txt` LOCO summary |

### 判定 Benchmarks (与 A4 baseline 0.6943 / Baseline 0.8154 对照)

| 标准 | 含义 |
|---|---|
| LODO mean PCC > 0.80 | 追平 baseline(0.8154)——成功 |
| LODO mean PCC > A4 + 0.05 = 0.7443 | 显著改进——继续此路线 |
| LODO mean PCC < A4 + 0.02 = 0.7143 | 改动无效——放弃路线 |
| LODO mean PCC < A4 − 0.02 = 0.6743 | 改动有害——回退 |
| paired test PCC > 0.93 | 不以 paired 换 LODO,overall fit 不退化 |
| LOCO mean PCC > 0.93 | LOCO 不能因改造而退化(>= A4 0.9274) |

### 可能的结论模式 (plan §8.4)

| 模式 | 含义 | 后续 |
|---|---|---|
| B1 > A4 + 0.05 | bypass + residual 路线有效 | 选 B1 作新基线,继续优化 bypass 投影 |
| B3 > B1 | cat 双通更好 | B3 成新 baseline,继续 cat 路线扩展 |
| B1 ≈ B3 < A4 + 0.02 | bypass 思路无效 | **已触发 → B4 dual encoder 已实施并跑完 (2026-08-01)** |
| B2 > B1 + 0.03 | 扩容有效 | B2 作新基线 |
| B2 ≈ B1 | 扩容无效,bypass 是瓶颈主因 | 优化 bypass 设计(pool 策略 + 投影) |
| 所有 B1-B3 LODO < 0.70 | 架构改造失败 | 换思路:DrugEncoder 扩容 + cross-attn 改 K/V 角色互换(gene 当 query 去查 drug),作 Phase 8 候选 |

### B4 判定锚 (plan §5.4 / §8.4) — 已回填

| 锚 | 实测 (2026-08-01) | 判定 |
|---|---|---|
| LODO mean PCC > 0.80 | B4 = **0.7784** | ✗ 未追平 baseline (差 0.037) |
| B4 LOCO mean < 0.92 | B4 = **0.9484** | ✅ LOCO 未退化 |
| B4 LODO < 0.70 | 0.7784 > 0.70 | ✅ 未判死,暂不转 Phase 8 |

> **B4 总结**: test PCC 0.9354 (≥0.93 达标)、LODO 0.7784 (GDGN 历史最佳,
> +0.084 vs A4)、LOCO 0.9484、7 无 DTI pooled 0.9134。flatten 主通道恢复
> overall fit 但 drug OOD 仍差 baseline 0.037。详见 summary.md §3.1 下一步选项。

## 跨变体对比矩阵 (summary.md 模板)

跑完后按 [Phase4.2架构改造规划.md] §8.3 模板回填 `data/model/arch/summary.md`:
含 Baseline / A0 / A1 / A4 / A5 / B1 / B2 / B3 / B4(-) / B5(-) / B6(-) 各列。