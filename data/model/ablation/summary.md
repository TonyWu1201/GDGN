# Phase 6 消融实验跨变体对比 summary

> **数据来源**:`data/model/baseline/` + `data/model/ablation/{01_main,04_no_pretrain,05_arch}/` 的
> `final_eval_report.txt` + `generalization_report.txt`(Phase 6 已跑结果)
> **对照文档**:[Phase4.2架构改造规划.md](../../guidance/Phase4.2架构改造规划.md) §1.1 / §1.2 / §8.3、
> [Phase6消融实验规划.md](../../guidance/Phase6消融实验规划.md) §8.4
> **状态**:Phase 6 已跑 A0 / A1 / A4 / A5;A2 / A3 / A6 未跑(§10.4 已知,诊断已完成)
> **回填日期**:2026-07-31

---

## 1. 主口径对比(plan §8.3 模板)

三个口径同时看:**paired test PCC**(overall fit)+ **LODO per-fold mean PCC**(drug-side 泛化,关键)
+ **LOCO per-fold mean PCC**(cell-side 泛化)。

| ID | val PCC | test PCC | LODO mean | LODO median | LOCO mean | LODO std | 参数量 | 配置要点 | 结论 |
|:--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| Baseline | 0.9387 | **0.9378** | **0.8154** | **0.8200** | **0.9584** | **0.0688** | baseline_simple | DeepCDR-molecularGCN + flatten MLP cell | (Phase 4 已测, target) |
| A0 | 0.9083 | 0.9055 | 0.6505 | 0.6675 | 0.9124 | 0.1213 | 1.77M | Phase 4 原配置(pretrain + lr=1e-5 + use_main_drug_emb=False) | (Phase 4 已训, 对照) |
| A1 | 0.9191 | 0.9162 | 0.6775 | 0.6922 | 0.9227 | 0.1104 | 1.83M | lr_encoder=1e-4 + use_main_drug_emb=True | (Phase 6 已测, best 旧) |
| A4 | 0.9225 | 0.9191 | **0.6943** | **0.7143** | **0.9274** | **0.0996** | 1.77M | pretrain_ckpt=null + lr_encoder=1e-4 | (Phase 6 已测, best 旧, A4>A1) |
| A5 | 0.9165 | 0.9131 | 0.6577 | 0.6773 | 0.9194 | 0.1186 | 2.47M | n_query_tokens=4 + predictor_hidden=512 | (Phase 6, 过参数化加剧 OOD) |

> 注:`LODO std` 取 per-fold PCC 的标准差,衡量跨药泛化的稳定性;越小越稳。

---

## 2. 辅助诊断指标(plan §1.1)

| 口径 | Baseline | A0 | A1 main | A4 no_pretrain | A5 arch |
|---|---:|---:|---:|---:|---:|
| Paired test PCC | **0.9378** | 0.9055 | 0.9162 | **0.9191** | 0.9131 |
| **LODO mean PCC** | **0.8154** | 0.6505 | 0.6775 | **0.6943** | 0.6577 |
| LODO median PCC | **0.8200** | 0.6675 | 0.6922 | **0.7143** | 0.6773 |
| LODO std | 0.0688 | 0.1213 | 0.1104 | **0.0996** | 0.1186 |
| LOCO mean PCC | **0.9584** | 0.9124 | 0.9227 | **0.9274** | 0.9194 |
| 7 无 DTI pooled PCC | **0.9291** | 0.8612 | 0.8838 | 0.8833 | 0.8749 |
| best epoch | 48 | 46 | 48 | 48 | 42 |
| train time | 1935s | 145388s | 38500s | 38018s | 38229s |
| lr_encoder 收敛到 | 1e-5 | 5e-6 | 5e-5 | 5e-5 | 2.5e-5 |

数据来源:
- `data/model/baseline/{final_eval_report.txt,generalization_report.txt}`
- `data/model/gdgn/{final_eval_report.txt,generalization_report.txt}`(A0 → Phase 4 产物)
- `data/model/ablation/{01_main,04_no_pretrain,05_arch}/{final_eval_report.txt,generalization_report.txt}`

---

## 3. 触发的结论模式(plan §1.2,参照 Phase6消融实验规划.md §8.4)

| 实测 | 触发模式 | 含义 |
|---|---|---|
| A4(no_pretrain) > A1(+0.017 LODO) | A1 ≈ A4 / A4 > A1 | **Phase 2 预训练不仅没帮助,反而引入有害 init bias**;PPI/DTI 重构任务与 IC50 严重不对齐 |
| A5(arch) < A1,接近 A0 | A5 ≈ A1 / A5 < A1 | 扩 cross-attn query + 加深 MLP 没解决瓶颈,**反而过参数化更易过拟合 OOD** |
| 所有变体 LODO < 0.75,最高 A4 = 0.6943,距 baseline 0.8154 差 0.121 | "所有变体 LODO < 0.75" | **结构性瓶颈**,重设计 DrugEncoder 或 GAT / cross-attn 主导路径 |
| lr 收敛 + A4/A5 train time 1/20 baseline 仍更高 | 非欠训练 | 不是训练问题,是路径设计问题 |

---

## 4. 结论(转入 Phase 4.2 架构改造)

不是模型没训好,是 **cross-attn drug-query 作主路径**这一架构决策在 LODO 下结构性失败:
继续扫架构超参(heads8 / freeze_main)已无诊断价值,OOD 漂移主因已定位,转入 Phase 4.2 架构改造。

详见 [Phase4.2架构改造规划.md](../../guidance/Phase4.2架构改造规划.md) §2 根因分析 + §3 改造方案路线。

**关键诊断**(plan §2):
1. baseline 强在"cell-side 不感知 drug"(`baseline_simple.py:40-66`),cell_emb 完全 drug-agnostic,
   留出新 drug 时 cell_emb 不变,drug 信号仅占 fusion 128/448 ≈ 28.6% 维度,OOD 不放大。
2. GDGN 崩在"cross-attn 让 drug 调制整个 cell-side 表征"(`dpredictor.py:115-167`):
   cross-attn 把 8412 gene 表征按 drug query 加权塌缩成 256 维,**整个 cell-side 256 维被 drug 调制**,
   留出 drug 时 attention 权重 OOD → attended_genes 表征漂移 → predictor 头输入分布漂移放大。
3. A4 > A1 指向 Phase 2 预训练任务重设计(§10.3):放弃 PPI/DTI 重构,改"drug→gene 设靶 co-occurrence"
   或"组学扰动恢复"自监督;**触发条件**:仅当 Phase 4.2 B1/B3 成功(LODO > 0.80)再重设计 Phase 2。

---

## 5. A2 / A3 / A6 未跑说明(plan §10.4)

| ID | 原计划 | 状态 |
|---|---|---|
| A2 | `02_lr_only.json` — 仅 `lr_encoder=1e-4`(隔离 use_main_drug_emb 影响) | **未跑** |
| A3 | `03_freeze_main.json` — `freeze_encoder=True` + `use_main_drug_emb=True` | **未跑** |
| A6 | `06_heads8.json` — `num_heads=8`(可选) | **未跑** |

诊断已完成,不跑也能下结论。若 Phase 4.2 B1-B6 全部失败,可回头跑 A2 单纯 lr 变量作兜底验证。

---

## 6. 下一步衔接

Phase 4.2 架构改造已开新分支 `feature/phase4.2-arch-bypass-cell-crossattn`,变体跨对比表见
`data/model/arch/summary.md`(B1 / B2 / B3 跑完后回填)。