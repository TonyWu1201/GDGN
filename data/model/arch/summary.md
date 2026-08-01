# Phase 4.2 架构改造跨变体对比 summary

> **实施分支**:`feature/phase4.2-arch-bypass-cell-crossattn`
> **本批范围**:B1 bypass_residual / B2 bypass_arch / B3 bypass_cat(B4/B5/B6 暂缓,plan §10.4)
> **对照文档**:[Phase4.2架构改造规划.md](../../guidance/Phase4.2架构改造规划.md) §1.1 / §8.3、
>   [data/model/ablation/summary.md](../ablation/summary.md)(Phase 6 已测结果)
> **状态**:B1 / B3 已完成并回填;B2 未跑(§8.4 判定 bypass 路线未达 bar,扩容无诊断价值);B4/B5/B6 暂缓。
> **回填日期**:2026-08-01(B1 / B3 实测回填)

---

## 1. 主口径对比 (plan §8.3 模板)

三个口径同时看:**paired test PCC**(overall fit)+ **LODO per-fold mean PCC**(drug-side 泛化,关键)
+ **LOCO per-fold mean PCC**(cell-side 泛化)。

| ID | val PCC | test PCC | LODO mean | LODO median | LOCO mean | LODO std | 参数量 | 配置要点 | 结论 |
|:--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| **Baseline** | 0.9387 | **0.9378** | **0.8154** | **0.8200** | **0.9584** | **0.0688** | baseline_simple | DeepCDR-molecularGCN + flatten MLP cell | (Phase 4 已测, target) |
| A0 | 0.9083 | 0.9055 | 0.6505 | 0.6675 | 0.9124 | 0.1213 | 1.77M | Phase 4 原配置(pretrain + lr=1e-5) | (Phase 6 已测, 对照) |
| A1 | 0.9191 | 0.9162 | 0.6775 | 0.6922 | 0.9227 | 0.1104 | 1.83M | lr=1e-4 + use_main_drug_emb=True | (Phase 6 已测, best 旧) |
| A4 | 0.9225 | 0.9191 | 0.6943 | 0.7143 | 0.9274 | 0.0996 | 1.77M | pretrain_ckpt=null + lr=1e-4 | (Phase 6 已测, **A4>A1**) |
| A5 | 0.9165 | 0.9131 | 0.6577 | 0.6773 | 0.9194 | 0.1186 | 2.47M | n_q=4 + hidden=512 | (Phase 6, 过参数化加剧 OOD) |
| **B1** | 0.9262 | 0.9232 | 0.7111 | 0.7282 | 0.9323 | 0.0889 | head ~510K (总 ~1.84M) | residual + n_q=1 + learnable_alpha=True + pretrain=null | LODO +0.017 vs A4, 未达 A4+0.02 bar; LOCO/7无DTI 微升 |
| **B2** | – | – | – | – | – | – | head ~1.46M (总 ~2.79M) | residual + n_q=4 + hidden=512 + pretrain=null | **未跑**(bypass 路线未达标, 扩容无优先) |
| **B3** | 0.9212 | 0.9175 | 0.6825 | 0.7031 | 0.9257 | 0.1083 | head ~576K (总 ~1.91M) | cat + n_q=1 + pretrain=null | LODO −0.012 vs A4, cat 显式双通反而更差 |
| B4 | – | – | – | – | – | – | head ~510K + dual enc ~2.3M | dual_cell(暂缓) | **暂缓**(plan §10.4, 待 B1/B3 决定) |
| B5 | – | – | – | – | – | – | head ~510K | B1 + cross-attn dropout=0.5(暂缓) | **暂缓** |
| B6 | – | – | – | – | – | – | head ~535K | bypass_gate(暂缓) | **暂缓** |

> **基准锚**(plan §8.2):
> - LODO baseline 0.8154 / A4 best 0.6943 → **目标 LODO ≥ 0.74** = 显著改进 = 继续(此路线)
> - paired test PCC ≥ 0.93 = overall fit 不退化
> - LOCO mean PCC ≥ 0.93 = LOCO 不退化(>= A4 0.9274)

---

## 2. 本批 B 变体设计要点对照

| ID | 路线 | bypass_mode | n_q | hidden | 关键设计 | plan §3 / §4.4 预期 |
|:--:|:---:|:---:|:---:|:---:|---|---|
| B1 | L1 | residual | 1 | 256 | bypass(gene mean pool)主 + cross-attn 残差调制, learnable_alpha 从 1.0 起步 | LODO 0.75-0.78, BP > A4 + 0.05 |
| B2 | L1+扩容 | residual | 4 | 512 | B1 基础扩容 (n_q=4 + MLP 加宽); bypass_proj 输出 = 1024, attended_genes = 1024 对齐残差和 | LODO 0.76-0.80, paired PCC > 0.93 |
| B3 | L2 | cat | 1 | 256 | 显式双通 cell_repr=cat([bypass 256, attended 256])=512, fusion=704 | LODO 0.78-0.82, 更接近 baseline |

详细设计见 [Phase4.2架构改造规划.md](../../guidance/Phase4.2架构改造规划.md) §4.1 §4.4 §5.1 §5.2 §5.3。

---

## 3. 判定 Benchmarks 回填

按 [Phase4.2架构改造规划.md](../../guidance/Phase4.2架构改造规划.md) §8.4 可能模式判定:

| 模式 | 实测(本批跑完填) | 含义 | 后续 |
|---|---|---|---|
| B1 > A4 + 0.05 | B1 LODO = **0.7111** < 0.7443 ✗ | bypass + residual 路线有效 | 不触发 |
| B3 > B1 | B3 0.6825 < B1 0.7111 ✗ | cat 双通更好 | 不触发 |
| B1 ≈ B3 < A4 + 0.02 | B1 = A4+0.017, B3 = A4−0.012 ✅ | **bypass 思路(mean-pool 主路径)无效** | **触发 → 实施 B4 dual encoder**(复制 baseline flatten 路径) |
| B2 > B1 + 0.03 | B2 未跑 | 扩容有效 | 不跑(B1 未达 bar, 扩容无诊断价值) |
| B2 ≈ B1 | B2 未跑 | 扩容无效, bypass 是瓶颈主因 | 不跑 |
| 所有 B1-B3 LODO < 0.70 | max(B1,B2,B3) LODO = **0.7111** > 0.70 | 架构改造失败 | 未触发; 若 B4 仍 < 0.70 再转 Phase 8(K/V 角色互换) |

**附加观察**(B1 vs B4 起步依据):
- paired test PCC 双双 < 0.93(B1 0.9232 / B3 0.9175):降级 cross-attn 调制也牺牲了已见 drug 的 overall fit → cross-attn 在已见 drug 上仍有真实贡献,**应保留作增强通道**,而非砍掉 → 指向 L3 dual encoder 方案
- B1 LOCO mean 0.9323 > A4 0.9274(+0.005),7 无 DTI LODO pooled 0.8910 > A4 0.8833:bypass 在 cell-side 与弱 drug 侧无害甚至微升,瓶颈确实只在 drug OOD 调制
- B1 LODO std 0.0889 < A4 0.0996:残差结构让 drug 泛化更稳,只是幅度不够

### 3.1 结论与下一步

1. **B1/B3 判定:bypass(mean-pool 主路径)思路无效** — B1 仅 +0.017、B3 −0.012,均未达 A4+0.02 bar,更距 baseline 0.8154 甚远
2. **下一步:实施 B4 dual encoder**(plan §5.4 / §10.4):GDGN 图路径 + `SimpleCellEncoder` flatten 路径并行 cat(总 ~5.0M),直接复制 baseline 的 drug-agnostic cell 优势
3. B4 预期 LODO 0.80-0.85,判定:> 0.80 追平 baseline 成功;< 0.70 放弃 L3,转 Phase 8(K/V 角色互换)

---

## 4. 数据来源 (跑完后核对)

- B1: `data/model/arch/B1_bypass_residual/{final_eval_report.txt,generalization_report.txt}`
- B2: `data/model/arch/B2_bypass_arch/{final_eval_report.txt,generalization_report.txt}`
- B3: `data/model/arch/B3_bypass_cat/{final_eval_report.txt,generalization_report.txt}`

Phase 6 对照:`data/model/ablation/summary.md`(已回填 A0/A1/A4/A5/Baseline + 辅助指标)。