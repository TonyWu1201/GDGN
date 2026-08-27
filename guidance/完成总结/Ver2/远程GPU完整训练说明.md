# GDGN Ver2 远程 GPU 完整训练说明

## 1. 文档用途

本文档用于在远程 Linux GPU 主机上复现 Ver2 的数据准备、折专属预训练、筛选实验、5 折 × 3 种子确认实验、模块消融、统计汇总和可解释性干预。

本地已完成代码与 CPU smoke test，没有运行可用于论文结论的完整训练。正式结果必须由本文档中的远程流程产生。

## 2. 固定版本

- Git 分支：`feature/ver2-strict-evaluation-model-improvement`
- 核心实现提交：`b735155`
- 固定 split 与数据审计提交：`e205200`
- Python 和依赖管理：`uv`
- 确认种子：`42`、`3407`、`8128`
- 确认折数：5

远程主机先检查：

```bash
git switch feature/ver2-strict-evaluation-model-improvement
git log --oneline -3
uv sync --all-groups
uv lock --check
```

不要在远程机上使用 `pip` 直接改变环境。

## 3. GPU 和磁盘建议

- 建议 4 块显存不低于 24 GB 的 NVIDIA GPU，每块 GPU 同时只运行一个 GDGN/DeepCDR 任务。
- M0–M4 可根据显存情况提高并行度；首次仍建议一卡一任务。
- 建议保留至少 150 GB 可用磁盘，用于原始数据、折专属预训练检查点、正式预测 parquet 和消融实验。
- 若 OOM，优先降低实验 YAML 中的 `batch_size`，不要改 split、seed 或 DTI 可见性规则。

## 4. 原始数据前置检查

`scripts/prepare_ver2_data.sh` 会重建正式未标准化特征。执行前至少检查下列文件：

| 类别 | 预期路径 |
| --- | --- |
| 表达 | `data/raw/cell_line_omics/DepMap_ExpressionTPMLogp1HumanProteinCodingGenes.csv` |
| 突变 | `data/raw/cell_line_omics/DepMap_SomaticMutations.csv` |
| CNV | `data/raw/cell_line_omics/DepMap_CNGeneWGS.csv` |
| 甲基化 | `data/raw/cell_line_omics/CCLE_DNA_methylation_TSS1kb.txt` |
| 细胞系注释 | `data/raw/cell_line_omics/Cell_lines_annotations_20181226.txt` |
| 药物结构 | `data/raw/drug_structures/compound_cid_smiles.csv` |
| UniProt（只在重建 ESM 时需要） | `data/raw/protein_sequence/uniprot_sprot.dat` |
| PPI | `data/processed/protein_protein_interaction/ppi_dg_filtered.csv` |
| DTI | `data/processed/drug_gene_interaction/interactions_filtered.csv` |
| 药物响应 | `data/processed/drug_sensitivity/ic50_matrix.csv` |

本地 data-card 中的两个 strict readiness 标志当前为 `False`，因此远程正式训练前必须重建：

```bash
bash scripts/prepare_ver2_data.sh
grep -E "Strict unscaled (cell|drug) artifact ready" data/model/data-card.md
```

预期两行均为 `True`。如果任一为 `False`，正式 runner 会主动拒绝启动；不要用 `--smoke` 绕过。

如需重建 ESM-MEAN：

```bash
REBUILD_ESM_MEAN=1 bash scripts/prepare_ver2_data.sh
```

新 ESM-MEAN 不包含 BOS/EOS 特殊 token。长序列的分块方案见第 8 节。

## 5. 训练前验收

```bash
uv run pytest tests/strict_eval -q
uv run python -m compileall -q program tests
bash -n scripts/*.sh
uv run python program/smoke_strict_pipeline.py
uv run python program/model/edge_mask.py
```

还应确认：

- `data/model/splits/*/manifest.json` 中的 `git_commit` 和数据校验值存在。
- 六类 `fold-00` 至 `fold-04` 均存在，`audit.passed=true`。
- LCO/LDO/LTO/DB 中 train、validation、test 的受控实体两两无交集。
- `edge_split_strict.pt` 的 validation/test 正边不出现在消息传递图。

## 6. 折专属图预训练

这一步是 Ver2 与旧实验的关键差异。LDO-SO 和 DB 不得复用全 DTI 图预训练检查点。折专属预训练仅保留响应训练折药物的 DTI，仅使用响应训练折细胞的组学，并复用同一折 scaler。

筛选阶段只生成 fold-00/seed-0042：

```bash
RUN_FOLD_GRAPH_PRETRAIN=1 \
PRETRAIN_FOLDS="0" \
PRETRAIN_SEEDS="42" \
bash scripts/run_ver2_pretraining.sh
```

确认阶段生成全部 5 折 × 3 种子：

```bash
RUN_FOLD_GRAPH_PRETRAIN=1 bash scripts/run_ver2_pretraining.sh
```

若需要对每个预训练检查点额外运行独立 test-edge 验证：

```bash
RUN_FOLD_GRAPH_PRETRAIN=1 \
VERIFY_FOLD_PRETRAIN=1 \
bash scripts/run_ver2_pretraining.sh
```

这会明显增加运行时间。全图预训练只用于历史链路预测诊断，不能用作 LDO-SO/DB 下游初始化：

```bash
RUN_GLOBAL_GRAPH_PRETRAIN=1 bash scripts/run_ver2_pretraining.sh
```

## 7. 单折单种子筛选

确保第 6 节的 fold-00/seed-0042 折专属检查点存在，然后执行：

```bash
bash scripts/run_screening.sh
```

`screening.yaml` 共 102 个运行：6 个协议 × 17 个基线/模型 × 1 折 × 1 种子。该阶段只用于淘汰明显无效方案，不进行论文主张。

结果位于：

```text
data/model/experiments/<experiment-id>/<protocol--model--variant>/fold-00/seed-0042/
```

每个运行至少应有 `config.json`、`status.json`、`checkpoint-best.pt`、`metrics.json`、`predictions.parquet` 和 `run.log`。

## 8. 可选 PaRTI 路线

PaRTI 不是默认必跑模块。只有 ESM-MEAN 与 ESM-ID 对照值得继续时才执行：

```bash
bash scripts/prepare_parti.sh
```

该流程对长蛋白分块，排除特殊 token，并用核心基因的通路成员多标签目标训练 chunk attention pooler。产物为 `esm2_parti_gene_embeddings.pt`。

## 9. 5 折 × 3 种子确认实验

下列命令中的最后一个参数是并行 GPU 数。脚本为每个折/种子启动独立进程，不在一个模型内做 DDP。

严格 benchmark：

```bash
bash scripts/run_confirmation.sh configs/sweeps/benchmark.yaml 4
```

M0–M4 逐级模型：

```bash
bash scripts/run_confirmation.sh configs/sweeps/model-pathway-residual.yaml 4
```

模块消融：

```bash
bash scripts/run_confirmation.sh configs/sweeps/abl-pretrain.yaml 4
bash scripts/run_confirmation.sh configs/sweeps/abl-graph.yaml 4
bash scripts/run_confirmation.sh configs/sweeps/abl-esm.yaml 4
bash scripts/run_confirmation.sh configs/sweeps/abl-modality.yaml 4
```

运行顺序必须遵守停止规则：

1. M0 未超过强简单基线时，先诊断数据与协议。
2. M2 未超过 M1 且真实图未超过 graph-none/degree/random 时，停止加深 PPI。
3. ESM-MEAN 未超过 ESM-ID 时，不主张序列语义贡献，不强制执行 PaRTI。
4. 预训练二乘二无稳定收益时，不把边重建作为主贡献。

## 10. 任务对齐的掩码组学预训练

这是阶段 5 的条件性实验，只在 M0–M4 还有明确表征瓶颈时执行。预训练同样按协议/折/种子隔离：

```bash
RUN_TASK_ALIGNED=1 \
PRETRAIN_PROTOCOLS="eval-lco eval-ldo-kt eval-ldo-so eval-db" \
bash scripts/run_ver2_pretraining.sh

bash scripts/run_confirmation.sh configs/sweeps/pretrain-task-aligned.yaml 4
```

对照包括随机初始化、冻结、全量微调和分阶段解冻。

## 11. 统计汇总

每个 sweep 脚本结束时会自动汇总。论文级汇总建议重跑 2,000 次层级 bootstrap：

```bash
uv run python program/aggregate_strict_results.py \
  --root data/model/experiments \
  --output-dir data/model/experiments \
  --hierarchical-bootstrap-runs 2000
```

产物：

- `benchmark-summary.csv`：折/种子均值、标准差和置信区间。
- `benchmark-report.md`：主表。
- `hierarchical-bootstrap.csv`：按药物和细胞系整群重采样的区间。
- `paired-comparisons.csv`：同折同种子配对差值，包括对双向加性基线和预指定消融参照的比较。

指标同时包括全局 PCC、Spearman、RMSE、MAE、R²，按药物/细胞系宏平均，以及低/中/高相似度分层。

## 12. 严格可解释性

只对通过严格性能门槛的 M0–M4 检查点执行。示例：

```bash
bash scripts/run_interpretation.sh \
  data/model/interpret-faithfulness \
  data/model/experiments/model-pathway-residual/eval-ldo-so--model-m4/fold-00/seed-0042 \
  data/model/experiments/model-pathway-residual/eval-ldo-so--model-m4/fold-00/seed-3407 \
  data/model/experiments/model-pathway-residual/eval-ldo-so--model-m4/fold-00/seed-8128
```

该流程会覆盖所有样本数足够的测试药物，各取敏感/耐药细胞系，并输出：

- 多 baseline IG 和 convergence delta。
- 保留正负归因。
- 删除曲线、comprehensiveness、sufficiency。
- 随机基因和表达/PPI 度匹配基因对照。
- 门控打乱。
- 已知靶点 AUPRC/NDCG/排名百分位。
- 靶点、PPI 一跳/二跳与通路富集，BH-FDR 校正。
- 跨种子 top-20 Jaccard 稳定性。

表达基因被干预时，程序会从原始表达重新计算 ssGSEA，再应用当前折 pathway scaler。若 top 删除未超过随机/匹配对照，或门控打乱几乎不改变预测，只能称为内部权重可视化。

## 13. 外部数据接口

仅在内部严格评估和消融通过后执行。数据标准化示例：

```bash
uv run python program/prepare_external_data.py \
  --dataset gdsc1 --input /path/to/gdsc1.csv \
  --output-dir data/model/external/gdsc1
```

支持的 schema 为 `gdsc1`、`ctrp`、`prism`、`pdx` 和 `tcga`。脚本不会默认把 AUC、LN(IC50)、PDX response 或临床结局当成同一终点，实验前仍需药物/基因/组织 crosswalk 审核。

## 14. 断点恢复与失败保留

- 已完成且配置一致的运行会直接跳过。
- 同一运行键已完成但配置不同时，runner 会拒绝覆盖。
- 失败或中断运行的原配置、状态和部分产物会移入 `attempts/attempt-XX/`，然后重跑。
- `status.json` 中的 `state` 为 `complete` 才会被汇总。

## 15. 正式结果验收清单

1. 六协议全部完成固定 5 折。
2. 确认实验每折有 3 个种子，没有用 smoke 输出补数。
3. LDO-KT 与 LDO-SO 分开汇报。
4. 每个 scaler 记录的 `train_cell_idx/train_drug_idx` 与 split 一致。
5. 每个 GDGN 预训练检查点路径包含对应协议/折/种子。
6. 汇总表同时有原始折结果、均值、标准差、95% CI 和配对差值。
7. 只有达到预注册停止规则的模块才进入最终模型。
8. 可解释性必须同时通过生物富集、干预忠实性和跨种子稳定性。
