#!/usr/bin/env bash
# =============================================================================
# GDGN Ver2 远程 GPU 无人值守训练脚本
# 依据 guidance/完成总结/Ver2/远程GPU完整训练说明.md 编写
#
# 用法:
#   bash scripts/run_remote_unattended.sh [--force]
#
# 常用环境变量（默认值见下方 CONFIG 段）:
#   PARALLEL_GPUS=4        确认实验并行 GPU 数（0=自动检测，上限 4）
#   RUN_SCREENING=0        跳过单折单种子筛选（默认已由本地前置完成，远程通常不需要）
#   RUN_PARTI=1            执行可选 PaRTI 路线
#   RUN_TASK_ALIGNED=1     执行任务对齐掩码组学预训练（阶段 5 条件性实验）
#   RUN_INTERPRETATION=0   跳过严格可解释性
#   REBUILD_ESM_MEAN=1     重建 ESM-MEAN（需 uniprot_sprot.dat）
#   VERIFY_FOLD_PRETRAIN=1 对每个预训练检查点额外跑 test-edge 验证
#   SKIP_ENV_CHECK=1       跳过环境检查（分支/提交/uv sync/GPU/磁盘）
#   SKIP_DATA=1            跳过数据重建（仅校验 data-card 就绪标志）
#   SKIP_ACCEPTANCE=1      跳过训练前验收
#   SKIP_PRETRAIN=1        跳过折专属图预训练（本地已完成时置 1）
#   SKIP_CONFIRMATION=1    跳过 5 折 x 3 种子确认实验
#   SKIP_AGGREGATE=1       跳过最终统计汇总
#   BOOTSTRAP_RUNS=2000    层级 bootstrap 重采样次数
#   STOP_AFTER_SCREENING=1 筛选结束后停止，人工核对再继续
#   STOP_AFTER_BENCHMARK=1  benchmark 结束后停止，按停止规则核对再继续
#   FORCE=1                忽略状态文件强制重跑全部阶段
#
# 说明:
#   - 全程只使用 uv，不调用 pip。
#   - 各阶段幂等：已完成且配置一致的运行自动跳过，失败运行移入 attempts/。
#   - 阶段完成记录在 data/model/remote-run.state，中断后重跑自动续跑。
#   - 停止规则（文档第 9 节）需人工核对中间结果，脚本只保证执行顺序。
#   - 数据处理与折专属预训练默认在本地完成（scripts/run_local_data_and_pretrain.ps1），
#     远程通常以 SKIP_DATA=1 SKIP_PRETRAIN=1 启动，只跑确认实验/消融/可解释性。
# =============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ---------------- 配置 ----------------
PARALLEL_GPUS="${PARALLEL_GPUS:-0}"
RUN_SCREENING="${RUN_SCREENING:-0}"
RUN_PARTI="${RUN_PARTI:-0}"
RUN_TASK_ALIGNED="${RUN_TASK_ALIGNED:-0}"
RUN_INTERPRETATION="${RUN_INTERPRETATION:-1}"
REBUILD_ESM_MEAN="${REBUILD_ESM_MEAN:-0}"
VERIFY_FOLD_PRETRAIN="${VERIFY_FOLD_PRETRAIN:-0}"
SKIP_ENV_CHECK="${SKIP_ENV_CHECK:-0}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_ACCEPTANCE="${SKIP_ACCEPTANCE:-0}"
SKIP_PRETRAIN="${SKIP_PRETRAIN:-0}"
SKIP_CONFIRMATION="${SKIP_CONFIRMATION:-0}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"
BOOTSTRAP_RUNS="${BOOTSTRAP_RUNS:-2000}"
STOP_AFTER_SCREENING="${STOP_AFTER_SCREENING:-0}"
STOP_AFTER_BENCHMARK="${STOP_AFTER_BENCHMARK:-0}"
FORCE="${FORCE:-0}"

BRANCH="feature/ver2-strict-evaluation-model-improvement"
COMMIT_CORE="b735155"
COMMIT_SPLIT="e205200"
MIN_DISK_GB=150
PROTOCOLS=(eval-lpo eval-lco eval-ldo-kt eval-ldo-so eval-lto eval-db)
SEEDS=(42 3407 8128)
LOG_DIR="data/model/remote-logs"
STATE_FILE="data/model/remote-run.state"
CURRENT_PHASE=""

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

trap 'echo "[FATAL] 阶段 ${CURRENT_PHASE:-?} 失败（行号 $LINENO），日志: $LOG_FILE" >&2; exit 1' ERR

# ---------------- 工具函数 ----------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "[FATAL] $*" >&2; exit 1; }

phase_done() { [[ "$FORCE" == "1" ]] && return 1; grep -q "^$1=done$" "$STATE_FILE" 2>/dev/null; }
mark_done() { echo "$1=done" >> "$STATE_FILE"; }

run_phase() {
  local name="$1" fn="$2"
  if phase_done "$name"; then
    log "跳过已完成阶段: $name（FORCE=1 可强制重跑）"
    return 0
  fi
  CURRENT_PHASE="$name"
  log "==================== 阶段开始: $name ===================="
  "$fn"
  mark_done "$name"
  log "==================== 阶段完成: $name ===================="
}

detect_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l || true
  else
    echo 0
  fi
}

# ---------------- 阶段 1: 环境检查（文档第 2 节） ----------------
phase_env_check() {
  log "当前分支: $(git branch --show-current)"
  if [[ "$(git branch --show-current)" != "$BRANCH" ]]; then
    [[ -z "$(git status --porcelain)" ]] || die "工作区有未提交改动，无法自动切换到 $BRANCH"
    git switch "$BRANCH"
  fi
  git merge-base --is-ancestor "$COMMIT_CORE" HEAD || die "缺少核心实现提交 $COMMIT_CORE"
  git merge-base --is-ancestor "$COMMIT_SPLIT" HEAD || die "缺少 split 提交 $COMMIT_SPLIT"
  git log --oneline -3
  uv sync --all-groups
  uv lock --check

  local n_gpus
  n_gpus=$(detect_gpus)
  if [[ "$PARALLEL_GPUS" == "0" ]]; then
    PARALLEL_GPUS=$(( n_gpus < 4 ? n_gpus : 4 ))
  fi
  if [[ "$PARALLEL_GPUS" -lt 1 ]]; then
    die "未检测到可用 GPU（nvidia-smi 报告 $n_gpus 块）"
  fi
  if [[ "$PARALLEL_GPUS" -gt "$n_gpus" ]]; then
    log "警告: PARALLEL_GPUS=$PARALLEL_GPUS 超过检测到的 $n_gpus 块 GPU，已降级"
    PARALLEL_GPUS="$n_gpus"
  fi
  log "GPU 数: $n_gpus，并行度: $PARALLEL_GPUS"

  local avail_gb
  avail_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  log "可用磁盘: ${avail_gb} GB"
  if [[ "$avail_gb" -lt "$MIN_DISK_GB" ]]; then
    die "可用磁盘不足 ${MIN_DISK_GB} GB（当前 ${avail_gb} GB）"
  fi
}

# ---------------- 阶段 1b: 跳过环境检查时的 GPU 探测 ----------------
phase_gpu_probe() {
  local n_gpus
  n_gpus=$(detect_gpus)
  if [[ "$PARALLEL_GPUS" == "0" ]]; then
    PARALLEL_GPUS=$(( n_gpus < 4 ? n_gpus : 4 ))
  fi
  if [[ "$PARALLEL_GPUS" -lt 1 ]]; then
    die "未检测到可用 GPU（nvidia-smi 报告 $n_gpus 块）"
  fi
  log "GPU 数: $n_gpus，并行度: $PARALLEL_GPUS"
}

# ---------------- 阶段 2: 原始数据前置检查（文档第 4 节） ----------------
phase_data_preflight() {
  local files=(
    data/raw/cell_line_omics/DepMap_ExpressionTPMLogp1HumanProteinCodingGenes.csv
    data/raw/cell_line_omics/DepMap_SomaticMutations.csv
    data/raw/cell_line_omics/DepMap_CNGeneWGS.csv
    data/raw/cell_line_omics/CCLE_DNA_methylation_TSS1kb.txt
    data/raw/cell_line_omics/Cell_lines_annotations_20181226.txt
    data/raw/drug_structures/compound_cid_smiles.csv
    data/processed/protein_protein_interaction/ppi_dg_filtered.csv
    data/processed/drug_gene_interaction/interactions_filtered.csv
    data/processed/drug_sensitivity/ic50_matrix.csv
  )
  if [[ "$REBUILD_ESM_MEAN" == "1" ]]; then
    files+=(data/raw/protein_sequence/uniprot_sprot.dat)
  fi
  local missing=0 f
  for f in "${files[@]}"; do
    if [[ ! -f "$f" ]]; then
      log "缺少原始数据: $f"
      missing=1
    fi
  done
  [[ "$missing" == "0" ]] || die "原始数据不完整，请先按文档第 4 节放置文件"
  log "原始数据前置检查通过"
}

# ---------------- 阶段 3: 数据重建与就绪校验 ----------------
check_data_card() {
  local card="data/model/data-card.md"
  [[ -f "$card" ]] || die "data-card.md 不存在，请先运行数据准备"
  local cell drug
  cell=$(grep -E "Strict unscaled cell artifact ready" "$card" | grep -oE "True|False" || true)
  drug=$(grep -E "Strict unscaled drug artifact ready" "$card" | grep -oE "True|False" || true)
  if [[ "$cell" != "True" || "$drug" != "True" ]]; then
    die "strict readiness 未就绪（cell=$cell drug=$drug），正式 runner 会拒绝启动"
  fi
  log "data-card strict readiness: cell=$cell drug=$drug"
}

phase_data_prepare() {
  if [[ "$SKIP_DATA" == "1" ]]; then
    log "SKIP_DATA=1，跳过数据重建"
    check_data_card
    return 0
  fi
  if [[ "$REBUILD_ESM_MEAN" == "1" ]]; then
    REBUILD_ESM_MEAN=1 bash scripts/prepare_ver2_data.sh
  else
    bash scripts/prepare_ver2_data.sh
  fi
  check_data_card
}

# ---------------- 阶段 4: 训练前验收（文档第 5 节） ----------------
phase_acceptance() {
  uv run pytest tests/strict_eval -q
  uv run python -m compileall -q program tests
  bash -n scripts/*.sh
  uv run python program/smoke_strict_pipeline.py
  uv run python program/model/edge_mask.py

  local p f j
  for p in "${PROTOCOLS[@]}"; do
    for f in 0 1 2 3 4; do
      j="data/model/splits/$p/fold-$(printf '%02d' "$f").json"
      [[ -f "$j" ]] || die "缺少 split 文件: $j"
      grep -q '"passed": true' "$j" || die "split audit 未通过: $j"
    done
  done
  [[ -f data/model/pretrain/edge_split_strict.pt ]] || die "缺少 data/model/pretrain/edge_split_strict.pt"
  log "训练前验收通过（6 协议 x 5 折 audit.passed=true）"
}

# ---------------- 阶段 5: 折专属图预训练（文档第 6 节） ----------------
phase_pretrain() {
  RUN_FOLD_GRAPH_PRETRAIN=1 \
  VERIFY_FOLD_PRETRAIN="$VERIFY_FOLD_PRETRAIN" \
  bash scripts/run_ver2_pretraining.sh

  local p f s ckpt missing=0
  for p in "${PROTOCOLS[@]}"; do
    for f in 0 1 2 3 4; do
      for s in "${SEEDS[@]}"; do
        ckpt="data/model/pretrain/strict/$p/fold-$(printf '%02d' "$f")/seed-$(printf '%04d' "$s")/best_encoder.pt"
        if [[ ! -f "$ckpt" ]]; then
          log "缺少预训练检查点: $ckpt"
          missing=1
        fi
      done
    done
  done
  [[ "$missing" == "0" ]] || die "折专属预训练检查点不完整（应为 90 个）"
  log "折专属预训练检查点完整（90 个）"
}

# ---------------- 阶段 6: 单折单种子筛选（文档第 7 节） ----------------
phase_screening() {
  local p ckpt
  for p in "${PROTOCOLS[@]}"; do
    ckpt="data/model/pretrain/strict/$p/fold-00/seed-0042/best_encoder.pt"
    [[ -f "$ckpt" ]] || die "缺少筛选所需预训练检查点: $ckpt"
  done

  local manifest="data/model/experiments/screening-manifest.jsonl"
  uv run python program/generate_run_manifest.py \
    --sweep configs/sweeps/screening.yaml --output "$manifest"
  local n_runs index gpu
  n_runs=$(wc -l < "$manifest")
  log "筛选: $n_runs 个运行，并行 $PARALLEL_GPUS 块 GPU"
  for ((index=0; index<n_runs; index++)); do
    gpu=$((index % PARALLEL_GPUS))
    CUDA_VISIBLE_DEVICES="$gpu" uv run python program/run_manifest_entry.py \
      --manifest "$manifest" --index "$index" &
    if (((index + 1) % PARALLEL_GPUS == 0)); then
      wait
    fi
  done
  wait
  uv run python program/aggregate_strict_results.py
}

# ---------------- 阶段 7: 可选 PaRTI 路线（文档第 8 节） ----------------
phase_parti() {
  bash scripts/prepare_parti.sh
}

# ---------------- 阶段 8: 5 折 x 3 种子确认实验（文档第 9 节） ----------------
phase_confirmation() {
  bash scripts/run_confirmation.sh configs/sweeps/benchmark.yaml "$PARALLEL_GPUS"
  if [[ "$STOP_AFTER_BENCHMARK" == "1" ]]; then
    log "STOP_AFTER_BENCHMARK=1，停止在 benchmark 之后，请按停止规则 1/2/3 人工核对后重跑"
    exit 0
  fi
  bash scripts/run_confirmation.sh configs/sweeps/model-pathway-residual.yaml "$PARALLEL_GPUS"
  bash scripts/run_confirmation.sh configs/sweeps/abl-pretrain.yaml "$PARALLEL_GPUS"
  bash scripts/run_confirmation.sh configs/sweeps/abl-graph.yaml "$PARALLEL_GPUS"
  bash scripts/run_confirmation.sh configs/sweeps/abl-esm.yaml "$PARALLEL_GPUS"
  bash scripts/run_confirmation.sh configs/sweeps/abl-modality.yaml "$PARALLEL_GPUS"
}

# ---------------- 阶段 9: 任务对齐掩码组学预训练（文档第 10 节，条件性） ----------------
phase_task_aligned() {
  RUN_TASK_ALIGNED=1 \
  PRETRAIN_PROTOCOLS="eval-lco eval-ldo-kt eval-ldo-so eval-db" \
  bash scripts/run_ver2_pretraining.sh
  bash scripts/run_confirmation.sh configs/sweeps/pretrain-task-aligned.yaml "$PARALLEL_GPUS"
}

# ---------------- 阶段 10: 统计汇总（文档第 11 节） ----------------
phase_aggregate() {
  uv run python program/aggregate_strict_results.py \
    --root data/model/experiments \
    --output-dir data/model/experiments \
    --hierarchical-bootstrap-runs "$BOOTSTRAP_RUNS"
}

# ---------------- 阶段 11: 严格可解释性（文档第 12 节） ----------------
phase_interpretation() {
  local out="data/model/interpret-faithfulness"
  local dirs=() s d
  for s in "${SEEDS[@]}"; do
    d="data/model/experiments/model-pathway-residual/eval-ldo-so--model-m4/fold-00/seed-$(printf '%04d' "$s")"
    if [[ -f "$d/checkpoint-best.pt" ]]; then
      dirs+=("$d")
    else
      log "跳过可解释性输入（缺少检查点）: $d"
    fi
  done
  if [[ "${#dirs[@]}" -eq 0 ]]; then
    log "没有可用的 M4 检查点，跳过可解释性"
    return 0
  fi
  bash scripts/run_interpretation.sh "$out" "${dirs[@]}"
}

# ---------------- 阶段 12: 最终报告（文档第 15 节） ----------------
phase_final_report() {
  local n_complete n_ckpt
  n_complete=$(grep -rl '"state": "complete"' data/model/experiments --include=status.json 2>/dev/null | wc -l || true)
  n_ckpt=$(find data/model/pretrain/strict -name best_encoder.pt 2>/dev/null | wc -l || true)
  log "===== 最终结果汇总 ====="
  log "完成运行数: $n_complete"
  log "折专属预训练检查点: $n_ckpt / 90"
  local f
  for f in benchmark-summary.csv benchmark-report.md hierarchical-bootstrap.csv paired-comparisons.csv; do
    if [[ -f "data/model/experiments/$f" ]]; then
      log "产物: data/model/experiments/$f"
    else
      log "警告: 缺少 data/model/experiments/$f"
    fi
  done
  log "验收清单（文档第 15 节）:"
  log "  1. 六协议全部完成固定 5 折 -> 检查 benchmark-summary.csv 的 n_folds"
  log "  2. 每折 3 种子 -> 检查 n_seeds"
  log "  3. LDO-KT 与 LDO-SO 分开汇报 -> 检查 split_id 列"
  log "  4. scaler 的 train_cell_idx/train_drug_idx 与 split 一致 -> 人工核对"
  log "  5. 预训练检查点路径含协议/折/种子 -> 已校验 90 个"
  log "  6. 汇总表含均值/标准差/95% CI/配对差值 -> 见 benchmark-summary.csv"
  log "  7. 停止规则 -> 人工核对（文档第 9 节）"
  log "  8. 可解释性三要素 -> 见 data/model/interpret-faithfulness/summary"
  log "完整日志: $LOG_FILE"
}

# ---------------- 主流程 ----------------
log "GDGN Ver2 远程无人值守训练开始（日志: $LOG_FILE）"
if [[ "$SKIP_ENV_CHECK" == "1" ]]; then
  log "SKIP_ENV_CHECK=1，跳过环境检查（分支/提交/uv sync/磁盘），仅探测 GPU"
  run_phase gpu_probe phase_gpu_probe
else
  run_phase env_check phase_env_check
fi
run_phase data_preflight phase_data_preflight
run_phase data_prepare phase_data_prepare
run_phase acceptance phase_acceptance
if [[ "$SKIP_PRETRAIN" != "1" ]]; then
  run_phase pretrain phase_pretrain
fi
if [[ "$RUN_SCREENING" == "1" ]]; then
  run_phase screening phase_screening
  if [[ "$STOP_AFTER_SCREENING" == "1" ]]; then
    log "STOP_AFTER_SCREENING=1，停止在筛选阶段，请人工核对结果后重跑（去掉该变量）"
    exit 0
  fi
fi
if [[ "$RUN_PARTI" == "1" ]]; then
  run_phase parti phase_parti
fi
if [[ "$SKIP_CONFIRMATION" != "1" ]]; then
  run_phase confirmation phase_confirmation
fi
if [[ "$RUN_TASK_ALIGNED" == "1" ]]; then
  run_phase task_aligned phase_task_aligned
fi
if [[ "$SKIP_AGGREGATE" != "1" ]]; then
  run_phase aggregate phase_aggregate
fi
if [[ "$RUN_INTERPRETATION" == "1" ]]; then
  run_phase interpretation phase_interpretation
fi
run_phase final_report phase_final_report
log "全部阶段完成。正式结果请按文档第 15 节验收清单核对。"
