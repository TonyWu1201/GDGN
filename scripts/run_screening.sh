#!/usr/bin/env bash
# =============================================================================
# GDGN Ver2 单折单种子筛选（阶段 5）
# 依据 guidance/完成总结/Ver2/远程GPU完整训练说明.md 第 7 节编写
#
# 用法:
#   bash scripts/run_screening.sh [--force]
#
# 环境变量:
#   PARALLEL_GPUS=4   并行 GPU 数（0=自动检测，上限 4）
#   FORCE=1           忽略已完成运行强制重跑
#
# 说明:
#   - 前置条件: 6 个协议的 fold-00/seed-0042 折专属预训练检查点已存在
#     （由 scripts/run_local_data_and_pretrain.ps1 阶段 4 产出）。
#   - 全程只使用 uv，不调用 pip。
#   - 幂等: 已完成且配置一致的运行自动跳过（program/strict_eval/registry.py），
#     中断后重跑自动续跑。
#   - 每个运行独立进程，不在一个模型内做 DDP；每块 GPU 同时只跑一个任务。
# =============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PARALLEL_GPUS="${PARALLEL_GPUS:-0}"
FORCE="${FORCE:-0}"
PROTOCOLS=(eval-lpo eval-lco eval-ldo-kt eval-ldo-so eval-lto eval-db)
MANIFEST="data/model/experiments/screening-manifest.jsonl"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo "[FATAL] $*" >&2; exit 1; }

# ---------------- 前置检查: 筛选所需预训练检查点 ----------------
for p in "${PROTOCOLS[@]}"; do
  ckpt="data/model/pretrain/strict/$p/fold-00/seed-0042/best_encoder.pt"
  [[ -f "$ckpt" ]] || die "缺少筛选所需预训练检查点: $ckpt"
done
log "筛选所需预训练检查点完整（${#PROTOCOLS[@]} 个）"

# ---------------- GPU 探测 ----------------
detect_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l || true
  else
    echo 0
  fi
}
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

# ---------------- 生成运行清单 ----------------
uv run python program/generate_run_manifest.py \
  --sweep configs/sweeps/screening.yaml --output "$MANIFEST"
n_runs=$(wc -l < "$MANIFEST")
log "筛选: $n_runs 个运行，并行 $PARALLEL_GPUS 块 GPU"

# ---------------- 并行执行 ----------------
force_flag=()
if [[ "$FORCE" == "1" ]]; then
  force_flag=(--force)
fi

pids=()
for ((index=0; index<n_runs; index++)); do
  gpu=$((index % PARALLEL_GPUS))
  log "筛选运行 #$((index + 1))/$n_runs -> GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
    uv run python program/run_manifest_entry.py \
      --manifest "$MANIFEST" --index "$index" "${force_flag[@]}" &
  pids+=("$!")
  if (((index + 1) % PARALLEL_GPUS == 0)); then
    for pid in "${pids[@]}"; do
      wait "$pid" || die "筛选任务失败（pid=$pid）"
    done
    pids=()
  fi
done
for pid in "${pids[@]}"; do
  wait "$pid" || die "筛选任务失败（pid=$pid）"
done

# ---------------- 汇总 ----------------
uv run python program/aggregate_strict_results.py
log "筛选完成，结果在 data/model/experiments/"
