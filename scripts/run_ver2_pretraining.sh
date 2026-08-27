#!/usr/bin/env bash
set -euo pipefail

uv run python program/model/edge_mask.py
if [[ "${RUN_GLOBAL_GRAPH_PRETRAIN:-0}" == "1" ]]; then
  uv run python program/model/pretrain.py --config data/model/pretrain/pretrain_config.json
  uv run python program/verify_pretrain.py --ckpt data/model/pretrain/best_encoder.pt
fi

if [[ "${RUN_FOLD_GRAPH_PRETRAIN:-0}" == "1" ]]; then
  protocols=(eval-lpo eval-lco eval-ldo-kt eval-ldo-so eval-lto eval-db)
  seeds=(42 3407 8128)
  for protocol in "${protocols[@]}"; do
    for fold in 0 1 2 3 4; do
      for seed in "${seeds[@]}"; do
        target=$(printf 'data/model/pretrain/strict/%s/fold-%02d/seed-%04d' "$protocol" "$fold" "$seed")
        uv run python program/model/pretrain.py \
          --config data/model/pretrain/pretrain_config.json \
          --split-id "$protocol" --fold "$fold" --seed "$seed" --output-dir "$target"
        uv run python program/verify_pretrain.py \
          --ckpt "$target/best_encoder.pt" --split-id "$protocol" --fold "$fold"
      done
    done
  done
fi

if [[ "${RUN_TASK_ALIGNED:-0}" == "1" ]]; then
  protocols=(eval-lco eval-ldo-kt eval-ldo-so eval-db)
  seeds=(42 3407 8128)
  for protocol in "${protocols[@]}"; do
    for fold in 0 1 2 3 4; do
      for seed in "${seeds[@]}"; do
        target=$(printf 'data/model/pretrain-masked-omics/%s/fold-%02d/seed-%04d' "$protocol" "$fold" "$seed")
        uv run python program/pretrain_task_aligned.py \
          --output-dir "$target" --split-id "$protocol" --fold "$fold" \
          --modalities expression copynumber methylation \
          --latent-dim 128 --epochs 50 --seed "$seed"
      done
    done
  done
fi

echo "[pretrain] strict graph and masked-omics pretraining complete"
