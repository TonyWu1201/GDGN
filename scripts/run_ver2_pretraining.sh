#!/usr/bin/env bash
set -euo pipefail

uv run python program/model/edge_mask.py
if [[ "${RUN_GLOBAL_GRAPH_PRETRAIN:-0}" == "1" ]]; then
  uv run python program/model/pretrain.py --config data/model/pretrain/pretrain_config.json
  uv run python program/verify_pretrain.py --ckpt data/model/pretrain/best_encoder.pt
fi

if [[ "${RUN_FOLD_GRAPH_PRETRAIN:-0}" == "1" ]]; then
  read -r -a protocols <<< "${PRETRAIN_PROTOCOLS:-eval-lpo eval-lco eval-ldo-kt eval-ldo-so eval-lto eval-db}"
  read -r -a folds <<< "${PRETRAIN_FOLDS:-0 1 2 3 4}"
  read -r -a seeds <<< "${PRETRAIN_SEEDS:-42 3407 8128}"
  for protocol in "${protocols[@]}"; do
    for fold in "${folds[@]}"; do
      for seed in "${seeds[@]}"; do
        target=$(printf 'data/model/pretrain/strict/%s/fold-%02d/seed-%04d' "$protocol" "$fold" "$seed")
        uv run python program/model/pretrain.py \
          --config data/model/pretrain/pretrain_config.json \
          --split-id "$protocol" --fold "$fold" --seed "$seed" --output-dir "$target"
        if [[ "${VERIFY_FOLD_PRETRAIN:-0}" == "1" ]]; then
          uv run python program/verify_pretrain.py \
            --ckpt "$target/best_encoder.pt" --split-id "$protocol" --fold "$fold"
        fi
      done
    done
  done
fi

if [[ "${RUN_TASK_ALIGNED:-0}" == "1" ]]; then
  read -r -a protocols <<< "${PRETRAIN_PROTOCOLS:-eval-lco eval-ldo-kt eval-ldo-so eval-db}"
  read -r -a folds <<< "${PRETRAIN_FOLDS:-0 1 2 3 4}"
  read -r -a seeds <<< "${PRETRAIN_SEEDS:-42 3407 8128}"
  for protocol in "${protocols[@]}"; do
    for fold in "${folds[@]}"; do
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
