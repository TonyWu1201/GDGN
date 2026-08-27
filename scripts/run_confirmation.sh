#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <sweep-yaml> [parallel-gpus]"
  exit 2
fi

sweep="$1"
parallel_gpus="${2:-1}"
manifest="data/model/experiments/run-manifest.jsonl"
uv run python program/generate_run_manifest.py --sweep "$sweep" --output "$manifest"
n_runs=$(wc -l < "$manifest")

for ((index=0; index<n_runs; index++)); do
  gpu=$((index % parallel_gpus))
  CUDA_VISIBLE_DEVICES="$gpu" uv run python program/run_manifest_entry.py \
    --manifest "$manifest" --index "$index" &
  if (((index + 1) % parallel_gpus == 0)); then
    wait
  fi
done
wait

uv run python program/aggregate_strict_results.py
