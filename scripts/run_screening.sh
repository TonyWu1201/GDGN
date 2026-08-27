#!/usr/bin/env bash
set -euo pipefail

manifest="data/model/experiments/screening-manifest.jsonl"
uv run python program/generate_run_manifest.py \
  --sweep configs/sweeps/screening.yaml --output "$manifest"
n_runs=$(wc -l < "$manifest")
for ((index=0; index<n_runs; index++)); do
  uv run python program/run_manifest_entry.py --manifest "$manifest" --index "$index"
done

uv run python program/aggregate_strict_results.py
