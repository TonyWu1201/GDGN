#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <output-root> <run-dir> [run-dir ...]"
  exit 2
fi

output_root="$1"
shift
input_dirs=()
for run_dir in "$@"; do
  seed=$(basename "$run_dir")
  fold=$(basename "$(dirname "$run_dir")")
  model=$(basename "$(dirname "$(dirname "$run_dir")")")
  target="${output_root}/${model}/${fold}/${seed}"
  uv run python program/run_strict_interpretation.py \
    --run-dir "$run_dir" --output-dir "$target"
  input_dirs+=("$target")
done

uv run python program/run_faithfulness.py \
  --input-dirs "${input_dirs[@]}" \
  --output-dir "${output_root}/summary"
