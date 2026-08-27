#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after placing the licensed/raw source files at
# the paths listed in guidance/完成总结/Ver2/远程GPU训练说明.md.
uv sync --all-groups

uv run python program/preprocess/preprocess_expression.py
uv run python program/preprocess/preprocess_mutation.py
uv run python program/preprocess/preprocess_cnv.py
uv run python program/preprocess/methylation_tss1kb.py
uv run python program/preprocess/compute_ssgsea.py

# Regenerate these only when the ESM/UniProt source changes. The mean path
# excludes BOS/EOS; PaRTI is prepared separately after core genes are known.
if [[ "${REBUILD_ESM_MEAN:-0}" == "1" ]]; then
  uv run python program/preprocess/protein_embedding.py
  uv run python program/preprocess/filter_gene_embeddings.py
fi

uv run python program/model/build_graph.py
uv run python program/preprocess/build_cell_line_features.py
uv run python program/model/edge_mask.py
uv run python program/build_strict_splits.py
uv run python program/build_data_audit_figures.py
uv run python program/build_data_card.py

uv run python - <<'PY'
from program.strict_eval.data import RAW_CELL_FEATURE_PATH, RAW_DRUG_FEATURE_PATH
import torch

assert RAW_CELL_FEATURE_PATH.exists(), RAW_CELL_FEATURE_PATH
assert RAW_DRUG_FEATURE_PATH.exists(), RAW_DRUG_FEATURE_PATH
cell = torch.load(RAW_CELL_FEATURE_PATH, map_location="cpu", weights_only=False)
assert cell["preprocessing_status"]["strict_unscaled"] is True
print("[prepare] strict unscaled cell/drug artifacts: OK")
PY

echo "[prepare] Ver2 data preparation complete"
