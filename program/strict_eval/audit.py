from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import (
    CELL_FEATURE_PATH, DTI_PATH, HETERO_GRAPH_PATH, IC50_PATH, PPI_PATH, PROJECT_ROOT,
)
from .data import RAW_CELL_FEATURE_PATH, RAW_DRUG_FEATURE_PATH
from .io import atomic_write_text, sha256_file
from .splits import load_pair_table


def build_data_card(output_path: str | Path) -> Path:
    pairs = load_pair_table()
    graph = torch.load(HETERO_GRAPH_PATH, map_location="cpu", weights_only=False)
    cell = torch.load(CELL_FEATURE_PATH, map_location="cpu", weights_only=False)
    duplicate_pairs = int(pairs.duplicated(["cell_idx", "drug_idx"]).sum())
    response = pairs["ic50"].to_numpy(dtype=float)
    raw_cell_ready = False
    raw_cell_status = {}
    if RAW_CELL_FEATURE_PATH.exists():
        raw_cell = torch.load(RAW_CELL_FEATURE_PATH, map_location="cpu", weights_only=False)
        raw_cell_status = raw_cell.get("preprocessing_status", {})
        raw_cell_ready = bool(raw_cell_status.get("strict_unscaled"))
    raw_drug_ready = RAW_DRUG_FEATURE_PATH.exists()
    lines = [
        "# Data quality report",
        "",
        "## Canonical cohort",
        "",
        f"- Observed response pairs: {len(pairs):,}",
        f"- Cell lines: {pairs['cell_idx'].nunique():,}",
        f"- Drugs: {pairs['drug_idx'].nunique():,}",
        f"- Tissues: {pairs['cancer_type'].nunique():,}",
        f"- Duplicate drug-cell pairs: {duplicate_pairs}",
        f"- Missing/non-finite responses: {int((~np.isfinite(pairs['ic50'])).sum())}",
        f"- Response mean/std: {response.mean():.4f} / {response.std():.4f}",
        f"- Response min/median/max: {response.min():.4f} / {np.median(response):.4f} / {response.max():.4f}",
        "",
        "## Graph and features",
        "",
        f"- Gene nodes: {int(graph['gene'].num_nodes):,}",
        f"- Drug nodes: {int(graph['drug'].num_nodes):,}",
        f"- PPI directed records: {graph['gene', 'ppi', 'gene'].edge_index.shape[1]:,}",
        f"- DTI edges: {graph['drug', 'targets', 'gene'].edge_index.shape[1]:,}",
    ]
    for name in ("expression", "mutation", "copynumber", "methylation", "pathway_activity"):
        values = cell[name]
        lines.append(f"- {name}: shape={tuple(values.shape)}, finite={bool(torch.isfinite(values).all())}")
    lines.extend([
        f"- Strict unscaled cell artifact ready: {raw_cell_ready}",
        f"- Strict unscaled drug artifact ready: {raw_drug_ready}",
        f"- Raw cell preprocessing status: `{json.dumps(raw_cell_status, ensure_ascii=False, sort_keys=True)}`",
    ])
    lines.extend(["", "## Source checksums", ""])
    source_paths = [IC50_PATH, PPI_PATH, DTI_PATH, HETERO_GRAPH_PATH, CELL_FEATURE_PATH]
    source_paths.extend(sorted((PROJECT_ROOT / "data/model/splits").glob("*/manifest.json")))
    for path in source_paths:
        if path.exists():
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            lines.append(f"- `{relative}`: `{sha256_file(path)}`")
    lines.extend([
        "",
        "## Leakage policy",
        "",
        "All response-dependent preprocessing and continuous-feature scaling in Ver2 are fitted on the training fold only. Mutation remains a discrete modality. Pathway scores are generated without response labels and standardized using the training fold.",
        "",
        "Legacy globally standardized artifacts remain only for old-checkpoint compatibility and are rejected by formal Ver2 runs.",
        "If either strict artifact readiness flag above is false, run the remote preprocessing sequence before any formal experiment. Smoke mode alone may use the legacy fallback and its metrics are not scientific results.",
        "",
    ])
    return atomic_write_text(output_path, "\n".join(lines))
