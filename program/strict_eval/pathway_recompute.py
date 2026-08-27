from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import CELL_ORDER_PATH, PROCESSED_DIR, PROJECT_ROOT


def ssgsea_sample(
    expression: np.ndarray,
    gene_names: list[str],
    gene_sets: dict[str, list[str]],
    pathway_names: list[str],
    alpha: float = 0.25,
) -> np.ndarray:
    """Vectorized single-sample equivalent of preprocess/compute_ssgsea.py."""
    values = np.asarray(expression, dtype=np.float64)
    order = np.argsort(-values)
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    gene_to_idx = {gene: index for index, gene in enumerate(gene_names)}
    scores = np.zeros(len(pathway_names), dtype=np.float32)
    for pathway_index, pathway in enumerate(pathway_names):
        members = np.asarray(
            [gene_to_idx[gene] for gene in gene_sets[pathway] if gene in gene_to_idx], dtype=np.int64
        )
        if members.size == 0:
            continue
        n_background = len(values) - len(members)
        increments = np.full(len(values), -1.0 / n_background if n_background else 0.0)
        weights = np.power((len(values) - ranks[members]).astype(np.float64), alpha)
        if weights.sum() > 0:
            increments[members] = weights / weights.sum()
        running = np.cumsum(increments[order])
        maximum, minimum = float(running.max()), float(running.min())
        scores[pathway_index] = maximum if abs(maximum) >= abs(minimum) else minimum
    return scores


class FoldPathwayRecomputer:
    """Recompute expression-derived pathways after a gene intervention."""

    def __init__(self, split_payload: dict, scaler_path: str | Path):
        expression_path = PROCESSED_DIR / "cell_line_omics" / "expression.csv"
        pathway_path = PROCESSED_DIR / "driver&pathway" / "pathway_gene_sets.json"
        pathway_names_path = PROCESSED_DIR / "driver&pathway" / "pathway_names.txt"
        core_gene_path = PROJECT_ROOT / "data/model/hetero_graph/core_gene_order.txt"
        self.expression = pd.read_csv(expression_path, index_col=0)
        self.cell_order = CELL_ORDER_PATH.read_text(encoding="utf-8").splitlines()
        self.gene_names = self.expression.columns.astype(str).tolist()
        self.pathway_sets = json.loads(pathway_path.read_text(encoding="utf-8"))
        self.pathway_names = pathway_names_path.read_text(encoding="utf-8").splitlines()
        core_genes = core_gene_path.read_text(encoding="utf-8").splitlines()
        full_index = {gene: index for index, gene in enumerate(self.gene_names)}
        self.core_to_full = np.asarray([full_index.get(gene, -1) for gene in core_genes], dtype=np.int64)
        train_names = [self.cell_order[index] for index in split_payload["splits"]["train"]["cell_idx"]]
        self.train_mean = self.expression.reindex(train_names).mean(axis=0, skipna=True).fillna(0.0).to_numpy()
        scaler = np.load(scaler_path)
        self.pathway_mean = scaler["pathway_activity__mean"]
        self.pathway_scale = scaler["pathway_activity__scale"]

    def recompute(self, cell_idx: int, gene_idx: np.ndarray, retain: bool = False) -> np.ndarray:
        name = self.cell_order[int(cell_idx)]
        values = self.expression.loc[name].to_numpy(dtype=np.float64, copy=True)
        mapped = self.core_to_full[np.asarray(gene_idx, dtype=int)]
        mapped = mapped[mapped >= 0]
        if retain:
            original = values.copy()
            all_core = self.core_to_full[self.core_to_full >= 0]
            values[all_core] = self.train_mean[all_core]
            values[mapped] = original[mapped]
        else:
            values[mapped] = self.train_mean[mapped]
        raw = ssgsea_sample(values, self.gene_names, self.pathway_sets, self.pathway_names)
        return ((raw - self.pathway_mean) / self.pathway_scale).astype(np.float32)
