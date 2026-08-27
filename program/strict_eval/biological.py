from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .constants import DTI_PATH, MODEL_DIR, PROCESSED_DIR

GENE_ORDER_PATH = MODEL_DIR / "hetero_graph" / "core_gene_order.txt"
PATHWAY_SET_PATH = PROCESSED_DIR / "driver&pathway" / "pathway_gene_sets.json"
PATHWAY_NAME_PATH = PROCESSED_DIR / "driver&pathway" / "pathway_names.txt"
DRUG_INDEX_PATH = PROCESSED_DIR / "drug_cid_to_idx.json"
PPI_EDGE_PATH = MODEL_DIR / "hetero_graph" / "ppi_edge_index.pt"
PPI_WEIGHT_PATH = MODEL_DIR / "hetero_graph" / "ppi_edge_weight.pt"


def pathway_membership() -> tuple[torch.Tensor, list[str], list[str]]:
    genes = GENE_ORDER_PATH.read_text(encoding="utf-8").splitlines()
    pathway_sets: dict[str, list[str]] = json.loads(PATHWAY_SET_PATH.read_text(encoding="utf-8"))
    preferred = PATHWAY_NAME_PATH.read_text(encoding="utf-8").splitlines()
    pathways = [name for name in preferred if name in pathway_sets]
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    membership = torch.zeros((len(pathways), len(genes)), dtype=torch.float32)
    for pathway_idx, name in enumerate(pathways):
        indices = [gene_to_idx[gene] for gene in pathway_sets[name] if gene in gene_to_idx]
        if indices:
            membership[pathway_idx, indices] = 1.0
    return membership, pathways, genes


def build_weighted_pathway_adjacency(
    edge_index: torch.Tensor | None = None,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project weighted PPI to pathways and return symmetric normalized weights."""
    membership, _, _ = pathway_membership()
    if edge_index is None:
        edge_index = torch.load(PPI_EDGE_PATH, map_location="cpu", weights_only=False)
    if edge_weight is None:
        edge_weight = torch.load(PPI_WEIGHT_PATH, map_location="cpu", weights_only=False)
    edge_index_np = edge_index.detach().cpu().numpy()
    weights = edge_weight.detach().cpu().numpy()
    n_genes = membership.shape[1]
    ppi = sp.coo_matrix(
        (weights, (edge_index_np[0], edge_index_np[1])), shape=(n_genes, n_genes)
    ).tocsr()
    member = sp.csr_matrix(membership.numpy())
    projected = (member @ ppi @ member.T).toarray().astype(np.float32)
    np.fill_diagonal(projected, 0.0)
    projected = np.maximum(projected, projected.T)
    if np.max(projected) > 0:
        projected /= np.max(projected)
    projected += np.eye(projected.shape[0], dtype=np.float32)
    degree = projected.sum(axis=1)
    inv_sqrt = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    normalized = inv_sqrt[:, None] * projected * inv_sqrt[None, :]
    return torch.from_numpy(normalized)


def known_target_pathway_distribution() -> torch.Tensor:
    """Map observed DTI targets to a normalized drug-by-pathway matrix."""
    membership, _, genes = pathway_membership()
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    drug_to_idx = json.loads(DRUG_INDEX_PATH.read_text(encoding="utf-8"))
    matrix = torch.zeros((len(drug_to_idx), membership.shape[0]), dtype=torch.float32)
    dti = pd.read_csv(DTI_PATH)
    cid_column = "cid" if "cid" in dti.columns else "drug_cid"
    gene_column = next((name for name in ("gene_name", "gene", "gene_symbol") if name in dti.columns), None)
    if gene_column is None:
        raise KeyError(f"cannot identify gene column in {DTI_PATH}: {list(dti.columns)}")
    for row in dti.itertuples(index=False):
        cid = str(getattr(row, cid_column))
        gene = str(getattr(row, gene_column))
        if cid not in drug_to_idx or gene not in gene_to_idx:
            continue
        matrix[int(drug_to_idx[cid])] += membership[:, gene_to_idx[gene]]
    row_sum = matrix.sum(dim=1, keepdim=True)
    fallback = torch.full_like(matrix, 1.0 / max(1, matrix.shape[1]))
    return torch.where(row_sum > 0, matrix / row_sum.clamp_min(1e-8), fallback)


def pathway_gene_mask() -> torch.Tensor:
    membership, _, _ = pathway_membership()
    return membership.sum(dim=0) > 0
