"""
Step 3a: 细胞系组学特征构建
===========================
产出 data/model/hetero_graph/cell_line_features.pt：
  dict of stacked tensors keyed by omics, plus reference orders.

  {
    'expression':      (404, 8412) float32   <- 按 core_gene 切片
    'mutation':        (404, 8412) float32   <- 按 core_gene 切片，缺失基因填 0
    'copynumber':      (404, 8412) float32
    'methylation':     (404, 8412) float32   <- CpG→基因 聚合后切片
    'pathway_activity':(404, 186)  float32
    'cell_line_order':  list[str]            404 标准细胞系名
    'gene_order':       list[str]            8412 核心基因名 (== core_gene_order.txt)
    'pathway_names':    list[str]            186 通路名
  }

约定：
  - 行序 = data/processed/cell_line_canonical_order.txt（404）
  - 基因列序 = data/model/hetero_graph/core_gene_order.txt（8412）
  - 矩阵在加载时按名 reindex；缺失表型/基因填 0；甲基化聚合为基因级均值；pathway 保持 Z-score（脚本内已标准化，不重复）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data"
PROCESSED = DATA / "processed"

EXPR_CSV = PROCESSED / "cell_line_omics" / "expression.csv"
MUT_CSV = PROCESSED / "cell_line_omics" / "mutation.csv"
CNV_CSV = PROCESSED / "cell_line_omics" / "copynumber.csv"
METH_CSV = PROCESSED / "cell_line_omics" / "methylation.csv"
METH_RAW_CSV = PROCESSED / "cell_line_omics" / "methylation_raw.csv"
PW_NPY = PROCESSED / "driver&pathway" / "pathway_activity.npy"
PW_RAW_NPY = PROCESSED / "driver&pathway" / "pathway_activity_raw.npy"
PW_NAMES = PROCESSED / "driver&pathway" / "pathway_names.txt"

CANON_PATH = PROCESSED / "cell_line_canonical_order.txt"
CORE_PATH = DATA / "model" / "hetero_graph" / "core_gene_order.txt"
OUT_PATH = DATA / "model" / "hetero_graph" / "cell_line_features.pt"
RAW_OUT_PATH = DATA / "model" / "hetero_graph" / "cell_line_features_raw.pt"


def _reindex_and_slice(
    df: pd.DataFrame,
    cell_order: list[str],
    gene_order: list[str],
    preserve_missing: bool = False,
) -> pd.DataFrame:
    """Reindex rows to cell_order, slice columns by gene_order; missing cols -> 0."""
    rows = df.reindex(cell_order)
    present = set(rows.columns)
    cols_needed = [g if g in present else None for g in gene_order]
    fill = np.nan if preserve_missing else 0.0
    out = pd.DataFrame(
        np.full((len(rows), len(gene_order)), fill, dtype=np.float32),
        index=cell_order,
        columns=gene_order,
    )
    # fill present genes
    filled_cols = [g for g in gene_order if g in present]
    if filled_cols:
        out[filled_cols] = rows[filled_cols].values
    return out if preserve_missing else out.fillna(0.0)


def _aggregate_methylation_by_gene(
    meth: pd.DataFrame, gene_order: list[str], preserve_missing: bool = False
) -> pd.DataFrame:
    """Methylation cols: 'GENE_chr_start_end' → mean per gene. Then slice to gene_order."""
    raw_cols = meth.columns.astype(str)
    gene_prefixes = [c.split("_")[0] for c in raw_cols]
    # use groupby on columns (mean of CpG clusters per gene, per cell line)
    df = meth.copy()
    df.columns = gene_prefixes
    # average duplicate gene columns
    grouped = df.T.groupby(level=0).mean().T
    # slice
    fill = np.nan if preserve_missing else 0.0
    out = pd.DataFrame(
        np.full((len(meth.index), len(gene_order)), fill, dtype=np.float32),
        index=meth.index,
        columns=gene_order,
    )
    filled = [g for g in gene_order if g in set(grouped.columns)]
    if filled:
        out[filled] = grouped[filled].values
    return out if preserve_missing else out.fillna(0.0)


def _reorder_pathways(values: np.ndarray, expr_index: list[str], cell_order: list[str]) -> np.ndarray:
    if values.shape[0] != len(expr_index):
        raise ValueError(f"pathway row count {values.shape[0]} != expression rows {len(expr_index)}")
    name_to_canon_pos = {name: index for index, name in enumerate(cell_order)}
    result = np.full((len(cell_order), values.shape[1]), np.nan, dtype=np.float32)
    for source_index, name in enumerate(expr_index):
        if name in name_to_canon_pos:
            result[name_to_canon_pos[name]] = values[source_index]
    return result


def main():
    cell_order = CANON_PATH.read_text().splitlines()
    gene_order = CORE_PATH.read_text().splitlines()
    n_cells = len(cell_order)
    n_genes = len(gene_order)
    print(f"cells={n_cells} genes={n_genes}")

    print(" loading expression...")
    expr = pd.read_csv(EXPR_CSV, index_col=0)
    expr_s = _reindex_and_slice(expr, cell_order, gene_order)
    expr_raw = _reindex_and_slice(expr, cell_order, gene_order, preserve_missing=True)
    print(" loading mutation...")
    mut = pd.read_csv(MUT_CSV, index_col=0)
    mut_s = _reindex_and_slice(mut, cell_order, gene_order)
    print(" loading copynumber...")
    cnv = pd.read_csv(CNV_CSV, index_col=0)
    cnv_s = _reindex_and_slice(cnv, cell_order, gene_order)
    cnv_raw = _reindex_and_slice(cnv, cell_order, gene_order, preserve_missing=True)
    print(" loading methylation (slow)...")
    meth_legacy = pd.read_csv(METH_CSV, index_col=0).reindex(cell_order)
    meth_s = _aggregate_methylation_by_gene(meth_legacy, gene_order)
    meth_path = METH_RAW_CSV if METH_RAW_CSV.exists() else METH_CSV
    meth_raw = _aggregate_methylation_by_gene(
        pd.read_csv(meth_path, index_col=0).reindex(cell_order),
        gene_order,
        preserve_missing=True,
    )

    print(" loading pathway_activity...")
    pw = np.load(PW_NPY)
    pw_path = PW_RAW_NPY if PW_RAW_NPY.exists() else PW_NPY
    pw_raw = np.load(pw_path)
    assert pw.shape == (n_cells, 186), f"pathway shape {pw.shape} != ({n_cells}, 186)"
    assert pw_raw.shape == (n_cells, 186), f"raw pathway shape {pw_raw.shape} != ({n_cells}, 186)"
    pw_names = PW_NAMES.read_text().splitlines()
    assert pw.shape[1] == len(pw_names), "pathway_names count mismatch"

    expr_index = list(expr.index)
    pw_reordered = _reorder_pathways(pw, expr_index, cell_order)
    pw_raw_reordered = _reorder_pathways(pw_raw, expr_index, cell_order)

    out = {
        "expression": torch.from_numpy(expr_s.values.astype(np.float32)),
        "mutation": torch.from_numpy(mut_s.values.astype(np.float32)),
        "copynumber": torch.from_numpy(cnv_s.values.astype(np.float32)),
        "methylation": torch.from_numpy(meth_s.values.astype(np.float32)),
        "pathway_activity": torch.from_numpy(pw_reordered.astype(np.float32)),
        "cell_line_order": cell_order,
        "gene_order": gene_order,
        "pathway_names": pw_names,
    }
    torch.save(out, OUT_PATH)
    raw_out = dict(out)
    raw_out.update({
        "expression": torch.from_numpy(expr_raw.values.astype(np.float32)),
        "copynumber": torch.from_numpy(cnv_raw.values.astype(np.float32)),
        "methylation": torch.from_numpy(meth_raw.values.astype(np.float32)),
        "pathway_activity": torch.from_numpy(pw_raw_reordered.astype(np.float32)),
    })
    raw_out["preprocessing_status"] = {
        "strict_unscaled": bool(METH_RAW_CSV.exists() and PW_RAW_NPY.exists()),
        "methylation_source": str(meth_path.relative_to(PROJECT_ROOT)),
        "pathway_source": str(pw_path.relative_to(PROJECT_ROOT)),
    }
    torch.save(raw_out, RAW_OUT_PATH)
    shapes = {k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)}
    print(" saved:", shapes)
    print(f" strict raw artifact: {RAW_OUT_PATH} status={raw_out['preprocessing_status']}")
    print(" done.")


if __name__ == "__main__":
    main()
