"""
Step 4: 数据验证 -> data_report.txt
==================================
对 Step 1–3 的所有产出做整体一致性校验，写入 data/processed/data_report.txt。

校验项（计划 §6.1）：
  维度一致 / PPI&DTI 索引合法 / 样本对标签合法 / 划分无泄漏 / 癌型分层 JS / 孤立基因 / 自环 /
  特征标准化 / 通路活性对齐 / GPU 内存估算(策略 B)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"
PROCESSED = DATA / "processed"
GRAPH = DATA / "model" / "hetero_graph"
DRUG = DATA / "model" / "drug_encoder"

OUT = PROCESSED / "data_report.txt"


def _ok(cond: bool) -> str:
    return "OK" if cond else "FAIL"


def _js(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64); q = q.astype(np.float64)
    p = p / (p.sum() + 1e-12); q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / (b[mask] + 1e-12))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    lines: list[str] = []
    L = lines.append

    # ===== load everything =====
    g = torch.load(GRAPH / "hetero_graph_base.pt", weights_only=False)
    gene_feat = torch.load(GRAPH / "gene_static_features.pt", weights_only=False)
    drug_feat = torch.load(GRAPH / "drug_features.pt", weights_only=False)
    clf = torch.load(GRAPH / "cell_line_features.pt", weights_only=False)
    core_gene_order = (GRAPH / "core_gene_order.txt").read_text().splitlines()
    core_mask = torch.load(GRAPH / "core_gene_mask.pt", weights_only=False)
    core_idx_esm = np.load(GRAPH / "core_gene_idx_in_esm.npy")
    splits = torch.load(PROCESSED / "sample_pairs_split.pt", weights_only=False)
    ldo = torch.load(PROCESSED / "ldo_splits.pt", weights_only=False)
    lco = torch.load(PROCESSED / "lco_splits.pt", weights_only=False)
    cell_idx_json = json.loads((PROCESSED / "cell_line_to_idx.json").read_text())
    drug_idx_json = json.loads((PROCESSED / "drug_cid_to_idx.json").read_text())
    canon = (PROCESSED / "cell_line_canonical_order.txt").read_text().splitlines()
    pw_orig = np.load(PROCESSED / "driver&pathway" / "pathway_activity.npy")

    n_genes = g["gene"].num_nodes
    n_drugs = g["drug"].num_nodes
    ppi_ei = g["gene", "ppi", "gene"].edge_index
    ppi_ew = g["gene", "ppi", "gene"].edge_weight
    dti_ei = g["drug", "targets", "gene"].edge_index
    dti_ew = g["drug", "targets", "gene"].edge_weight

    L("=== GDGN Phase 1 — 数据验证报告 ===")
    L("")
    L("## 1. 维度一致性")
    L(f"  hetero gene.num_nodes  = {n_genes}")
    L(f"  hetero drug.num_nodes  = {n_drugs}")
    L(f"  len(core_gene_order)   = {len(core_gene_order)}")
    L(f"  gene_static_features   = {tuple(gene_feat.shape)}")
    L(f"  drug_features          = {tuple(drug_feat.shape)}")
    L(f"  core_gene_mask.sum()   = {int(core_mask.sum())}")
    L(f"  core_gene_idx_in_esm   = {core_idx_esm.shape}")
    L(f"  cell_line_features:")
    for k in ["expression", "mutation", "copynumber", "methylation", "pathway_activity"]:
        L(f"    {k:<18} {tuple(clf[k].shape)}")
    L(f"  n_cells (canonical)    = {len(canon)}")
    L(f"  n_drugs (cid_to_idx)   = {len(drug_idx_json)}")
    L(f"  n_cells (cell_to_idx)  = {len(cell_idx_json)}")
    L("")

    # ===== 2. 边索引合法 =====
    L("## 2. PPI / DTI 边索引合法性")
    L(f"  PPI edge_index shape     = {tuple(ppi_ei.shape)}")
    L(f"  PPI max gene idx = {int(ppi_ei.max())}  (< {n_genes} ? {_ok(int(ppi_ei.max()) < n_genes)})")
    L(f"  PPI min gene idx = {int(ppi_ei.min())}  (>= 0 ? {_ok(int(ppi_ei.min()) >= 0)})")
    L(f"  DTI edge_index shape     = {tuple(dti_ei.shape)}")
    L(f"  DTI drug max = {int(dti_ei[0].max())}  (< {n_drugs} ? {_ok(int(dti_ei[0].max()) < n_drugs)})")
    L(f"  DTI gene max = {int(dti_ei[1].max())}  (< {n_genes} ? {_ok(int(dti_ei[1].max()) < n_genes)})")
    L(f"  DTI drug min = {int(dti_ei[0].min())}  (>= 0 ? {_ok(int(dti_ei[0].min()) >= 0)})")
    L(f"  DTI gene min = {int(dti_ei[1].min())}  (>= 0 ? {_ok(int(dti_ei[1].min()) >= 0)})")
    L("")

    # ===== 3. 样本对标签合法 =====
    L("## 3. 样本对标签合法性")
    ok_label = True
    n_nan = 0
    for sp, items in splits.items():
        for p in items:
            if math.isnan(p["ic50"]):
                n_nan += 1
            if not (0 <= p["cell_idx"] < len(canon) and 0 <= p["drug_idx"] < n_drugs):
                ok_label = False
        L(f"  {sp:<6}: n={len(items)} | cell_idx∈[0,{len(canon)-1}] drug_idx∈[0,{n_drugs-1}]")
    L(f"  NaN ic50 count: {n_nan}  (expected 0 ? {_ok(n_nan == 0)})")
    L(f"  label range OK: {_ok(ok_label)}")
    L("")

    # ===== 4. 划分无泄漏 =====
    L("## 4. 划分无泄漏 (random 80/10/10)")
    def keyset(items):
        return {(p["cell_idx"], p["drug_idx"]) for p in items}
    s_tr, s_va, s_te = keyset(splits["train"]), keyset(splits["val"]), keyset(splits["test"])
    L(f"  |train ∩ val|   = {len(s_tr & s_va)} (expected 0)")
    L(f"  |train ∩ test|  = {len(s_tr & s_te)} (expected 0)")
    L(f"  |val   ∩ test|  = {len(s_va & s_te)} (expected 0)")
    L(f"  LODO folds = {len(ldo)} | LOCO folds = {len(lco)}")
    # LODO fold closure check (no overlap train/test within fold)
    ldo_ok = True
    for f in ldo:
        st, te = set(f["train"]), set(f["test"])
        if st & te:
            ldo_ok = False
            break
    lco_ok = True
    for f in lco:
        st, te = set(f["train"]), set(f["test"])
        if st & te:
            lco_ok = False
            break
    L(f"  LODO folds closure OK: {_ok(ldo_ok)} | LOCO folds closure OK: {_ok(lco_ok)}")
    L("")

    # ===== 5. 癌型分层 JS =====
    L("## 5. 癌型分层质量")
    from collections import Counter
    cancer_types = sorted({p["cancer_type"] for p in splits["train"]})
    vec_all = np.array([Counter(p["cancer_type"] for p in (splits["train"]+splits["val"]+splits["test"]))[c] for c in cancer_types], dtype=np.float64)
    vec_tr = np.array([Counter(p["cancer_type"] for p in splits["train"])[c] for c in cancer_types], dtype=np.float64)
    vec_va = np.array([Counter(p["cancer_type"] for p in splits["val"])[c] for c in cancer_types], dtype=np.float64)
    vec_te = np.array([Counter(p["cancer_type"] for p in splits["test"])[c] for c in cancer_types], dtype=np.float64)
    js_tr, js_va, js_te = _js(vec_tr, vec_all), _js(vec_va, vec_all), _js(vec_te, vec_all)
    L(f"  JS(train, all) = {js_tr:.4f}  (< 0.05 ? {_ok(js_tr < 0.05)})")
    L(f"  JS(val,   all) = {js_va:.4f}  (< 0.05 ? {_ok(js_va < 0.05)})")
    L(f"  JS(test,  all) = {js_te:.4f}  (< 0.05 ? {_ok(js_te < 0.05)})")
    L("")

    # ===== 6. 孤立基因节点 =====
    L("## 6. 孤立基因 / 自环")
    # degree of each gene = count of unique edges (gene appears in PPI src/dst or DTI gene side)
    gene_deg = torch.zeros(n_genes, dtype=torch.long)
    gene_deg.scatter_add_(0, ppi_ei[0], torch.ones(ppi_ei.shape[1], dtype=torch.long))
    gene_deg.scatter_add_(0, ppi_ei[1], torch.ones(ppi_ei.shape[1], dtype=torch.long))
    gene_deg.scatter_add_(0, dti_ei[1], torch.ones(dti_ei.shape[1], dtype=torch.long))
    n_iso = int((gene_deg == 0).sum())
    L(f"  isolated gene nodes: {n_iso} / {n_genes}  ({n_iso/n_genes:.3%}, <=5% ? {_ok(n_iso/n_genes <= 0.05)})")
    # self loops (same-type only — cross-type DTI drug<->gene index spaces are disjoint in meaning)
    ppi_sl = int((ppi_ei[0] == ppi_ei[1]).sum())
    L(f"  PPI self-loops: {ppi_sl}  ({ppi_sl/ppi_ei.shape[1]:.4%}, <0.1% ? {_ok(ppi_sl/ppi_ei.shape[1] < 0.001)})")
    L(f"  DTI self-loops: not applicable (cross-type drug<->gene edges; drug idx ∈[0,{n_drugs-1}], gene idx ∈[0,{n_genes-1}] overlap by index but refer to different node types)")
    L("")

    # ===== 7. 特征标准化 =====
    L("## 7. 特征标准化检查")
    L(f"  drug_features (ECFP+理化, Z-score): mean={drug_feat.mean():.4f} std={drug_feat.std():.4f} "
      f"min={drug_feat.min():.3f} max={drug_feat.max():.3f}")
    L(f"  gene ESM-2 static: mean={gene_feat.mean():.4f} std={gene_feat.std():.4f} "
      f"min={gene_feat.min():.3f} max={gene_feat.max():.3f} (range [-12,12]? "
      f"{_ok(gene_feat.min() >= -12 and gene_feat.max() <= 12)})")
    L(f"  pathway_activity: shape={tuple(clf['pathway_activity'].shape)} mean={clf['pathway_activity'].mean():.4f} "
      f"std={clf['pathway_activity'].std():.4f}")
    L("")

    # ===== 8. 通路活性对齐 =====
    L("## 8. 通路活性对齐 (与 ssGSEA 输出 vs canonical 顺序)")
    # pathway_activity in clf was reordered to canonical; pw_orig was generated against expression.csv row order
    # verify: for every canonical cell, row matches the original at the corresponding expr-row position
    import pandas as pd
    expr = pd.read_csv(PROCESSED / "cell_line_omics" / "expression.csv", index_col=0)
    expr_index = list(expr.index)
    canon_pos_of_expr = {c: i for i, c in enumerate(canon)}
    aligned_ok = True
    pw_clf = clf["pathway_activity"].numpy()
    for i, c in enumerate(expr_index):
        cp = canon_pos_of_expr[c]
        if not np.allclose(pw_clf[cp], pw_orig[i]):
            aligned_ok = False
            break
    L(f"  pathway shape: clf={pw_clf.shape} orig={pw_orig.shape} | aligned to canonical: {_ok(aligned_ok)}")
    L(f"  pathway Z-score mean~0 std~1: {_ok(abs(pw_clf.mean()) < 1e-3 and abs(pw_clf.std() - 1.0) < 0.05)}")
    L("")

    # ===== 9. GPU 内存估算（策略 B） =====
    L("## 9. GPU 内存估算（策略 B：静态基图常驻 + batch 动态注入）")
    g_bytes = gene_feat.numel() * 4
    d_bytes = drug_feat.numel() * 4
    ppi_ei_bytes = ppi_ei.numel() * 8
    ppi_ew_bytes = ppi_ew.numel() * 4
    dti_ei_bytes = dti_ei.numel() * 8
    dti_ew_bytes = dti_ew.numel() * 4
    clf_bytes = sum(clf[k].numel() * 4 for k in ["expression", "mutation", "copynumber", "methylation", "pathway_activity"])
    total = g_bytes + d_bytes + ppi_ei_bytes + ppi_ew_bytes + dti_ei_bytes + dti_ew_bytes + clf_bytes
    L(f"  gene static   : {g_bytes/1e6:>7.2f} MB")
    L(f"  drug features : {d_bytes/1e6:>7.2f} MB")
    L(f"  PPI edge_index: {ppi_ei_bytes/1e6:>7.2f} MB  weight: {ppi_ew_bytes/1e6:>7.2f} MB")
    L(f"  DTI edge_index: {dti_ei_bytes/1e6:>7.2f} MB  weight: {dti_ew_bytes/1e6:>7.2f} MB")
    L(f"  cell_line_features (5 omics): {clf_bytes/1e6:>7.2f} MB")
    L(f"  ----------------------------------")
    L(f"  total static常驻             : {total/1e6:>7.2f} MB")
    L(f"  (训练激活另算；策略 B 下无 batch × 整图 clone，估算远低于方案 A)")
    L("")

    # ===== summary =====
    L("## 10. 最终结论")
    all_ok = (
        int(ppi_ei.max()) < n_genes and int(ppi_ei.min()) >= 0
        and int(dti_ei[0].max()) < n_drugs and int(dti_ei[1].max()) < n_genes
        and not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)
        and ldo_ok and lco_ok and n_nan == 0 and ok_label
        and max(js_tr, js_va, js_te) < 0.05
        and n_iso / n_genes <= 0.05
        and aligned_ok
    )
    L(f"  All critical checks passed: {_ok(all_ok)}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"Report written to {OUT}")
    print(f"  Overall: {'OK' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()