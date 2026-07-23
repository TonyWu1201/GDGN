"""
Step 2: 异构图构建
=================
节点：gene (8412) + drug (184)
边类型：(gene, ppi, gene) + (drug, targets, gene)

产出（写入 data/model/hetero_graph/ 与 data/model/drug_encoder/）：
  - core_gene_order.txt            8412 行核心基因符号列表
  - core_gene_idx_in_esm.npy       8412 行 ESM 矩阵中的行索引 (int64)
  - core_gene_mask.pt              (15632,) 布尔 mask
  - gene_static_features.pt        (8412, 1280) ESM-2 子集
  - drug_features.pt               (184, 1030) ECFP4(1024) + 理化(6) Z-score
  - ppi_edge_index.pt              (2, 2*E_ppi) 双向
  - ppi_edge_weight.pt
  - dti_edge_index.pt              (2, E_dti) 单向 (drug -> gene)
  - dti_edge_weight.pt
  - hetero_graph_base.pt           PyG HeteroData 组装结果

核心基因集（候选 A, 8412）：ESM-2 ∩ (PPI_genes ∪ DTI_targets) ∩ expression_genes
  排序：CGC 驱动基因优先（保 8412 内 CGC 在前），其余按字母序
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors
from rdkit import RDLogger
from torch_geometric.data import HeteroData

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data"
PROCESSED = DATA / "processed"
RAW = DATA / "raw"

ESM_NPY = PROCESSED / "gene_embeddings" / "filtered_esm2_gene_embeddings.npy"
ESM_ORDER = PROCESSED / "gene_embeddings" / "filtered_gene_order.txt"
DRIVER_TXT = PROCESSED / "driver&pathway" / "driver_genes.txt"
EXPR_CSV = PROCESSED / "cell_line_omics" / "expression.csv"
PPI_CSV = PROCESSED / "protein_protein_interaction" / "ppi_dg_filtered.csv"
DTI_MATRIX_CSV = PROCESSED / "drug_gene_interaction" / "interaction_matrix.csv"
FEATS_PKL = PROCESSED / "drug_structures" / "feats.pkl"
SMILES_CSV = RAW / "drug_structures" / "compound_cid_smiles.csv"

OUT_GRAPH = DATA / "model" / "hetero_graph"
OUT_DRUG = DATA / "model" / "drug_encoder"
OUT_GRAPH.mkdir(parents=True, exist_ok=True)
OUT_DRUG.mkdir(parents=True, exist_ok=True)

CELL_IDX_JSON = PROCESSED / "cell_line_to_idx.json"
DRUG_IDX_JSON = PROCESSED / "drug_cid_to_idx.json"


# ---------------- Step 2a: 核心基因集 ----------------

def determine_core_gene_set() -> tuple[list[str], np.ndarray, torch.Tensor]:
    """Returns (core_gene_order, indices_into_ESM, mask_over_ESM)."""
    esm_order: list[str] = ESM_ORDER.read_text().splitlines()
    esm_set: set[str] = set(esm_order)
    name_to_idx = {g: i for i, g in enumerate(esm_order)}

    driver_set: set[str] = set(DRIVER_TXT.read_text().splitlines())
    ppi = pd.read_csv(PPI_CSV)
    ppi_genes: set[str] = set(ppi["gene1"].astype(str)) | set(ppi["gene2"].astype(str))
    dti = pd.read_csv(DTI_MATRIX_CSV, index_col=0)
    dti_targets: set[str] = set(dti.columns.astype(str))
    # expression gene symbols (header row only)
    expr_header = pd.read_csv(EXPR_CSV, index_col=0, nrows=1)
    expr_genes: set[str] = set(expr_header.columns.astype(str))

    network_genes = ppi_genes | dti_targets
    candidate = (esm_set & network_genes & expr_genes) or (esm_set & network_genes)
    print(f" ESM={len(esm_set)} PPI={len(ppi_genes)} DTI_targets={len(dti_targets)} expr={len(expr_genes)}")
    print(f" network(PPI∪DTI)={len(network_genes)} | ESM∩network={len(esm_set & network_genes)} | +expr={len(esm_set & network_genes & expr_genes)}")

    in_driver = sorted(candidate & driver_set, key=str.upper)
    rest = sorted(candidate - driver_set, key=str.upper)
    core_order = in_driver + rest

    core_idx_in_esm = np.array([name_to_idx[g] for g in core_order], dtype=np.int64)
    mask = torch.zeros(len(esm_order), dtype=torch.bool)
    mask[core_idx_in_esm] = True

    print(f" core_gene_set size: {len(core_order)} (driver first: {len(in_driver)}, rest: {len(rest)})")
    return core_order, core_idx_in_esm, mask


# ---------------- Step 2b: 基因节点静态特征 ----------------

def build_gene_static_features(core_idx_in_esm: np.ndarray) -> torch.Tensor:
    esm = np.load(ESM_NPY)  # (15632, 1280) float32
    x = torch.from_numpy(esm[core_idx_in_esm]).float()  # (N_core, 1280)
    print(f" gene_static_features: {tuple(x.shape)} dtype={x.dtype}")
    return x


# ---------------- Step 2c: 药物特征 ----------------

def _drug_phychem(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(6, dtype=np.float32)
    feats = [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Chem.rdMolDescriptors.CalcNumRotatableBonds(mol),
    ]
    return np.asarray(feats, dtype=np.float32)


def _drug_ecfp(smiles: str, n_bits: int = 1024) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


def build_drug_features(drug_cid_to_idx: dict) -> tuple[torch.Tensor, dict]:
    """ECFP + 理化 -> (184, 1030) Z-score. Returns (features,DrugEcFailures)."""
    smiles_df = pd.read_csv(SMILES_CSV)
    smiles_df["cid"] = smiles_df["cid"].astype(str)
    cid_to_smiles = dict(zip(smiles_df["cid"], smiles_df["smiles"]))

    n_drugs = len(drug_cid_to_idx)
    # cid(str) -> drug_idx(int)
    dim = 1024 + 6
    mat = np.zeros((n_drugs, dim), dtype=np.float32)
    failures: dict[str, str] = {}

    for cid, idx in drug_cid_to_idx.items():
        smi = cid_to_smiles.get(cid)
        if smi is None or pd.isna(smi):
            failures[cid] = "no SMILES"
            continue
        ecfp = _drug_ecfp(smi)
        phy = _drug_phychem(smi)
        mat[idx] = np.concatenate([ecfp, phy])
        if ecfp.sum() == 0:
            failures[cid] = "ECFP all-zero (parse fail)"

    # Z-score across drugs
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    mat = (mat - mu) / sd

    if failures:
        print(f" [WARN] {len(failures)} drug feature failures: {[f'{k}:{v}' for k,v in failures.items()][:5]}")
    return torch.from_numpy(mat).float(), failures


def reorganize_mol_graphs(drug_cid_to_idx: dict) -> dict:
    """feats.pkl keyed by CID(int) -> dict keyed by drug_idx(int)."""
    with open(FEATS_PKL, "rb") as f:
        feats = pickle.load(f)
    # feats keys are int CIDs (drug_feat.py iterated drug.cid which is int)
    out: dict[int, list] = {}
    mismatch = []
    for k, v in feats.items():
        cid_s = str(k)
        if cid_s not in drug_cid_to_idx:
            mismatch.append(cid_s)
            continue
        out[drug_cid_to_idx[cid_s]] = v
    if mismatch:
        print(f" [WARN] {len(mismatch)} mol-graphs whose CID not in ic50 (ignored): {mismatch[:5]}")
    print(f" drug_mol_graphs: {len(out)} entries | atoms[0] shape: "
          f"{getattr(feats[list(feats.keys())[0]][0], 'shape', 'NA')}")
    return out


# ---------------- Step 2d: PPI 边 ----------------

def build_ppi_edges(core_order: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    name_to_idx = {g: i for i, g in enumerate(core_order)}
    df = pd.read_csv(PPI_CSV)

    g1 = df["gene1"].astype(str).values
    g2 = df["gene2"].astype(str).values
    score = df["combined_score"].astype(np.float32).values

    keep = np.zeros(len(df), dtype=bool)
    who1 = np.zeros(len(df), dtype=np.int64)
    who2 = np.zeros(len(df), dtype=np.int64)
    for i in range(len(df)):
        if g1[i] in name_to_idx and g2[i] in name_to_idx:
            keep[i] = True
            who1[i] = name_to_idx[g1[i]]
            who2[i] = name_to_idx[g2[i]]
    src = who1[keep]
    dst = who2[keep]
    w = score[keep] / 1000.0
    edges = np.stack([src, dst], axis=0)
    # reverse (add the other direction)
    both_edges = np.concatenate([edges, edges[[1, 0]]], axis=1)
    both_w = np.concatenate([w, w], axis=0)
    # remove self-loops
    mask_no_sl = both_edges[0] != both_edges[1]
    both_edges = both_edges[:, mask_no_sl]
    both_w = both_w[mask_no_sl]
    eidx = torch.from_numpy(both_edges).long()
    ew = torch.from_numpy(both_w).float()
    print(f" PPI: kept {keep.sum()}/{len(df)} -> bidirectional+no-self-loop edge_index {tuple(eidx.shape)}")
    return eidx, ew, eidx.unique()


# ---------------- Step 2e: DTI 边 ----------------

def build_dti_edges(
    core_order: list[str],
    drug_cid_to_idx: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    name_to_idx = {g: i for i, g in enumerate(core_order)}
    imat = pd.read_csv(DTI_MATRIX_CSV, index_col=0)
    # rows are CID (int64) -> str
    edges_src = []
    edges_dst = []
    edges_w = []
    n_drugs_in_dti = 0
    for cid, row in imat.iterrows():
        cid_s = str(cid)
        if cid_s not in drug_cid_to_idx:
            continue
        di = drug_cid_to_idx[cid_s]
        n_drugs_in_dti += 1
        # iterate only positive cells
        pos = row[row > 0]
        for gname, val in pos.items():
            gn = str(gname)
            if gn in name_to_idx:
                edges_src.append(di)
                edges_dst.append(name_to_idx[gn])
                edges_w.append(float(val))

    eidx = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    ew = torch.tensor(edges_w, dtype=torch.float32)
    print(f" DTI drugs covered: {n_drugs_in_dti}/{len(drug_cid_to_idx)} | edges (after core filter): {eidx.shape[1]}")
    return eidx, ew


# ---------------- Step 2f: 组装 HeteroData ----------------

def assemble_hetero(
    n_drugs: int,
    gene_feat: torch.Tensor,
    drug_feat: torch.Tensor,
    ppi_eidx: torch.Tensor,
    ppi_ew: torch.Tensor,
    dti_eidx: torch.Tensor,
    dti_ew: torch.Tensor,
    extra_meta: dict,
) -> HeteroData:
    data = HeteroData()
    data["gene"].x = gene_feat.contiguous()
    data["gene"].num_nodes = int(gene_feat.shape[0])
    data["drug"].x = drug_feat.contiguous()
    data["drug"].num_nodes = int(n_drugs)
    data["gene", "ppi", "gene"].edge_index = ppi_eidx.contiguous()
    data["gene", "ppi", "gene"].edge_weight = ppi_ew.contiguous()
    data["drug", "targets", "gene"].edge_index = dti_eidx.contiguous()
    data["drug", "targets", "gene"].edge_weight = dti_ew.contiguous()
    # attach metadata for downstream reference
    data.meta = extra_meta
    return data


# ---------------- main ----------------

def main():
    drug_idx_to_cid: dict = json.loads(DRUG_IDX_JSON.read_text())
    cid_to_idx = {k: int(v) for k, v in drug_idx_to_cid.items()}
    n_drugs = len(cid_to_idx)
    print(f" n_drugs = {n_drugs}")

    # 2a core gene set
    core_order, core_idx, mask = determine_core_gene_set()
    (OUT_GRAPH / "core_gene_order.txt").write_text("\n".join(core_order) + "\n")
    np.save(OUT_GRAPH / "core_gene_idx_in_esm.npy", core_idx)
    torch.save(mask, OUT_GRAPH / "core_gene_mask.pt")

    # 2b gene static features
    gene_feat = build_gene_static_features(core_idx)
    torch.save(gene_feat, OUT_GRAPH / "gene_static_features.pt")

    # 2c drug features (ECFP + physchem) + mol graphs
    drug_feat, drug_failures = build_drug_features(cid_to_idx)
    torch.save(drug_feat, OUT_GRAPH / "drug_features.pt")
    mol_graphs = reorganize_mol_graphs(cid_to_idx)
    with open(OUT_DRUG / "drug_mol_graphs.pkl", "wb") as f:
        pickle.dump(mol_graphs, f)

    # 2d PPI edges
    ppi_eidx, ppi_ew, _ = build_ppi_edges(core_order)
    torch.save(ppi_eidx, OUT_GRAPH / "ppi_edge_index.pt")
    torch.save(ppi_ew, OUT_GRAPH / "ppi_edge_weight.pt")

    # 2e DTI edges
    dti_eidx, dti_ew = build_dti_edges(core_order, cid_to_idx)
    torch.save(dti_eidx, OUT_GRAPH / "dti_edge_index.pt")
    torch.save(dti_ew, OUT_GRAPH / "dti_edge_weight.pt")

    # 2f assemble
    meta = {
        "n_genes": int(len(core_order)),
        "n_drugs": int(n_drugs),
        "gene_feat_dim": int(gene_feat.shape[1]),
        "drug_feat_dim": int(drug_feat.shape[1]),
        "ppi_edge_count": int(ppi_eidx.shape[1]),
        "dti_edge_count": int(dti_eidx.shape[1]),
        "drug_feature_failures": drug_failures,
        "feature_pipeline": "ECFP4(1024, r=2) + 6 physchem, Z-score across 184 drugs; mol_graphs from DeepChem ConvMolFeaturizer",
    }
    data = assemble_hetero(
        n_drugs=n_drugs,
        gene_feat=gene_feat,
        drug_feat=drug_feat,
        ppi_eidx=ppi_eidx,
        ppi_ew=ppi_ew,
        dti_eidx=dti_eidx,
        dti_ew=dti_ew,
        extra_meta=meta,
    )
    torch.save(data, OUT_GRAPH / "hetero_graph_base.pt")
    print("\n== Summary ==")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Saved hetero_graph_base.pt to {OUT_GRAPH}")
    print("Step 2 done.")


if __name__ == "__main__":
    main()