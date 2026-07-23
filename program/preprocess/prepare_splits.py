"""
Step 1: 样本对划分
=================
产出（写入 data/processed/）：
  - cell_line_canonical_order.txt       404 行细胞系标准名顺序
  - cell_line_to_idx.json               name -> 0..403
  - drug_cid_to_idx.json               cid(str) -> 0..183
  - cell_line_cancer_types.json        name -> cancer_type (33~35 类)
  - sample_pairs_split.pt              {train, val, test} 每项 list of dicts
  - ldo_splits.pt                      Leave-One-Drug-Out 划分列表
  - lco_splits.pt                      Leave-One-Cancer-Out 划分列表
  - split_stats.txt                     各划分统计 + JS 散度

约定：
  - 细胞系标准顺序 = common_cell_lines.csv['Name']（404，已与 ic50_matrix 行序一致）
  - 药物标准顺序   = ic50_matrix.csv 列序（184，全部为 PubChem CID 字符串）
  - random_state=42，比例 80/10/10，分层键 cancer__drug_idx，稀有 <3 -> __other__
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data"
PROCESSED = DATA / "processed"
RAW = DATA / "raw"

IC50_PATH = PROCESSED / "drug_sensitivity" / "ic50_matrix.csv"
ANNO_PATH = RAW / "cell_line_omics" / "Cell_lines_annotations_20181226.txt"
COMMON_CL_PATH = DATA / "common_cell_lines.csv"

OUT_DIR = PROCESSED
RANDOM_STATE = 42
RARE_THRESHOLD = 3  # stratify key count < this -> __other__

# ---------- helpers ----------

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base-2) between two discrete distributions.
    p, q must be aligned over the same label set (raw counts OK).
    """
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / (b[mask] + 1e-12))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ---------- step 1.2 cancer type ----------

def get_cancer_types(cell_lines: list[str]) -> dict[str, str]:
    anno = pd.read_csv(ANNO_PATH, sep="\t")
    name_to_type = dict(zip(anno["Name"], anno["type"]))
    out = {}
    unknown = []
    for name in cell_lines:
        t = name_to_type.get(name, None)
        if isinstance(t, str) and pd.notna(t) and t != "":
            out[name] = t
        else:
            out[name] = "other"
            unknown.append(name)
    if unknown:
        print(f"[WARN] {len(unknown)} cell lines without cancer type; -> 'other': {unknown[:10]}")
    return out


# ---------- step 1.3 pair list ----------

def collect_pairs(ic50: pd.DataFrame, cell_idx: dict, drug_idx: dict, cancer_of: dict):
    pairs: list[dict] = []
    for cell_name, row in ic50.iterrows():
        cidx = cell_idx[cell_name]
        ctype = cancer_of[cell_name]
        for drug_cid, val in row.items():
            if pd.isna(val):
                continue
            pairs.append(
                {
                    "cell_idx": int(cidx),
                    "drug_idx": int(drug_idx[str(drug_cid)]),
                    "ic50": float(val),
                    "cancer_type": ctype,
                }
            )
    return pairs


# ---------- step 1.3 stratified split ----------

def stratified_split(pairs: list[dict]):
    # build stratify key = cancer__drug_idx
    keys = [f"{p['cancer_type']}__{p['drug_idx']}" for p in pairs]
    counter = Counter(keys)
    keys = ["__other__" if counter[k] < RARE_THRESHOLD else k for k in keys]

    idx_all = np.arange(len(pairs))
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=0.1, random_state=RANDOM_STATE, stratify=keys
    )
    # stratify on val split using same rare-merged keys restricted to trainval
    keys_trainval = [keys[i] for i in idx_trainval]
    # recompute rare within trainval to avoid single-sample strata after split
    sub_counter = Counter(keys_trainval)
    keys_trainval = [
        "__other__" if sub_counter[k] < 2 else k for k in keys_trainval
    ]
    idx_train_rel, idx_val_rel = train_test_split(
        np.arange(len(idx_trainval)),
        test_size=1 / 9,
        random_state=RANDOM_STATE,
        stratify=keys_trainval,
    )
    idx_train = idx_trainval[idx_train_rel]
    idx_val = idx_trainval[idx_val_rel]

    return (
        [pairs[i] for i in idx_train],
        [pairs[i] for i in idx_val],
        [pairs[i] for i in idx_test],
    )


# ---------- step 1.4 LODO / LOCO ----------

def build_ldo_splits(pairs: list[dict], n_drugs: int) -> list[dict]:
    by_drug: dict[int, list[int]] = {}
    for i, p in enumerate(pairs):
        by_drug.setdefault(p["drug_idx"], []).append(i)
    folds = []
    for d in range(n_drugs):
        test_idx = by_drug.get(d, [])
        train_idx = [i for i in range(len(pairs)) if i not in set(test_idx)]
        folds.append(
            {
                "held_out_drug_idx": d,
                "train": train_idx,
                "test": test_idx,
            }
        )
    return folds


def build_lco_splits(pairs: list[dict], cancer_of_cell: dict) -> list[dict]:
    """Leave-One-Cancer-Out: each fold holds out one cancer type's all cells.
    Only cancer types with >= 1 cell line produce a fold.
    """
    cancer_types = sorted({p["cancer_type"] for p in pairs})
    cell_idx_to_ctype = cancer_of_cell  # name -> type; we need idx -> type
    # build cell_idx -> cancer_type
    idx_to_ctype = {}
    for p in pairs:
        idx_to_ctype.setdefault(p["cell_idx"], p["cancer_type"])
    folds = []
    for ct in cancer_types:
        test_idx = [i for i, p in enumerate(pairs) if p["cancer_type"] == ct]
        if not test_idx:
            continue
        train_idx = [i for i in range(len(pairs)) if i not in set(test_idx)]
        folds.append(
            {
                "held_out_cancer_type": ct,
                "train": train_idx,
                "test": test_idx,
            }
        )
    return folds


# ---------- step 1 stats ----------

def write_stats(
    path: Path,
    pairs: list[dict],
    train: list[dict],
    val: list[dict],
    test: list[dict],
    cancer_types: list[str],
    n_drugs: int,
):
    lines = []
    lines.append("=== Sample Pair Split Statistics ===\n")
    lines.append(f"Total pairs: {len(pairs)}")
    lines.append(f"Train: {len(train)} ({len(train)/len(pairs):.3%})")
    lines.append(f"Val:   {len(val)}   ({len(val)/len(pairs):.3%})")
    lines.append(f"Test:  {len(test)}  ({len(test)/len(pairs):.3%})\n")

    # cancer distribution
    lines.append("Cancer-type distribution (count):")
    all_ct = Counter(p["cancer_type"] for p in pairs)
    tr_ct = Counter(p["cancer_type"] for p in train)
    va_ct = Counter(p["cancer_type"] for p in val)
    te_ct = Counter(p["cancer_type"] for p in test)
    lines.append(f"{'cancer_type':<28}{'all':>8}{'train':>8}{'val':>8}{'test':>8}")
    # Compute JS divergence over vector distributions
    vec_all = np.array([all_ct[c] for c in cancer_types], dtype=np.float64)
    vec_tr = np.array([tr_ct[c] for c in cancer_types], dtype=np.float64)
    vec_va = np.array([va_ct[c] for c in cancer_types], dtype=np.float64)
    vec_te = np.array([te_ct[c] for c in cancer_types], dtype=np.float64)
    js_tr = js_divergence(vec_tr, vec_all)
    js_va = js_divergence(vec_va, vec_all)
    js_te = js_divergence(vec_te, vec_all)
    for ct in cancer_types:
        lines.append(
            f"{ct:<28}{all_ct[ct]:>8}{tr_ct[ct]:>8}{va_ct[ct]:>8}{te_ct[ct]:>8}"
        )
    lines.append("")
    lines.append(f"JS(train, all) = {js_tr:.4f}")
    lines.append(f"JS(val,   all) = {js_va:.4f}")
    lines.append(f"JS(test,  all) = {js_te:.4f}")
    lines.append(f"(threshold < 0.05 -> {'OK' if max(js_tr, js_va, js_te) < 0.05 else 'WARN'})\n")

    # drug distribution
    all_d = Counter(p["drug_idx"] for p in pairs)
    tr_d = Counter(p["drug_idx"] for p in train)
    va_d = Counter(p["drug_idx"] for p in val)
    te_d = Counter(p["drug_idx"] for p in test)
    drugs_universe = list(range(n_drugs))
    v_all = np.array([all_d[i] for i in drugs_universe], dtype=np.float64)
    v_tr = np.array([tr_d[i] for i in drugs_universe], dtype=np.float64)
    v_va = np.array([va_d[i] for i in drugs_universe], dtype=np.float64)
    v_te = np.array([te_d[i] for i in drugs_universe], dtype=np.float64)
    lines.append(
        f"JS drug-distribution: train={js_divergence(v_tr, v_all):.4f} "
        f"val={js_divergence(v_va, v_all):.4f} test={js_divergence(v_te, v_all):.4f}\n"
    )

    # leakage check (no (cell, drug) overlap)
    def keyset(lst):
        return {(p["cell_idx"], p["drug_idx"]) for p in lst}
    s_tr, s_va, s_te = keyset(train), keyset(val), keyset(test)
    lines.append(f"Leakage: |train∩val|={len(s_tr & s_va)}, |train∩test|={len(s_tr & s_te)}, |val∩test|={len(s_va & s_te)}")

    path.write_text("\n".join(lines) + "\n")
    print(f"Stats written to {path}")
    print(f"  JS(train,all)={js_tr:.4f} JS(val,all)={js_va:.4f} JS(test,all)={js_te:.4f}")


# ---------- main ----------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ic50 = pd.read_csv(IC50_PATH, index_col=0)
    common = pd.read_csv(COMMON_CL_PATH)

    # canonical cell-line order
    cell_lines = common["Name"].astype(str).tolist()
    assert len(cell_lines) == 404, f"expected 404 cells, got {len(cell_lines)}"
    # verify cell lines ⊆ ic50 rows
    missing_in_ic50 = [c for c in cell_lines if c not in ic50.index]
    assert not missing_in_ic50, f"{len(missing_in_ic50)} cells missing in ic50 rows: {missing_in_ic50[:5]}"
    # canonical drug order = ic50 columns
    drug_cids = [str(c) for c in ic50.columns]
    assert len(drug_cids) == 184, f"expected 184 drugs, got {len(drug_cids)}"

    cell_idx = {name: i for i, name in enumerate(cell_lines)}
    drug_idx = {cid: i for i, cid in enumerate(drug_cids)}

    # cancer types restricted to 404
    cancer_of = get_cancer_types(cell_lines)
    cancer_universe = sorted(set(cancer_of.values()))

    print(f"Cell lines: {len(cell_lines)} | drugs: {len(drug_cids)} | cancer types: {len(cancer_universe)}")

    pairs = collect_pairs(ic50, cell_idx, drug_idx, cancer_of)
    print(f"Total non-NaN pairs: {len(pairs)}")

    # stratified split
    train, val, test = stratified_split(pairs)
    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    # leakage check (absolute guarantee: no pair index overlap)
    s_tr = {(p["cell_idx"], p["drug_idx"]) for p in train}
    s_va = {(p["cell_idx"], p["drug_idx"]) for p in val}
    s_te = {(p["cell_idx"], p["drug_idx"]) for p in test}
    assert not (s_tr & s_va), "train/val leakage"
    assert not (s_tr & s_te), "train/test leakage"
    assert not (s_va & s_te), "val/test leakage"
    print("Leakage check: OK (no (cell,drug) overlap)")

    # LODO / LOCO
    ldo = build_ldo_splits(pairs, n_drugs=len(drug_cids))
    lco = build_lco_splits(pairs, cancer_of_cell=cancer_of)
    print(f"LODO folds: {len(ldo)} | LOCO folds: {len(lco)}")

    # save tensors
    torch.save(
        {"train": train, "val": val, "test": test},
        OUT_DIR / "sample_pairs_split.pt",
    )
    torch.save(ldo, OUT_DIR / "ldo_splits.pt")
    torch.save(lco, OUT_DIR / "lco_splits.pt")

    # save json mappings
    (OUT_DIR / "cell_line_to_idx.json").write_text(json.dumps(cell_idx, ensure_ascii=False))
    (OUT_DIR / "drug_cid_to_idx.json").write_text(json.dumps(drug_idx, ensure_ascii=False))
    (OUT_DIR / "cell_line_cancer_types.json").write_text(
        json.dumps(cancer_of, ensure_ascii=False)
    )
    (OUT_DIR / "cell_line_canonical_order.txt").write_text("\n".join(cell_lines) + "\n")

    # stats
    write_stats(
        OUT_DIR / "split_stats.txt",
        pairs, train, val, test,
        cancer_universe, len(drug_cids),
    )

    print("Step 1 done.")


if __name__ == "__main__":
    main()