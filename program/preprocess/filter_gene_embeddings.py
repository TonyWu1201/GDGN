import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMB_DIR = PROJECT_ROOT / "data" / "processed" / "gene_embeddings"
EMB_NPY = EMB_DIR / "esm2_gene_embeddings.npy"
ORDER_TXT = EMB_DIR / "gene_order.txt"

DG_FILTERED = PROJECT_ROOT / "data" / "processed" / "drug_gene_interaction" / "interactions_filtered.csv"
PPI_FILTERED = PROJECT_ROOT / "data" / "processed" / "protein_protein_interaction" / "ppi_filtered.csv"

OUT_DIR = EMB_DIR
OUT_ORDER = OUT_DIR / "filtered_gene_order.txt"
OUT_EMB = OUT_DIR / "filtered_esm2_gene_embeddings.npy"

print("Loading gene_order.txt ...")
with open(ORDER_TXT, encoding="utf-8") as f:
    lines = [l.rstrip("\n") for l in f]
print(f"Total lines: {len(lines)}")

prefix_to_rows = {}
for idx, line in enumerate(lines):
    if not line:
        continue
    prefix = line.split(" ", 1)[0]
    key = prefix.lower()
    is_exact = " " not in line
    has_hgnc = "HGNC:" in line
    is_same_case = prefix == prefix.upper()
    prefix_to_rows.setdefault(key, []).append((idx, prefix, is_exact, has_hgnc, is_same_case))
print(f"Unique case-insensitive prefixes: {len(prefix_to_rows)}")

print("Loading target gene set from DTI + PPI ...")
dg = pd.read_csv(DG_FILTERED)
ppi = pd.read_csv(PPI_FILTERED)
target_genes = set(dg["gene_name"].dropna()) | set(ppi["gene1"]) | set(ppi["gene2"])
print(f"Target genes: {len(target_genes)}")

emb = np.load(EMB_NPY, mmap_mode="r")
assert emb.shape[0] == len(lines), f"Shape mismatch: {emb.shape[0]} vs {len(lines)}"

selected = []
missing = []
for gene in sorted(target_genes):
    rows = prefix_to_rows.get(gene.lower())
    if not rows:
        missing.append(gene)
        continue

    candidates = [(idx, prefix, is_exact, has_hgnc) for idx, prefix, is_exact, has_hgnc, _ in rows]

    def rank(c):
        idx, prefix, is_exact, has_hgnc = c
        return (
            prefix == gene,
            is_exact,
            has_hgnc,
            prefix == prefix.upper(),
            -idx,
        )

    candidates.sort(key=rank, reverse=True)
    selected.append((gene, candidates[0][0]))

print(f"Matched: {len(selected)} / {len(target_genes)}")
print(f"Missing: {len(missing)}")
if missing:
    print(f"  Sample missing: {missing[:20]}")

filtered_genes = [g for g, _ in selected]
filtered_indices = [i for _, i in selected]

filtered_emb = np.asarray(emb[filtered_indices], dtype=emb.dtype)
print(f"Filtered embedding matrix: {filtered_emb.shape}")

np.save(OUT_EMB, filtered_emb)
with open(OUT_ORDER, "w", encoding="utf-8") as f:
    for g in filtered_genes:
        f.write(f"{g}\n")

print(f"Saved gene order to {OUT_ORDER}")
print(f"Saved filtered embeddings to {OUT_EMB}")
