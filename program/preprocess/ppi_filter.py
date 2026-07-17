import pickle
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ppi_raw = PROJECT_ROOT / "data" / "raw" / "protein_protein_interaction" / "9606.protein.links.v12.0.txt"
alias_raw = PROJECT_ROOT / "data" / "raw" / "protein_protein_interaction" / "9606.protein.aliases.v12.0.txt"
dg_filtered = PROJECT_ROOT / "data" / "processed" / "drug_gene_interaction" / "interactions_filtered.csv"
output_dir = PROJECT_ROOT / "data" / "processed" / "protein_protein_interaction"
output_dir.mkdir(parents=True, exist_ok=True)

SCORE_THRESHOLD = 700

MAX_HOP = 1

print("Building STRING ID -> HGNC symbol mapping ...")
id_to_symbol = {}
for chunk in pd.read_csv(
    alias_raw, sep="\t", comment="#",
    names=["string_protein_id", "alias", "source"],
    chunksize=500000,
):
    hs = chunk[chunk["source"] == "Ensembl_HGNC_symbol"]
    for pid, alias in zip(hs["string_protein_id"], hs["alias"]):
        if pid not in id_to_symbol:
            id_to_symbol[pid] = alias
print(f"Mapping built: {len(id_to_symbol)} unique STRING IDs mapped to HGNC symbols")

print(f"Filtering PPI with combined_score > {SCORE_THRESHOLD} ...")
filtered_rows = []
total = 0
for chunk in pd.read_csv(ppi_raw, sep=" ", chunksize=1000000):
    total += len(chunk)
    high_score = chunk[chunk["combined_score"] > SCORE_THRESHOLD]
    filtered_rows.append(high_score)
    if total % 5000000 == 0:
        print(f"  Processed {total:,} rows ...")

ppi = pd.concat(filtered_rows, ignore_index=True)
print(f"Total PPI rows: {total:,}, with score > {SCORE_THRESHOLD}: {len(ppi):,}")

print("Mapping protein IDs to HGNC symbols ...")
ppi["gene1"] = ppi["protein1"].map(id_to_symbol)
ppi["gene2"] = ppi["protein2"].map(id_to_symbol)

before = len(ppi)
ppi = ppi.dropna(subset=["gene1", "gene2"])
print(f"Mapped both proteins: {len(ppi):,} (dropped {before - len(ppi):,} unmappable)")

self_loop_mask = ppi["gene1"] == ppi["gene2"]
if self_loop_mask.any():
    print(f"Dropping {self_loop_mask.sum()} self-loops")
    ppi = ppi[~self_loop_mask]

ppi = ppi[["gene1", "gene2", "combined_score"]].copy()

csv_path = output_dir / "ppi_filtered.csv"
ppi.to_csv(csv_path, index=False)
print(f"Saved to {csv_path}")

ppi_list = ppi.values.tolist()
list_path = output_dir / "ppi_list.pkl"
with open(list_path, "wb") as f:
    pickle.dump(ppi_list, f)
print(f"PPI list saved to {list_path}, length: {len(ppi_list)}")

print("\n=== Filtering PPI by drug-gene associations (MAX_HOP = %d) ===" % MAX_HOP)

print("Loading drug-gene interactions ...")
dg = pd.read_csv(dg_filtered)
dg_genes = set(dg["gene_name"].dropna())
print(f"Drug-target genes: {len(dg_genes)}")

ppi_edges = ppi[["gene1", "gene2"]].values.tolist()
adj = {}
for u, v in ppi_edges:
    adj.setdefault(u, set()).add(v)
    adj.setdefault(v, set()).add(u)
print(f"PPI graph: {len(adj)} nodes, {len(ppi_edges)} edges")

seed_genes = dg_genes & set(adj)
print(f"Seed genes (DG genes in PPI): {len(seed_genes)}")

all_nodes = set(seed_genes)
frontier = set(seed_genes)
for hop in range(MAX_HOP):
    next_nodes = set()
    for node in frontier:
        next_nodes.update(adj.get(node, set()))
    frontier = next_nodes - all_nodes
    all_nodes.update(frontier)
    print(f"  Hop {hop + 1}: added {len(frontier):,} nodes, total {len(all_nodes):,}")

ppi_dg = ppi[ppi["gene1"].isin(all_nodes) & ppi["gene2"].isin(all_nodes)]
in_dg_count = (ppi_dg["gene1"].isin(seed_genes) & ppi_dg["gene2"].isin(seed_genes)).sum()
print(f"PPI edges within DG-anchored subgraph: {len(ppi_dg):,} "
      f"(both sides DG: {in_dg_count:,})")

csv_path = output_dir / "ppi_dg_filtered.csv"
ppi_dg.to_csv(csv_path, index=False)
print(f"Saved to {csv_path}")

ppi_dg_list = ppi_dg.values.tolist()
list_path = output_dir / "ppi_dg_list.pkl"
with open(list_path, "wb") as f:
    pickle.dump(ppi_dg_list, f)
print(f"DG-filtered PPI list saved to {list_path}, length: {len(ppi_dg_list)}")
