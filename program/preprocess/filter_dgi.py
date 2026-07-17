import pickle
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

raw_path = PROJECT_ROOT / "data" / "raw" / "drug_gene_interaction" / "interactions.tsv"
cid_map_path = PROJECT_ROOT / "data" / "processed" / "drug_structures" / "dgidb_to_cid.csv"
output_dir = PROJECT_ROOT / "data" / "processed" / "drug_gene_interaction"
output_dir.mkdir(parents=True, exist_ok=True)
filtered_path = output_dir / "interactions_filtered.csv"
matrix_path = output_dir / "interaction_matrix.csv"
list_path = output_dir / "interaction_list.pkl"

df = pd.read_csv(raw_path, sep="\t")
print(f"Total interactions: {len(df)}")

df = df.dropna(subset=["interaction_score"]).copy()
print(f"After dropping NaN scores: {len(df)}")

scores = df["interaction_score"].values.astype(np.float64)
log_scores = np.log1p(scores)

log_min = log_scores.min()
log_max = log_scores.max()
print(f"Log-score range: [{log_min:.4f}, {log_max:.4f}]")

norm_scores = (log_scores - log_min) / (log_max - log_min)
print(f"Normalized score range: [{norm_scores.min():.4f}, {norm_scores.max():.4f}]")

score_threshold = 0.175
print(f"raw_score > {score_threshold:.4f}")

mask = scores > score_threshold
df = df[mask].copy()
df["score_norm"] = norm_scores[mask]
print(f"After score > {score_threshold} filter: {len(df)}")

cid_map = pd.read_csv(cid_map_path).dropna(subset=["cid"])
cid_map = cid_map[["dgidb_name", "cid"]].drop_duplicates()
print(f"DGIdb->CID map: {len(cid_map)} rows, {cid_map['cid'].nunique()} unique CIDs")

df = df.merge(cid_map, left_on="drug_name", right_on="dgidb_name", how="left")
df = df.dropna(subset=["cid"]).copy()
df["cid"] = df["cid"].astype(int)
print(f"After mapping to CID: {len(df)} rows, {df['cid'].nunique()} drugs, {df['gene_name'].nunique()} genes")

agg = df.groupby(["cid", "gene_name"], as_index=False).agg(
    score_norm=("score_norm", "max"),
    interaction_score=("interaction_score", "max"),
    drug_name=("drug_name", lambda s: "; ".join(sorted(set(s)))),
    gene_name=("gene_name", "first"),
)
print(f"Unique (CID, gene) pairs after max aggregation: {len(agg)}")

agg.to_csv(filtered_path, index=False)
print(f"Saved filtered long-form to {filtered_path}")

matrix = agg.pivot_table(
    index="cid",
    columns="gene_name",
    values="score_norm",
    aggfunc="first",
    fill_value=0.0,
)
matrix.index = [int(i) for i in matrix.index]
matrix.index.name = "cid"
print(f"Interaction matrix: {matrix.shape[0]} drugs x {matrix.shape[1]} genes")
print(f"Non-zero entries: {(matrix > 0).sum().sum()} / {matrix.size}")

matrix.to_csv(matrix_path)
print(f"Saved matrix to {matrix_path}")

interaction_list = [
    [row.gene_name, int(row.cid), float(row.score_norm)]
    for row in agg.itertuples(index=False)
]
print(f"Interaction list length: {len(interaction_list)}")
with open(list_path, "wb") as f:
    pickle.dump(interaction_list, f)
print(f"List saved to {list_path}")
