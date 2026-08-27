import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

common_cl_path = PROJECT_ROOT / "data" / "common_cell_lines.csv"
mut_raw_path = PROJECT_ROOT / "data" / "raw" / "cell_line_omics" / "DepMap_SomaticMutations.csv"
output_dir = PROJECT_ROOT / "data" / "processed" / "cell_line_omics"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "mutation.csv"

common_cl = pd.read_csv(common_cl_path)
depmap_to_name = dict(zip(common_cl["depMapID"], common_cl["Name"]))

CHUNK_SIZE = 100_000
USECOLS = ["ModelID", "HugoSymbol", "VepImpact"]

mutated_pairs = set()

for chunk in pd.read_csv(mut_raw_path, usecols=USECOLS, chunksize=CHUNK_SIZE):
    chunk = chunk[chunk["ModelID"].isin(depmap_to_name)]
    chunk = chunk[chunk["VepImpact"].isin(["HIGH", "MODERATE"])]
    for _, row in chunk.iterrows():
        mutated_pairs.add((row["ModelID"], row["HugoSymbol"]))

cell_lines = list(depmap_to_name.keys())
all_genes = sorted({gene for _, gene in mutated_pairs})

cl_to_name = {cl: depmap_to_name[cl] for cl in cell_lines}

matrix = pd.DataFrame(0, index=cell_lines, columns=all_genes)

for depmap_id, gene in mutated_pairs:
    if depmap_id in matrix.index and gene in matrix.columns:
        matrix.at[depmap_id, gene] = 1

matrix.index = matrix.index.map(cl_to_name)
matrix.index.name = "cell_line_name"
matrix.reset_index(inplace=True)

print(f"Output shape: {matrix.shape[0]} cell lines x {matrix.shape[1] - 1} genes")
print(f"Total mutation events (gene-level): {len(mutated_pairs)}")
print(f"Cell lines covered: {matrix.shape[0]}")

matrix.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")
