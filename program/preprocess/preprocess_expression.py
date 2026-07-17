import pandas as pd
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

common_cl_path = PROJECT_ROOT / "data" / "common_cell_lines.csv"
expr_raw_path = PROJECT_ROOT / "data" / "raw" / "cell_line_omics" / "DepMap_ExpressionTPMLogp1HumanProteinCodingGenes.csv"
output_dir = PROJECT_ROOT / "data" / "processed" / "cell_line_omics"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "expression_processed.csv"

common_cl = pd.read_csv(common_cl_path)
depmap_to_name = dict(zip(common_cl["depMapID"], common_cl["Name"]))

expr = pd.read_csv(expr_raw_path)

expr = expr[expr["ModelID"].isin(depmap_to_name)]
expr = expr[expr["IsDefaultEntryForModel"] == "Yes"].reset_index(drop=True)

expr.insert(0, "cell_line_name", expr["ModelID"].map(depmap_to_name))

meta_cols = ["", "SequencingID", "ModelConditionID", "ModelID", "IsDefaultEntryForMC", "IsDefaultEntryForModel"]
expr = expr.drop(columns=meta_cols, errors="ignore")

def clean_gene_name(col: str) -> str:
    return re.sub(r"\s*\(\d+\)$", "", col)

expr = expr.rename(columns=clean_gene_name)

expr = expr.drop(columns=["Unnamed: 0"], errors="ignore")

expr.to_csv(output_path, index=False)

print(f"Output shape: {expr.shape[0]} rows x {expr.shape[1] - 1} columns")
print(f"Saved to: {output_path}")
