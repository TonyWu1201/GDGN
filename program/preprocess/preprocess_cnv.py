import pandas as pd
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

common_cl_path = PROJECT_ROOT / "data" / "common_cell_lines.csv"
cnv_raw_path = PROJECT_ROOT / "data" / "raw" / "cell_line_omics" / "DepMap_CNGeneWGS.csv"
output_dir = PROJECT_ROOT / "data" / "processed" / "cell_line_omics"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "copynumber.csv"

common_cl = pd.read_csv(common_cl_path)
depmap_to_name = dict(zip(common_cl["depMapID"], common_cl["Name"]))

cnv = pd.read_csv(cnv_raw_path)

cnv = cnv[cnv["ModelID"].isin(depmap_to_name)]

cnv.insert(0, "cell_line_name", cnv["ModelID"].map(depmap_to_name))

meta_cols = ["", "SequencingID", "ModelConditionID", "ModelID", "IsDefaultEntryForMC", "IsDefaultEntryForModel"]
cnv = cnv.drop(columns=meta_cols, errors="ignore")

def clean_gene_name(col: str) -> str:
    return re.sub(r"\s*\(\d+\)$", "", col)

cnv = cnv.rename(columns=clean_gene_name)

cnv = cnv.drop(columns=["Unnamed: 0"], errors="ignore")

cnv.to_csv(output_path, index=False)

print(f"Output shape: {cnv.shape[0]} rows x {cnv.shape[1]} columns")
print(f"Saved to: {output_path}")
