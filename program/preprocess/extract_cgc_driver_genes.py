import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CMC_PATH = PROJECT_ROOT / "data" / "raw" / "cancer_driver_mutations" / "cmc_export.tsv"
EXPR_PATH = PROJECT_ROOT / "data" / "processed" / "cell_line_omics" / "expression.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "driver&pathway"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "driver_genes.csv"
OUTPUT_TXT = OUTPUT_DIR / "driver_genes.txt"


def main():
    df = pd.read_csv(
        CMC_PATH,
        sep="\t",
        usecols=["GENE_NAME", "ONC_TSG", "CGC_TIER"],
        low_memory=False,
    )

    tier1 = df[df["CGC_TIER"] == 1.0].copy()
    print(f"Rows with CGC_TIER == 1: {len(tier1)}")

    driver_genes = tier1.groupby("GENE_NAME")["ONC_TSG"].agg(
        lambda x: "|".join(sorted(set(str(v) for v in x if str(v) != "nan")))
    ).reset_index()
    driver_genes.columns = ["gene_name", "onc_tsg"]
    driver_genes["onc_tsg"] = driver_genes["onc_tsg"].replace("", "unknown")
    print(f"Unique driver genes: {len(driver_genes)}")

    expr_df = pd.read_csv(EXPR_PATH, nrows=0)
    expr_genes = set(c for c in expr_df.columns if c != "cell_line_name")

    driver_genes["in_expression_matrix"] = driver_genes["gene_name"].isin(expr_genes)
    overlap_count = driver_genes["in_expression_matrix"].sum()
    print(f"Driver genes in expression matrix: {overlap_count}/{len(driver_genes)}")

    driver_genes.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_TXT, "w") as f:
        for gene in driver_genes["gene_name"]:
            f.write(gene + "\n")

    print(f"Saved to: {OUTPUT_CSV}")
    print(f"Saved to: {OUTPUT_TXT}")

    onc_tsg_counts = driver_genes["onc_tsg"].value_counts()
    print("\nONC_TSG distribution:")
    for k, v in onc_tsg_counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
