import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GMT_PATH = PROJECT_ROOT / "data" / "raw" / "pathway" / "msigdb_v2026.1.Hs_files_to_download_locally" / "msigdb_v2026.1.Hs_GMTs" / "c2.cp.kegg_legacy.v2026.1.Hs.symbols.gmt"
EXPR_PATH = PROJECT_ROOT / "data" / "processed" / "cell_line_omics" / "expression.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "driver&pathway"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "pathway_gene_sets.json"
OUTPUT_FULL_PATH = OUTPUT_DIR / "pathway_gene_sets_full.json"

MIN_GENES = 10


def load_expression_genes(path: Path) -> set[str]:
    import pandas as pd

    df = pd.read_csv(path, nrows=0)
    genes = [c for c in df.columns if c != "cell_line_name"]
    return set(genes)


def parse_gmt(path: Path) -> dict[str, list[str]]:
    pathways: dict[str, list[str]] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            pathway_name = parts[0]
            genes = parts[2:]
            pathways[pathway_name] = genes
    return pathways


def main():
    expr_genes = load_expression_genes(EXPR_PATH)
    print(f"Expression matrix genes: {len(expr_genes)}")

    all_pathways = parse_gmt(GMT_PATH)
    print(f"Total pathways in GMT: {len(all_pathways)}")

    filtered: dict[str, list[str]] = {}
    full_info: dict[str, dict] = {}

    for name, genes in all_pathways.items():
        overlap = [g for g in genes if g in expr_genes]
        overlap_count = len(overlap)
        full_info[name] = {
            "total_genes": len(genes),
            "overlap_genes": overlap_count,
            "genes": genes,
        }
        if overlap_count >= MIN_GENES:
            filtered[name] = overlap

    print(f"Pathways with >= {MIN_GENES} expression genes: {len(filtered)}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_FULL_PATH, "w") as f:
        json.dump(full_info, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Saved to: {OUTPUT_FULL_PATH}")


if __name__ == "__main__":
    main()
