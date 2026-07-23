import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPR_PATH = PROJECT_ROOT / "data" / "processed" / "cell_line_omics" / "expression.csv"
PATHWAY_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "driver&pathway" / "pathway_gene_sets.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "driver&pathway"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACTIVITY_OUTPUT = OUTPUT_DIR / "pathway_activity.npy"
NAMES_OUTPUT = OUTPUT_DIR / "pathway_names.txt"


def ssgsea(
    expr: np.ndarray,
    gene_names: list[str],
    gene_sets: dict[str, list[str]],
    alpha: float = 0.25,
) -> np.ndarray:
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes, n_samples = expr.shape
    n_pathways = len(gene_sets)
    scores = np.zeros((n_samples, n_pathways), dtype=np.float64)

    pathway_names = list(gene_sets.keys())
    for p_idx, p_name in enumerate(pathway_names):
        gs_genes = gene_sets[p_name]
        gs_indices = np.array([gene_to_idx[g] for g in gs_genes if g in gene_to_idx], dtype=np.int64)
        if len(gs_indices) == 0:
            continue
        n_gs = len(gs_indices)
        n_not_gs = n_genes - n_gs
        decr = 1.0 / n_not_gs if n_not_gs > 0 else 0.0

        for s_idx in range(n_samples):
            sample_expr = expr[:, s_idx]
            sorted_idx = np.argsort(-sample_expr)
            ranks = np.argsort(sorted_idx)

            ranks_in_gs = ranks[gs_indices]
            r_alpha = np.power(np.abs(ranks_in_gs.astype(np.float64)), alpha)
            sum_r_alpha = r_alpha.sum()
            incr = r_alpha / sum_r_alpha if sum_r_alpha > 0 else np.zeros_like(r_alpha)

            is_in_gs = np.zeros(n_genes, dtype=np.float64)
            is_in_gs[gs_indices] = incr

            running_sum = 0.0
            max_es = 0.0
            min_es = 0.0
            for i in range(n_genes):
                idx = sorted_idx[i]
                if is_in_gs[idx] > 0:
                    running_sum += is_in_gs[idx]
                else:
                    running_sum -= decr
                if running_sum > max_es:
                    max_es = running_sum
                if running_sum < min_es:
                    min_es = running_sum

            es = max_es if abs(max_es) >= abs(min_es) else min_es
            scores[s_idx, p_idx] = es

    return scores


def main():
    expr_df = pd.read_csv(EXPR_PATH, index_col=0)
    cell_lines = expr_df.index.tolist()
    gene_names = expr_df.columns.tolist()
    expr = expr_df.to_numpy(dtype=np.float64).T
    print(f"Cell lines: {len(cell_lines)}, Genes: {len(gene_names)}")

    with open(PATHWAY_JSON_PATH, "r") as f:
        pathway_sets: dict[str, list[str]] = json.load(f)
    print(f"Pathway gene sets: {len(pathway_sets)}")

    activity = ssgsea(expr, gene_names, pathway_sets, alpha=0.25)
    print(f"Pathway activity matrix shape: {activity.shape}")

    pathway_names = list(pathway_sets.keys())

    mean = activity.mean(axis=0)
    std = activity.std(axis=0)
    std[std == 0] = 1.0
    activity = (activity - mean) / std
    print(f"Z-score normalized: mean={activity.mean():.6f}, std={activity.std():.6f}")

    np.save(ACTIVITY_OUTPUT, activity.astype(np.float32))

    with open(NAMES_OUTPUT, "w") as f:
        for name in pathway_names:
            f.write(name + "\n")

    print(f"Saved activity matrix to: {ACTIVITY_OUTPUT}")
    print(f"Saved pathway names to: {NAMES_OUTPUT}")


if __name__ == "__main__":
    main()
