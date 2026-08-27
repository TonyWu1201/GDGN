from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.strict_eval.constants import PROTOCOLS, SPLITS_DIR
from program.strict_eval.splits import load_pair_table


def build_figures(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_pair_table()
    paths = []

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    drug_counts = pairs.groupby("drug_idx").size()
    cell_counts = pairs.groupby("cell_idx").size()
    axes[0, 0].hist(drug_counts, bins=25, color="#3b82f6", edgecolor="white")
    axes[0, 0].set(title="Observed pairs per drug", xlabel="Pair count", ylabel="Drugs")
    axes[0, 1].hist(cell_counts, bins=25, color="#10b981", edgecolor="white")
    axes[0, 1].set(title="Observed pairs per cell line", xlabel="Pair count", ylabel="Cell lines")
    axes[1, 0].hist(pairs["ic50"], bins=50, color="#8b5cf6", edgecolor="white")
    axes[1, 0].set(title="LN(IC50) distribution", xlabel="LN(IC50)", ylabel="Pairs")
    tissue_median = pairs.groupby("cancer_type")["ic50"].median().sort_values()
    axes[1, 1].barh(tissue_median.index, tissue_median.values, color="#f59e0b")
    axes[1, 1].set(title="Median response by tissue", xlabel="Median LN(IC50)")
    axes[1, 1].tick_params(axis="y", labelsize=6)
    figure.tight_layout()
    cohort_path = output_dir / "cohort-response-audit.pdf"
    figure.savefig(cohort_path, bbox_inches="tight")
    plt.close(figure)
    paths.append(cohort_path)

    figure, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for axis, protocol in zip(axes.flat, PROTOCOLS):
        path = SPLITS_DIR / protocol / "fold-00.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        mappings = payload.get("similarity_by_split", {}).get("test", {})
        drug = np.asarray(list(mappings.get("drug_max_to_train", {}).values()), dtype=float)
        cell = np.asarray(list(mappings.get("cell_max_to_train", {}).values()), dtype=float)
        if drug.size:
            axis.hist(drug, bins=np.linspace(0, 1, 21), alpha=0.65, label="drug")
        if cell.size:
            axis.hist(cell, bins=np.linspace(0, 1, 21), alpha=0.55, label="cell")
        axis.axvline(0.4, color="black", linestyle="--", linewidth=0.8)
        axis.axvline(0.7, color="black", linestyle=":", linewidth=0.8)
        axis.set_title(protocol)
        axis.set_xlim(0, 1.01)
        axis.legend(fontsize=7)
    figure.supxlabel("Maximum similarity to a training entity")
    figure.supylabel("Entity count")
    figure.tight_layout()
    similarity_path = output_dir / "split-similarity-audit.pdf"
    figure.savefig(similarity_path, bbox_inches="tight")
    plt.close(figure)
    paths.append(similarity_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/model/audit-figures"))
    args = parser.parse_args()
    for path in build_figures(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
