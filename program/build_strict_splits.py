from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.strict_eval.constants import PROTOCOLS, SPLITS_DIR
from program.strict_eval.splits import SplitOptions, build_all_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe GDGN evaluation splits")
    parser.add_argument("--output-root", type=Path, default=SPLITS_DIR)
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=list(PROTOCOLS))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-similarity-control", action="store_true")
    parser.add_argument("--drug-clusters", type=int, default=60)
    parser.add_argument("--cell-clusters", type=int, default=60)
    args = parser.parse_args()
    options = SplitOptions(
        n_folds=args.n_folds,
        seed=args.seed,
        similarity_control=not args.no_similarity_control,
        n_drug_clusters=args.drug_clusters,
        n_cell_clusters=args.cell_clusters,
    )
    outputs = build_all_splits(args.output_root, options, args.protocols)
    for protocol, directory in outputs.items():
        print(f"[split] {protocol}: {directory}")


if __name__ == "__main__":
    main()
