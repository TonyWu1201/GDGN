from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.strict_eval.aggregate import aggregate_experiments
from program.strict_eval.constants import EXPERIMENTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--hierarchical-bootstrap-runs", type=int, default=500)
    args = parser.parse_args()
    for path in aggregate_experiments(
        args.root, args.output_dir, args.hierarchical_bootstrap_runs
    ):
        print(path)


if __name__ == "__main__":
    main()
