from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.strict_eval.training import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one leakage-safe fold/seed experiment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variant")
    parser.add_argument("--graph-policy")
    parser.add_argument("--gene-feature-mode")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key, value in {
        "model_id": args.model_id, "fold": args.fold, "seed": args.seed,
        "output_dir": str(args.output_dir) if args.output_dir else None,
        "variant": args.variant, "graph_policy": args.graph_policy,
        "gene_feature_mode": args.gene_feature_mode,
    }.items():
        if value is not None:
            config[key] = value
    metrics = run_experiment(config, smoke=args.smoke)
    print(yaml.safe_dump(metrics, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
