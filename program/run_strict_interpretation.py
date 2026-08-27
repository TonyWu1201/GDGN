from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.strict_eval.strict_interpretation import run_strict_interpretation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict M0-M4 attribution and intervention tests")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-drug-samples", type=int, default=5)
    parser.add_argument("--max-drugs", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--n-attention-random", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    written = run_strict_interpretation(
        args.run_dir, args.output_dir, args.min_drug_samples, args.max_drugs,
        args.n_steps, args.n_attention_random, args.smoke,
    )
    print(f"wrote {len(written)} cases to {args.output_dir}")


if __name__ == "__main__":
    main()
