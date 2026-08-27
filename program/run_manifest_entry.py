from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.strict_eval.training import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one zero-based line from a JSONL run manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.index < 0:
        raise ValueError("index must be non-negative")
    with args.manifest.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == args.index:
                config = json.loads(line)
                run_experiment(config, smoke=args.smoke)
                return
    raise IndexError(f"manifest has no line {args.index}")


if __name__ == "__main__":
    main()
