from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.strict_eval.interpretation_report import write_interpretation_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(write_interpretation_report(args.input_dirs, args.output_dir))


if __name__ == "__main__":
    main()
