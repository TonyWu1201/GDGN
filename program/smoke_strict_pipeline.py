from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.strict_eval.constants import PROTOCOLS
from program.strict_eval.splits import SplitOptions, build_all_splits
from program.strict_eval.training import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Ver2 end-to-end CPU smoke test")
    parser.add_argument("--keep-output", type=Path)
    args = parser.parse_args()
    if args.keep_output:
        root = args.keep_output.resolve()
        root.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        root = Path(tempfile.mkdtemp(prefix="gdgn-ver2-smoke-"))
        cleanup = True
    try:
        split_root = root / "splits"
        outputs = build_all_splits(
            split_root, SplitOptions(n_folds=3, seed=42, similarity_control=False), PROTOCOLS
        )
        assert set(outputs) == set(PROTOCOLS)
        for protocol in PROTOCOLS:
            manifest = outputs[protocol] / "manifest.json"
            assert manifest.exists()
            print(f"[smoke] split {protocol}: OK")

        base = {
            "experiment_id": "eval-lco", "split_id": "eval-lco", "fold": 0, "seed": 42,
            "split_path": str(split_root / "eval-lco" / "fold-00.json"),
        }
        statistical = base | {"model_id": "global-mean", "output_dir": str(root / "global")}
        run_experiment(statistical, smoke=True)
        print("[smoke] global baseline artifacts: OK")

        neural = base | {
            "model_id": "model-m0", "output_dir": str(root / "m0"), "batch_size": 4,
            "hidden_dim": 16, "modalities": ["pathway_activity"], "graph_policy": "graph-real",
            "lr": 1e-3, "max_epochs": 1,
        }
        run_experiment(neural, smoke=True)
        print("[smoke] M0 train/checkpoint/predict/parquet: OK")
        print(f"[smoke] ALL OK output={root}")
    finally:
        if cleanup:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
