from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.strict_eval.constants import CELL_FEATURE_PATH
from program.strict_eval.data import RAW_CELL_FEATURE_PATH, load_fold_features
from program.strict_eval.io import read_json
from program.strict_eval.task_pretraining import train_masked_omics


def main() -> None:
    parser = argparse.ArgumentParser(description="Masked-omics task-aligned pretraining")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--modalities", nargs="+",
        default=["expression", "copynumber", "methylation"],
        choices=["expression", "mutation", "copynumber", "methylation", "pathway_activity"],
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--split-id")
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    output_dir = args.output_dir or Path(
        "data/model/smoke/pretrain-masked-omics" if args.smoke else "data/model/pretrain-masked-omics"
    )
    train_idx = val_idx = None
    if (args.split_id is None) != (args.fold is None):
        raise ValueError("--split-id and --fold must be supplied together")
    if args.split_id is not None:
        split_path = Path(f"data/model/splits/{args.split_id}/fold-{args.fold:02d}.json")
        payload = read_json(split_path)
        features, _, _ = load_fold_features(
            payload, output_dir, allow_legacy_fallback=args.smoke
        )
        values = torch.cat([features[name] for name in args.modalities], dim=1)
        train_cells = set(payload["splits"]["train"]["cell_idx"])
        validation_cells = set(payload["splits"]["val"]["cell_idx"]) - train_cells
        if not validation_cells:
            ordered = torch.tensor(sorted(train_cells))
            permutation = ordered[torch.randperm(len(ordered), generator=torch.Generator().manual_seed(args.seed))]
            n_val = max(1, round(0.2 * len(permutation)))
            val_idx, train_idx = permutation[:n_val], permutation[n_val:]
        else:
            train_idx = torch.tensor(sorted(train_cells))
            val_idx = torch.tensor(sorted(validation_cells))
    else:
        source = RAW_CELL_FEATURE_PATH
        if not source.exists():
            if not args.smoke:
                raise FileNotFoundError(
                    f"{source} is required for formal pretraining; rebuild unscaled features first"
                )
            source = CELL_FEATURE_PATH
            print(f"[smoke] strict raw artifact absent; using legacy feature tensor only for code-path validation: {source}")
        obj = torch.load(source, map_location="cpu", weights_only=False)
        values = torch.cat([obj[name] for name in args.modalities], dim=1)
    print(train_masked_omics(
        values, output_dir, args.latent_dim, args.mask_ratio,
        args.epochs, seed=args.seed, smoke=args.smoke, modalities=tuple(args.modalities),
        train_idx=train_idx, val_idx=val_idx,
    ))


if __name__ == "__main__":
    main()
