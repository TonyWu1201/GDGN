from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.strict_eval.models import ChunkAttentionPooler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Task-trained ChunkAttentionPooler checkpoint")
    parser.add_argument("--output", type=Path, default=Path("data/processed/gene_embeddings/esm2_parti_gene_embeddings.pt"))
    args = parser.parse_args()
    obj = torch.load(args.chunks, map_location="cpu", weights_only=False)
    pooler = ChunkAttentionPooler(obj["chunks"].shape[-1])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pooler.load_state_dict(checkpoint["pooler_state_dict"])
    pooler.eval()
    with torch.no_grad():
        embeddings, weights = pooler(obj["chunks"], obj["valid_mask"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, args.output)
    torch.save(weights, args.output.with_name(args.output.stem + "_weights.pt"))
    print(args.output)


if __name__ == "__main__":
    main()
