from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.strict_eval.biological import pathway_membership
from program.strict_eval.io import seed_everything, write_json
from program.strict_eval.models import ChunkAttentionPooler


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PaRTI pooling from pathway-membership supervision")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/model/pretrain-parti"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    obj = torch.load(args.chunks, map_location="cpu", weights_only=False)
    membership, pathways, genes = pathway_membership()
    if obj["gene_order"] != genes:
        raise ValueError("PaRTI chunk gene order must equal core_gene_order.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chunks, mask = obj["chunks"].to(device), obj["valid_mask"].to(device)
    target = membership.T.to(device)
    permutation = torch.randperm(len(genes), generator=torch.Generator().manual_seed(args.seed))
    n_val = max(1, round(0.2 * len(genes)))
    val_idx, train_idx = permutation[:n_val].to(device), permutation[n_val:].to(device)
    pooler = ChunkAttentionPooler(chunks.shape[-1]).to(device)
    decoder = torch.nn.Linear(chunks.shape[-1], len(pathways)).to(device)
    optimizer = torch.optim.AdamW([*pooler.parameters(), *decoder.parameters()], lr=args.lr)
    best_loss, best_state, history = float("inf"), None, []
    for epoch in range(1 if args.smoke else args.epochs):
        pooler.train(); decoder.train()
        pooled, _ = pooler(chunks[train_idx], mask[train_idx])
        loss = F.binary_cross_entropy_with_logits(decoder(pooled), target[train_idx])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        pooler.eval(); decoder.eval()
        with torch.no_grad():
            val_pooled, _ = pooler(chunks[val_idx], mask[val_idx])
            val_loss = float(F.binary_cross_entropy_with_logits(decoder(val_pooled), target[val_idx]).item())
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in pooler.state_dict().items()})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint-best.pt"
    torch.save({
        "pooler_state_dict": best_state, "embedding_dim": int(chunks.shape[-1]),
        "objective": "core-gene pathway-membership multilabel prediction",
        "seed": args.seed, "smoke": args.smoke,
    }, checkpoint)
    write_json(args.output_dir / "metrics.json", {"best_val_loss": best_loss, "history": history})
    print(checkpoint)


if __name__ == "__main__":
    main()
