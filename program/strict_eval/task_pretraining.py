from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .io import seed_everything, write_json
from .models import MaskedOmicsAutoencoder


def random_feature_mask(values: torch.Tensor, ratio: float, generator: torch.Generator) -> torch.Tensor:
    if not 0 < ratio < 1:
        raise ValueError("mask ratio must be between zero and one")
    return torch.rand(values.shape, generator=generator, device="cpu").to(values.device) < ratio


def train_masked_omics(
    values: torch.Tensor,
    output_dir: str | Path,
    latent_dim: int = 128,
    mask_ratio: float = 0.15,
    epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 42,
    smoke: bool = False,
    modalities: tuple[str, ...] = ("expression", "copynumber", "methylation"),
    train_idx: torch.Tensor | None = None,
    val_idx: torch.Tensor | None = None,
) -> dict:
    """Task-aligned cell-state pretraining with a held-out cell validation set."""
    seed_everything(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = output / "checkpoint-best.pt"
    if checkpoint_path.exists() and not smoke:
        report = {"skipped": True, "checkpoint": str(checkpoint_path)}
        write_json(output / "metrics.json", report)
        print(f"[masked-omics] SKIP: 已有检查点 {checkpoint_path}")
        return report
    values = values.float()
    if train_idx is None or val_idx is None:
        permutation = torch.randperm(len(values), generator=torch.Generator().manual_seed(seed))
        n_val = max(1, int(round(0.2 * len(values))))
        val_idx, train_idx = permutation[:n_val], permutation[n_val:]
    train_idx = torch.as_tensor(train_idx, dtype=torch.long)
    val_idx = torch.as_tensor(val_idx, dtype=torch.long)
    if len(train_idx) == 0 or len(val_idx) == 0 or set(train_idx.tolist()) & set(val_idx.tolist()):
        raise ValueError("masked-omics train/validation cell indices must be nonempty and disjoint")
    model = MaskedOmicsAutoencoder(values.shape[1], latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_state = float("inf"), None
    history = []
    for epoch in range(1 if smoke else epochs):
        model.train()
        train = values[train_idx].to(device)
        mask = random_feature_mask(train, mask_ratio, generator)
        reconstruction = model(train, mask)
        loss = F.mse_loss(reconstruction[mask], train[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = values[val_idx].to(device)
            val_mask = random_feature_mask(val, mask_ratio, generator)
            val_pred = model(val, val_mask)
            val_loss = float(F.mse_loss(val_pred[val_mask], val[val_mask]).item())
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    torch.save({
        "model_state_dict": best_state,
        "input_dim": int(values.shape[1]),
        "latent_dim": latent_dim,
        "seed": seed,
        "mask_ratio": mask_ratio,
        "smoke": smoke,
        "modalities": list(modalities),
        "train_cell_idx": train_idx.tolist(),
        "val_cell_idx": val_idx.tolist(),
    }, output / "checkpoint-best.pt")
    report = {"best_val_loss": best_loss, "history": history, "smoke": smoke}
    write_json(output / "metrics.json", report)
    return report


def set_finetune_policy(model: torch.nn.Module, policy: str, unfreeze_last_n: int = 0) -> None:
    if policy == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif policy == "freeze":
        for parameter in model.parameters():
            parameter.requires_grad = False
    elif policy == "staged":
        parameters = list(model.parameters())
        for parameter in parameters:
            parameter.requires_grad = False
        for parameter in parameters[-max(1, unfreeze_last_n):]:
            parameter.requires_grad = True
    else:
        raise ValueError(f"unknown finetune policy: {policy}")
