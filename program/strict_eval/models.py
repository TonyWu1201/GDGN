from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IDPairMLP(nn.Module):
    """Identity-memory diagnostic; unseen entities map to explicit UNK tokens."""

    def __init__(self, n_cells: int, n_drugs: int, hidden_dim: int = 128):
        super().__init__()
        self.cell_unk, self.drug_unk = n_cells, n_drugs
        self.cell_embedding = nn.Embedding(n_cells + 1, hidden_dim)
        self.drug_embedding = nn.Embedding(n_drugs + 1, hidden_dim)
        self.head = MLP(hidden_dim * 2, hidden_dim, 1)

    def forward(self, cell_idx: torch.Tensor, drug_idx: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.cell_embedding(cell_idx), self.drug_embedding(drug_idx)], dim=-1))


class WeightedPathwayLayer(nn.Module):
    """Shared-parameter, weighted, undirected pathway propagation."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbor = torch.einsum("ij,bjh->bih", adjacency, tokens)
        return self.norm(tokens + torch.nn.functional.gelu(
            self.self_linear(tokens) + self.neighbor_linear(neighbor)
        ))


class PathwayResidualModel(nn.Module):
    """M0-M4 minimal sufficient family from the Ver2 master plan."""

    def __init__(
        self,
        cell_features: dict[str, torch.Tensor],
        drug_features: torch.Tensor,
        model_id: str = "model-m0",
        hidden_dim: int = 128,
        pathway_adjacency: torch.Tensor | None = None,
        known_target_pathways: torch.Tensor | None = None,
        modalities: tuple[str, ...] = ("expression",),
        dropout: float = 0.2,
        omics_pretrain_ckpt: str | Path | None = None,
        omics_finetune_policy: str = "full",
    ):
        super().__init__()
        if model_id not in {f"model-m{i}" for i in range(5)}:
            raise ValueError(f"unknown model_id: {model_id}")
        self.model_id = model_id
        self.level = int(model_id[-1])
        self.modalities = modalities
        for name, values in cell_features.items():
            self.register_buffer(f"cell_{name}", values.float())
        self.register_buffer("drug_features", drug_features.float())
        n_pathways = int(cell_features["pathway_activity"].shape[1])
        cell_dim = sum(int(cell_features[name].shape[1]) for name in modalities)
        self.cell_encoder = MLP(cell_dim, hidden_dim * 2, hidden_dim, dropout)
        if omics_pretrain_ckpt:
            checkpoint = torch.load(omics_pretrain_ckpt, map_location="cpu", weights_only=False)
            if int(checkpoint["input_dim"]) != cell_dim or int(checkpoint["latent_dim"]) != hidden_dim:
                raise ValueError(
                    "masked-omics checkpoint dimensions do not match the downstream cell encoder"
                )
            encoder_state = {
                key.removeprefix("encoder."): value
                for key, value in checkpoint["model_state_dict"].items()
                if key.startswith("encoder.")
            }
            self.cell_encoder.load_state_dict(encoder_state)
        if omics_finetune_policy not in {"full", "freeze", "staged"}:
            raise ValueError(f"unknown omics_finetune_policy: {omics_finetune_policy}")
        self.omics_finetune_policy = omics_finetune_policy
        if omics_finetune_policy in {"freeze", "staged"}:
            for parameter in self.cell_encoder.parameters():
                parameter.requires_grad_(False)
        self.drug_encoder = MLP(int(drug_features.shape[1]), hidden_dim * 2, hidden_dim, dropout)
        self.base_head = MLP(hidden_dim * 2, hidden_dim, 1, dropout)

        if self.level >= 1:
            self.pathway_id = nn.Parameter(torch.randn(n_pathways, hidden_dim) / math.sqrt(hidden_dim))
            self.pathway_value = nn.Linear(1, hidden_dim)
            self.pathway_pool = nn.Linear(hidden_dim, 1)
        if self.level >= 2:
            if pathway_adjacency is None:
                raise ValueError("M2-M4 require pathway_adjacency")
            self.register_buffer("pathway_adjacency", pathway_adjacency.float())
            self.pathway_graph = WeightedPathwayLayer(hidden_dim)
            self.graph_head = MLP(hidden_dim * 2, hidden_dim, 1, dropout)
            self.alpha = nn.Parameter(torch.tensor(-4.0))
        if self.level >= 3:
            if known_target_pathways is None:
                raise ValueError("M3-M4 require known_target_pathways")
            self.register_buffer("known_target_pathways", known_target_pathways.float())
        if self.level >= 4:
            self.target_predictor = nn.Sequential(
                nn.Linear(int(drug_features.shape[1]), hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_pathways)
            )

    def _cell_input(self, cell_idx: torch.Tensor) -> torch.Tensor:
        return torch.cat([getattr(self, f"cell_{name}")[cell_idx] for name in self.modalities], dim=-1)

    def _pathway_tokens(self, cell_idx: torch.Tensor) -> torch.Tensor:
        return self._pathway_tokens_from_values(self.cell_pathway_activity[cell_idx])

    def _pathway_tokens_from_values(self, pathway_activity: torch.Tensor) -> torch.Tensor:
        values = pathway_activity.unsqueeze(-1)
        return self.pathway_value(values) + self.pathway_id.unsqueeze(0)

    def forward_features(
        self,
        cell_input: torch.Tensor,
        pathway_activity: torch.Tensor,
        drug_idx: torch.Tensor,
        dti_policy: str = "known-targets",
        pathway_gate_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward with explicit cell inputs for attribution and intervention tests."""
        cell = self.cell_encoder(cell_input)
        drug_raw = self.drug_features[drug_idx]
        drug = self.drug_encoder(drug_raw)
        base = self.base_head(torch.cat([cell, drug], dim=-1))
        extras: dict[str, torch.Tensor] = {"base_prediction": base}
        if self.level == 0:
            return base, extras

        tokens = self._pathway_tokens_from_values(pathway_activity)
        if self.level == 1:
            pathway = (torch.softmax(self.pathway_pool(tokens).squeeze(-1), dim=-1).unsqueeze(-1) * tokens).sum(1)
            pred = base + 0.1 * self.base_head(torch.cat([pathway, drug], dim=-1))
            extras["pathway_tokens"] = tokens
            return pred, extras

        graph_tokens = self.pathway_graph(tokens, self.pathway_adjacency)
        if pathway_gate_override is not None:
            gate = pathway_gate_override
        elif self.level == 2:
            gate = torch.full(
                graph_tokens.shape[:2], 1.0 / graph_tokens.shape[1],
                device=graph_tokens.device,
            )
        elif self.level == 3:
            gate = self.known_target_pathways[drug_idx]
        else:
            if dti_policy in {"known-targets", "training-entities-only"}:
                gate = self.known_target_pathways[drug_idx]
            elif dti_policy == "structure-only":
                gate = torch.softmax(self.target_predictor(drug_raw), dim=-1)
            else:
                raise ValueError(f"unknown dti_policy: {dti_policy}")
        gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        graph_cell = (gate.unsqueeze(-1) * graph_tokens).sum(dim=1)
        residual = self.graph_head(torch.cat([graph_cell, drug], dim=-1))
        alpha = torch.sigmoid(self.alpha)
        prediction = base + alpha * residual
        extras.update({"pathway_gate": gate, "graph_residual": residual, "alpha": alpha})
        return prediction, extras

    def forward(
        self,
        cell_idx: torch.Tensor,
        drug_idx: torch.Tensor,
        dti_policy: str = "known-targets",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.forward_features(
            self._cell_input(cell_idx), self.cell_pathway_activity[cell_idx], drug_idx, dti_policy
        )


class MaskedOmicsAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        self.encoder = MLP(input_dim, latent_dim * 2, latent_dim)
        self.decoder = MLP(latent_dim, latent_dim * 2, input_dim)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        corrupted = values.masked_fill(mask, 0.0)
        return self.decoder(self.encoder(corrupted))


class ChunkAttentionPooler(nn.Module):
    """Optional PaRTI-style attention pooling over cached sequence chunks."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(embedding_dim, max(32, embedding_dim // 4)),
            nn.Tanh(),
            nn.Linear(max(32, embedding_dim // 4), 1),
        )

    def forward(self, chunks: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(chunks).squeeze(-1).masked_fill(~valid_mask, float("-inf"))
        all_invalid = ~valid_mask.any(dim=-1)
        if all_invalid.any():
            logits = logits.clone()
            logits[all_invalid, 0] = 0.0
        weights = torch.softmax(logits, dim=-1)
        return torch.einsum("bc,bcd->bd", weights, chunks), weights
