"""
Step 3a: Edge Predictors
==========================

PPIEdgePredictor (条件无关, §5.2.1):
    pair = concat(gene_emb[src], gene_emb[dst]) -> MLP -> logit
    pos/neg 边单向 (无向边的 i<j 形式), 每个 pos+neg 给一条 logit.

DTIConditionedPredictor (bilinear, 修复 R1):
    cond = condition_net(omics_one)                       # (N_gene, H)
    gene_c = gene_emb + cond                              # 显式条件通路, 留 Phase 5 IG 归因
    score = (drug_h @ W) * gene_h  ->  sum / sqrt(H) + bias
    强制 drug-gene 双向交互, 避免 concat-MLP drug 通路退化.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PPIEdgePredictor(nn.Module):
    def __init__(self, hidden_dim: int, inner_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, inner_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, 1),
        )

    def forward(self, gene_emb: torch.Tensor, pos_edges: torch.Tensor, neg_edges: torch.Tensor) -> torch.Tensor:
        all_edges = torch.cat([pos_edges, neg_edges], dim=1).to(gene_emb.device)
        src, dst = all_edges[0], all_edges[1]
        pair = torch.cat([gene_emb[src], gene_emb[dst]], dim=-1)
        return self.mlp(pair).squeeze(-1)


class DTIConditionedPredictor(nn.Module):
    """Bilinear 打分: score = drug_h^T W gene_h + bias.

    强制 drug 与 gene 双向交互, 避免 concat-MLP 让 drug 通路退化为死神经元
    (R1 根因: 旧实现 drug 半边权重趋 0, 11 个无 DTI 药物 top-10 完全雷同).
    保留 condition_net 作 Phase 5 IG 归因入口 (per-gene 组学贡献残差).

    omics_per_gene 支持两种 shape:
        (N_gene, 4)       —— 单条件 (训练 per-cell 调用 / 评估单 cell omics)
        (B, N_gene, 4)    —— 多条件, 自动取 [0] (与旧 API 兼容)

    Parameters
    ----------
    hidden_dim : int
    omics_per_gene_dim : int
        4 (expr/mut/cnv/meth)
    inner_dim : int
        condition_net 内部维度 (仅影响条件网络容量)
    dropout : float
    """

    def __init__(self, hidden_dim: int, omics_per_gene_dim: int = 4, inner_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.condition_net = nn.Sequential(
            nn.Linear(omics_per_gene_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, hidden_dim),
        )
        self.W = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W)
        self.b_drug = nn.Parameter(torch.zeros(hidden_dim))
        self.b_gene = nn.Parameter(torch.zeros(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        gene_emb: torch.Tensor,
        drug_emb: torch.Tensor,
        omics_per_gene: torch.Tensor,
        pos_edges: torch.Tensor,
        neg_edges: torch.Tensor,
    ) -> torch.Tensor:
        if omics_per_gene.dim() == 3:
            omics_one = omics_per_gene[0]
        else:
            omics_one = omics_per_gene
        cond = self.condition_net(omics_one)
        gene_c = gene_emb + cond

        all_edges = torch.cat([pos_edges, neg_edges], dim=1).to(gene_emb.device)
        drug_idx = all_edges[0]
        gene_idx = all_edges[1]
        drug_h = drug_emb[drug_idx] + self.b_drug
        gene_h = self.dropout(gene_c[gene_idx]) + self.b_gene
        scale = 1.0 / math.sqrt(self.W.shape[0])
        score = ((drug_h @ self.W) * gene_h).sum(dim=-1) * scale + self.bias.squeeze()
        return score


def _smoke_test() -> None:
    H = 64
    N_gene, N_drug = 8412, 184
    n_pos, n_neg = 100, 100

    gene_emb = torch.randn(N_gene, H)
    drug_emb = torch.randn(N_drug, H)
    omics = torch.randn(2, N_gene, 4)

    pos_g = torch.randint(0, N_gene, (2, n_pos))
    neg_g = torch.randint(0, N_gene, (2, n_neg))
    pos_d = torch.stack([torch.randint(0, N_drug, (n_pos,)), torch.randint(0, N_gene, (n_pos,))], dim=0)
    neg_d = torch.stack([torch.randint(0, N_drug, (n_neg,)), torch.randint(0, N_gene, (n_neg,))], dim=0)

    ppi_pred = PPIEdgePredictor(H, inner_dim=H, dropout=0.1)
    dti_pred = DTIConditionedPredictor(H, omics_per_gene_dim=4, inner_dim=H, dropout=0.1)

    ppi_logits = ppi_pred(gene_emb, pos_g, neg_g)
    assert ppi_logits.shape == (n_pos + n_neg,), ppi_logits.shape
    assert torch.isfinite(ppi_logits).all()
    print(f"[smoke] PPI logits shape={tuple(ppi_logits.shape)} mean={ppi_logits.mean():.4f} std={ppi_logits.std():.4f}")

    dti_logits = dti_pred(gene_emb, drug_emb, omics, pos_d, neg_d)
    assert dti_logits.shape == (n_pos + n_neg,), dti_logits.shape
    assert torch.isfinite(dti_logits).all()
    print(f"[smoke] DTI logits shape={tuple(dti_logits.shape)} mean={dti_logits.mean():.4f} std={dti_logits.std():.4f}")

    loss = ppi_logits.mean() ** 2 + dti_logits.mean() ** 2
    loss.backward()
    n_grad_ppi = sum(int(p.grad is not None) for p in ppi_pred.parameters())
    n_grad_dti = sum(int(p.grad is not None) for p in dti_pred.parameters())
    print(f"[smoke] backward OK: PPI {n_grad_ppi}/{sum(1 for _ in ppi_pred.parameters())} | DTI {n_grad_dti}/{sum(1 for _ in dti_pred.parameters())} params grad")

    print("[smoke] ALL OK | edge_predictor.py")


if __name__ == "__main__":
    _smoke_test()