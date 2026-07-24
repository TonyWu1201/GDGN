"""
Step 2: PretrainGNNEncoder (HeteroConv + GAT)
===============================================

架构 (策略 B - 计划 §0.2.1):
    gene 输入 = proj( concat(ESM-2 1280, omics_gene 4) )  # per-cell, 每条件一次前向
    drug 输入 = proj(ECFP+理化 1030)

    HeteroConv 关系 (4 个 GAT, §0.2.3):
        ('gene', 'ppi', 'gene'):           GATConv(H, H/k, heads=k_ppi)
        ('gene', 'rev_ppi', 'gene'):        GATConv(H, H/k, heads=k_ppi)   PPI 反向
        ('drug', 'targets', 'gene'):        GATConv(H, H/k, heads=k_dti)   DTI 单向
        ('gene', 'rev_targets', 'drug'):    GATConv(H, H/k, heads=k_dti)   DTI 反向 (drug 接收 gene 消息, §2.2)

    每层: HeteroConv -> BatchNorm1d per-node-type -> ELU -> Dropout

复用 (Phase 3 GeneEncoder):
    本类与 `best_encoder.pt` 权重独立可 import, Phase 3 加载 init.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class PretrainGNNEncoder(nn.Module):
    """HeteroConv GAT encoder. 策略 B: omics 在 GNN 输入即注入.

    Parameters
    ----------
    gene_static_dim : int
        ESM-2 静态维度 (1280).
    drug_dim : int
        药物 ECFP+理化维度 (1030).
    omics_per_gene_dim : int
        组学按基因对齐的通道数 (expr/mut/cnv/meth = 4).
    hidden_dim : int
        统一隐藏维度, 必须可被每个 heads 整除.
    n_layers : int
        HeteroConv 层数, 推荐 2-3.
    heads_ppi : int
        PPI GAT 头数.
    heads_dti : int
        DTI GAT 头数.
    dropout : float
    """

    def __init__(
        self,
        gene_static_dim: int = 1280,
        drug_dim: int = 1030,
        omics_per_gene_dim: int = 4,
        hidden_dim: int = 128,
        n_layers: int = 2,
        heads_ppi: int = 4,
        heads_dti: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert hidden_dim % heads_ppi == 0, f"hidden_dim {hidden_dim} must be divisible by heads_ppi {heads_ppi}"
        assert hidden_dim % heads_dti == 0, f"hidden_dim {hidden_dim} must be divisible by heads_dti {heads_dti}"

        self.gene_static_dim = gene_static_dim
        self.drug_dim = drug_dim
        self.omics_per_gene_dim = omics_per_gene_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.heads_ppi = heads_ppi
        self.heads_dti = heads_dti

        self.proj_gene = nn.Linear(gene_static_dim + omics_per_gene_dim, hidden_dim)
        self.proj_drug = nn.Linear(drug_dim, hidden_dim)

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for li in range(n_layers):
            rels = {
                ("gene", "ppi", "gene"): GATConv(
                    hidden_dim, hidden_dim // heads_ppi, heads=heads_ppi,
                    add_self_loops=False, dropout=0.0,
                ),
                ("gene", "rev_ppi", "gene"): GATConv(
                    hidden_dim, hidden_dim // heads_ppi, heads=heads_ppi,
                    add_self_loops=False, dropout=0.0,
                ),
                ("drug", "targets", "gene"): GATConv(
                    hidden_dim, hidden_dim // heads_dti, heads=heads_dti,
                    add_self_loops=False, dropout=0.0,
                ),
                ("gene", "rev_targets", "drug"): GATConv(
                    hidden_dim, hidden_dim // heads_dti, heads=heads_dti,
                    add_self_loops=False, dropout=0.0,
                ),
            }
            self.layers.append(HeteroConv(rels, aggr="sum"))
            self.bns.append(nn.ModuleDict({
                "gene": nn.BatchNorm1d(hidden_dim),
                "drug": nn.BatchNorm1d(hidden_dim),
            }))

        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward_single(
        self,
        gene_x_static: torch.Tensor,
        drug_x: torch.Tensor,
        omics_one: torch.Tensor,
        ppi_vis_fwd: torch.Tensor,
        ppi_vis_rev: torch.Tensor,
        dti_vis_fwd: torch.Tensor,
        dti_vis_rev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """per-cell 前向.

        Parameters
        ----------
        gene_x_static : (N_gene, 1280) on device
        drug_x : (N_drug, 1030) on device
        omics_one : (N_gene, 4)  single-condition 组学
        ppi_vis_fwd, ppi_vis_rev : (2, E_ppi) visible PPI 双向 (拆成 fwd/rev)
        dti_vis_fwd, dti_vis_rev : (2, E_dti) visible DTI 单向 + 手工反向

        Returns
        -------
        gene_emb : (N_gene, hidden_dim)
        drug_emb : (N_drug, hidden_dim)
        """
        gene_in = torch.cat([gene_x_static, omics_one], dim=-1)
        gene_h = self.proj_gene(gene_in)
        drug_h = self.proj_drug(drug_x)

        x_dict = {"gene": gene_h, "drug": drug_h}
        edge_index_dict = {
            ("gene", "ppi", "gene"): ppi_vis_fwd,
            ("gene", "rev_ppi", "gene"): ppi_vis_rev,
            ("drug", "targets", "gene"): dti_vis_fwd,
            ("gene", "rev_targets", "drug"): dti_vis_rev,
        }

        last = n_layers = self.n_layers
        for li in range(n_layers):
            x_dict = self.layers[li](x_dict, edge_index_dict)
            new_dict = {}
            for k, v in x_dict.items():
                v = self.bns[li][k](v)
                if li < last - 1:
                    v = self.act(v)
                    v = self.dropout(v)
                else:
                    v = self.dropout(v)
                new_dict[k] = v
            x_dict = new_dict
        return x_dict["gene"], x_dict["drug"]

    def forward(
        self,
        gene_x_static: torch.Tensor,
        drug_x: torch.Tensor,
        omics_per_gene: torch.Tensor,
        ppi_vis_fwd: torch.Tensor,
        ppi_vis_rev: torch.Tensor,
        dti_vis_fwd: torch.Tensor,
        dti_vis_rev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """batch 条件 (策略 B per-cell loop). Return stacked (B, N_gene, H), (B, N_drug, H).

        训练时为节省显存通常直接调用 forward_single 在 batch 内逐 cell 反向累积.
        """
        B = omics_per_gene.shape[0]
        gene_outs = []
        drug_outs = []
        for ci in range(B):
            g, d = self.forward_single(
                gene_x_static, drug_x, omics_per_gene[ci],
                ppi_vis_fwd, ppi_vis_rev, dti_vis_fwd, dti_vis_rev,
            )
            gene_outs.append(g)
            drug_outs.append(d)
        return torch.stack(gene_outs, dim=0), torch.stack(drug_outs, dim=0)


def _smoke_test() -> None:
    from program.model.dataset import load_hetero_graph
    from program.model.edge_mask import EdgeMaskSampler
    from program.model.edge_mask import _make_cfg as make_cfg

    print("[smoke] building tiny encoder (hidden=64, heads_ppi=4, heads_dti=2, n_layers=2) ...")
    enc = PretrainGNNEncoder(gene_static_dim=1280, drug_dim=1030, omics_per_gene_dim=4,
                             hidden_dim=64, n_layers=2, heads_ppi=4, heads_dti=2, dropout=0.3)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"[smoke] encoder params: {n_params:,}")

    print("[smoke] loading base graph + sampling edges for a forward ...")
    hetero = load_hetero_graph()
    sampler = EdgeMaskSampler(hetero, make_cfg())
    _ = sampler.load_or_build_heldout()

    ppi = sampler.sample_ppi()
    dti = sampler.sample_dti()
    ppi_fwd = ppi["visible_fwd"]
    ppi_rev = ppi["visible_rev"]
    dti_fwd = dti["visible_fwd"]
    dti_rev = dti["visible_rev"]

    for name, t in [("ppi_fwd", ppi_fwd), ("ppi_rev", ppi_rev), ("dti_fwd", dti_fwd), ("dti_rev", dti_rev)]:
        assert t.dtype == torch.long and t.shape[0] == 2, f"{name} bad shape/dtype {t.shape} {t.dtype}"
        if name in ("ppi_fwd", "ppi_rev"):
            assert int(t.max()) < 8412 and int(t.min()) >= 0, f"{name} out of gene range"
        elif name == "dti_fwd":
            assert int(t[0].max()) < 184 and int(t[1].max()) < 8412, f"{name} out of range"
        elif name == "dti_rev":
            assert int(t[0].max()) < 8412 and int(t[1].max()) < 184, f"{name} out of range"

    gene_static = hetero["gene"].x
    drug_x = hetero["drug"].x
    omics = torch.randn(1, 8412, 4)

    print(f"[smoke] forward_single with single condition (B=1) ...")
    enc.eval()
    with torch.no_grad():
        g_emb, d_emb = enc.forward_single(gene_static, drug_x, omics[0], ppi_fwd, ppi_rev, dti_fwd, dti_rev)
    assert g_emb.shape == (8412, 64), g_emb.shape
    assert d_emb.shape == (184, 64), d_emb.shape
    assert torch.isfinite(g_emb).all(), "gene_emb has NaN/inf"
    assert torch.isfinite(d_emb).all(), "drug_emb has NaN/inf"
    print(f"[smoke] forward_single OK: gene_emb={tuple(g_emb.shape)} drug_emb={tuple(d_emb.shape)} "
          f"mean={g_emb.mean():.4f} std={g_emb.std():.4f}")

    print("[smoke] forward with B=2 (per-cell loop) ...")
    with torch.no_grad():
        gs, ds = enc.forward(gene_static, drug_x, omics.repeat(2, 1, 1),
                             ppi_fwd, ppi_rev, dti_fwd, dti_rev)
    assert gs.shape == (2, 8412, 64) and ds.shape == (2, 184, 64)
    print(f"[smoke] forward B=2 OK: gene эмб={tuple(gs.shape)} drug_emb={tuple(ds.shape)}")

    print("[smoke] backward (loss = -g_emb.sum() - d_emb.sum()) to check grads ...")
    enc.train()
    g_emb, d_emb = enc.forward_single(gene_static, drug_x, omics[0], ppi_fwd, ppi_rev, dti_fwd, dti_rev)
    loss = g_emb.sum() + d_emb.sum() * 0.1
    loss.backward()
    n_grad = sum(int(p.grad is not None) for p in enc.parameters())
    n_total = sum(1 for _ in enc.parameters())
    assert n_grad == n_total, f"only {n_grad}/{n_total} params have grad"
    max_grad = max(p.grad.abs().max().item() for p in enc.parameters() if p.grad is not None)
    print(f"[smoke] backward OK: {n_grad}/{n_total} params have grad | max grad={max_grad:.4f}")

    print("[smoke] ALL OK | pretrain_encoder.py")


if __name__ == "__main__":
    _smoke_test()