"""
Phase 4 Step 4: 简化基线 BaselineSimpleModel (DeepCDR-molecularGCN + flatten MLP cell)
==================================================================================

设计依据: Phase 4 计划 §6.2, §0.2.7 方案 A (新框架下简化基线, 不复现完整
DeepCDR Keras 原版 / DeepCCDS).

架构
----
- 药物编码: 直接复用 Phase 3 DrugEncoder (DeepCDR 风格 4 层分子图 GCN, 185K params)
- 细胞系编码: SimpleCellEncoder = 组学 (B, 8412, 4) flatten + Linear(8412*4+186, 512)
  + BN + ReLU + Dropout + Linear(512, 256) + BN + ReLU  (无 GNN, 无 PPI, 无 pLM)
- 通路编码: 复用 Phase 3 PathwayEncoder (Linear+BN+ReLU+Dropout, 12K params)
- 融合: cat([cell_emb (B,256), drug_emb (B,128), pathway_emb (B,64)]) -> (B, 448)
  -> 3 层 MLP (Linear+BN+ReLU+Dropout*2) -> (B, 1)
- forward 返回 (ic50_pred, None)  None 占位与 GDGNModel 对齐 (有 attn_weights 的接口)

只读消费 Phase 3 模块 (DrugEncoder / PathwayEncoder) + Phase 1 数据 (drug_mol_graphs).
不修改 Phase 1-3 任何代码 / 数据;

⚠️ BatchNorm1d B=1 隐患: trainer 必须 drop_last=True + 评估切 eval(). 单元 smoke 用 B>=4.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.model.drug_encoder import DrugEncoder, drug_encoder_forward_batch, load_drug_mol_graphs
from program.model.pathway_encoder import PathwayEncoder
from program.model.pretrain import omics_to_per_gene


class SimpleCellEncoder(nn.Module):
    """组学 flatten MLP 基线 (无 GNN, 无 PPI, 无 pLM)."""

    def __init__(
        self,
        n_genes: int = 8412,
        n_omics_ch: int = 4,
        pathway_in_dim: int = 186,
        out_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        in_dim = n_genes * n_omics_ch + pathway_in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

    def forward(self, omics_4: torch.Tensor, pathway: torch.Tensor) -> torch.Tensor:
        x = torch.cat([omics_4.flatten(1), pathway], dim=-1)
        return self.net(x)


class BaselineSimpleModel(nn.Module):
    """新框架下简化基线: DeepCDR-molecularGCN + flatten MLP cell encoding.

    与 GDGNModel 一样持 self._device 而非 self.device (Phase 4 评估项 S2).
    """

    def __init__(
        self,
        hetero,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self._device = torch.device(device)
        self.drug_enc = DrugEncoder(atom_dim=75, hidden=256, out_dim=128).to(device)
        self.cell_enc = SimpleCellEncoder(n_genes=8412, n_omics_ch=4,
                                          pathway_in_dim=186, out_dim=256).to(device)
        self.pathway_enc = PathwayEncoder(pathway_in_dim=186, pathway_dim=64).to(device)

        self.head = nn.Sequential(
            nn.Linear(256 + 128 + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        ).to(device)

        self.drug_mol_graphs = load_drug_mol_graphs()
        _ = hetero

    def forward(
        self,
        cell_idx: torch.Tensor,
        drug_idx: torch.Tensor,
        omics: dict,
    ) -> tuple[torch.Tensor, None]:
        """返回 (ic50_pred (B,1), None). None 占位与 GDGNModel 对齐."""
        omics_4 = omics_to_per_gene(omics).to(self._device)
        pathway = omics["pathway"].to(self._device)
        drug_emb = drug_encoder_forward_batch(self.drug_enc, drug_idx, self.drug_mol_graphs, self._device)
        cell_emb = self.cell_enc(omics_4, pathway)
        pathway_emb = self.pathway_enc(pathway)
        fused = torch.cat([cell_emb, drug_emb, pathway_emb], dim=-1)
        return self.head(fused), None

    def encoder_parameters(self):
        params = list(self.drug_enc.parameters()) + \
                 list(self.cell_enc.parameters()) + \
                 list(self.pathway_enc.parameters())
        return params

    def head_parameters(self):
        return list(self.head.parameters())


def _smoke_test() -> None:
    from program.model.dataset import inject_batch_omics, load_cell_line_features, load_hetero_graph

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")

    hetero = load_hetero_graph(device)
    clf = load_cell_line_features(device)
    n_genes = int(hetero["gene"].num_nodes)

    model = BaselineSimpleModel(hetero=hetero, device=device).to(device)
    n_enc = sum(p.numel() for p in model.encoder_parameters())
    n_head = sum(p.numel() for p in model.head_parameters())
    print(f"[smoke] encoder_params={n_enc:,} head_params={n_head:,}")

    cell_idx = torch.tensor([0, 1, 2, 3], device=device)
    drug_idx = torch.tensor([10, 50, 100, 150], device=device)
    omics = inject_batch_omics(clf, cell_idx)

    model.train()
    ic50_pred, attn = model(cell_idx, drug_idx, omics)
    assert attn is None
    assert ic50_pred.shape == (4, 1), ic50_pred.shape
    assert torch.isfinite(ic50_pred).all()
    print(f"[smoke] forward OK: ic50_pred={tuple(ic50_pred.shape)} "
          f"mean={ic50_pred.mean():.4f} std={ic50_pred.std():.4f}")

    loss = ic50_pred.sum()
    loss.backward()
    n_total_enc = sum(1 for _ in model.encoder_parameters())
    n_grad_enc = sum(1 for p in model.encoder_parameters() if p.grad is not None and p.requires_grad)
    n_total_head = sum(1 for _ in model.head_parameters())
    n_grad_head = sum(1 for p in model.head_parameters() if p.grad is not None and p.requires_grad)
    print(f"[smoke] backward: encoder {n_grad_enc}/{n_total_enc}, head {n_grad_head}/{n_total_head}")
    assert n_grad_enc == n_total_enc, f"encoder only {n_grad_enc}/{n_total_enc} have grad"
    assert n_grad_head == n_total_head, f"head only {n_grad_head}/{n_total_head} have grad"

    model.eval()
    with torch.no_grad():
        ic50_eval, _ = model(cell_idx, drug_idx, omics)
    assert ic50_eval.shape == (4, 1)
    print(f"[smoke] eval forward OK (BN running stats): ic50 std={ic50_eval.std():.4f}")

    print(f"[smoke] n_genes={n_genes} (data integrity, expected 8412)")
    assert n_genes == 8412, f"n_genes {n_genes} != 8412 (Phase 1 hetero graph)"

    print("[smoke] ALL OK | baseline_simple.py")


if __name__ == "__main__":
    _smoke_test()