"""
Phase 3 Step 1: GeneEncoder (Phase 2 编码器薄包装 + 满图边 buffer 缓存)
=======================================================================

设计依据: Phase 3 计划 §3.2 选项 2 (buffer 缓存方案), §0.2.1 方案 A (完整复用
Phase 2 的 PretrainGNNEncoder 含 4 个 GAT: PPI fwd/rev + DTI drug->gene + gene->drug).

职责
----
- 加载 Phase 2 `best_encoder.pt` 权重作初始化 (Phase 4 warm start)
- 持 `PretrainGNNEncoder` 实例 + 4 个 buffer 缓存满图 PPI/DTI 的 fwd/rev 边索引,
  使 `forward(gene_x_static, drug_x, omics_4)` 仅需 3 参数 — Phase 4 trainer 不再
  需要管理 edge_index.
- 暴露 `freeze` flag:
    * False (默认): 端到端微调, 所有 GAT + proj 参数 trainable.
    * True:         冻结权重 + 固定 eval 模式 (BN 用 running stats, 即使父级 train()).
                    用于 Phase 6 ablation "frozen vs finetuned encoder".
- 暴露 `pretrain_ckpt=None` 路径: 随机初始化作 Phase 6 "no-pretrain" baseline.

不修改 `PretrainGNNEncoder` 类自身 (保持 Phase 2 兼容); 同时不修改 Phase 2 ckpt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.model.graph_utils import split_full_edges
from program.model.pretrain_encoder import PretrainGNNEncoder


def load_pretrained_gene_encoder(
    ckpt_path: str | Path,
    device: torch.device | str = "cpu",
    freeze: bool = False,
) -> tuple[PretrainGNNEncoder, dict]:
    """从 Phase 2 ckpt 加载 PretrainGNNEncoder.

    Parameters
    ----------
    ckpt_path : str | Path
        Phase 2 `best_encoder.pt` 路径.
    device : torch.device | str
        加载到的设备.
    freeze : bool
        True 时冻结所有参数 + 切 eval 模式.

    Returns
    -------
    (encoder, encoder_kwargs) : (PretrainGNNEncoder, dict)
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    kw = ckpt["encoder_kwargs"]
    assert kw["hidden_dim"] == 256, (
        f"expect full-trained ckpt hidden_dim=256, got {kw['hidden_dim']}; "
        "请先按 Phase 2 §10.1 跑全量训练再启动 Phase 3")
    assert int(ckpt.get("epoch", -1)) >= 10, (
        f"smoke ckpt 停在 epoch<10, 实际 epoch={ckpt.get('epoch')}; "
        "请先按 Phase 2 §10.1 跑全量训练再启动 Phase 3")
    enc = PretrainGNNEncoder(**kw).to(device)
    enc.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    if freeze:
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()
    return enc, kw


class GeneEncoder(nn.Module):
    """Phase 3 基因编码器: 薄包装 PretrainGNNEncoder + 满图边 buffer.

    Parameters
    ----------
    hetero : PyG HeteroData
        Phase 1 基图, 用以抽取并以 buffer 缓存满图 PPI/DTI 的 fwd/rev 边索引.
    pretrain_ckpt : str | Path | None
        Phase 2 ckpt 路径; None 表示随机初始化 (Phase 6 ablation "no-pretrain" baseline).
    device : torch.device | str
        外部 device.
    freeze : bool
        True 冻结内层 GAT + proj 参数, 并固定内层为 eval 模式 (BN 用 running stats).
    """

    def __init__(
        self,
        hetero,
        pretrain_ckpt: str | Path | None = None,
        device: torch.device | str = "cpu",
        freeze: bool = False,
    ):
        super().__init__()
        self.frozen_encoder: bool = bool(freeze)

        if pretrain_ckpt is not None:
            self.encoder, self.pretrain_kwargs = load_pretrained_gene_encoder(
                pretrain_ckpt, device=device, freeze=freeze,
            )
        else:
            # Phase 6 ablation "无预训练" baseline: 随机初始化 (Phase 2 默认架构)
            self.encoder = PretrainGNNEncoder(
                gene_static_dim=int(hetero["gene"].x.shape[1]),
                drug_dim=int(hetero["drug"].x.shape[1]),
                omics_per_gene_dim=4,
                hidden_dim=256, n_layers=2, heads_ppi=4, heads_dti=2,
            ).to(device)
            self.pretrain_kwargs = None
            if freeze:
                for p in self.encoder.parameters():
                    p.requires_grad_(False)
                self.encoder.eval()

        # 拆 fwd/rev 并 buffer 化 (Phase 3 / 4 用满图, 不做 mask, 与 Phase 2 验证阶段一致)
        ppi_fwd, ppi_rev, dti_fwd, dti_rev = split_full_edges(hetero)
        for name, t in [("ppi_fwd", ppi_fwd), ("ppi_rev", ppi_rev),
                        ("dti_fwd", dti_fwd), ("dti_rev", dti_rev)]:
            self.register_buffer(name, t.long().to(device), persistent=False)

    def forward(
        self,
        gene_x_static: torch.Tensor,
        drug_x: torch.Tensor,
        omics_per_gene: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """per-cell loop batch 前向 (委托给 PretrainGNNEncoder.forward).

        Parameters
        ----------
        gene_x_static : (N_gene, 1280)  ESM-2 静态特征.
        drug_x        : (N_drug, 1030)  ECFP+理化 Z-score 主图节点特征.
        omics_per_gene : (B, N_gene, 4) per-cell 组学 (expr/mut/cnv/meth stacked on last dim).

        Returns
        -------
        gene_emb : (B, N_gene, hidden_dim=256)
        drug_emb : (B, N_drug, hidden_dim=256)   主图 drug 节点嵌入, Phase 4 ablation 可选用
        """
        return self.encoder.forward(
            gene_x_static, drug_x, omics_per_gene,
            self.ppi_fwd, self.ppi_rev, self.dti_fwd, self.dti_rev,
        )

    def train(self, mode: bool = True) -> "GeneEncoder":
        """覆盖 train: 被冻结时内层 encoder 始终保持 eval 模式 (BN 用 running stats),
        避免父级 trainer.train() 时 BN 错误地用 batch 统计并更新 running 统计."""
        super().train(mode)
        if self.frozen_encoder:
            self.encoder.eval()
        return self


def _smoke_test() -> None:
    from program.model.dataset import load_hetero_graph, load_cell_line_features, inject_batch_omics
    from program.model.pretrain import omics_to_per_gene

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hetero = load_hetero_graph(device)
    clf = load_cell_line_features(device)
    n_genes = int(hetero["gene"].num_nodes)
    n_drugs = int(hetero["drug"].num_nodes)
    gene_static = hetero["gene"].x
    drug_x = hetero["drug"].x

    ckpt_path = "data/model/pretrain/best_encoder.pt"

    # ==== A. 加载 Phase 2 ckpt 路径 ====
    print(f"[smoke] GeneEncoder(hetero, pretrain_ckpt={ckpt_path}, device={device}, freeze=False)")
    gene_enc = GeneEncoder(hetero, pretrain_ckpt=ckpt_path, device=device, freeze=False).to(device)
    n_params = sum(p.numel() for p in gene_enc.parameters())
    n_buffers = len(list(gene_enc.buffers()))
    print(f"[smoke] params={n_params:,} buffers={n_buffers} "
          f"(expect 4 buffers: ppi_fwd/rev, dti_fwd/rev)")
    assert n_buffers == 4, f"expect 4 buffers, got {n_buffers}"
    # 与 Phase 2 verify_pretrain 报告对照: encoder params 应 ~1.13M
    print(f"[smoke] ppi_fwd={tuple(gene_enc.ppi_fwd.shape)} ppi_rev={tuple(gene_enc.ppi_rev.shape)} "
          f"dti_fwd={tuple(gene_enc.dti_fwd.shape)} dti_rev={tuple(gene_enc.dti_rev.shape)}")

    # B=1 forward
    omics = inject_batch_omics(clf, torch.tensor([0], device=device))
    omics_4 = omics_to_per_gene(omics).to(device)  # (1, 8412, 4)
    with torch.no_grad():
        g_emb, d_emb = gene_enc(gene_static, drug_x, omics_4)
    assert g_emb.shape == (1, n_genes, 256), g_emb.shape
    assert d_emb.shape == (1, n_drugs, 256), d_emb.shape
    assert torch.isfinite(g_emb).all() and torch.isfinite(d_emb).all()
    print(f"[smoke] forward B=1 OK: gene_emb={tuple(g_emb.shape)} drug_emb={tuple(d_emb.shape)} "
          f"std={g_emb.std():.4f}")

    # B=2 forward (per-cell loop)
    omics2 = inject_batch_omics(clf, torch.tensor([0, 1], device=device))
    omics_4_2 = omics_to_per_gene(omics2).to(device)
    with torch.no_grad():
        g2, d2 = gene_enc(gene_static, drug_x, omics_4_2)
    assert g2.shape == (2, n_genes, 256) and d2.shape == (2, n_drugs, 256)
    print(f"[smoke] forward B=2 OK: gene_emb={tuple(g2.shape)} drug_emb={tuple(d2.shape)}")

    # backward (freeze=False, 所有权重应有梯度)
    gene_enc.train()
    g, d = gene_enc(gene_static, drug_x, omics_4_2)
    loss = g.sum() + d.sum() * 0.1
    loss.backward()
    n_total = sum(1 for _ in gene_enc.parameters())
    n_grad = sum(1 for p in gene_enc.parameters() if p.grad is not None and p.requires_grad)
    print(f"[smoke] backward (freeze=False): {n_grad}/{n_total} params have grad")
    assert n_grad == n_total, f"only {n_grad}/{n_total} params have grad (expect all)"

    # ==== B. freeze=True 路径 ====
    gene_enc_fr = GeneEncoder(hetero, pretrain_ckpt=ckpt_path, device=device, freeze=True).to(device)
    n_train = sum(int(p.requires_grad) for p in gene_enc_fr.parameters())
    print(f"[smoke] freeze=True: trainable params={n_train}")
    assert n_train == 0, f"expect 0 trainable when freeze=True, got {n_train}"
    # 被冻结时父级 train() 不应让 BN 进入 train 模式
    gene_enc_fr.train()
    assert not gene_enc_fr.encoder.training, "frozen encoder must stay in eval mode after parent.train()"
    print("[smoke] freeze=True: encoder.training stays False after train() (BN uses running stats)")

    # ==== C. pretrain_ckpt=None 路径 (Phase 6 ablation baseline) ====
    gene_enc_rand = GeneEncoder(hetero, pretrain_ckpt=None, device=device, freeze=False).to(device)
    with torch.no_grad():
        g_r, d_r = gene_enc_rand(gene_static, drug_x, omics_4)
    assert g_r.shape == (1, n_genes, 256) and d_r.shape == (1, n_drugs, 256)
    print(f"[smoke] random-init OK: gene_emb std={g_r.std():.4f} "
          f"(pretrained std={g_emb.std():.4f})")

    print("[smoke] ALL OK | gene_encoder.py")


if __name__ == "__main__":
    _smoke_test()