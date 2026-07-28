"""
Phase 4 Step 1: DrugResponsePredictor (Cross-Attention 融合 + IC50 头)
=====================================================================

设计依据: Phase 4 计划 §3.2, §0.2.4 (cross-attention 默认 nn.MultiheadAttention
embed_dim=256 / num_heads=4), §0.2.5 (3 层 MLP + BN1d + Dropout(0.3), 输出不激活),
§0.4.1 (use_main_drug_emb 默认 False, Phase 6 ablation flag).

职责
----
- 输入: gene_emb (B, N_gene, 256) + drug_emb (B, 128) + pathway_emb (B, 64),
  可选 main_drug_emb (B, N_drug, 256) + drug_idx (B,)
- 输出: ic50_pred (B, 1) + attn_weights (B, 1, N_gene) — Phase 5 IG 直接消费
- drug 作 query, 须经 drug_proj Linear(128, 256) 投影到 gene 维度;
  gene 作为 key/value; cross-attention average_attn_weights=True 跨 4 head 取平均,
  便于 Phase 5 单一权重直接归因.

参数量 (use_main_drug_emb=False): 446,209 ≈ 446K
    drug_proj Linear(128, 256)                     33,024
    cross_attn MultiheadAttention(256, 4 heads)   264,448
    predictor[0] Linear(448, 256)                 114,944
    predictor[1] BatchNorm1d(256)                     512
    predictor[4] Linear(256, 128)                  32,896
    predictor[5] BatchNorm1d(128)                     256
    predictor[8] Linear(128, 1)                      129

⚠️ BatchNorm1d B=1 隐患: train 模式下 B=1 必触发 ValueError. Phase 4 trainer 必须
drop_last=True + 评估切 eval(). 单元 smoke 用 B>=4 远离此隐患.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class DrugResponsePredictor(nn.Module):
    """Phase 4 主任务预测器: Cross-Attention 融合 + IC50 MLP 头.

    Parameters
    ----------
    gene_dim : int
        GeneEncoder 输出特征维度 (默认 256).
    drug_dim : int
        DrugEncoder 输出特征维度 (默认 128).
    pathway_dim : int
        PathwayEncoder 输出特征维度 (默认 64).
    proj_dim : int
        cross-attention 内统一投影维度 (默认 256, 必须 == gene_dim 因 K/V 同体).
    num_heads : int
        cross-attention 头数 (默认 4; proj_dim 必须可被 num_heads 整除).
    hidden_dim : int
        MLP 中间维度 (默认 256).
    dropout : float
        MLP dropout (默认 0.3).
    use_main_drug_emb : bool
        是否融合主图 drug_emb (默认 False; True 时追加 main_drug_proj Linear(256, 128))
    """

    def __init__(
        self,
        gene_dim: int = 256,
        drug_dim: int = 128,
        pathway_dim: int = 64,
        proj_dim: int = 256,
        num_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        use_main_drug_emb: bool = False,
        n_query_tokens: int = 1,
    ):
        super().__init__()
        assert proj_dim % num_heads == 0, f"proj_dim {proj_dim} must be divisible by num_heads {num_heads}"
        assert proj_dim == gene_dim, f"proj_dim {proj_dim} must equal gene_dim {gene_dim} (cross-attn K/V == gene_emb)"
        assert n_query_tokens >= 1, f"n_query_tokens {n_query_tokens} must be >= 1"
        self.use_main_drug_emb = use_main_drug_emb
        self.n_query_tokens = n_query_tokens
        self.gene_dim = gene_dim
        self.drug_dim = drug_dim
        self.pathway_dim = pathway_dim

        # n_query_tokens=1 时与原行为严格一致 (Phase 4 ckpt 可直接 load)
        # n_query_tokens>1 时 drug_proj 输出 (B, proj_dim * n_query) 后 reshape 成 (B, n_query, proj_dim)
        self.drug_proj = nn.Linear(drug_dim, proj_dim * n_query_tokens)

        if use_main_drug_emb:
            self.main_drug_proj = nn.Linear(gene_dim, drug_dim)
            fusion_dim = proj_dim * n_query_tokens + drug_dim + drug_dim + pathway_dim
        else:
            fusion_dim = proj_dim * n_query_tokens + drug_dim + pathway_dim
        self.fusion_dim = fusion_dim

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads, batch_first=True,
        )

        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        gene_emb: torch.Tensor,
        drug_emb: torch.Tensor,
        pathway_emb: torch.Tensor,
        main_drug_emb: torch.Tensor | None = None,
        drug_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        gene_emb : (B, N_gene, gene_dim)        Cross-attention K/V
        drug_emb : (B, drug_dim)               从 DrugEncoder 分子图 GCN
        pathway_emb : (B, pathway_dim)
        main_drug_emb : (B, N_drug, gene_dim) | None  主图 drug 节点嵌入 (Phase 6 ablation)
        drug_idx : (B,) LongTensor | None             指明 batch 内每样本对应主图哪个 drug

        Returns
        -------
        ic50_pred : (B, 1)             回归值, 不激活
        attn_weights : (B, 1, N_gene)  跨 4 head 平均注意力权重, Phase 5 IG 直接消费
        """
        drug_query = self.drug_proj(drug_emb)                       # (B, proj_dim * n_query)
        B = drug_emb.size(0)
        if self.n_query_tokens > 1:
            drug_query = drug_query.view(B, self.n_query_tokens, -1)  # (B, n_q, proj_dim)
        else:
            drug_query = drug_query.unsqueeze(1)                # (B, 1, proj_dim) — 与原一致
        attended, attn_weights = self.cross_attn(
            drug_query, gene_emb, gene_emb,
            need_weights=True,
            average_attn_weights=True,
        )
        # attended: (B, n_query, proj_dim); attn_weights: (B, n_query, N_gene)
        if self.n_query_tokens > 1:
            attended_genes = attended.reshape(B, -1)            # (B, n_query * proj_dim)
            # attn_weights 多 query 时跨 query 平均到单 token, 保持 Phase 5 IG 接口 (B, 1, N_gene)
            attn_weights = attn_weights.mean(dim=1, keepdim=True)
        else:
            attended_genes = attended.squeeze(1)                # (B, proj_dim) — 与原一致

        parts = [attended_genes, drug_emb, pathway_emb]
        if self.use_main_drug_emb:
            assert main_drug_emb is not None and drug_idx is not None, \
                "use_main_drug_emb=True requires main_drug_emb + drug_idx"
            B = gene_emb.size(0)
            batch_arange = torch.arange(B, device=main_drug_emb.device)
            main_drug_selected = main_drug_emb[batch_arange, drug_idx]   # (B, gene_dim)
            main_drug_proj = self.main_drug_proj(main_drug_selected)     # (B, drug_dim)
            parts.insert(2, main_drug_proj)
        fused = torch.cat(parts, dim=-1)                          # (B, fusion_dim)
        ic50_pred = self.predictor(fused)                         # (B, 1)
        return ic50_pred, attn_weights


def _smoke_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")

    B, N_gene, N_drug = 4, 8412, 184
    gene_emb = torch.randn(B, N_gene, 256, device=device)
    drug_emb = torch.randn(B, 128, device=device)
    pathway_emb = torch.randn(B, 64, device=device)

    print("[smoke] use_main_drug_emb=False ...")
    predictor = DrugResponsePredictor(use_main_drug_emb=False).to(device)
    n_params = sum(p.numel() for p in predictor.parameters())
    print(f"[smoke] params={n_params:,} (expect ~446K)")
    assert 400_000 < n_params < 500_000, f"params {n_params} out of expected range"

    predictor.train()
    ic50_pred, attn_weights = predictor(gene_emb, drug_emb, pathway_emb)
    assert ic50_pred.shape == (B, 1), f"ic50_pred shape {ic50_pred.shape} != ({B}, 1)"
    assert attn_weights.shape == (B, 1, N_gene), f"attn_weights shape {attn_weights.shape} != ({B}, 1, {N_gene})"
    assert torch.isfinite(ic50_pred).all() and torch.isfinite(attn_weights).all()
    attn_row_sum = attn_weights.sum(dim=-1)
    print(f"[smoke] forward OK: ic50_pred={tuple(ic50_pred.shape)} attn_weights={tuple(attn_weights.shape)} "
          f"attn_row_sum_mean={attn_row_sum.mean():.4f} (softmax expect ~1.0)")
    assert torch.allclose(attn_row_sum, torch.ones_like(attn_row_sum), atol=1e-4), \
        f"attn weights row sum != 1: min={attn_row_sum.min():.4f} max={attn_row_sum.max():.4f}"

    loss = ic50_pred.sum()
    loss.backward()
    n_total = sum(1 for _ in predictor.parameters())
    n_grad = sum(1 for p in predictor.parameters() if p.grad is not None and p.requires_grad)
    print(f"[smoke] backward: {n_grad}/{n_total} params have grad")
    assert n_grad == n_total, f"only {n_grad}/{n_total} params have grad"

    print("[smoke] use_main_drug_emb=True ...")
    predictor_main = DrugResponsePredictor(use_main_drug_emb=True).to(device)
    n_params_main = sum(p.numel() for p in predictor_main.parameters())
    print(f"[smoke] params={n_params_main:,} (expect ~512K, +66K vs default)")
    assert n_params_main > n_params, "main_drug_proj should add params"

    main_drug_emb = torch.randn(B, N_drug, 256, device=device)
    drug_idx = torch.tensor([10, 50, 100, 150], device=device)
    predictor_main.train()
    ic50_pred2, attn_weights2 = predictor_main(gene_emb, drug_emb, pathway_emb,
                                              main_drug_emb=main_drug_emb, drug_idx=drug_idx)
    assert ic50_pred2.shape == (B, 1)
    assert attn_weights2.shape == (B, 1, N_gene)
    print(f"[smoke] forward (use_main_drug_emb=True) OK: ic50_pred={tuple(ic50_pred2.shape)}")
    loss2 = ic50_pred2.sum()
    loss2.backward()
    n_grad2 = sum(1 for p in predictor_main.parameters() if p.grad is not None and p.requires_grad)
    n_total2 = sum(1 for _ in predictor_main.parameters())
    print(f"[smoke] backward: {n_grad2}/{n_total2} params have grad")
    assert n_grad2 == n_total2, f"only {n_grad2}/{n_total2} params have grad"

    print("[smoke] ALL OK | dpredictor.py")


if __name__ == "__main__":
    _smoke_test()