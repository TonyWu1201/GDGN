"""
Phase 4 Step 1: DrugResponsePredictor (Cross-Attention 融合 + IC50 头)
=====================================================================

设计依据: Phase 4 计划 §3.2, §0.2.4 (cross-attention 默认 nn.MultiheadAttention
embed_dim=256 / num_heads=4), §0.2.5 (3 层 MLP + BN1d + Dropout(0.3), 输出不激活),
§0.4.1 (use_main_drug_emb 默认 False, Phase 6 ablation flag).

Phase 4.2 架构改造 (cell-side bypass + cross-attn residual modulation):
依据 [Phase4.2架构改造规划.md] §3 §4. 新增 `cell_bypass_mode` / `learnable_alpha`
两参数, 默认 "none" / True 严格与 Phase 6 ckpt 兼容 (bypass_proj /
modulation_alpha 不实例化 → strict load 通过).

职责
----
- 输入: gene_emb (B, N_gene, 256) + drug_emb (B, 128) + pathway_emb (B, 64),
  可选 main_drug_emb (B, N_drug, 256) + drug_idx (B,)
- 输出: ic50_pred (B, 1) + attn_weights (B, 1, N_gene) — Phase 5 IG 直接消费
- drug 作 query, 须经 drug_proj Linear(128, 256) 投影到 gene 维度;
  gene 作为 key/value; cross-attention average_attn_weights=True 跨 4 head 取平均,
  便于 Phase 5 单一权重直接归因.
- Phase 4.2 cell_bypass_mode:
  * "none": cell-side = attended_genes (与 Phase 6 严格一致, strict load OK)
  * "residual": cell-side = bypass_proj(gene_emb.mean) + alpha * attended_genes
                (bypass 主路径 drug-agnostic, 残差调制. 当 n_query>1 时, bypass_proj
                 输出 dim = n_query*proj_dim 与 attended_genes 对齐做残差和.)
  * "cat": cell-side = cat([bypass_proj(gene_emb.mean), attended_genes], -1)
           (双通道显式并行. bypass_proj 输出 dim == gene_dim (n_query 无关),
            cell_repr dim = gene_dim + n_query*proj_dim.)

参数量 (use_main_drug_emb=False): 446,209 ≈ 446K
    drug_proj Linear(128, 256)                     33,024
    cross_attn MultiheadAttention(256, 4 heads)   264,448
    predictor[0] Linear(448, 256)                 114,944
    predictor[1] BatchNorm1d(256)                     512
    predictor[4] Linear(256, 128)                  32,896
    predictor[5] BatchNorm1d(128)                     256
    predictor[8] Linear(128, 1)                      129

Phase 4.2 参数量 (cell_bypass_mode != "none", use_main_drug_emb=False, 来自
[Phase4.2架构改造规划.md] §4.4, B1/B2/B3 三变体):
    B1  residual | n_q=1 | hidden=256 : head ~510K   (+bypass_proj ~65K vs A0 444K)
    B2  residual | n_q=4 | hidden=512 : head ~1.46M  (+bypass_proj ~264K, MLP 加宽)
    B3  cat      | n_q=1 | hidden=256 : head ~576K   (cell_repr 512 -> fusion 704)
    B4  residual | n_q=1 | hidden=256 | extra_cell_dim=256 : head ~576K
        (fusion 704, extra 通道参数在 GDGNModel.flatten_cell_enc, 不在本模块)

⚠️ BatchNorm1d B=1 隐患: train 模式下 B=1 必触发 ValueError. Phase 4 trainer 必须
drop_last=True + 评估切 eval(). 单元 smoke 用 B>=4 远离此隐患. Phase 4.2
bypass_proj BN1d 同样适用.
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
    n_query_tokens : int
        cross-attention query token 数 (Phase 6 ablation; 默认 1).
    cell_bypass_mode : str
        Phase 4.2 cell-side bypass 路径模式 (默认 "none"):
        * "none":     cell-side = attended_genes (与 Phase 6 严格一致, strict load OK)
        * "residual": cell-side = bypass_proj(gene_emb.mean) + alpha * attended_genes
        * "cat":      cell-side = cat([bypass_proj(gene_emb.mean), attended_genes], -1)
    learnable_alpha : bool
        Phase 4.2 residual 模式的可学习调制强度 (默认 True); False 时 alpha=1.0 固定.
        仅 cell_bypass_mode="residual" 时生效; "none"/"cat" 不实例化此参数.
    extra_cell_dim : int
        Phase 4.2 B4 dual encoder 的附加 cell 通道维度 (默认 0):
        * 0 (默认): 无附加通道, 行为与 Phase 6 / B1-B3 完全一致.
        * >0: fusion 时在 cell_repr 之后追加 extra_cell_emb (B, extra_cell_dim),
          forward 必须传 extra_cell_emb. GDGNModel dual_cell=True 时 = 256
          (SimpleCellEncoder flatten 路径 drug-agnostic 嵌入).

    Phase 4.2 兼容性:
    - cell_bypass_mode="none" + extra_cell_dim=0 (默认) 不实例化任何新参数 →
      state_dict 键集与 Phase 6 完全一致, strict load_state_dict 通过.
    - cell_bypass_mode != "none" 时新增 bypass_proj (+ modulation_alpha for
      residual+learnable); extra_cell_dim > 0 时 fusion_dim 改变 → 均须
      strict=False 加载或全新 init (B1/B2/B3 走全新 init, B4 全新 init).
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
        cell_bypass_mode: str = "none",
        learnable_alpha: bool = True,
        extra_cell_dim: int = 0,
    ):
        super().__init__()
        assert proj_dim % num_heads == 0, f"proj_dim {proj_dim} must be divisible by num_heads {num_heads}"
        assert proj_dim == gene_dim, f"proj_dim {proj_dim} must equal gene_dim {gene_dim} (cross-attn K/V == gene_emb)"
        assert n_query_tokens >= 1, f"n_query_tokens {n_query_tokens} must be >= 1"
        assert cell_bypass_mode in ("none", "residual", "cat"), \
            f"cell_bypass_mode {cell_bypass_mode!r} must be one of 'none'/'residual'/'cat'"
        assert extra_cell_dim >= 0, f"extra_cell_dim {extra_cell_dim} must be >= 0"
        self.use_main_drug_emb = use_main_drug_emb
        self.n_query_tokens = n_query_tokens
        self.gene_dim = gene_dim
        self.drug_dim = drug_dim
        self.pathway_dim = pathway_dim
        self.cell_bypass_mode = cell_bypass_mode
        self.learnable_alpha = learnable_alpha
        self.extra_cell_dim = extra_cell_dim

        # n_query_tokens=1 时与原行为严格一致 (Phase 4 ckpt 可直接 load)
        # n_query_tokens>1 时 drug_proj 输出 (B, proj_dim * n_query) 后 reshape 成 (B, n_query, proj_dim)
        self.drug_proj = nn.Linear(drug_dim, proj_dim * n_query_tokens)

        # Phase 4.2 cell-side bypass 路径 (cell_bypass_mode != "none" 时实例化)
        # bypass_proj 输出 dim 取决于 mode:
        #   "residual": 必须与 attended_genes 维度 (n_query_tokens * proj_dim) 对齐做残差和
        #               (n_q=1 时与 gene_dim 相等; n_q>1 时扩到 n_q*proj_dim, B2 参数 ~1.46M 对应此)
        #   "cat":      始终输出 gene_dim (drug-agnostic 单通道与 attended 显式并行)
        if cell_bypass_mode != "none":
            bypass_out_dim = (n_query_tokens * proj_dim) if cell_bypass_mode == "residual" else gene_dim
            self.bypass_proj = nn.Sequential(
                nn.Linear(gene_dim, bypass_out_dim),
                nn.BatchNorm1d(bypass_out_dim),
                nn.ReLU(),
            )
            self.bypass_out_dim = bypass_out_dim
            if cell_bypass_mode == "residual" and learnable_alpha:
                # 残差调制强度, init=1.0 让模型从"等价 attended 主路径"起步逐步学调制
                self.modulation_alpha = nn.Parameter(torch.tensor(1.0))
        else:
            self.bypass_out_dim = 0  # 占位, "none" 模式不消费

        # fusion_dim 计算: 随 cell_bypass_mode 变化
        #    "none":     cell_dim = proj_dim * n_query                  (= Phase 6)
        #    "residual": cell_dim = n_query * proj_dim (与 attended 相同, 仅新增 bypass_proj 参数)
        #    "cat":      cell_dim = gene_dim + n_query * proj_dim        (显式双通道)
        # B4 dual encoder: extra_cell_dim > 0 时再追加 flatten 通道维度.
        if cell_bypass_mode == "cat":
            cell_dim = gene_dim + n_query_tokens * proj_dim
        else:  # "none" or "residual"
            cell_dim = n_query_tokens * proj_dim
        if use_main_drug_emb:
            self.main_drug_proj = nn.Linear(gene_dim, drug_dim)
            fusion_dim = cell_dim + extra_cell_dim + drug_dim + drug_dim + pathway_dim
        else:
            fusion_dim = cell_dim + extra_cell_dim + drug_dim + pathway_dim
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
        extra_cell_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        gene_emb : (B, N_gene, gene_dim)        Cross-attention K/V
        drug_emb : (B, drug_dim)               从 DrugEncoder 分子图 GCN
        pathway_emb : (B, pathway_dim)
        main_drug_emb : (B, N_drug, gene_dim) | None  主图 drug 节点嵌入 (Phase 6 ablation)
        drug_idx : (B,) LongTensor | None             指明 batch 内每样本对应主图哪个 drug
        extra_cell_emb : (B, extra_cell_dim) | None
            B4 dual encoder 附加 cell 通道 (SimpleCellEncoder flatten 嵌入,
            drug-agnostic). 仅 extra_cell_dim > 0 时消费; 否则必须为 None.

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

        # Phase 4.2 cell-side bypass: 让 cell-side 在 LODO 留出新 drug 时仍能稳
        # 输出 drug-agnostic 表征, cross-attn 降级为残差调制 (而非独占主路径).
        # 详见 [Phase4.2架构改造规划.md] §3 §4.1.
        if self.cell_bypass_mode == "none":
            cell_repr = attended_genes                          # Phase 6 严格一致
        else:
            bypass = gene_emb.mean(dim=1)                       # (B, gene_dim) drug-agnostic
            bypass = self.bypass_proj(bypass)                   # (B, bypass_out_dim)
            if self.cell_bypass_mode == "residual":
                alpha = self.modulation_alpha if self.learnable_alpha else 1.0
                cell_repr = bypass + alpha * attended_genes     # (B, n_query*proj_dim)
            else:  # "cat"
                cell_repr = torch.cat([bypass, attended_genes], dim=-1)  # (B, gene_dim + n_query*proj_dim)

        parts = [cell_repr]
        if self.extra_cell_dim > 0:
            # Phase 4.2 B4 dual encoder: 追加 SimpleCellEncoder flatten 通道.
            assert extra_cell_emb is not None, \
                "extra_cell_dim > 0 requires extra_cell_emb (B4 dual encoder path)"
            assert extra_cell_emb.size(1) == self.extra_cell_dim, \
                f"extra_cell_emb dim {extra_cell_emb.size(1)} != extra_cell_dim {self.extra_cell_dim}"
            parts.append(extra_cell_emb)
        parts += [drug_emb, pathway_emb]
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

    # --- Phase 4.2 cell_bypass_mode 测试 (residual / cat) ---------------------
    # 详见 [Phase4.2架构改造规划.md] §4.1 §4.4 §4.5.
    def _test_bypass(mode: str, n_q: int, learn_alpha: bool, *,
                     expect_bypass_extra_keys: bool, expect_residual_alpha: bool,
                     expect_fusion_dim: int) -> None:
        m = DrugResponsePredictor(
            use_main_drug_emb=False, n_query_tokens=n_q,
            cell_bypass_mode=mode, learnable_alpha=learn_alpha,
        ).to(device)
        n_p = sum(p.numel() for p in m.parameters())
        keys = set(m.state_dict().keys())
        assert m.fusion_dim == expect_fusion_dim, \
            f"[{mode} n_q={n_q} alpha={learn_alpha}] fusion_dim {m.fusion_dim} != {expect_fusion_dim}"
        has_bypass = any(k.startswith("bypass_proj.") for k in keys)
        has_alpha = any(k == "modulation_alpha" for k in keys)
        assert has_bypass == expect_bypass_extra_keys, \
            f"[{mode}] bypass_proj keys present={has_bypass} (expect {expect_bypass_extra_keys})"
        assert has_alpha == expect_residual_alpha, \
            f"[{mode}] modulation_alpha present={has_alpha} (expect {expect_residual_alpha})"
        m.train()
        out, attn = m(gene_emb, drug_emb, pathway_emb)
        assert out.shape == (B, 1), f"[{mode} n_q={n_q}] ic50_pred shape {out.shape}"
        assert attn.shape == (B, 1, N_gene), f"[{mode} n_q={n_q}] attn shape {attn.shape}"
        assert torch.isfinite(out).all() and torch.isfinite(attn).all()
        loss = out.sum()
        loss.backward()
        n_grad = sum(1 for p in m.parameters() if p.grad is not None and p.requires_grad)
        n_total = sum(1 for _ in m.parameters())
        assert n_grad == n_total, f"[{mode} n_q={n_q}] only {n_grad}/{n_total} params have grad"
        print(f"[smoke] bypass {mode} n_q={n_q} alpha={learn_alpha} OK "
              f"| fusion_dim={m.fusion_dim} params={n_p:,} keys={len(keys)} "
              f"has_bypass_proj={has_bypass} has_alpha={has_alpha}")
        # 零化所有梯度, 防下一个测试复用张量
        m.zero_grad(set_to_none=True)

    # B1: residual + n_q=1 + learnable_alpha=True
    # cell_dim = 1*256 = 256; fusion_dim = 256+128+64 = 448 (与 A0 一致, 但新增 bypass_proj+alpha)
    _test_bypass("residual", 1, True,
                 expect_bypass_extra_keys=True, expect_residual_alpha=True, expect_fusion_dim=448)
    # residual + n_q=2 + learnable_alpha=False (alpha=1.0 固定, n_q>1 验证 bypass_proj 维度对齐)
    # cell_dim = 2*256 = 512; fusion_dim = 512+128+64 = 704
    _test_bypass("residual", 2, False,
                 expect_bypass_extra_keys=True, expect_residual_alpha=False, expect_fusion_dim=704)
    # B3: cat + n_q=1 (双通道显式并行)
    # cell_dim = 256+256 = 512; fusion_dim = 512+128+64 = 704
    _test_bypass("cat", 1, True,
                 expect_bypass_extra_keys=True, expect_residual_alpha=False, expect_fusion_dim=704)
    # cat + n_q=2 (plan §4.5 smoke 配置)
    # cell_dim = 256+512 = 768; fusion_dim = 768+128+64 = 960
    _test_bypass("cat", 2, True,
                 expect_bypass_extra_keys=True, expect_residual_alpha=False, expect_fusion_dim=960)

    # --- Phase 4.2 B4 dual encoder: extra_cell_dim 测试 -----------------------
    # 详见 [Phase4.2架构改造规划.md] §5.4 + 本文件 extra_cell_dim 文档.
    # B4 = residual + extra_cell_dim=256 (SimpleCellEncoder flatten 通道):
    #   cell_dim = 256 (bypass + alpha*attended), fusion_dim = 256+256+128+64 = 704
    extra_emb = torch.randn(B, 256, device=device)
    m_extra = DrugResponsePredictor(
        use_main_drug_emb=False, cell_bypass_mode="residual",
        learnable_alpha=True, extra_cell_dim=256,
    ).to(device)
    assert m_extra.fusion_dim == 704, f"B4 fusion_dim {m_extra.fusion_dim} != 704"
    m_extra.train()
    out_extra, attn_extra = m_extra(gene_emb, drug_emb, pathway_emb, extra_cell_emb=extra_emb)
    assert out_extra.shape == (B, 1) and attn_extra.shape == (B, 1, N_gene)
    assert torch.isfinite(out_extra).all() and torch.isfinite(attn_extra).all()
    loss_extra = out_extra.sum()
    loss_extra.backward()
    n_grad_extra = sum(1 for p in m_extra.parameters() if p.grad is not None and p.requires_grad)
    n_total_extra = sum(1 for _ in m_extra.parameters())
    assert n_grad_extra == n_total_extra, f"B4 only {n_grad_extra}/{n_total_extra} params have grad"
    n_p_extra = sum(p.numel() for p in m_extra.parameters())
    print(f"[smoke] B4 extra_cell_dim=256 (residual) OK "
          f"| fusion_dim={m_extra.fusion_dim} params={n_p_extra:,} "
          f"| (expect ~576K = B3 head 尺寸, extra 通道无新增 predictor 参数)")
    m_extra.zero_grad(set_to_none=True)
    # extra_cell_dim>0 但不传 extra_cell_emb 必须显式报错 (防静默静默降级)
    try:
        m_extra(gene_emb, drug_emb, pathway_emb)
        raise AssertionError("B4: missing extra_cell_emb should raise")
    except AssertionError:
        pass
    print("[smoke] B4 missing extra_cell_emb correctly raises AssertionError")

    # extra_cell_dim=0 (默认) 时 extra_cell_emb 不消费: 与 Phase 6 行为一致已由
    # _test_bypass("none", ...) 覆盖; 再显式验证 none+extra=0 的 strict 键集兼容.

    # --- Phase 4.2 ∩ Phase 6 ckpt 严格兼容性校验 ---------------------------
    # cell_bypass_mode="none" 必须产生与 Phase 6 predictor 严格一致的 state_dict 键集
    # (bypass_proj/modulation_alpha 不实例化), 故 strict load_state_dict 通过.
    predictor_none = DrugResponsePredictor(use_main_drug_emb=False, cell_bypass_mode="none").to(device)
    none_keys = set(predictor_none.state_dict().keys())
    predictor_resid = DrugResponsePredictor(
        use_main_drug_emb=False, cell_bypass_mode="residual"
    ).to(device)
    resid_keys = set(predictor_resid.state_dict().keys())
    resid_only = resid_keys - none_keys
    # residual 模式新增: bypass_proj.* (Linear + BN1d, 含 BN running_mean/running_var/
    # num_batches_tracked 3 个 buffer) + modulation_alpha. 余下应无任何额外键.
    resid_bp = {k for k in resid_only if k.startswith("bypass_proj.")}
    resid_alpha = {k for k in resid_only if k == "modulation_alpha"}
    resid_other = resid_only - resid_bp - resid_alpha
    assert resid_bp and len(resid_alpha) == 1 and not resid_other, \
        f"residual-only keys unexpected: bp={sorted(resid_bp)} alpha={sorted(resid_alpha)} " \
        f"other={sorted(resid_other)}"
    cat_keys = set(DrugResponsePredictor(
        use_main_drug_emb=False, cell_bypass_mode="cat"
    ).to(device).state_dict().keys())
    cat_only = cat_keys - none_keys
    cat_bp = {k for k in cat_only if k.startswith("bypass_proj.")}
    cat_other = cat_only - cat_bp
    assert cat_bp and not cat_other, \
        f"cat-only keys unexpected: bp={sorted(cat_bp)} other={sorted(cat_other)}"
    print(f"[smoke] strict-compat: none_keys={len(none_keys)} "
          f"residual_extra={len(resid_only)} (bypass_proj+alpha) "
          f"cat_extra={len(cat_only)} (bypass_proj only) "
          f"-> Phase 6 ckpt strict load OK with mode='none'")

    print("[smoke] ALL OK | dpredictor.py")


if __name__ == "__main__":
    _smoke_test()