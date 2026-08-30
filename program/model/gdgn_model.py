"""
Phase 4 Step 2: GDGNModel (三编码器 + DrugResponsePredictor 整合容器)
=====================================================================

设计依据: Phase 4 计划 §4.2, §2.5 (forward 顺序: omics 注入 -> DrugEncoder ->
GeneEncoder -> PathwayEncoder -> Predictor -> loss), §0.2.2 (param group 切分:
encoder vs head).

职责
----
- 把 Phase 3 三编码器 (GeneEncoder / DrugEncoder / PathwayEncoder) + Phase 4
  DrugResponsePredictor 装进一个 nn.Module, 供 trainer 一次调用管理设备 / train-eval.
- 缓存静态数据: 基图 gene.x / drug.x 节点特征 + 184 药物分子图, 避免每 step 重读盘.
- forward(cell_idx, drug_idx, omics) -> (ic50_pred (B,1), attn_weights (B,1,N_gene))
  omics 是 inject_batch_omics 的返回 dict {'expr','mut','cnv','meth','pathway'}

⚠️ Phase 4 评估项 S2 修正: 计划 §4.2 伪代码用 self.device, nn.Module 没有此属性;
本实现存 self._device = torch.device(device), forward 内用 self._device.

⚠️ Phase 4 评估项 M2 已知行为: use_main_drug_emb=False 时 main_drug_emb 是死分支,
但 PretrainGNNEncoder 内 drug-side BatchNorm1d 仍会在 train 模式下更新 running
stats (没有梯度反流去校正). 这是 Phase 3 编码器固有行为, Phase 6 ablation 用
freeze_encoder=True 可消除. 本模块不修改 Phase 3 编码器.

⚠️ 计划 §4.2 偏离: 伪代码写 load_random_gene_encoder 参数, 但 Phase 3 实际
GeneEncoder 用 pretrain_ckpt=None 表达随机初始化. 本实现按 Phase 3 实际签名:
pretrain_ckpt=None 即随机初始化 (Phase 6 ablation "no-pretrain" baseline),
不再单独开 load_random_gene_encoder 参数.

⚠️ BatchNorm1d B=1 隐患: 训练时 B=1 会触发 ValueError. trainer 必须 drop_last=True
+ 评估切 eval(). 单元 smoke 用 B>=4.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.model.dataset import load_drug_emb_matrix
from program.model.dpredictor import DrugResponsePredictor
from program.model.drug_encoder import DrugEncoder, drug_encoder_forward_batch, load_drug_mol_graphs
from program.model.gene_encoder import GeneEncoder
from program.model.pathway_encoder import PathwayEncoder
from program.model.pretrain import omics_to_per_gene
from program.model.baseline_simple import SimpleCellEncoder


def initialize_gene_features(
    original: torch.Tensor, mode: str, seed: int = 42
) -> tuple[torch.Tensor, bool]:
    """Return the controlled static feature tensor and whether it is trainable."""
    if mode in {"esm-mean", "esm-parti"}:
        return original, False
    if mode == "esm-none":
        return torch.zeros_like(original), False
    if mode in {"esm-random", "esm-id"}:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        values = torch.randn(original.shape, generator=generator)
        values = values / original.shape[1] ** 0.5
        return values.to(original.device), mode == "esm-id"
    raise ValueError(f"unknown gene_feature_mode: {mode}")


class GDGNModel(nn.Module):
    """Phase 4 整合模型: 三编码器 + DrugResponsePredictor.

    Parameters
    ----------
    hetero : PyG HeteroData
        Phase 1 基图 (含 gene.x / drug.x + PPI / DTI 边).
    pretrain_ckpt : str | Path | None
        Phase 2 best_encoder.pt 路径; None 表示随机初始化 (Phase 6 ablation "no-pretrain").
    device : torch.device | str
    freeze_encoder : bool
        True 冻结三编码器 (Phase 6 ablation "frozen vs finetuned").
    use_main_drug_emb : bool
        True 让 Predictor 融合主图 drug_emb (Phase 6 ablation "主图融合 vs 不融合").
    num_heads : int
        cross-attention 头数 (默认 4).
    n_query_tokens : int
        cross-attention query token 数 (Phase 6 ablation; 默认 1).
    predictor_hidden : int
        Predictor MLP 隐层维度 (默认 256).
    cell_bypass_mode : str
        Phase 4.2 cell-side bypass 模式 (默认 "none"):
        "none" / "residual" / "cat". 详见 DrugResponsePredictor 文档字符串 +
        [Phase4.2架构改造规划.md] §4.1.
    learnable_alpha : bool
        Phase 4.2 residual 模式的可学习调制强度 (默认 True).
    dual_cell : bool
        Phase 4.2 B4 dual encoder 开关 (默认 False):
        True 时在 GDGN 图路径 (gene_enc -> cross-attn/bypass) 之外并行一个
        baseline 原版 SimpleCellEncoder flatten 路径 (drug-agnostic 256 维),
        cell_repr = cat([图路径 cell_repr (residual/bypass 输出), flatten 嵌入])
        -> fusion_dim = 704. 复制 baseline 已证明的 drug-agnostic cell 优势
        (plan §5.4 L3). flatten 参数计入 encoder_parameters() (lr_encoder 组).
    """

    def __init__(
        self,
        hetero,
        pretrain_ckpt: str | Path | None = None,
        device: torch.device | str = "cpu",
        freeze_encoder: bool = False,
        use_main_drug_emb: bool = False,
        num_heads: int = 4,
        n_query_tokens: int = 1,
        predictor_hidden: int = 256,
        cell_bypass_mode: str = "none",
        learnable_alpha: bool = True,
        dual_cell: bool = False,
        gene_feature_mode: str = "esm-mean",
        gene_feature_seed: int = 42,
    ):
        super().__init__()
        self._device = torch.device(device)
        self.use_main_drug_emb = bool(use_main_drug_emb)
        self.freeze_encoder = bool(freeze_encoder)
        self.cell_bypass_mode = cell_bypass_mode
        self.learnable_alpha = learnable_alpha
        self.dual_cell = bool(dual_cell)
        self.gene_feature_mode = gene_feature_mode

        self.gene_enc = GeneEncoder(
            hetero, pretrain_ckpt=pretrain_ckpt, device=device, freeze=freeze_encoder,
        )
        self.drug_enc = DrugEncoder(atom_dim=75, hidden=256, out_dim=128).to(device)
        self.pathway_enc = PathwayEncoder(pathway_in_dim=186, pathway_dim=64).to(device)

        # Phase 4.2 B4 dual encoder: 并行 baseline 原版 flatten cell 路径
        # (SimpleCellEncoder, 无 GNN / 无 PPI, drug-agnostic; 复制 baseline_simple
        # 原始实现 = ~17.4M 参数). extra_cell_dim 与 cell_bypass_mode 正交:
        # 默认按 plan §5.4 组合 residual + extra 256; 其它 bypass 模式亦可配.
        if self.dual_cell:
            self.flatten_cell_enc = SimpleCellEncoder(
                n_genes=int(hetero["gene"].num_nodes), n_omics_ch=4,
                pathway_in_dim=186, out_dim=256,
            ).to(device)
            extra_cell_dim = 256
        else:
            self.flatten_cell_enc = None
            extra_cell_dim = 0

        self.predictor = DrugResponsePredictor(
            gene_dim=256, drug_dim=128, pathway_dim=64,
            proj_dim=256, num_heads=num_heads, hidden_dim=predictor_hidden, dropout=0.3,
            use_main_drug_emb=self.use_main_drug_emb,
            n_query_tokens=n_query_tokens,
            cell_bypass_mode=cell_bypass_mode,
            learnable_alpha=learnable_alpha,
            extra_cell_dim=extra_cell_dim,
        ).to(device)

        original_gene_x = hetero["gene"].x.to(self._device)
        gene_values, trainable_gene_values = initialize_gene_features(
            original_gene_x, gene_feature_mode, gene_feature_seed
        )
        # esm-parti expects the caller to replace hetero['gene'].x with the
        # cached, biologically supervised chunk-attention embedding first.
        if trainable_gene_values:
            self.gene_x_static = nn.Parameter(gene_values)
        else:
            self.register_buffer("gene_x_static", gene_values)
        self.drug_x = hetero["drug"].x.to(self._device)
        self.drug_mol_graphs = load_drug_mol_graphs()

    def forward(
        self,
        cell_idx: torch.Tensor,
        drug_idx: torch.Tensor,
        omics: dict,
        micro_batch: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """端到端前向.

        Parameters
        ----------
        cell_idx : (B,) LongTensor   — 仅作 ABI 占位, trainer 调用前已用其取 omics, 本函数不直接消费
        drug_idx : (B,) LongTensor
        omics : dict from inject_batch_omics(clf, cell_idx)
            {'expr','mut','cnv','meth' 各 (B, 8412); 'pathway' (B, 186)}
        micro_batch : int | None
            None 时整批前向 (原 Phase 4 行为); >0 时把 batch 按 micro_batch 切块,
            逐块 (encoder + predictor) 前向, 输出等价于整批 (BN/Dropout 除外),
            峰值激活显存与 micro_batch 成正比, 用于大图 GAT 下的 OOM 缓解.

        Returns
        -------
        ic50_pred : (B, 1)
        attn_weights : (B, 1, N_gene)   use_main_drug_emb=False 时也无 None
        """
        B = int(cell_idx.shape[0])
        if not micro_batch or micro_batch >= B:
            return self._forward_batch(cell_idx, drug_idx, omics)
        preds, attns = [], []
        for start in range(0, B, micro_batch):
            stop = min(start + micro_batch, B)
            micro_omics = {key: value[start:stop] for key, value in omics.items()}
            p_slice, a_slice = self._forward_batch(
                cell_idx[start:stop], drug_idx[start:stop], micro_omics
            )
            preds.append(p_slice)
            attns.append(a_slice)
        return torch.cat(preds, dim=0), torch.cat(attns, dim=0)

    def _forward_batch(
        self,
        cell_idx: torch.Tensor,
        drug_idx: torch.Tensor,
        omics: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        omics_4 = omics_to_per_gene(omics).to(self._device)
        gene_emb, main_drug_emb = self.gene_enc(self.gene_x_static, self.drug_x, omics_4)
        drug_emb = drug_encoder_forward_batch(self.drug_enc, drug_idx, self.drug_mol_graphs, self._device)
        pathway_emb = self.pathway_enc(omics["pathway"].to(self._device))

        # Phase 4.2 B4 dual encoder: flatten 路径与 baseline_simple 完全一致
        # (输入 omics_4 flatten + raw pathway), 输出 drug-agnostic 256 维.
        extra_cell_emb = None
        if self.dual_cell:
            extra_cell_emb = self.flatten_cell_enc(
                omics_4, omics["pathway"].to(self._device))

        if self.use_main_drug_emb:
            ic50_pred, attn_weights = self.predictor(
                gene_emb, drug_emb, pathway_emb,
                main_drug_emb=main_drug_emb, drug_idx=drug_idx,
                extra_cell_emb=extra_cell_emb,
            )
        else:
            ic50_pred, attn_weights = self.predictor(
                gene_emb, drug_emb, pathway_emb, extra_cell_emb=extra_cell_emb)
        return ic50_pred, attn_weights

    def encoder_parameters(self):
        """Phase 4 trainer 用: 返回三编码器参数 (param group 0, lr_encoder).

        dual_cell=True (B4) 时追加 flatten_cell_enc 参数 (与 baseline_simple 的
        cell_enc 分组一致, lr_encoder=1e-4).
        """
        params = list(self.gene_enc.parameters()) + \
                 list(self.drug_enc.parameters()) + \
                 list(self.pathway_enc.parameters())
        if self.dual_cell:
            params += list(self.flatten_cell_enc.parameters())
        if isinstance(self.gene_x_static, nn.Parameter):
            params += [self.gene_x_static]
        return params

    def head_parameters(self):
        """Phase 4 trainer 用: 返回 Predictor 参数 (param group 1, lr_head)."""
        return list(self.predictor.parameters())


def _smoke_test() -> None:
    from program.model.dataset import inject_batch_omics, load_cell_line_features, load_hetero_graph

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")

    print("[smoke] loading hetero + cell_line_features ...")
    hetero = load_hetero_graph(device)
    clf = load_cell_line_features(device)
    n_genes = int(hetero["gene"].num_nodes)
    n_drugs = int(hetero["drug"].num_nodes)

    ckpt_path = "data/model/pretrain/best_encoder.pt"
    pretrain_ckpt = None
    if Path(ckpt_path).exists():
        info = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        kw = info.get("encoder_kwargs", {})
        if int(kw.get("hidden_dim", 0)) == 256 and not info.get("cfg", {}).get("smoke", False):
            pretrain_ckpt = ckpt_path
            print(f"[smoke] using pretrained ckpt (epoch={info.get('epoch')}, hidden=256)")
        else:
            print(f"[smoke] NOTE best_encoder.pt is incompatible/smoke (hidden={kw.get('hidden_dim')}, "
                  f"smoke={info.get('cfg', {}).get('smoke')}); using pretrain_ckpt=None for smoke. "
                  f"Phase 4 全量训练需先重跑 Phase 2 全量 (hidden=256) 产出 ckpt.")

    print(f"[smoke] GDGNModel(hetero, pretrain_ckpt={pretrain_ckpt}, device={device}, freeze_encoder=False)")
    model = GDGNModel(hetero=hetero, pretrain_ckpt=pretrain_ckpt, device=device, freeze_encoder=False).to(device)

    n_enc = sum(p.numel() for p in model.encoder_parameters())
    n_head = sum(p.numel() for p in model.head_parameters())
    print(f"[smoke] encoder_params={n_enc:,} head_params={n_head:,} "
          f"(expect ~1.33M enc / ~445K head)")

    cell_idx = torch.tensor([0, 1, 2, 3], device=device)
    drug_idx = torch.tensor([10, 50, 100, 150], device=device)
    omics = inject_batch_omics(clf, cell_idx)
    assert omics["expr"].shape == (4, n_genes)
    assert omics["pathway"].shape == (4, 186)

    model.train()
    ic50_pred, attn_weights = model(cell_idx, drug_idx, omics)
    assert ic50_pred.shape == (4, 1), ic50_pred.shape
    assert attn_weights.shape == (4, 1, n_genes), attn_weights.shape
    assert torch.isfinite(ic50_pred).all() and torch.isfinite(attn_weights).all()
    print(f"[smoke] forward OK: ic50_pred={tuple(ic50_pred.shape)} attn={tuple(attn_weights.shape)} "
          f"mean={ic50_pred.mean():.4f} std={ic50_pred.std():.4f}")

    loss = ic50_pred.sum()
    loss.backward()
    n_total_enc = sum(1 for _ in model.encoder_parameters())
    n_grad_enc = sum(1 for p in model.encoder_parameters() if p.grad is not None and p.requires_grad)
    n_total_head = sum(1 for _ in model.head_parameters())
    n_grad_head = sum(1 for p in model.head_parameters() if p.grad is not None and p.requires_grad)
    print(f"[smoke] backward: encoder {n_grad_enc}/{n_total_enc}, head {n_grad_head}/{n_total_head}")
    assert n_grad_head == n_total_head, f"head only {n_grad_head}/{n_total_head} have grad"

    print("[smoke] use_main_drug_emb=True variant ...")
    model_main = GDGNModel(hetero=hetero, pretrain_ckpt=pretrain_ckpt, device=device,
                           freeze_encoder=False, use_main_drug_emb=True).to(device)
    model_main.train()
    ic50_pred2, attn_weights2 = model_main(cell_idx, drug_idx, omics)
    assert ic50_pred2.shape == (4, 1)
    assert attn_weights2.shape == (4, 1, n_genes)
    loss2 = ic50_pred2.sum()
    loss2.backward()
    n_total_head2 = sum(1 for _ in model_main.head_parameters())
    n_grad_head2 = sum(1 for p in model_main.head_parameters() if p.grad is not None and p.requires_grad)
    print(f"[smoke] use_main_drug_emb=True backward head: {n_grad_head2}/{n_total_head2}")
    assert n_grad_head2 == n_total_head2

    print("[smoke] pretrain_ckpt=None (Phase 6 ablation random-init) ...")
    model_rand = GDGNModel(hetero=hetero, pretrain_ckpt=None, device=device, freeze_encoder=False).to(device)
    model_rand.eval()
    with torch.no_grad():
        ic50_r, attn_r = model_rand(cell_idx, drug_idx, omics)
    assert ic50_r.shape == (4, 1) and attn_r.shape == (4, 1, n_genes)
    print(f"[smoke] random-init OK: ic50 std={ic50_r.std():.4f} "
          f"(pretrained std={ic50_pred.std():.4f})")

    # --- Phase 4.2 cell_bypass_mode 端到端校验 ---------------------------
    # 详见 [Phase4.2架构改造规划.md] §4.1 §4.2 §4.5.
    for mode in ("residual", "cat"):
        model_bp = GDGNModel(
            hetero=hetero, pretrain_ckpt=None, device=device, freeze_encoder=False,
            cell_bypass_mode=mode, learnable_alpha=True,
        ).to(device)
        n_enc_bp = sum(p.numel() for p in model_bp.encoder_parameters())
        n_head_bp = sum(p.numel() for p in model_bp.head_parameters())
        model_bp.train()
        ic50_bp, attn_bp = model_bp(cell_idx, drug_idx, omics)
        assert ic50_bp.shape == (4, 1), f"{mode}: ic50_pred shape {ic50_bp.shape}"
        assert attn_bp.shape == (4, 1, n_genes), f"{mode}: attn shape {attn_bp.shape}"
        assert torch.isfinite(ic50_bp).all() and torch.isfinite(attn_bp).all()
        loss_bp = ic50_bp.sum()
        loss_bp.backward()
        n_grad_head_bp = sum(
            1 for p in model_bp.head_parameters() if p.grad is not None and p.requires_grad)
        n_total_head_bp = sum(1 for _ in model_bp.head_parameters())
        assert n_grad_head_bp == n_total_head_bp, \
            f"{mode}: head only {n_grad_head_bp}/{n_total_head_bp} have grad"
        print(f"[smoke] cell_bypass_mode={mode} OK | encoder={n_enc_bp:,} head={n_head_bp:,} "
              f"(expect residual>B1 ~510K head, cat>B3 ~576K head)")

    # --- Phase 4.2 B4 dual encoder 端到端校验 -----------------------------
    # 详见 [Phase4.2架构改造规划.md] §5.4. B4 = dual_cell=True +
    # cell_bypass_mode="residual": 图路径 (bypass+alpha*attended, 256) ‖
    # SimpleCellEncoder flatten 路径 (drug-agnostic, 256) -> cell 512 -> fusion 704.
    # cat 模式: cell 512 (bypass‖attended) + flatten 256 -> fusion 960.
    for mode in ("residual", "cat"):
        expect_fusion = 704 if mode == "residual" else 960
        model_dual = GDGNModel(
            hetero=hetero, pretrain_ckpt=None, device=device, freeze_encoder=False,
            cell_bypass_mode=mode, learnable_alpha=True, dual_cell=True,
        ).to(device)
        n_enc_dual = sum(p.numel() for p in model_dual.encoder_parameters())
        n_head_dual = sum(p.numel() for p in model_dual.head_parameters())
        assert model_dual.predictor.fusion_dim == expect_fusion, \
            f"dual_cell({mode}) fusion_dim {model_dual.predictor.fusion_dim} != {expect_fusion}"
        n_flatten = sum(p.numel() for p in model_dual.flatten_cell_enc.parameters())
        model_dual.train()
        ic50_dual, attn_dual = model_dual(cell_idx, drug_idx, omics)
        assert ic50_dual.shape == (4, 1), f"dual {mode}: ic50_pred shape {ic50_dual.shape}"
        assert attn_dual.shape == (4, 1, n_genes), f"dual {mode}: attn shape {attn_dual.shape}"
        assert torch.isfinite(ic50_dual).all() and torch.isfinite(attn_dual).all()
        loss_dual = ic50_dual.sum()
        loss_dual.backward()
        # use_main_drug_emb=False 时 proj_drug/drug-BN 死分支无梯度 (M2 已知行为),
        # 只对新增的 flatten_cell_enc 做全梯度断言.
        n_grad_flatten = sum(
            1 for p in model_dual.flatten_cell_enc.parameters()
            if p.grad is not None and p.requires_grad)
        n_total_flatten = sum(1 for _ in model_dual.flatten_cell_enc.parameters())
        assert n_grad_flatten == n_total_flatten, \
            f"dual {mode}: flatten_cell_enc only {n_grad_flatten}/{n_total_flatten} have grad"
        n_grad_head_dual = sum(
            1 for p in model_dual.head_parameters() if p.grad is not None and p.requires_grad)
        n_total_head_dual = sum(1 for _ in model_dual.head_parameters())
        assert n_grad_head_dual == n_total_head_dual, \
            f"dual {mode}: head only {n_grad_head_dual}/{n_total_head_dual} have grad"
        print(f"[smoke] dual_cell={mode} OK | encoder={n_enc_dual:,} "
              f"(flatten_cell_enc={n_flatten:,}) head={n_head_dual:,} fusion_dim={model_dual.predictor.fusion_dim}")

    print("[smoke] ALL OK | gdgn_model.py")


if __name__ == "__main__":
    _smoke_test()
