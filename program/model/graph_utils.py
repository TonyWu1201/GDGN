"""
Phase 3 Step 1: 公共图边拆分 helper (技术债清理)
=================================================

Phase 2 中"满图 PPI/DTI 拆 fwd/rev"逻辑原本分散在两处:
- `edge_mask.py:216-217` (visible 边拆 fwd/rev, 算法略不同 — 给训练用)
- `pretrain.py:131-139` (满图按 i<=j 拆 fwd/rev + DTI 手工反向 — 给评估用)

Phase 3 GeneEncoder 也需要同一种"满图拆分"用于 buffer 缓存 (计划 §3.3).
为避免 3 处副本漂移, 抽出公共 helper.

拆分算法 (与 `pretrain.py:131-139` 对齐, 详见 Phase 3 计划 §0.3.3):
    PPI:  fwd = edges[:, src<=dst];  rev = edges[:, src>dst]
          若 rev 为空 (极端无 src>dst 边), 用 fwd 手工反向填 rev.
    DTI:  fwd = hetero['drug','targets','gene'].edge_index (单向 drug->gene)
          rev = torch.stack([fwd[1], fwd[0]])    (手工翻转, 让 drug 节点接收 gene 消息)

输入 `hetero` 由 `program.model.dataset.load_hetero_graph` 加载的 PyG HeteroData.
"""
from __future__ import annotations

import torch


def split_full_edges(hetero) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """从基图 HeteroData 拆出满图 PPI/DTI 的 fwd/rev 边索引.

    Parameters
    ----------
    hetero : torch_geometric.data.HeteroData
        Phase 1 基图, 必须含两类边:
        - hetero["gene", "ppi", "gene"].edge_index   (2, E_ppi) 双向无向 PPI
        - hetero["drug", "targets", "gene"].edge_index (2, E_dti) 单向 DTI (drug->gene)

    Returns
    -------
    (ppi_fwd, ppi_rev, dti_fwd, dti_rev) : 每个 (2, E) LongTensor
        ppi_fwd : 仅含 src<=dst 的 PPI 边 (i->j 形式, i<=j)
        ppi_rev : 仅含 src>dst 的 PPI 边 (i->j 形式, i>j); 若原全图无 src>dst 边, 用 fwd 反向填充
        dti_fwd : 原始单向 DTI (drug->gene)
        dti_rev : dti_fwd 手工反向 (gene->drug), 让 drug 节点接收 gene 消息
    """
    ppi_full = hetero["gene", "ppi", "gene"].edge_index
    kh = ppi_full[0] <= ppi_full[1]
    ppi_fwd = ppi_full[:, kh]
    ppi_rev = ppi_full[:, ~kh]
    if ppi_rev.shape[1] == 0:
        ppi_rev = torch.stack([ppi_fwd[1], ppi_fwd[0]], dim=0)
    dti_fwd = hetero["drug", "targets", "gene"].edge_index
    dti_rev = torch.stack([dti_fwd[1], dti_fwd[0]], dim=0)
    return ppi_fwd, ppi_rev, dti_fwd, dti_rev


def _smoke_test() -> None:
    from program.model.dataset import load_hetero_graph

    print("[smoke] loading hetero_graph_base.pt ...")
    hetero = load_hetero_graph()
    ppi_fwd, ppi_rev, dti_fwd, dti_rev = split_full_edges(hetero)

    n_genes = int(hetero["gene"].num_nodes)
    n_drugs = int(hetero["drug"].num_nodes)
    ppi_full = hetero["gene", "ppi", "gene"].edge_index
    dti_full = hetero["drug", "targets", "gene"].edge_index

    print(f"[smoke] ppi_fwd={tuple(ppi_fwd.shape)} ppi_rev={tuple(ppi_rev.shape)} "
          f"(|ppi_full|/2 should ~= fwd: {ppi_full.shape[1] // 2})")
    print(f"[smoke] dti_fwd={tuple(dti_fwd.shape)} dti_rev={tuple(dti_rev.shape)} "
          f"(should equal dti_full: {tuple(dti_full.shape)})")

    assert ppi_fwd.dtype == torch.long and ppi_fwd.shape[0] == 2
    assert ppi_rev.dtype == torch.long and ppi_rev.shape[0] == 2
    assert dti_fwd.dtype == torch.long and dti_fwd.shape[0] == 2
    assert dti_rev.dtype == torch.long and dti_rev.shape[0] == 2

    assert int(ppi_fwd[0].min()) >= 0 and int(ppi_fwd[1].max()) < n_genes, "ppi_fwd out of gene range"
    assert int(ppi_rev[0].min()) >= 0 and int(ppi_rev[1].max()) < n_genes, "ppi_rev out of gene range"
    assert (ppi_fwd[0] <= ppi_fwd[1]).all(), "ppi_fwd must have src<=dst"
    if ppi_rev.shape[1] > 0 and ppi_rev is not ppi_fwd:
        assert (ppi_rev[0] >= ppi_rev[1]).all() or ppi_rev.shape[1] == 0, \
            "ppi_rev must have src>=dst (or be empty backfilled)"

    assert int(dti_fwd[0].min()) >= 0 and int(dti_fwd[0].max()) < n_drugs, "dti_fwd drug out of range"
    assert int(dti_fwd[1].min()) >= 0 and int(dti_fwd[1].max()) < n_genes, "dti_fwd gene out of range"
    assert int(dti_rev[0].min()) >= 0 and int(dti_rev[0].max()) < n_genes, "dti_rev gene out of range"
    assert int(dti_rev[1].min()) >= 0 and int(dti_rev[1].max()) < n_drugs, "dti_rev drug out of range"
    assert torch.equal(dti_fwd[0], dti_rev[1]) and torch.equal(dti_fwd[1], dti_rev[0]), \
        "dti_rev must be exact flip of dti_fwd"

    print("[smoke] ALL OK | graph_utils.py")


if __name__ == "__main__":
    _smoke_test()