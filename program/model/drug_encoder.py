"""
Phase 3 Step 2: DrugEncoder (DeepCDR 风格分子图 GCN, PyG+PyTorch 化)
=====================================================================

设计依据: Phase 3 计划 §4.2 (路径 2: 独立分子图 GCN), §4.3 (feats.pkl -> PyG Data),
§4.4 (batch 推理), §0.2.4 (输出 drug_dim=128, Phase 4 cross-attention 投影).

职责
----
- _build_mol_data(drug_idx, drug_mol_graphs) → PyG Data
    把 `data/model/drug_encoder/drug_mol_graphs.pkl` 单个药物条目
    `[atom_features(N,75), adj_list(list[list[int]]), degree_list]` 转为 PyG Data.
    邻接表双向化 (PyG GCNConv 期望), torch.unique(dim=1) 去重.
- DrugEncoder(4 层 GCN(75→256→256→256→128) + BN + ReLU + Dropout)
    forward(atom_feats, edge_index, batch) -> global_max_pool -> (B, 128)
- drug_encoder_forward_batch(enc, drug_idx_batch, drug_mol_graphs, device)
    用 Batch.from_data_list 把变长 N 个分子图堆成单 PyG Batch, 一次前向.

不读取主图 drug_emb (路径 2, 主图嵌入留 Phase 4 ablation 可选融合).
Overlap with Phase 2 §10.7.3: 11 个无 DTI 药物在主图嵌入退化, 但分子图 GCN 给每个
药物独立结构表征, 完全补救.

⚠️ BatchNorm1d B=1 隐患 (Phase 3 计划 §0.3.5): 节点级 BN 在 train 模式遇单原子图
会触发 ValueError. Phase 4 trainer 必须 drop_last=True + 评估切 eval(). Phase 3
smoke 用 B>=2 + 经多药物 Batch 后总原子数通常 >1 不触发.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_max_pool

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_DRUG_MOL_PKL = _PROJECT_ROOT / "data" / "model" / "drug_encoder" / "drug_mol_graphs.pkl"


def _build_mol_data(drug_idx: int, drug_mol_graphs: Mapping[int, Sequence] | Sequence[Sequence]) -> Data:
    """把 drug_mol_graphs[drug_idx] 转为 PyG Data.

    输入条目结构 (Phase 1 §3.4 实际 probe):
        item[0] = np.ndarray (N, 75)    atom features
        item[1] = list[list[int]]        adj_list, adj_list[i] 是 atom i 的邻居列表
        item[2] = list[np.int32]         degree list (本函数不消费, 仅记录)

    边双向化: 对每条 (src -> dst) 同时添加 (dst -> src), 用 PyG GCNConv 默认 expectation.
    torch.unique(dim=1) 去重 (DeepChem 原结构可能含重复边).

    Parameters
    ----------
    drug_idx : int
    drug_mol_graphs : dict[int, [...]] | list[[...]]
        184 个药物的分子图集合; 支持 dict (实际 pickle 格式) 或 list 形式.

    Returns
    -------
    PyG Data with x=(N,75) float, edge_index=(2, E) long (双向无重复)
    """
    item = drug_mol_graphs[drug_idx]
    atom_feats = torch.as_tensor(item[0], dtype=torch.float)   # (N, 75)
    adj_list = item[1]
    edges = []
    for src, nbrs in enumerate(adj_list):
        if hasattr(nbrs, "tolist"):
            nbrs = nbrs.tolist()
        for dst in nbrs:
            edges.append((src, dst))
            edges.append((dst, src))   # 双向化 (GCNConv 默认 expectation)
    if len(edges) == 0:
        edge_index = torch.empty(2, 0, dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = torch.unique(edge_index, dim=1)           # 去重
    return Data(x=atom_feats, edge_index=edge_index)


class DrugEncoder(nn.Module):
    """DeepCDR 风格 4 层 GCN + global max pool.

    Parameters
    ----------
    atom_dim : int
        原子特征维度 (DeepCDR 原 75).
    hidden : int
        中间 3 层 GCN 输出维度 (DeepCDR §4.3.2 原 256; 计划 §4.2 修订).
    out_dim : int
        末层 GCN 输出维度 (DeepCDR 原 128, Phase 4 cross-attention 投影到 256).
    dropout : float
        前 3 层 GCN 之间的 dropout (DeepCDR 0.1).
    """

    def __init__(self, atom_dim: int = 75, hidden: int = 256, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.atom_dim = atom_dim
        self.hidden = hidden
        self.out_dim = out_dim

        self.gcn1 = GCNConv(atom_dim, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.gcn3 = GCNConv(hidden, hidden)
        self.gcn4 = GCNConv(hidden, out_dim)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, atom_feats: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """PyG Batch-级前向.

        Parameters
        ----------
        atom_feats : (N_total, atom_dim)   Batch 拼接后所有原子
        edge_index : (2, E_total)          Batch 拼接后所有边 (含 atom 跨分子偏移)
        batch      : (N_total,)            每个原子归属 (0..B-1)

        Returns
        -------
        drug_emb : (B, out_dim)   B 个分子的池化表示
        """
        h = self.gcn1(atom_feats, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.gcn2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.gcn3(h, edge_index)
        h = self.bn3(h)
        h = F.relu(h)
        # 末层前不加 dropout, 与 DeepCDR 原版对齐

        h = self.gcn4(h, edge_index)            # 末层不激活 (DeepCDR 风格)
        return global_max_pool(h, batch)          # (B, out_dim)


def drug_encoder_forward_batch(
    drug_encoder: DrugEncoder,
    drug_idx_batch: torch.Tensor | Sequence[int],
    drug_mol_graphs: Mapping[int, Sequence] | Sequence[Sequence],
    device: torch.device | str,
) -> torch.Tensor:
    """便利: 给定 drug_idx 批次, 转成 PyG Batch 后一次前向.

    Parameters
    ----------
    drug_encoder : DrugEncoder
        已实例化 + 设备迁移的模块.
    drug_idx_batch : LongTensor | list[int]
        B 个 drug_idx (0..183).
    drug_mol_graphs : dict | list
        184 个药物的分子图集合.

    Returns
    -------
    drug_emb : (B, out_dim)
    """
    datas = [_build_mol_data(int(d), drug_mol_graphs) for d in drug_idx_batch]
    batch = Batch.from_data_list(datas).to(device)
    return drug_encoder(batch.x, batch.edge_index, batch.batch)


def load_drug_mol_graphs(path: str | Path = _DRUG_MOL_PKL) -> dict:
    """加载 drug_mol_graphs.pkl (184 个 dict[int -> [atom_feats, adj_list, degree_list]])."""
    with open(path, "rb") as f:
        graphs = pickle.load(f)
    return graphs


def _smoke_test() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")
    print(f"[smoke] loading drug_mol_graphs.pkl ...")
    graphs = load_drug_mol_graphs()
    n_graphs = len(graphs)
    print(f"[smoke] n_graphs={n_graphs} (expect 184)")
    assert n_graphs == 184, f"expect 184 graphs, got {n_graphs}"

    # ==== 1. 184 个药物全部能转 PyG Data ====
    max_atoms = 0
    total_edges = 0
    for d in range(184):
        data = _build_mol_data(d, graphs)
        assert data.x.shape[1] == 75, f"drug {d} atom dim {data.x.shape[1]} != 75"
        assert data.edge_index.shape[0] == 2, f"drug {d} edge_index shape[0] {data.edge_index.shape[0]} != 2"
        max_atoms = max(max_atoms, int(data.x.shape[0]))
        total_edges += int(data.edge_index.shape[1])
    print(f"[smoke] 184 graphs -> Data OK | max_atoms={max_atoms} total_directed_edges={total_edges}")

    # 单药物 shape 检查
    d0 = _build_mol_data(0, graphs)
    print(f"[smoke] drug 0: x={tuple(d0.x.shape)} edge_index={tuple(d0.edge_index.shape)}")
    assert d0.x.shape[0] == d0.edge_index.max().item() + 1 or d0.edge_index.shape[1] == 0, \
        "edge_index node ids out of range"

    # ==== 2. DrugEncoder 参数量与 forward ====
    enc = DrugEncoder(atom_dim=75, hidden=256, out_dim=128).to(device)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"[smoke] DrugEncoder params={n_params:,} (expect ~185K)")
    assert 100_000 < n_params < 400_000, f"params {n_params} out of expected range"

    # eval 模式 forward (BN 用 running stats 安全; 远离 B=1 训练隐患)
    enc.eval()
    drug_idx = torch.tensor([0, 1, 2, 3, 4], device=device)
    with torch.no_grad():
        emb = drug_encoder_forward_batch(enc, drug_idx, graphs, device)
    assert emb.shape == (5, 128), f"forward batch shape {emb.shape} != (5, 128)"
    assert torch.isfinite(emb).all(), "forward has NaN/Inf"
    print(f"[smoke] forward batch (B=5) OK: emb={tuple(emb.shape)} mean={emb.mean():.4f} std={emb.std():.4f}")

    # 单药物 batch (B=1, eval mode 安全)
    with torch.no_grad():
        emb1 = drug_encoder_forward_batch(enc, [10], graphs, device)
    assert emb1.shape == (1, 128)
    print(f"[smoke] forward batch (B=1) OK: emb={tuple(emb1.shape)}")

    # ==== 3. backward 梯度 ====
    enc.train()
    emb_g = drug_encoder_forward_batch(enc, drug_idx, graphs, device)
    loss = emb_g.sum()
    loss.backward()
    n_total = sum(1 for _ in enc.parameters())
    n_grad = sum(1 for p in enc.parameters() if p.grad is not None)
    print(f"[smoke] backward: {n_grad}/{n_total} params have grad")
    assert n_grad == n_total, f"only {n_grad}/{n_total} params have grad"

    print("[smoke] ALL OK | drug_encoder.py")


if __name__ == "__main__":
    _smoke_test()