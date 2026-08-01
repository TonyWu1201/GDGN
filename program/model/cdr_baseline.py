"""
Phase 4.2 补充: CDRBaselineModel (DeepCDR 原版逐层忠实 PyTorch 复刻)
=====================================================================

设计依据: DeepCDR (Liu et al., Bioinformatics 2020) `prog/model.py`
`KerasMultiSourceGCNModel` + `prog/layers/graph.py` `GraphConv`, 原样映射到
PyTorch, 喂 GDGN 数据管线 (drug_mol_graphs.pkl + cell_line_features.pt).

与 baseline_simple (简化版) 的区别
----------------------------------
baseline_simple 只借用了 DeepCDR 的分子图 GCN + flatten MLP cell; 本文件
CDRBaselineModel 是**完整原版复刻**: 四分支 + 1D-CNN head 逐层对齐 Keras 代码.

逐层映射表 (原版 Keras -> 本文件 PyTorch)
------------------------------------------
| 原版 (model.py)                        | 本文件                                  |
|---                                     |---                                      |
| GraphConv(256)x3 + ReLU+BN+Dropout0.1  | GraphConv(256)x3 + ReLU+BN+Dropout0.1  |
| GraphConv(100) + ReLU+BN+Dropout0.1    | GraphConv(100) + ReLU+BN+Dropout0.1    |
| GlobalMaxPooling1D -> (B,100)          | max(dim=1) -> (B,100)                  |
| mut: Conv2D(50,(1,700),s(1,5),tanh)    | Conv1d(1,50,700,s5)+tanh               |
|      MaxPool(1,5)                      |      MaxPool1d(5)                      |
|      Conv2D(30,(1,5),s(1,2),relu)      |      Conv1d(50,30,5,s2)+relu           |
|      MaxPool(1,10) -> Flatten          |      MaxPool1d(10) -> flatten           |
|      Dense(100,relu)+Dropout0.1        |      Linear(450,100)+relu+Dropout0.1   |
| gexp: Dense(256,tanh)+BN+Drop0.1       | Linear(8412,256)+tanh+BN+Dropout0.1   |
|       Dense(100,relu)                  |       Linear(256,100)+relu             |
| methy: 同 gexp                         | 同 gexp                               |
| cat([drug,mut,gexp,methy]) -> 400      | cat -> 400                            |
| Dense(300,tanh)+Dropout0.1             | Linear(400,300)+tanh+Dropout0.1       |
| reshape (B,1,300,1)                    | unsqueeze(1) -> (B,1,300)              |
| Conv2D(30,(1,150),relu)+MaxPool(1,2)   | Conv1d(1,30,150)+relu+MaxPool1d(2)    |
| Conv2D(10,(1,5),relu)+MaxPool(1,3)     | Conv1d(30,10,5)+relu+MaxPool1d(3)     |
| Conv2D(5,(1,5),relu)+MaxPool(1,3)      | Conv1d(10,5,5)+relu+MaxPool1d(3)      |
| Flatten+Dropout0.2 -> Dense(1)         | flatten+Dropout0.2 -> Linear(30,1)    |

与数据相关的适配 (原版 -> GDGN)
-------------------------------
- Max_atoms: 100 -> 100 (GDGN max=96, 兼容).
- mutation_dim: 34673 (位点) -> 8412 (per-gene mutation 通道, 0/1).
  卷积核/步长/pool 不变: conv 后 L = 1543 -> /5=308 -> (308-5)/2+1=152 -> /10=15,
  flatten = 30*15 = 450 (原版 30*67=2010), Dense(100) 不变.
- gexp_dim / methy_dim: 697 / 808 -> 8412 (per-gene 通道), Dense(256) 不变.
- cnv 通道不用 (原版无 cnv 分支, use_mut/use_gexp/use_methy 三开关忠实保留).
- 药物图预处理忠实复刻 CalculateGraphFeat: 原子特征零填充至 100, 邻接矩阵
  真实块 NormalizeAdj (A+I 对称归一化) + 填充块单位阵, 拼成 (100,100) 张量.
- GraphConv 原版消息传递: h = A^T (H W + b), **不**除以度 (原版注释掉了除法),
  A 已含自环 + 对称归一化. kernel 初始化 glorot_uniform (xavier_uniform).

训练建议 (承自原版 run_DeepCDR.py: Adam lr=0.001, batch=64, patience=10)
-----------------------------------------------------------------------
- 全网络从零训练, encoder/head 都用大 lr (默认 lr_encoder=1e-3, lr_head=1e-3),
  与 baseline_simple (lr=1e-5) 不同 —— 本模型无预训练 encoder.
- 若 7 无 DTI / LODO 表现不如 baseline_simple, 属预期 (原版就是成对随机分口径
  的模型, 药物 GCN 纯结构特征无 DTI 先验).

接口 (与 GDGNModel / BaselineSimpleModel 同 ABI)
------------------------------------------------
forward(cell_idx, drug_idx, omics) -> (ic50_pred (B,1), None)
encoder_parameters() / head_parameters() 供 trainer 分组 lr.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.model.drug_encoder import load_drug_mol_graphs

MAX_ATOMS = 100  # 原版 Max_atoms, GDGN 分子图 max=96
N_GENES = 8412  # GDGN per-gene omics 通道数 (expr/mut/cnv/meth 共用)


def normalize_adj(adj: np.ndarray) -> np.ndarray:
    """复刻 run_DeepCDR.py NormalizeAdj: A + I, D^-1/2 A D^-1/2 对称归一化."""
    adj = adj + np.eye(adj.shape[0])
    d = np.power(np.array(adj.sum(1)), -0.5).flatten()
    d[np.isinf(d)] = 0.0
    d_mat = np.diag(d)
    return d_mat @ adj @ d_mat


def build_padded_drug_tensor(drug_mol_graphs: dict) -> tuple[np.ndarray, np.ndarray]:
    """复刻 CalculateGraphFeat: 全部药物预计算 (N,100,75) 特征 + (N,100,100) 邻接.

    真实原子块: 特征原样 + 邻接 NormalizeAdj (含自环对称归一化);
    填充块: 特征 0 + 邻接单位阵 (NormalizeAdj(zeros) = I, 与原版一致).
    """
    n_drugs = len(drug_mol_graphs)
    feat_all = np.zeros((n_drugs, MAX_ATOMS, 75), dtype="float32")
    adj_all = np.zeros((n_drugs, MAX_ATOMS, MAX_ATOMS), dtype="float32")
    for d in range(n_drugs):
        feat_mat, adj_list, _ = drug_mol_graphs[d]
        n_atom = int(feat_mat.shape[0])
        feat_all[d, :n_atom, :] = feat_mat
        adj = np.zeros((MAX_ATOMS, MAX_ATOMS), dtype="float32")
        for i in range(n_atom):
            nbrs = adj_list[i]
            if hasattr(nbrs, "tolist"):
                nbrs = nbrs.tolist()
            for j in nbrs:
                adj[i, int(j)] = 1.0
        adj_all[d, :n_atom, :n_atom] = normalize_adj(adj[:n_atom, :n_atom])
        adj_all[d, n_atom:, n_atom:] = normalize_adj(adj[n_atom:, n_atom:])
    return feat_all, adj_all


class GraphConv(nn.Module):
    """复刻原版 GraphConv: h = A^T (H W + b), 不除以度, glorot_uniform 初始化.

    Keras 实现:
        features = dot(features, W); features += b
        return batch_dot(permute_dimensions(edges, (0,2,1)), features)
    A 为预归一化邻接 (含自环 + 对称归一化), A^T = A (对称), 此处保留转置写法.
    """

    def __init__(self, in_dim: int, units: int):
        super().__init__()
        bound = math.sqrt(6.0 / (in_dim + units))
        self.W = nn.Parameter(torch.empty(in_dim, units).uniform_(-bound, bound))
        self.b = nn.Parameter(torch.zeros(units))

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.bmm(adj.transpose(1, 2), h @ self.W + self.b)


class CDRBaselineModel(nn.Module):
    """DeepCDR 原版完整复刻 (regression), 喂 GDGN 数据管线.

    与 GDGNModel 一样持 self._device 而非 self.device (Phase 4 评估项 S2).
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        unit_list: list[int] | None = None,
        use_relu: bool = True,
        use_bn: bool = True,
        use_GMP: bool = True,
        use_mut: bool = True,
        use_gexp: bool = True,
        use_methy: bool = True,
    ):
        super().__init__()
        self._device = torch.device(device)
        self.use_relu = use_relu
        self.use_bn = use_bn
        self.use_GMP = use_GMP
        self.use_mut = use_mut
        self.use_gexp = use_gexp
        self.use_methy = use_methy
        units = unit_list if unit_list is not None else [256, 256, 256]
        assert len(units) >= 1

        # ---- drug GCN 分支 (原版: 3x256 + 100, 每层后 relu/bn/dropout0.1) ----
        conv_specs = units + [100]
        in_dim = 75
        self.gcn_convs = nn.ModuleList()
        self.gcn_bns = nn.ModuleList()
        for u in conv_specs:
            self.gcn_convs.append(GraphConv(in_dim, u))
            self.gcn_bns.append(nn.BatchNorm1d(u))
            in_dim = u
        self.gcn_dropout = nn.Dropout(0.1)

        # ---- mutation CNN 分支 (原版: 2xConv2D -> Flatten -> Dense100) ----
        # 输入 (B,1,8412); conv1 k=700 s=5 -> 1543; pool5 -> 308; conv2 k=5 s=2 -> 152;
        # pool10 -> 15; flatten 30*15=450
        self.mut_conv1 = nn.Conv1d(1, 50, kernel_size=700, stride=5)
        self.mut_conv2 = nn.Conv1d(50, 30, kernel_size=5, stride=2)
        self.mut_fc = nn.Linear(30 * 15, 100)

        # ---- gexp / methy MLP 分支 (原版: Dense256 tanh+BN+Dropout0.1 -> Dense100 relu) ----
        self.gexp_fc1 = nn.Linear(N_GENES, 256)
        self.gexp_bn1 = nn.BatchNorm1d(256)
        self.gexp_fc2 = nn.Linear(256, 100)
        self.methy_fc1 = nn.Linear(N_GENES, 256)
        self.methy_bn1 = nn.BatchNorm1d(256)
        self.methy_fc2 = nn.Linear(256, 100)

        # ---- 融合 head (原版: Dense300 tanh -> 1D-CNN x3 -> Dense1) ----
        branch_dim = 100 + (100 if use_mut else 0) + (100 if use_gexp else 0) + (100 if use_methy else 0)
        self.head_fc1 = nn.Linear(branch_dim, 300)
        self.head_conv1 = nn.Conv1d(1, 30, kernel_size=150)   # 300 -> 151 -> /2=75
        self.head_conv2 = nn.Conv1d(30, 10, kernel_size=5)    # 75 -> 71 -> /3=23
        self.head_conv3 = nn.Conv1d(10, 5, kernel_size=5)     # 23 -> 19 -> /3=6
        self.head_out = nn.Linear(5 * 6, 1)
        self.dropout01 = nn.Dropout(0.1)
        self.dropout02 = nn.Dropout(0.2)

        # ---- 药物图预处理 (预计算全部 184 个药物的 padded 特征 + 归一化邻接) ----
        self.drug_mol_graphs = load_drug_mol_graphs()
        feat_all, adj_all = build_padded_drug_tensor(self.drug_mol_graphs)
        self.register_buffer("drug_feat_all", torch.from_numpy(feat_all))
        self.register_buffer("drug_adj_all", torch.from_numpy(adj_all))

    # ---- 分支实现 (与 Keras 代码逐行对应) ----

    def _drug_gcn(self, drug_idx: torch.Tensor) -> torch.Tensor:
        feat = self.drug_feat_all[drug_idx].to(self._device)  # (B,100,75)
        adj = self.drug_adj_all[drug_idx].to(self._device)    # (B,100,100)
        h = feat
        for conv, bn in zip(self.gcn_convs, self.gcn_bns):
            h = conv(h, adj)
            if self.use_relu:
                h = F.relu(h)
            if self.use_bn:
                h = bn(h.transpose(1, 2)).transpose(1, 2)  # BN per-feature over B*N
            h = self.gcn_dropout(h)
        if self.use_GMP:
            return h.max(dim=1).values  # (B,100) GlobalMaxPooling1D
        return h.mean(dim=1)  # GlobalAveragePooling1D (use_GMP=False)

    def _mut_branch(self, omics: dict) -> torch.Tensor:
        x = omics["mut"].to(self._device).unsqueeze(1)  # (B,1,8412)
        x = torch.tanh(self.mut_conv1(x))
        x = F.max_pool1d(x, 5)
        x = F.relu(self.mut_conv2(x))
        x = F.max_pool1d(x, 10)
        x = F.relu(self.mut_fc(x.flatten(1)))
        return self.dropout01(x)  # (B,100)

    def _mlp_branch(self, x: torch.Tensor, fc1, bn1, fc2) -> torch.Tensor:
        x = torch.tanh(fc1(x.to(self._device)))
        if self.use_bn:
            x = bn1(x)
        x = self.dropout01(x)
        return F.relu(fc2(x))

    def _head(self, fused: torch.Tensor) -> torch.Tensor:
        x = self.dropout01(torch.tanh(self.head_fc1(fused)))  # (B,300)
        x = x.unsqueeze(1)  # (B,1,300) <-> Keras (B,1,300,1)
        x = F.max_pool1d(F.relu(self.head_conv1(x)), 2)       # (B,30,75)
        x = F.max_pool1d(F.relu(self.head_conv2(x)), 3)       # (B,10,23)
        x = F.max_pool1d(F.relu(self.head_conv3(x)), 3)       # (B,5,6)
        x = self.dropout02(x.flatten(1))                      # (B,30)
        return self.head_out(x)  # (B,1)

    # ---- 主 ABI ----

    def forward(
        self,
        cell_idx: torch.Tensor,
        drug_idx: torch.Tensor,
        omics: dict,
    ) -> tuple[torch.Tensor, None]:
        """返回 (ic50_pred (B,1), None). None 占位与 GDGNModel 对齐."""
        x_drug = self._drug_gcn(drug_idx)
        parts = [x_drug]
        if self.use_mut:
            parts.append(self._mut_branch(omics))
        if self.use_gexp:
            parts.append(self._mlp_branch(omics["expr"], self.gexp_fc1, self.gexp_bn1, self.gexp_fc2))
        if self.use_methy:
            parts.append(self._mlp_branch(omics["meth"], self.methy_fc1, self.methy_bn1, self.methy_fc2))
        fused = torch.cat(parts, dim=-1)
        return self._head(fused), None

    def encoder_parameters(self):
        params = (
            list(self.gcn_convs.parameters())
            + list(self.gcn_bns.parameters())
            + list(self.mut_conv1.parameters())
            + list(self.mut_conv2.parameters())
            + [self.mut_fc.weight, self.mut_fc.bias]
            + list(self.gexp_fc1.parameters()) + list(self.gexp_bn1.parameters())
            + [self.gexp_fc2.weight, self.gexp_fc2.bias]
            + list(self.methy_fc1.parameters()) + list(self.methy_bn1.parameters())
            + [self.methy_fc2.weight, self.methy_fc2.bias]
        )
        return params

    def head_parameters(self):
        return (
            list(self.head_fc1.parameters())
            + list(self.head_conv1.parameters())
            + list(self.head_conv2.parameters())
            + list(self.head_conv3.parameters())
            + [self.head_out.weight, self.head_out.bias]
        )


def _smoke_test() -> None:
    from program.model.dataset import inject_batch_omics, load_cell_line_features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")

    clf = load_cell_line_features(device)
    model = CDRBaselineModel(device=device).to(device)
    n_enc = sum(p.numel() for p in model.encoder_parameters())
    n_head = sum(p.numel() for p in model.head_parameters())
    print(f"[smoke] cdr_baseline encoder_params={n_enc:,} head_params={n_head:,}")

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

    # 开关组合: 只开 mut + gexp (use_methy=False) 的 ABI 兼容
    model2 = CDRBaselineModel(device=device, use_methy=False).to(device)
    model2.eval()
    with torch.no_grad():
        ic50_2, _ = model2(cell_idx, drug_idx, omics)
    assert ic50_2.shape == (4, 1)
    print(f"[smoke] use_methy=False forward OK: ic50={tuple(ic50_2.shape)}")

    # 预计算张量完整性
    print(f"[smoke] drug_feat_all={tuple(model.drug_feat_all.shape)} "
          f"drug_adj_all={tuple(model.drug_adj_all.shape)}")
    assert model.drug_feat_all.shape == (184, 100, 75)
    assert model.drug_adj_all.shape == (184, 100, 100)

    print("[smoke] ALL OK | cdr_baseline.py")


if __name__ == "__main__":
    _smoke_test()
