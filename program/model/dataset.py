"""
Step 3b/3c: GDGNDataset + DataLoader (策略 B：静态基图常驻 + batch 动态注入)
================================================================================

核心机制
--------
- 基图 `hetero_graph_base.pt` 全程常驻，不每样本 clone
- __getitem__ 只返回索引元组（cell_idx, drug_idx, y）
- 每个训练/推理步：训练器按 batch 的 cell_idx/drug_idx 从预量化的
  cell_line_features / drug 嵌入矩阵中依索引取行（dynamic injection），注入基因节点
  x_dynamic / 药物 target 嵌入；pathway_activity 取 (B, 186) 作 batch 级特征

Dataset 返回与 collate
----------------------
- __getitem__: {'cell_idx': int, 'drug_idx': int, 'y': tensor scalar}
- collate (默认): 简单堆成 batch → {'cell_idx': LongTensor(B), 'drug_idx': LongTensor(B), 'y': FloatTensor(B)}
- 实际组学/药物特征 → by `extract_batch_features(batch, drug_emb_matrix, drug_feat_keys …)` 由 training step 调用

设计的便利函数
-------------
  get_dataloaders(batch_size=32, split_seed=…, num_workers=0, pin_memory=True)
  iter_batch_omics(loader, cell_line_features, device)
       -> yields (cell_idx_batch, drug_idx_batch, omics dict {'expr','mut','cnv','meth'}, pathway_batch, y)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data"
PROCESSED = DATA / "processed"
MODEL_GRAPH = DATA / "model" / "hetero_graph"
MODEL_DRUG = DATA / "model" / "drug_encoder"

HETERO_PT = MODEL_GRAPH / "hetero_graph_base.pt"
CELL_FEAT_PT = MODEL_GRAPH / "cell_line_features.pt"
DRUG_FEAT_PT = MODEL_GRAPH / "drug_features.pt"
DRUG_MOL_PKL = MODEL_DRUG / "drug_mol_graphs.pkl"

SPLIT_PT = PROCESSED / "sample_pairs_split.pt"
CELL_IDX_JSON = PROCESSED / "cell_line_to_idx.json"
DRUG_IDX_JSON = PROCESSED / "drug_cid_to_idx.json"
CANON_TXT = PROCESSED / "cell_line_canonical_order.txt"


class GDGNDataset(Dataset):
    """策略 B 的样本对数据集：基图常驻 + 索引式 __getitem__。

    Parameters
    ----------
    split : 'train' | 'val' | 'test'
        sample_pairs_split.pt 中的子集。
    load_hetero : bool
        若 True，则在 __init__ 时把 HeteroData 基图加载到 self.hetero（常驻 CPU 内存）；
        训练器可再 .to(device) 搬走。若为 False，则跳过（仅取样本对）。
    load_drugs_mol : bool
        是否加载 drug_mol_graphs.pkl（变长分子图，Phase 3 DrugEncoder 用）。
    """

    def __init__(self, split: str = "train", load_hetero: bool = True, load_drugs_mol: bool = False):
        assert split in {"train", "val", "test"}, split
        self.split = split
        splits = torch.load(SPLIT_PT, weights_only=False)
        assert split in splits, f"split '{split}' not in {list(splits.keys())}"
        self.pairs = splits[split]
        self.n_pairs = len(self.pairs)

        meta = torch.load(HETERO_PT, weights_only=False).meta if Path(HETERO_PT).exists() else {}
        self.n_genes: int = int(meta.get("n_genes", -1))
        self.n_drugs: int = int(meta.get("n_drugs", -1))

        self.cell_line_order: list[str] = CANON_TXT.read_text().splitlines() if Path(CANON_TXT).exists() else []
        if not self.cell_line_order:
            # fall back via json
            ci = json.loads(CELL_IDX_JSON.read_text())
            self.cell_line_order = [k for k, _ in sorted(ci.items(), key=lambda kv: kv[1])]

        # cell_line_features.pt: dict of stacked tensors + orders
        if Path(CELL_FEAT_PT).exists():
            clf = torch.load(CELL_FEAT_PT, weights_only=False)
            self.cell_line_features = clf
        else:
            self.cell_line_features = None

        # drug features (ECFP-style fixed-length, 主图节点的 x)
        if Path(DRUG_FEAT_PT).exists():
            self.drug_features = torch.load(DRUG_FEAT_PT, weights_only=False)
        else:
            self.drug_features = None

        # 基图常驻（可选）
        self.hetero = None
        if load_hetero and Path(HETERO_PT).exists():
            self.hetero = torch.load(HETERO_PT, weights_only=False)

        # 分子图（可选）
        self.drug_mol_graphs = None
        if load_drugs_mol and Path(DRUG_MOL_PKL).exists():
            import pickle
            with open(DRUG_MOL_PKL, "rb") as f:
                self.drug_mol_graphs = pickle.load(f)

    def __len__(self) -> int:
        return self.n_pairs

    def __getitem__(self, idx: int) -> dict:
        p = self.pairs[idx]
        return {
            "cell_idx": int(p["cell_idx"]),
            "drug_idx": int(p["drug_idx"]),
            "y": torch.tensor(float(p["ic50"]), dtype=torch.float32),
        }


def gdgn_collate(batch: list[dict]) -> dict:
    """默认 collate：堆索引 + 标签；不在此处注入组学/分子图，留给训练步。"""
    cell_idx = torch.tensor([b["cell_idx"] for b in batch], dtype=torch.long)
    drug_idx = torch.tensor([b["drug_idx"] for b in batch], dtype=torch.long)
    y = torch.stack([b["y"] for b in batch], dim=0)
    return {"cell_idx": cell_idx, "drug_idx": drug_idx, "y": y}


# ---------------- 工厂函数 ----------------

def get_dataloaders(
    batch_size: int = 32,
    split_seed: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    load_hetero: bool = False,  # DataLoader 内通常不加载基图，避免 worker 重复
) -> dict:
    """Return {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}."""
    loaders = {}
    for sp in ["train", "val", "test"]:
        ds = GDGNDataset(split=sp, load_hetero=load_hetero, load_drugs_mol=False)
        shuffle = (sp == "train")
        loader = TorchDataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            collate_fn=gdgn_collate,
            drop_last=(sp == "train"),
        )
        loaders[sp] = loader
    return loaders


# ---------------- 便利：动态注入辅助 ----------------

def load_hetero_graph(device: torch.device | str | None = None):
    """加载基图（常驻），命中的话搬到 device。返回 PyG HeteroData。"""
    data = torch.load(HETERO_PT, weights_only=False)
    if device is not None:
        data = data.to(device)
    return data


def load_drug_emb_matrix(path: Path = DRUG_FEAT_PT, device: torch.device | str | None = None) -> torch.Tensor:
    """加载药物嵌入矩阵 (184, 1030)。供训练步按 drug_idx 取 (B, drug_dim)。"""
    m = torch.load(path, weights_only=False)
    if device is not None:
        m = m.to(device)
    return m


def load_cell_line_features(device: torch.device | str | None = None) -> dict:
    """加载细胞系组学 dict。键：expression/mutation/copynumber/methylation/pathway_activity + order lists."""
    clf = torch.load(CELL_FEAT_PT, weights_only=False)
    if device is not None:
        for k, v in clf.items():
            if isinstance(v, torch.Tensor):
                clf[k] = v.to(device)
    return clf


def inject_batch_omics(
    cell_line_features: dict,
    cell_idx_batch: torch.LongTensor,
) -> dict:
    """从预加载的 dict 中按 batch 取出组学切片。

    Return
    ------
        {
          'expr':    FloatTensor (B, n_genes),
          'mut':     FloatTensor (B, n_genes),
          'cnv':     FloatTensor (B, n_genes),
          'meth':    FloatTensor (B, n_genes),
          'pathway': FloatTensor (B, n_pathways),
        }
    """
    return {
        "expr": cell_line_features["expression"][cell_idx_batch],
        "mut": cell_line_features["mutation"][cell_idx_batch],
        "cnv": cell_line_features["copynumber"][cell_idx_batch],
        "meth": cell_line_features["methylation"][cell_idx_batch],
        "pathway": cell_line_features["pathway_activity"][cell_idx_batch],
    }


# ---------------- smoke test / 单元测试 ----------------

def _smoke_test() -> None:
    """小测试：加载 train/val/test，跑一个 batch 检查形状与索引合法性。"""
    loaders = get_dataloaders(batch_size=32, num_workers=0, pin_memory=False, load_hetero=False)
    clf = load_cell_line_features()
    drug_emb = load_drug_emb_matrix()
    loader = loaders["train"]
    batch = next(iter(loader))
    assert batch["cell_idx"].shape[0] == batch["drug_idx"].shape[0] == batch["y"].shape[0] == 32
    assert int(batch["cell_idx"].min()) >= 0 and int(batch["cell_idx"].max()) < 404
    assert int(batch["drug_idx"].min()) >= 0 and int(batch["drug_idx"].max()) < 184
    omics = inject_batch_omics(clf, batch["cell_idx"])
    n_genes = clf["expression"].shape[1]
    n_pw = clf["pathway_activity"].shape[1]
    for k in ["expr", "mut", "cnv", "meth"]:
        assert omics[k].shape == (32, n_genes), f"{k} shape {omics[k].shape} != (32, {n_genes})"
    assert omics["pathway"].shape == (32, n_pw)
    drug_target = drug_emb[batch["drug_idx"]]
    assert drug_target.shape == (32, drug_emb.shape[1])
    print(f"[smoke] OK | batch_size=32 y mean={batch['y'].mean():.3f} "
          f"drug_emb sampled={drug_target.shape} "
          f"omics expr mean={omics['expr'].mean():.3f} pathway shape={omics['pathway'].shape}")

    # base graph load test
    g = load_hetero_graph()
    assert g["gene"].x.shape[0] == clf["gene_order"].__len__()
    assert g["drug"].x.shape[0] == drug_emb.shape[0] == 184
    print(f"[smoke] hetero: gene.x={tuple(g['gene'].x.shape)} drug.x={tuple(g['drug'].x.shape)} "
          f"ppi={g['gene','ppi','gene'].edge_index.shape[1]} dti={g['drug','targets','gene'].edge_index.shape[1]}")


if __name__ == "__main__":
    _smoke_test()