"""
Step 1c: PretrainCellSampler
=============================

每个预训练 epoch 打乱 404 个细胞系, 按 batch_condition 输出 cell_idx batch.
策略 B (计划 §0.3.2)下, GNN 前向对一个 batch 内各 cell 分别注入各自 omics (per-cell forward).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class PretrainCellSampler:
    """Yield LongTensor(B,) cell-index mini-batches each epoch.

    Parameters
    ----------
    n_cells : int
        cell_line_canonical_order 行数 (404)
    batch_condition : int
        每个 step 使用的细胞系数 (显存紧张时降至 4 或 1)
    seed : int
        每个 epoch 内打乱用的 RNG seed (seed + epoch_offset 保证 epoch 间不同)
    shuffle : bool
    """

    def __init__(
        self, n_cells: int = 404, batch_condition: int = 16, seed: int = 42,
        shuffle: bool = True, cell_indices: list[int] | np.ndarray | None = None,
    ):
        self.indices = (
            np.arange(int(n_cells), dtype=np.int64)
            if cell_indices is None else np.asarray(cell_indices, dtype=np.int64)
        )
        if self.indices.size == 0 or len(np.unique(self.indices)) != len(self.indices):
            raise ValueError("cell_indices must be nonempty and unique")
        self.n_cells = int(len(self.indices))
        self.batch_condition = int(batch_condition)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self._epoch = 0

    def __len__(self) -> int:
        return (self.n_cells + self.batch_condition - 1) // self.batch_condition

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        if self.shuffle:
            idx = rng.permutation(self.indices)
        else:
            idx = self.indices.copy()
        for i in range(0, self.n_cells, self.batch_condition):
            yield torch.from_numpy(idx[i:i + self.batch_condition]).long()
        self._epoch += 1


def get_n_cells() -> int:
    """读取 cell_line_canonical_order.txt 行数."""
    canon = _PROJECT_ROOT / "data" / "processed" / "cell_line_canonical_order.txt"
    if canon.exists():
        return sum(1 for _ in canon.read_text().splitlines() if _.strip())
    return 404


def _smoke_test() -> None:
    n_cells = get_n_cells()
    sampler = PretrainCellSampler(n_cells=n_cells, batch_condition=16, seed=42)
    print(f"[smoke] n_cells={n_cells} expect batches={(n_cells + 15) // 16}")

    all_idx = []
    n_batches = 0
    for batch in sampler:
        all_idx.append(batch)
        n_batches += 1
        assert batch.dtype == torch.long
        assert int(batch.min()) >= 0 and int(batch.max()) < n_cells
    cat = torch.cat(all_idx)
    uniq = torch.unique(cat)
    assert uniq.numel() == n_cells, f"missing cells: {uniq.numel()} != {n_cells}"
    last = all_idx[-1].shape[0]
    print(f"[smoke] iter: {n_batches} batches | last batch size={last} | unique cells={uniq.numel()}")

    next_iter = list(sampler)
    perm_first = torch.cat(all_idx).tolist()
    perm_second = torch.cat(next_iter).tolist()
    assert perm_first != perm_second, "epoch 0 and epoch 1 must differ (shuffle changes with epoch)"
    print(f"[smoke] epoch 0 / epoch 1 permutation differ: True")

    small = PretrainCellSampler(n_cells=5, batch_condition=2, seed=1)
    bs = list(small)
    assert len(bs) == 3, f"ceil(5/2)=3, got {len(bs)}"
    assert bs[0].shape[0] == 2 and bs[-1].shape[0] == 1
    print(f"[smoke] ceil(5/2) split OK: shapes={[tuple(b.shape) for b in bs]}")

    print("[smoke] ALL OK | pretrain_dataloader.py")


if __name__ == "__main__":
    _smoke_test()
