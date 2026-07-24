"""
Step 1: 边 mask + 负采样 + 固定验证边集 (EdgeMaskSampler)
=========================================================

职责
----
- 从 Phase 1 基图 (HeteroData) 读出双向 PPI (2, 672784) 与单向 DTI (2, 1124)
- 每步动态采样:
    * PPI: 成对 mask 20% 无向边 -> 双向可见边移除被 mask 的对
            拆出 visible_fwd (i->j, i<j) 与 visible_rev (j->i, i<j) 给 HeteroConv
    * DTI: 单向 mask 40% (drug->gene) -> visible_fwd 是剩余 (drug->gene), visible_rev 是手工翻转 (gene->drug)
- 负采样: 排除全集边集合 (ppi_full_set / dti_full_set) 防假阴性 (见计划 §0.3.1)
    * PPI 负样本候选 = (gene, gene) \\ ppi_full_set, 无向双向采样
    * DTI 负样本候选 = (drug in 177, gene) \\ dti_full_set (§0.2.4: 7 无 DTI 药物不进负采样)
- held-out 边集: PPI 1% / DTI 20% 一次性冻结全程不进训练 mask; 同步生成固定负样本攻击集

输出格式 (sample_ppi / sample_dti 返回)
---------------------------------------
{
  'ppi': {'visible_fwd': (2,E), 'visible_rev': (2,E), 'pos': (2,P), 'neg': (2,N)},
  'dti': {'visible_fwd': (2,B), 'visible_rev': (2,B), 'pos': (2,Q), 'neg': (2,M)},
}
其中 pos/neg 均用单向表示, P:N = 1:neg_ratio
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_GRAPH = PROJECT_ROOT / "data" / "model" / "hetero_graph"
PRETRAIN_DIR = _PROJECT_ROOT / "data" / "model" / "pretrain"
PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)

HETERO_PT = MODEL_GRAPH / "hetero_graph_base.pt"
HELDOUT_PT = PRETRAIN_DIR / "heldout_edges.pt"


def _edge_key_sym(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    return lo * np.int64(n) + hi


def _edge_key_asym(src: np.ndarray, dst: np.ndarray, n_dst: int) -> np.ndarray:
    return src.astype(np.int64) * np.int64(n_dst) + dst.astype(np.int64)


class EdgeMaskSampler:
    """采样 mask 边 + 负样本 + 持有 held-out 验证边集.

    Parameters
    ----------
    hetero : PyG HeteroData
        由 program/model/dataset.load_hetero_graph() 加载
    cfg : dict-like
        必须含: mask_ppi_ratio, mask_dti_ratio, neg_ratio, heldout_ppi_ratio, heldout_dti_ratio, seed
    """

    def __init__(self, hetero, cfg):
        self.cfg = cfg
        seed = int(getattr(cfg, "seed", 42))
        self.rng = np.random.default_rng(seed)

        ppi_eidx = hetero["gene", "ppi", "gene"].edge_index.long().cpu().numpy()
        dti_eidx = hetero["drug", "targets", "gene"].edge_index.long().cpu().numpy()

        self.n_genes = int(hetero["gene"].num_nodes)
        self.n_drugs = int(hetero["drug"].num_nodes)

        sorted_eidx = np.sort(ppi_eidx, axis=0)
        no_sl = sorted_eidx[0] != sorted_eidx[1]
        sorted_eidx = sorted_eidx[:, no_sl]
        self.ppi_unique: np.ndarray = np.unique(sorted_eidx, axis=1)

        self.dti_edges: np.ndarray = dti_eidx
        self.drug_idx_with_dti: np.ndarray = np.unique(dti_eidx[0])

        ppi_keys = _edge_key_sym(self.ppi_unique[0], self.ppi_unique[1], self.n_genes)
        self.ppi_full_keys: np.ndarray = ppi_keys

        dti_keys = _edge_key_asym(self.dti_edges[0], self.dti_edges[1], self.n_genes)
        self.dti_full_keys: np.ndarray = np.unique(dti_keys)

        self._ppi_train_idx: np.ndarray | None = None
        self._dti_train_idx: np.ndarray | None = None

    def _assert_heldout_exists(self):
        if self._ppi_train_idx is None:
            raise RuntimeError("heldout not built/loaded; call load_or_build_heldout() first")

    def build_heldout(self) -> dict:
        cfg = self.cfg
        ppi_ratio = float(cfg.heldout_ppi_ratio)
        dti_ratio = float(cfg.heldout_dti_ratio)
        neg_ratio = float(cfg.neg_ratio)

        n_ppi = self.ppi_unique.shape[1]
        n_dti = self.dti_edges.shape[1]
        n_ppi_held = max(1, int(round(n_ppi * ppi_ratio)))
        n_dti_held = max(1, int(round(n_dti * dti_ratio)))

        ppi_perm = self.rng.permutation(n_ppi)
        ppi_held_idx = ppi_perm[:n_ppi_held]
        ppi_train_idx = ppi_perm[n_ppi_held:]
        self._ppi_train_idx = np.sort(ppi_train_idx)

        dti_perm = self.rng.permutation(n_dti)
        dti_held_idx = dti_perm[:n_dti_held]
        dti_train_idx = dti_perm[n_dti_held:]
        self._dti_train_idx = np.sort(dti_train_idx)

        ppi_pos = self.ppi_unique[:, ppi_held_idx]
        ppi_neg = self._sample_ppi_negatives(n_ppi_held * neg_ratio)
        dti_pos = self.dti_edges[:, dti_held_idx]
        dti_neg = self._sample_dti_negatives(n_dti_held * neg_ratio)

        heldout = {
            "ppi_pos": torch.from_numpy(ppi_pos).long(),
            "ppi_neg": torch.from_numpy(ppi_neg).long(),
            "dti_pos": torch.from_numpy(dti_pos).long(),
            "dti_neg": torch.from_numpy(dti_neg).long(),
            "ppi_train_idx": torch.from_numpy(ppi_train_idx).long(),
            "dti_train_idx": torch.from_numpy(dti_train_idx).long(),
            "meta": {
                "n_ppi_held": int(n_ppi_held),
                "n_dti_held": int(n_dti_held),
                "ppi_ratio": ppi_ratio,
                "dti_ratio": dti_ratio,
                "n_ppi_train": int(ppi_train_idx.shape[0]),
                "n_dti_train": int(dti_train_idx.shape[0]),
            },
        }
        return heldout

    def load_or_build_heldout(self, path: Path = HELDOUT_PT, rebuild: bool = False) -> dict:
        if path.exists() and not rebuild:
            obj = torch.load(path, weights_only=False)
            if "ppi_train_idx" not in obj or "dti_train_idx" not in obj:
                rebuild = True
            else:
                self._ppi_train_idx = obj["ppi_train_idx"].numpy()
                self._dti_train_idx = obj["dti_train_idx"].numpy()
                return obj
        heldout = self.build_heldout()
        torch.save(heldout, path)
        return heldout

    def _sample_ppi_negatives(self, n_samples: int) -> np.ndarray:
        n_samples = int(n_samples)
        if n_samples <= 0:
            return np.zeros((2, 0), dtype=np.int64)
        collected = []
        needed = n_samples
        nG = self.n_genes
        while needed > 0:
            cand = self.rng.integers(0, nG, size=(needed * 3, 2))
            cand = cand[cand[:, 0] != cand[:, 1]]
            if cand.shape[0] == 0:
                continue
            keys = _edge_key_sym(cand[:, 0], cand[:, 1], nG)
            keep = ~np.isin(keys, self.ppi_full_keys)
            kept = cand[keep][:needed]
            collected.append(kept)
            needed -= kept.shape[0]
        out = np.concatenate(collected, axis=0)[:n_samples]
        return out.T.astype(np.int64)

    def _sample_dti_negatives(self, n_samples: int) -> np.ndarray:
        n_samples = int(n_samples)
        if n_samples <= 0:
            return np.zeros((2, 0), dtype=np.int64)
        collected_drug = []
        collected_gene = []
        needed = n_samples
        nG = self.n_genes
        while needed > 0:
            drugs = self.rng.choice(self.drug_idx_with_dti, size=needed * 3)
            genes = self.rng.integers(0, nG, size=needed * 3)
            keys = _edge_key_asym(drugs, genes, nG)
            keep = ~np.isin(keys, self.dti_full_keys)
            kept_d = drugs[keep][:needed]
            kept_g = genes[keep][:needed]
            collected_drug.append(kept_d)
            collected_gene.append(kept_g)
            needed -= kept_d.shape[0]
        d = np.concatenate(collected_drug, axis=0)[:n_samples]
        g = np.concatenate(collected_gene, axis=0)[:n_samples]
        return np.stack([d, g], axis=0).astype(np.int64)

    def sample_ppi(self, mask_ratio: float | None = None, neg_ratio: float | None = None) -> dict:
        self._assert_heldout_exists()
        mr = float(mask_ratio if mask_ratio is not None else self.cfg.mask_ppi_ratio)
        nr = float(neg_ratio if neg_ratio is not None else self.cfg.neg_ratio)

        pool = self.ppi_unique[:, self._ppi_train_idx]
        n_pool = pool.shape[1]
        n_pos = int(round(n_pool * mr))
        if n_pos < 1:
            n_pos = 1
        perm = self.rng.permutation(n_pool)
        pos_idx = perm[:n_pos]
        vis_idx = perm[n_pos:]

        pos = pool[:, pos_idx]
        vis = pool[:, vis_idx]

        visible_fwd = np.stack([vis[0], vis[1]], axis=0)
        visible_rev = np.stack([vis[1], vis[0]], axis=0)
        neg = self._sample_ppi_negatives(n_pos * nr)

        return {
            "visible_fwd": torch.from_numpy(visible_fwd).long(),
            "visible_rev": torch.from_numpy(visible_rev).long(),
            "pos": torch.from_numpy(pos).long(),
            "neg": torch.from_numpy(neg).long(),
        }

    def sample_dti(self, mask_ratio: float | None = None, neg_ratio: float | None = None) -> dict:
        self._assert_heldout_exists()
        mr = float(mask_ratio if mask_ratio is not None else self.cfg.mask_dti_ratio)
        nr = float(neg_ratio if neg_ratio is not None else self.cfg.neg_ratio)

        pool = self.dti_edges[:, self._dti_train_idx]
        n_pool = pool.shape[1]
        n_pos = int(round(n_pool * mr))
        if n_pos < 1:
            n_pos = 1
        perm = self.rng.permutation(n_pool)
        pos_idx = perm[:n_pos]
        vis_idx = perm[n_pos:]

        pos = pool[:, pos_idx]
        vis = pool[:, vis_idx]

        visible_fwd = vis
        visible_rev = np.stack([vis[1], vis[0]], axis=0)
        neg = self._sample_dti_negatives(n_pos * nr)

        return {
            "visible_fwd": torch.from_numpy(visible_fwd).long(),
            "visible_rev": torch.from_numpy(visible_rev).long(),
            "pos": torch.from_numpy(pos).long(),
            "neg": torch.from_numpy(neg).long(),
        }

    def heldout_negatives(self, heldout: dict) -> tuple[torch.Tensor, torch.Tensor]:
        h_keys = _edge_key_sym(heldout["ppi_neg"][0].numpy(), heldout["ppi_neg"][1].numpy(), self.n_genes)
        d_keys = _edge_key_asym(heldout["dti_neg"][0].numpy(), heldout["dti_neg"][1].numpy(), self.n_genes)
        assert (~np.isin(h_keys, self.ppi_full_keys)).all(), "held-out ppi negatives must not be true edges"
        assert (~np.isin(d_keys, self.dti_full_keys)).all(), "held-out dti negatives must not be true edges"
        return heldout["ppi_neg"], heldout["dti_neg"]


def _make_cfg(**kw):
    defaults = dict(
        mask_ppi_ratio=0.20, mask_dti_ratio=0.40, neg_ratio=1.0,
        heldout_ppi_ratio=0.01, heldout_dti_ratio=0.20, seed=42,
    )
    defaults.update(kw)
    return type("Cfg", (), defaults)


def _smoke_test() -> None:
    from program.model.dataset import load_hetero_graph

    cfg = _make_cfg()
    print("[smoke] loading hetero_graph_base.pt ...")
    hetero = load_hetero_graph()
    sampler = EdgeMaskSampler(hetero, cfg)
    print(f"[smoke] n_genes={sampler.n_genes} n_drugs={sampler.n_drugs}")
    print(f"[smoke] ppi_unique={sampler.ppi_unique.shape} dti_edges={sampler.dti_edges.shape}")
    print(f"[smoke] drug_with_dti={sampler.drug_idx_with_dti.shape[0]} (DTI-covered drugs after core gene slice)")
    print(f"[smoke] ppi_full_keys={sampler.ppi_full_keys.shape} dti_full_keys={sampler.dti_full_keys.shape}")

    heldout_path = HELDOUT_PT
    rebuild = False
    heldout = sampler.load_or_build_heldout(heldout_path, rebuild=rebuild)
    meta = heldout["meta"]
    print(f"[smoke] heldout meta: {json.dumps(meta)}")
    print(f"[smoke] heldout shapes: ppi_pos={tuple(heldout['ppi_pos'].shape)} ppi_neg={tuple(heldout['ppi_neg'].shape)} "
          f"dti_pos={tuple(heldout['dti_pos'].shape)} dti_neg={tuple(heldout['dti_neg'].shape)}")

    assert heldout["ppi_pos"].shape[0] == 2 and heldout["dti_pos"].shape[0] == 2
    assert int(heldout["ppi_pos"].min()) >= 0 and int(heldout["ppi_pos"].max()) < sampler.n_genes
    assert int(heldout["dti_pos"][0].min()) >= 0 and int(heldout["dti_pos"][0].max()) < sampler.n_drugs
    assert int(heldout["dti_pos"][1].min()) >= 0 and int(heldout["dti_pos"][1].max()) < sampler.n_genes

    ppi = sampler.sample_ppi()
    print(f"[smoke] ppi: vis_fwd={tuple(ppi['visible_fwd'].shape)} vis_rev={tuple(ppi['visible_rev'].shape)} "
          f"pos={tuple(ppi['pos'].shape)} neg={tuple(ppi['neg'].shape)}")
    assert ppi["visible_fwd"].shape == ppi["visible_rev"].shape, "fwd/rev should match"
    assert ppi["visible_fwd"].shape[0] == 2 and ppi["pos"].shape[0] == 2 and ppi["neg"].shape[0] == 2
    assert int(ppi["pos"].min()) >= 0 and int(ppi["pos"].max()) < sampler.n_genes
    assert int(ppi["neg"].min()) >= 0 and int(ppi["neg"].max()) < sampler.n_genes
    assert ppi["visible_fwd"].shape[1] + ppi["pos"].shape[1] == int(meta["n_ppi_train"])

    dti = sampler.sample_dti()
    print(f"[smoke] dti: vis_fwd={tuple(dti['visible_fwd'].shape)} vis_rev={tuple(dti['visible_rev'].shape)} "
          f"pos={tuple(dti['pos'].shape)} neg={tuple(dti['neg'].shape)}")
    assert dti["visible_fwd"].shape == dti["visible_rev"].shape
    assert int(dti["pos"][0].min()) >= 0 and int(dti["pos"][0].max()) < sampler.n_drugs
    assert int(dti["neg"][0].min()) >= 0 and int(dti["neg"][0].max()) < sampler.n_drugs
    assert dti["visible_fwd"].shape[1] + dti["pos"].shape[1] == int(meta["n_dti_train"])
    neg_drug_set = set(int(x) for x in dti["neg"][0].tolist())
    assert neg_drug_set.issubset(set(int(x) for x in sampler.drug_idx_with_dti.tolist())), "neg drug must be in 177"

    neg_ppi_keys = _edge_key_sym(ppi["neg"][0].numpy(), ppi["neg"][1].numpy(), sampler.n_genes)
    overlap = np.intersect1d(neg_ppi_keys, sampler.ppi_full_keys).size
    print(f"[smoke] ppi neg overlap with full_set = {overlap} (should be 0)")
    assert overlap == 0, "PPI neg samples must not collide with full_set (plan §0.3.1)"

    neg_dti_keys = _edge_key_asym(dti["neg"][0].numpy(), dti["neg"][1].numpy(), sampler.n_genes)
    overlap_d = np.intersect1d(neg_dti_keys, sampler.dti_full_keys).size
    print(f"[smoke] dti neg overlap with full_set = {overlap_d} (should be 0)")
    assert overlap_d == 0

    print("[smoke] ALL OK | edge_mask.py")


if __name__ == "__main__":
    _smoke_test()