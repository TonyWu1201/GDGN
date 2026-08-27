"""
Phase 5.2: 图结构消融验证 eval_graph_structure.py
===================================================

目标: 验证 GDGN 的图结构 (DTI + PPI 边) 是否对预测起正向作用.
方法: **推理时扰动** (零重训) —— 用已训 ckpt, 替换 gene_enc 的边 buffer,
在 test 集上对比 PCC/RMSE. 若扰动后性能显著下降 -> 模型在利用图结构;
若几乎不变 -> 图结构对预测无实质贡献 (训练已把信息蒸馏进权重).

扰动模式 (mode)
---------------
- original        : 原图 (对照)
- permute_ppi     : PPI 边随机重连 (保边数, 度分布近似)  —— PPI 特异性
- permute_dti     : DTI 边随机重连 (保边数)              —— DTI 特异性
- permute_both    : 两者都置换                          —— 图结构整体
- remove_ppi      : PPI 边置空 (0 边)
- remove_dti      : DTI 边置空 (0 边)
- remove_both     : 全部置空 (等价无图 message passing)
- shuffle_gene_x  : gene 静态特征 (ESM-2 1280) 行置换    —— 节点特征对照
- shuffle_drug_x  : drug 节点特征行置换                  —— 节点特征对照

局限 (如实报告)
---------------
- 推理时扰动 ≠ 训练时消融: 权重是在原图上训出来的, 扰动只测"推理是否依赖边的
  具体连接", 不测"图结构是否帮助了训练". 若 permute 不掉点, 只能说明
  模型已不依赖图; 要严格归因需训练级消融 (重训无图变体, 计划 §2 待跑).
- GAT 无 self-loop, 边置空后节点仍保留自身 proj 特征 (bias 保留),
  因此 remove 不等于"删掉节点信息".

Usage
-----
    uv run python program/model/eval_graph_structure.py \
        --ckpt data/model/arch/B4_dual_encoder/best_model.pt \
        --output_dir data/model/interpretability/graph_ablation_b4 \
        --modes original permute_ppi permute_dti permute_both remove_ppi remove_dti

    uv run python program/model/eval_graph_structure.py \
        --ckpt data/model/ablation/04_no_pretrain/best_model.pt \
        --output_dir data/model/interpretability/graph_ablation_a4 \
        --modes original permute_ppi permute_dti permute_both

smoke (val 前 5 batch):
    uv run python program/model/eval_graph_structure.py \
        --ckpt data/model/arch/B4_dual_encoder/best_model.pt \
        --output_dir C:/temp/graph_ablation_smoke --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.model.dataset import get_dataloaders, inject_batch_omics, load_cell_line_features, load_hetero_graph
from program.model.train_gdgn import build_model, compute_metrics

MODES = ["original", "permute_ppi", "permute_dti", "permute_both",
         "remove_ppi", "remove_dti", "remove_both",
         "shuffle_gene_x", "shuffle_drug_x"]


def permute_edges(edge_index: torch.Tensor, n_nodes: int, rng: np.random.RandomState) -> torch.Tensor:
    """随机重连: 保持边数, 目标节点 (dst) 打乱重排. 源节点与度数分布近似保留."""
    e = edge_index.clone()
    src = e[0].cpu().numpy()
    new_dst = rng.randint(0, n_nodes, size=src.shape[0])
    e[0] = torch.from_numpy(src).to(edge_index.device)
    e[1] = torch.from_numpy(new_dst).to(edge_index.device)
    return e


def self_loop_edges(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """替换为自环边 (i,i): 无邻居消息传递 (GAT 空边会 crash, 自环是最小安全图)."""
    idx = torch.arange(n_nodes, device=edge_index.device)
    return torch.stack([idx, idx], dim=0)


def dummy_edge(edge_index: torch.Tensor) -> torch.Tensor:
    """单条 dummy 边 (0,0): DTI 是异质边 (drug->gene 两端维度不同, 无法自环),
    用单条边保留 shape 合法性, 其余节点无邻居消息 (GAT 无入边节点 = lin 自身)."""
    return torch.zeros(2, 1, dtype=edge_index.dtype, device=edge_index.device)


def apply_perturbation(model, mode: str, seed: int = 42) -> str:
    """替换 model.gene_enc 的 4 个边 buffer. 返回描述."""
    ge = model.gene_enc
    rng = np.random.RandomState(seed)
    n_gene = int(ge.ppi_fwd.max().item()) + 1 if ge.ppi_fwd.numel() else 8412
    n_drug = int(ge.dti_fwd[0].max().item()) + 1 if ge.dti_fwd.numel() else 184

    if mode == "original":
        return "no change"
    if mode == "permute_ppi":
        ge.ppi_fwd.data = permute_edges(ge.ppi_fwd, n_gene, rng)
        ge.ppi_rev.data = permute_edges(ge.ppi_rev, n_gene, rng)
        return f"ppi permuted ({ge.ppi_fwd.shape[1]} edges)"
    if mode == "permute_dti":
        # dti_fwd: drug->gene (dst 是 gene); dti_rev: gene->drug (dst 是 drug)
        ge.dti_fwd.data = permute_edges(ge.dti_fwd, n_gene, rng)
        ge.dti_rev.data = permute_edges(ge.dti_rev, n_drug, rng)
        return f"dti permuted ({ge.dti_fwd.shape[1]} edges)"
    if mode == "permute_both":
        ge.ppi_fwd.data = permute_edges(ge.ppi_fwd, n_gene, rng)
        ge.ppi_rev.data = permute_edges(ge.ppi_rev, n_gene, rng)
        ge.dti_fwd.data = permute_edges(ge.dti_fwd, n_gene, rng)
        ge.dti_rev.data = permute_edges(ge.dti_rev, n_drug, rng)
        return "ppi+dti permuted"
    if mode == "remove_ppi":
        ge.ppi_fwd.data = self_loop_edges(ge.ppi_fwd, n_gene)
        ge.ppi_rev.data = self_loop_edges(ge.ppi_rev, n_gene)
        return "ppi removed (self-loops only)"
    if mode == "remove_dti":
        ge.dti_fwd.data = dummy_edge(ge.dti_fwd)
        ge.dti_rev.data = dummy_edge(ge.dti_rev)
        return "dti removed (dummy edge)"
    if mode == "remove_both":
        ge.ppi_fwd.data = self_loop_edges(ge.ppi_fwd, n_gene)
        ge.ppi_rev.data = self_loop_edges(ge.ppi_rev, n_gene)
        ge.dti_fwd.data = dummy_edge(ge.dti_fwd)
        ge.dti_rev.data = dummy_edge(ge.dti_rev)
        return "ppi+dti removed"
    if mode == "shuffle_gene_x":
        perm = torch.randperm(model.gene_x_static.size(0), device=model.gene_x_static.device)
        model.gene_x_static.data = model.gene_x_static[perm]
        return "gene static x shuffled"
    if mode == "shuffle_drug_x":
        perm = torch.randperm(model.drug_x.size(0), device=model.drug_x.device)
        model.drug_x.data = model.drug_x[perm]
        return "drug node x shuffled"
    raise ValueError(f"unknown mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--modes", nargs="*", default=None,
                    help="缺省: original + 全部扰动 (shuffle_* 除外)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[graph_ablation] device={device} ckpt={args.ckpt} smoke={args.smoke}")

    hetero = load_hetero_graph(device)
    clf = load_cell_line_features(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    config.setdefault("model", "gdgn")
    model = build_model(config["model"], hetero, config, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loaders = get_dataloaders(batch_size=args.batch_size, num_workers=0,
                              pin_memory=torch.cuda.is_available(), load_hetero=False)
    max_batches = 5 if args.smoke else None

    modes = args.modes or [m for m in MODES if m != "original"]
    modes = ["original"] + modes

    # 基线 (原图) 先跑, 存原始 buffer 以备复位
    orig_state = {n: b.detach().clone() for n, b in model.gene_enc.named_buffers()
                  if n.startswith(("ppi_", "dti_"))}
    orig_gene_x = model.gene_x_static.detach().clone()
    orig_drug_x = model.drug_x.detach().clone()

    rows = []
    for mode in modes:
        # 复位到原图
        for n, t in orig_state.items():
            model.gene_enc.get_buffer(n).data = t
        model.gene_x_static.data = orig_gene_x
        model.drug_x.data = orig_drug_x
        desc = apply_perturbation(model, mode, seed=args.seed)
        print(f"[graph_ablation] mode={mode}: {desc}", flush=True)

        preds, trues = [], []
        t0 = time.time()
        with torch.no_grad():
            for i, batch in enumerate(loaders["test"]):
                if max_batches is not None and i >= max_batches:
                    break
                cell_idx = batch["cell_idx"].to(device)
                drug_idx = batch["drug_idx"].to(device)
                omics = inject_batch_omics(clf, cell_idx)
                ic50_pred, _ = model(cell_idx, drug_idx, omics)
                preds.append(ic50_pred.detach().cpu().squeeze(-1))
                trues.append(batch["y"].detach().cpu())
        preds = torch.cat(preds).numpy()
        trues = torch.cat(trues).numpy()
        m = compute_metrics(preds, trues)
        rows.append({"mode": mode, "desc": desc, "pcc": m["pcc"],
                     "spearman": m["spearman"], "rmse": m["rmse"], "n": m["n"],
                     "dt_sec": round(time.time() - t0, 1)})
        print(f"[graph_ablation] mode={mode}: test_pcc={m['pcc']:.4f} "
              f"rmse={m['rmse']:.4f} n={m['n']} ({rows[-1]['dt_sec']}s)")

    # 报告
    base_pcc = next(r["pcc"] for r in rows if r["mode"] == "original")
    lines = ["=" * 78, f"Graph structure ablation (inference-time perturbation)",
             f"ckpt: {args.ckpt}", f"device: {device}", "=" * 78]
    lines.append(f"{'mode':<15}{'PCC':>10}{'dPCC':>10}{'Spearman':>10}{'RMSE':>10}{'N':>8}{'note':<30}")
    for r in rows:
        dp = r["pcc"] - base_pcc
        lines.append(f"{r['mode']:<15}{r['pcc']:>10.4f}{dp:>+10.4f}{r['spearman']:>10.4f}"
                     f"{r['rmse']:>10.4f}{r['n']:>8}{r['desc']:<30}")
    lines.append("")
    lines.append("[Note] 推理时扰动口径: 权重在原图上训练, 扰动只测推理是否依赖")
    lines.append("边的具体连接. PCC 显著下降 -> 图结构被利用; 几乎不变 -> 预测不依赖")
    lines.append("图 (训练级消融需重训无图变体, 见 guidance/Phase5可解释性分析规划.md).")

    output_dir = Path(args.output_dir or "data/model/interpretability/graph_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "graph_ablation_report.txt").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "graph_ablation_report.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"[graph_ablation] report -> {output_dir}")


if __name__ == "__main__":
    main()
