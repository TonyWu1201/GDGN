"""
Phase 5: 可解释性分析 interpret.py
=====================================

设计依据: [guidance/Phase5可解释性分析规划.md] (本地文档, guidance/ 不进 git)

职责 (纯推理归因, 不改任何模型/训练代码)
----------------------------------------
- L1 基因级: captum IntegratedGradients on omics_per_gene (1, 8412, 4),
  三模型同口径 (gdgn / baseline_simple / cdr_baseline); GDGN 额外读内建
  cross-attn 权重 attn_weights (1, 8412) 并做 IG-attn 一致性相关
- L2 通路级: 基因归因聚合到 186 条 KEGG 通路 (mean |attr|) + 超几何富集
- L3 分子级: GDGN/baseline_simple 用 GNNExplainer (DrugEncoder 分子图,
  edge_mask); cdr_baseline 的 GraphConv 栈用 atom-feature IG (padded 100x75)
- 验证: DTI 靶点命中率 (interactions_filtered.csv) + CGC driver 富集
  (driver_genes.csv), top-20/top-100 vs 随机基线
- CLI: 单模型跑 case 或 --aggregate 汇总三模型对比

用法
----
单模型归因 (自动选 3 有 DTI + 2 无 DTI 药物 × 敏感/耐药细胞系):
    uv run python program/model/interpret.py --model gdgn \
        --ckpt data/model/arch/B4_dual_encoder/best_model.pt \
        --output_dir data/model/interpretability/gdgn_b4

指定药物/细胞系 + 只跑某层面:
    uv run python program/model/interpret.py --model baseline_simple \
        --ckpt data/model/baseline/best_model.pt \
        --output_dir data/model/interpretability/baseline \
        --drug_idx 10 50 --cell_idx 0 3 --kinds ig pathway

smoke (1 case, 少步数):
    uv run python program/model/interpret.py --model gdgn \
        --ckpt data/model/arch/B4_dual_encoder/best_model.pt \
        --output_dir C:/temp/interpret_smoke --smoke

汇总对比:
    uv run python program/model/interpret.py --aggregate \
        --input_dir data/model/interpretability

注意
----
- 归因全程 model.eval() (BN running stats), 规避 B=1 BN 训练隐患
- mut/cnv 是 0/1 离散输入, IG 视为连续变量; 归因按通道分开放,
  主结论以 expr/meth 连续通道为准
- IG baseline 默认 per-channel 全细胞系均值 (--baseline mean),
  可选 zero (mut/cnv 语义"无突变"更直观)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import hypergeom, spearmanr

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.model.dataset import inject_batch_omics, load_cell_line_features, load_hetero_graph
from program.model.drug_encoder import _build_mol_data, load_drug_mol_graphs
from program.model.pretrain import omics_to_per_gene
from program.model.train_gdgn import build_model, resolve_no_dti_drug_idx

PROC = _PROJECT_ROOT / "data" / "processed"
MODEL_DIR = _PROJECT_ROOT / "data" / "model"

GENE_ORDER_TXT = MODEL_DIR / "hetero_graph" / "core_gene_order.txt"
PATHWAY_SETS_JSON = PROC / "driver&pathway" / "pathway_gene_sets.json"
DRIVER_CSV = PROC / "driver&pathway" / "driver_genes.csv"
DTI_CSV = PROC / "drug_gene_interaction" / "interactions_filtered.csv"
IC50_CSV = PROC / "drug_sensitivity" / "ic50_matrix.csv"
CID_TO_IDX_JSON = PROC / "drug_cid_to_idx.json"
CELL_TO_IDX_JSON = PROC / "cell_line_to_idx.json"
CELL_ORDER_TXT = PROC / "cell_line_canonical_order.txt"
NO_DTI_IDX_JSON = MODEL_DIR / "gdgn" / "no_dti_drug_idx.json"

N_GENES = 8412
N_OMICS_CH = 4  # expr/mut/cnv/meth


# ---------------- 元数据加载 ----------------

def load_gene_order() -> list[str]:
    return GENE_ORDER_TXT.read_text(encoding="utf-8").splitlines()


def load_pathway_sets() -> dict[str, list[str]]:
    return json.loads(PATHWAY_SETS_JSON.read_text(encoding="utf-8"))


def load_driver_genes() -> set[str]:
    import csv
    genes = set()
    with open(DRIVER_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("in_expression_matrix", "True").lower() == "true":
                genes.add(row["gene_name"])
    return genes


def load_dti_targets() -> dict[int, set[str]]:
    """cid -> 已知靶基因集合 (interactions_filtered.csv)."""
    import csv
    dti: dict[int, set[str]] = {}
    with open(DTI_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = int(row["cid"])
            dti.setdefault(cid, set()).add(row["gene_name"])
    return dti


def load_cid_to_idx() -> dict[str, int]:
    return {k: int(v) for k, v in json.loads(CID_TO_IDX_JSON.read_text(encoding="utf-8")).items()}


def load_cell_idx_map() -> tuple[dict[str, int], list[str]]:
    idx = {k: int(v) for k, v in json.loads(CELL_TO_IDX_JSON.read_text(encoding="utf-8")).items()}
    order = CELL_ORDER_TXT.read_text(encoding="utf-8").splitlines()
    return idx, order


def load_ic50_matrix() -> "np.ndarray":
    import pandas as pd
    df = pd.read_csv(IC50_CSV, index_col=0)
    return df.values.astype(np.float64)  # (404, 184)


# ---------------- 归因 forward 闭包 ----------------

def build_forward_fn(model, clf, device, cell_idx: int, drug_idx: int):
    """Joint IG target over omics and its derived pathway representation."""
    cell_t = torch.tensor([cell_idx], dtype=torch.long, device=device)
    drug_t = torch.tensor([drug_idx], dtype=torch.long, device=device)

    def forward_fn(omics_4: torch.Tensor, pathway_input: torch.Tensor) -> torch.Tensor:
        omics = {
            "expr": omics_4[..., 0],
            "mut": omics_4[..., 1],
            "cnv": omics_4[..., 2],
            "meth": omics_4[..., 3],
            # Pathway is interpolated with expression instead of being held fixed.
            "pathway": pathway_input,
        }
        ic50_pred, _ = model(cell_t.expand(omics_4.shape[0]), drug_t.expand(omics_4.shape[0]), omics)
        return ic50_pred  # (B,1)

    return forward_fn


def ig_attribution(model, clf, device, cell_idx: int, drug_idx: int,
                   n_steps: int = 50, baseline: str = "mean") -> dict:
    """Joint IG on omics and expression-derived pathway inputs with convergence delta."""
    cell_t = torch.tensor([cell_idx], dtype=torch.long, device=device)
    omics = inject_batch_omics(clf, cell_t)
    pathway = omics["pathway"]
    omics_4 = omics_to_per_gene(omics).to(device)  # (1, 8412, 4)

    if baseline == "mean":
        ch_map = {"expr": "expression", "mut": "mutation",
                  "cnv": "copynumber", "meth": "methylation"}
        baseline_4 = torch.stack([
            clf[ch_map[k]].mean(dim=0) for k in ["expr", "mut", "cnv", "meth"]
        ], dim=-1).unsqueeze(0).to(device)
        baseline_pathway = clf["pathway_activity"].mean(dim=0, keepdim=True).to(device)
    else:
        baseline_4 = torch.zeros_like(omics_4)
        baseline_pathway = torch.zeros_like(pathway)

    from captum.attr import IntegratedGradients
    model.eval()
    original_requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # 全程 eval 模式: BN 用 running stats (可导, 且无 B=1 校验), dropout 关闭
    omics_4_in = omics_4.clone().requires_grad_(True)
    try:
        ig = IntegratedGradients(build_forward_fn(model, clf, device, cell_idx, drug_idx))
        (attr, pathway_attr), delta = ig.attribute(
            (omics_4_in, pathway.clone().requires_grad_(True)),
            baselines=(baseline_4, baseline_pathway),
            n_steps=n_steps, internal_batch_size=1,
            return_convergence_delta=True,
        )
    finally:
        for parameter, state in zip(model.parameters(), original_requires_grad):
            parameter.requires_grad_(state)
        model.eval()
    attr = attr.detach().cpu().squeeze(0)  # (8412, 4)
    per_gene = torch.sqrt((attr ** 2).sum(dim=-1))  # (8412,)
    signed_per_gene = attr.sum(dim=-1)
    return {
        "attr": attr,
        "per_gene": per_gene,
        "signed_per_gene": signed_per_gene,
        "pathway_attr": pathway_attr.detach().cpu().squeeze(0),
        "convergence_delta": float(delta.detach().abs().mean().cpu()),
        "pathway_jointly_interpolated": True,
    }


def attn_importance(model, clf, device, cell_idx: int, drug_idx: int) -> torch.Tensor | None:
    """GDGN 内建 cross-attn 权重 (1, 8412); 非 GDGN 返回 None."""
    cell_t = torch.tensor([cell_idx], dtype=torch.long, device=device)
    drug_t = torch.tensor([drug_idx], dtype=torch.long, device=device)
    omics = inject_batch_omics(clf, cell_t)
    model.eval()
    with torch.no_grad():
        _, attn = model(cell_t, drug_t, omics)
    if attn is None:
        return None
    return attn.detach().cpu().squeeze(0).squeeze(0)  # (8412,)


# ---------------- L2 通路聚合 ----------------

def pathway_aggregate(per_gene: torch.Tensor, gene_order: list[str],
                      pathway_sets: dict[str, list[str]]) -> dict:
    """基因归因 -> 通路得分 (member 基因 |attr| 均值, 只统计 8412 内交集)."""
    gene2idx = {g: i for i, g in enumerate(gene_order)}
    scores = {}
    for pw, genes in pathway_sets.items():
        idxs = [gene2idx[g] for g in genes if g in gene2idx]
        if not idxs:
            continue
        scores[pw] = float(per_gene[idxs].abs().mean().item())
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def hypergeom_enrich(per_gene: torch.Tensor, gene_order: list[str],
                     pathway_sets: dict[str, list[str]], top_k: int = 100) -> list[dict]:
    """top-k 基因 (按 |attr|) 与每条通路的超几何富集. 返回按 p 升序列表."""
    gene2idx = {g: i for i, g in enumerate(gene_order)}
    N = len(gene_order)
    order = per_gene.abs().argsort(descending=True).tolist()
    top = set(order[:top_k])
    results = []
    for pw, genes in pathway_sets.items():
        idxs = set(gene2idx[g] for g in genes if g in gene2idx)
        if not idxs:
            continue
        k = len(idxs & top)
        M = len(idxs)
        if M == 0:
            continue
        pval = float(hypergeom.sf(k - 1, N, M, top_k))
        results.append({"pathway": pw, "n_member": M, "n_hit": k, "p_value": pval,
                        "genes": [gene_order[i] for i in sorted(idxs & top)]})
    results.sort(key=lambda r: r["p_value"])
    return results


# ---------------- L3 分子归因 ----------------

class _DrugEncWrapper(nn.Module):
    """把 DrugEncoder.forward(x, edge_index, batch) 包装成单图 graph-level."""

    def __init__(self, drug_enc: nn.Module):
        super().__init__()
        self.drug_enc = drug_enc

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.drug_enc(x, edge_index, batch)  # (1, out_dim)


def drug_gnn_explainer(model, drug_idx: int, device, epochs: int = 200) -> dict:
    """GNNExplainer on DrugEncoder 分子图 (GDGN / baseline_simple 共用)."""
    from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

    drug_enc = model.drug_enc.to(device)
    wrapper = _DrugEncWrapper(drug_enc).eval()
    graphs = load_drug_mol_graphs()
    data = _build_mol_data(int(drug_idx), graphs).to(device)

    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        model_config=ModelConfig(mode="regression", task_level="graph"),
        node_mask_type="attributes",
        edge_mask_type="object",
    )
    explanation = explainer(data.x, data.edge_index)
    edge_mask = explanation.edge_mask.detach().cpu().numpy()  # (E,)
    node_mask = explanation.node_mask.detach().cpu().numpy()  # (N, 75)
    atom_importance = np.sqrt((node_mask ** 2).sum(axis=1))    # (N,)
    edges = data.edge_index.detach().cpu().numpy().T           # (E, 2)

    n_edges_keep = max(1, int(0.1 * len(edges)))
    top_edge_idx = np.argsort(edge_mask)[::-1][:n_edges_keep]
    top_edges = [{"src": int(edges[i, 0]), "dst": int(edges[i, 1]),
                  "mask": float(edge_mask[i])} for i in top_edge_idx]
    return {"n_atoms": int(data.x.size(0)), "n_edges": int(edges.shape[0]),
            "top_edges": top_edges,
            "atom_importance": atom_importance.tolist()}


def drug_atom_ig(model, drug_idx: int, device, n_steps: int = 50) -> dict:
    """cdr_baseline: atom-feature IG on padded (100, 75) 输入.

    cdr_baseline 的 drug 分支是 dense bmm GraphConv, 无 edge_index 接口,
    故用 IG 代替 GNNExplainer. 仅真原子部分 (前 n_atom) 报告.
    """
    from captum.attr import IntegratedGradients

    graphs = load_drug_mol_graphs()
    n_atom = int(graphs[drug_idx][0].shape[0])
    feat_in = model.drug_feat_all[drug_idx].clone().to(device).requires_grad_(True)
    adj = model.drug_adj_all[drug_idx].to(device)

    def forward_fn(atom_feat: torch.Tensor) -> torch.Tensor:
        h = atom_feat.unsqueeze(0)     # (1, 100, 75)
        a = adj.unsqueeze(0)           # (1, 100, 100)
        for conv, bn in zip(model.gcn_convs, model.gcn_bns):
            h = conv(h, a)
            if model.use_relu:
                h = torch.relu(h)
            if model.use_bn:
                h = bn(h.transpose(1, 2)).transpose(1, 2)
            h = model.gcn_dropout(h)
        if model.use_GMP:
            pooled = h.max(dim=1).values  # (1, 100)
        else:
            pooled = h.mean(dim=1)
        return pooled.sum().reshape(1)  # (1,) 单输出 (captum 需要非 0-dim)

    model.eval()
    ig = IntegratedGradients(forward_fn)
    attr = ig.attribute(feat_in, n_steps=n_steps, internal_batch_size=1)
    attr = attr.detach().cpu().numpy()  # (100, 75)
    atom_importance = np.sqrt((attr ** 2).sum(axis=1))[:n_atom]  # 仅真原子
    top_atoms = np.argsort(atom_importance)[::-1][:20].tolist()
    return {"n_atoms": n_atom, "atom_importance": atom_importance.tolist(),
            "top_atoms": top_atoms}


# ---------------- 验证 ----------------

def dti_validation(per_gene: torch.Tensor, gene_order: list[str], cid: int,
                   dti: dict[int, set[str]], top_k: int = 20) -> dict:
    """top-k 基因 ∩ 已知 DTI 靶点. 随机基线期望 = top_k * |dti| / 8412."""
    gene2idx = {g: i for i, g in enumerate(gene_order)}
    order = per_gene.abs().argsort(descending=True).tolist()[:top_k]
    top_genes = [gene_order[i] for i in order]
    targets = dti.get(cid, set())
    hits = [g for g in top_genes if g in targets]
    expect = top_k * len(targets) / len(gene_order)
    return {"top_k": top_k, "n_targets_known": len(targets),
            "n_hits": len(hits), "random_expect": round(expect, 3),
            "hit_genes": hits, "top_genes": top_genes}


def driver_validation(per_gene: torch.Tensor, gene_order: list[str],
                      drivers: set[str], top_k: int = 100) -> dict:
    """top-k 基因 ∩ CGC driver; 超几何 p 值."""
    gene2idx = {g: i for i, g in enumerate(gene_order)}
    order = per_gene.abs().argsort(descending=True).tolist()[:top_k]
    top = set(order)
    driver_idx = {gene2idx[g] for g in drivers if g in gene2idx}
    k = len(driver_idx & top)
    M = len(driver_idx)
    pval = float(hypergeom.sf(k - 1, len(gene_order), M, top_k)) if M > 0 else 1.0
    hits = [gene_order[i] for i in sorted(driver_idx & top)]
    return {"top_k": top_k, "n_drivers": M, "n_hits": k, "p_value": pval,
            "hit_genes": hits}


# ---------------- case 流程 ----------------

def auto_select_cases(n_dti: int = 3, n_no_dti: int = 2) -> list[tuple[int, int, int]]:
    """自动选 (drug_idx, cell_sensitive, cell_resistant) case 列表.

    有 DTI 药: interactions_filtered.csv 按 interaction_score 降序取 top-n (cid 去重);
    无 DTI 药: no_dti_drug_idx.json 前 n 个.
    细胞系: ic50_matrix 该药列 min (敏感) / max (耐药).
    """
    import pandas as pd
    dti_df = pd.read_csv(DTI_CSV)
    cid_to_idx = load_cid_to_idx()
    scores = dti_df.groupby("cid")["interaction_score"].max().sort_values(ascending=False)
    dti_cids = [int(c) for c in scores.index if str(c) in cid_to_idx][:n_dti]
    dti_idx = [cid_to_idx[str(c)] for c in dti_cids]
    no_dti = resolve_no_dti_drug_idx()[:n_no_dti]

    ic50 = load_ic50_matrix()
    cases = []
    for drug_idx in dti_idx + no_dti:
        col = ic50[:, drug_idx]
        sens = int(np.nanargmin(col)) if not np.all(np.isnan(col)) else 0
        res = int(np.nanargmax(col)) if not np.all(np.isnan(col)) else 1
        cases.append((int(drug_idx), sens, res))
    return cases


def run_case(model, clf, device, cell_idx: int, drug_idx: int, cid: int,
             gene_order, pathway_sets, drivers, dti,
             kinds, n_steps, expl_epochs, baseline) -> dict:
    result = {"drug_idx": drug_idx, "cell_idx": cell_idx, "cid": cid}

    if "ig" in kinds or "pathway" in kinds or "all" in kinds:
        ig = ig_attribution(model, clf, device, cell_idx, drug_idx,
                            n_steps=n_steps, baseline=baseline)
        result["ig_per_gene"] = ig["per_gene"].tolist()
        result["ig_signed_per_gene"] = ig["signed_per_gene"].tolist()
        result["ig_pathway"] = ig["pathway_attr"].tolist()
        result["ig_convergence_delta"] = ig["convergence_delta"]
        result["pathway_jointly_interpolated"] = ig["pathway_jointly_interpolated"]
        result["ig_channels"] = {
            ch: ig["attr"][:, c].tolist() for c, ch in enumerate(["expr", "mut", "cnv", "meth"])
        }
        result["dti_validation"] = dti_validation(ig["per_gene"], gene_order, cid, dti)
        result["driver_validation"] = driver_validation(ig["per_gene"], gene_order, drivers)
        if "pathway" in kinds or "all" in kinds:
            result["pathway_top"] = list(pathway_aggregate(
                ig["per_gene"], gene_order, pathway_sets).items())[:10]
            result["pathway_enrich"] = hypergeom_enrich(
                ig["per_gene"], gene_order, pathway_sets)[:10]

    if "attn" in kinds or "all" in kinds:
        attn = attn_importance(model, clf, device, cell_idx, drug_idx)
        if attn is not None:
            igv = torch.tensor(result.get("ig_per_gene", [0.0]))
            rho = float(spearmanr(attn.numpy(), igv.numpy())[0]) if igv.numel() > 1 else float("nan")
            result["attn_top"] = [gene_order[i] for i in
                                  attn.argsort(descending=True)[:20].tolist()]
            result["attn_ig_spearman"] = rho
        result["has_attn"] = attn is not None

    if "molecule" in kinds or "all" in kinds:
        if hasattr(model, "drug_enc") and type(model).__name__ in ("GDGNModel", "BaselineSimpleModel"):
            result["molecule"] = drug_gnn_explainer(model, drug_idx, device, epochs=expl_epochs)
        else:
            result["molecule"] = drug_atom_ig(model, drug_idx, device, n_steps=n_steps)
    return result


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Phase 5 可解释性分析")
    ap.add_argument("--model", choices=["gdgn", "baseline_simple", "cdr_baseline"],
                    default="gdgn")
    ap.add_argument("--ckpt", type=str, required=False)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--drug_idx", type=int, nargs="*", default=None,
                    help="指定药物 (覆盖自动选择); 细胞系仍按敏感/耐药自动配")
    ap.add_argument("--cell_idx", type=int, nargs="*", default=None,
                    help="指定细胞系 (须与 drug_idx 等长, 成对使用)")
    ap.add_argument("--kinds", nargs="*", default=["all"],
                    choices=["all", "ig", "attn", "pathway", "molecule"])
    ap.add_argument("--n_steps", type=int, default=50)
    ap.add_argument("--expl_epochs", type=int, default=200)
    ap.add_argument("--baseline", choices=["mean", "zero"], default="mean")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aggregate", action="store_true",
                    help="汇总 input_dir 下各模型 cases.json -> aggregate.md")
    ap.add_argument("--input_dir", type=str, default="data/model/interpretability")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.input_dir)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[interpret] device={device} model={args.model} smoke={args.smoke}")

    if args.smoke:
        args.n_steps = min(args.n_steps, 10)
        args.expl_epochs = min(args.expl_epochs, 20)

    hetero = load_hetero_graph(device)
    clf = load_cell_line_features(device)
    ckpt_path = args.ckpt or _default_ckpt(args.model)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    config.setdefault("model", args.model)
    model = build_model(args.model, hetero, config, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[interpret] loaded ckpt {ckpt_path} (epoch={ckpt.get('epoch')})")

    gene_order = load_gene_order()
    pathway_sets = load_pathway_sets()
    drivers = load_driver_genes()
    dti = load_dti_targets()
    cid_to_idx = load_cid_to_idx()
    cid_of = {v: k for k, v in cid_to_idx.items()}

    if args.drug_idx:
        cases = []
        for i, d in enumerate(args.drug_idx):
            col = load_ic50_matrix()[:, d]
            sens = int(np.nanargmin(col)); res = int(np.nanargmax(col))
            if args.cell_idx:
                sens = res = args.cell_idx[i % len(args.cell_idx)]
            cases.append((d, sens, res))
    else:
        cases = auto_select_cases()
    print(f"[interpret] {len(cases)} cases: " +
          ", ".join(f"(drug={d}, sens={s}, res={r})" for d, s, r in cases))

    default_output = (
        f"data/model/smoke/interpretability/{args.model}" if args.smoke
        else f"data/model/interpretability/{args.model}"
    )
    output_dir = Path(args.output_dir or default_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases_out = []
    t0 = time.time()
    mol_cache: dict[int, dict] = {}
    for i, (drug_idx, cell_sens, cell_res) in enumerate(cases):
        cid = int(cid_of.get(drug_idx, drug_idx))
        for tag, cell_idx in (("sensitive", cell_sens), ("resistant", cell_res)):
            key = f"drug{drug_idx}_cell{cell_idx}_{tag}"
            print(f"[interpret] [{i*2+1}/{(len(cases))*2}] {key} ...", flush=True)
            res = run_case(model, clf, device, cell_idx, drug_idx, cid,
                           gene_order, pathway_sets, drivers, dti,
                           args.kinds, args.n_steps, args.expl_epochs, args.baseline)
            if "molecule" in res and drug_idx not in mol_cache:
                mol_cache[drug_idx] = res["molecule"]
            if drug_idx in mol_cache:
                res["molecule"] = mol_cache[drug_idx]
            res["case"] = key
            res["cell_type"] = tag
            cases_out.append(res)
            (output_dir / f"{key}.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[interpret] {len(cases_out)} cases done in {time.time()-t0:.0f}s")

    # 汇总 + 验证统计
    summary = summarize(cases_out, gene_order)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    (output_dir / "cases.json").write_text(
        json.dumps(cases_out, ensure_ascii=False, indent=2))
    print(f"[interpret] summary -> {output_dir / 'summary.md'}")


def _default_ckpt(model: str) -> Path:
    if model == "gdgn":
        return MODEL_DIR / "arch" / "B4_dual_encoder" / "best_model.pt"
    if model == "baseline_simple":
        return MODEL_DIR / "baseline" / "best_model.pt"
    return MODEL_DIR / "cdr_baseline" / "best_model.pt"


def summarize(cases_out: list[dict], gene_order) -> str:
    lines = ["# Phase 5 Interpretability Summary", ""]
    lines.append(f"cases: {len(cases_out)}")
    hits_total = sum(1 for c in cases_out if c.get("dti_validation", {}).get("n_hits", 0) > 0)
    lines.append(f"cases with >=1 known DTI hit in top-20: {hits_total}/{len(cases_out)}")
    lines.append("")
    lines.append("| case | DTI hit | driver hit | driver p | attn-IG rho | top pathway |")
    lines.append("|---|---|---|---|---|---|")
    for c in cases_out:
        dv = c.get("dti_validation", {})
        drv = c.get("driver_validation", {})
        pw = c.get("pathway_top", [])
        lines.append(f"| {c['case']} | {dv.get('n_hits', 'N/A')} "
                     f"({', '.join(dv.get('hit_genes', [])[:2])}) | "
                     f"{drv.get('n_hits', 'N/A')} | {drv.get('p_value', float('nan')):.2e} | "
                     f"{c.get('attn_ig_spearman', float('nan')):.3f} | "
                     f"{pw[0][0] if pw else 'N/A'} |")
    return "\n".join(lines)


def aggregate(input_dir: str | Path) -> None:
    input_dir = Path(input_dir)
    models = [p.name for p in input_dir.iterdir()
              if p.is_dir() and (p / "cases.json").exists()]
    lines = ["# Phase 5 三模型对比汇总", ""]
    for m in models:
        cases = json.loads((input_dir / m / "cases.json").read_text(encoding="utf-8"))
        lines.append(f"## {m} ({len(cases)} cases)")
        dti_hits = [c["dti_validation"]["n_hits"] for c in cases
                    if "dti_validation" in c]
        driver_p = [c["driver_validation"]["p_value"] for c in cases
                    if "driver_validation" in c]
        rhos = [c["attn_ig_spearman"] for c in cases if "attn_ig_spearman" in c]
        lines.append(f"- DTI top-20 命中: mean={np.mean(dti_hits):.2f} "
                     f"(n={len(dti_hits)})" if dti_hits else "- DTI: N/A")
        lines.append(f"- driver top-100 p: mean={np.mean(driver_p):.2e} "
                     f"(n={len(driver_p)})" if driver_p else "- driver: N/A")
        lines.append(f"- attn-IG spearman: mean={np.mean(rhos):.3f} (GDGN only)"
                     if rhos else "- attn: N/A")
        lines.append("")
    out = input_dir / "aggregate.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[aggregate] -> {out}")


if __name__ == "__main__":
    main()
