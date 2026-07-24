"""
Step 5: 预训练验证与评估
=========================

按计划 §7 验收清单执行:
1. 加载 best_encoder.pt (或随机初始化作 baseline 对比) -> 在 held-out 上算 AUC / AP
2. DTI Hits@K (K=10) -- 每个有 held-out 边的 drug 排序 candidate gene, 看真值落在 top-K
3. 嵌入质量: mean / std / effective rank (奇异值法); 可选 t-SNE 可视化 CGC 驱动 vs 非驱动
4. 已知 DTI 恢复: 对 11 个无 DTI 药物 (184 - 173) 的 top-10 候选基因 -> recovered_dti_candidates.csv
5. 生成 pretrain_report.txt

Usage
-----
全量评估:    uv run python program/verify_pretrain.py --ckpt data/model/pretrain/best_encoder.pt
smoke:       uv run python program/verify_pretrain.py --smoke
随机基线:    uv run python program/verify_pretrain.py --random_only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.model.dataset import load_hetero_graph, load_cell_line_features, inject_batch_omics
from program.model.edge_mask import EdgeMaskSampler, HELDOUT_PT, PRETRAIN_DIR
from program.model.pretrain_encoder import PretrainGNNEncoder
from program.model.edge_predictor import PPIEdgePredictor, DTIConditionedPredictor

BEST_ENCODER_PT = PRETRAIN_DIR / "best_encoder.pt"
REPORT_TXT = PRETRAIN_DIR / "pretrain_report.txt"
TSNE_PNG = PRETRAIN_DIR / "tsne_gene_embeddings.png"
HITS_CSV = PRETRAIN_DIR / "hits_k_dti.csv"
RECOVERED_CSV = PRETRAIN_DIR / "recovered_dti_candidates.csv"

DRIVER_TXT = _PROJECT_ROOT / "data" / "processed" / "driver&pathway" / "driver_genes.txt"
CORE_GENE_ORDER_TXT = _PROJECT_ROOT / "data" / "model" / "hetero_graph" / "core_gene_order.txt"
DRUG_CID_TO_IDX_JSON = _PROJECT_ROOT / "data" / "processed" / "drug_cid_to_idx.json"


def build_encoders(ckpt_path: Path | None, hetero, cfg: SimpleNamespace, device, random_only: bool = False):
    encoder = PretrainGNNEncoder(
        gene_static_dim=int(hetero["gene"].x.shape[1]),
        drug_dim=int(hetero["drug"].x.shape[1]),
        omics_per_gene_dim=4,
        hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers,
        heads_ppi=cfg.heads_ppi, heads_dti=cfg.heads_dti, dropout=cfg.dropout,
    ).to(device)

    if random_only:
        ppi_pred = PPIEdgePredictor(cfg.hidden_dim, inner_dim=min(128, cfg.hidden_dim * 2), dropout=cfg.dropout).to(device)
        dti_pred = DTIConditionedPredictor(cfg.hidden_dim, 4, inner_dim=min(128, cfg.hidden_dim * 2), dropout=cfg.dropout).to(device)
        name = "random_init"
        return encoder, ppi_pred, dti_pred, name

    encoder_random_state = {k: v.clone() for k, v in encoder.state_dict().items()}

    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    obj = torch.load(ckpt_path, weights_only=False, map_location=device)
    encoder.load_state_dict(obj["encoder_state_dict"])
    kw = obj.get("encoder_kwargs", {})
    encoder_random_init = PretrainGNNEncoder(**kw).to(device)
    ppi_pred = PPIEdgePredictor(cfg.hidden_dim, inner_dim=min(128, cfg.hidden_dim * 2), dropout=cfg.dropout).to(device)
    dti_pred = DTIConditionedPredictor(cfg.hidden_dim, 4, inner_dim=min(128, cfg.hidden_dim * 2), dropout=cfg.dropout).to(device)
    if "ppi_pred_state_dict" in obj:
        ppi_pred.load_state_dict(obj["ppi_pred_state_dict"])
    if "dti_pred_state_dict" in obj:
        dti_pred.load_state_dict(obj["dti_pred_state_dict"])
    encoder_random_init.load_state_dict(encoder_random_state)

    return encoder, ppi_pred, dti_pred, "pretrained"


def eval_aucs(encoder, ppi_pred, dti_pred, hetero, clf, heldout, cfg, device) -> dict:
    encoder.eval(); ppi_pred.eval(); dti_pred.eval()
    omics = inject_batch_omics(clf, torch.arange(min(cfg.eval_cell_batch_size, clf["expression"].shape[0])))
    omics_4 = torch.stack([omics["expr"], omics["mut"], omics["cnv"], omics["meth"]], dim=-1).to(device)
    omics_mean = omics_4.mean(dim=0, keepdim=True)

    ppi_pos = heldout["ppi_pos"].to(device)
    ppi_neg = heldout["ppi_neg"].to(device)
    dti_pos = heldout["dti_pos"].to(device)
    dti_neg = heldout["dti_neg"].to(device)

    ppi_full_fwd = hetero["gene", "ppi", "gene"].edge_index
    kh = ppi_full_fwd[0] <= ppi_full_fwd[1]
    ppi_eval_fwd = ppi_full_fwd[:, kh]
    ppi_eval_rev = ppi_full_fwd[:, ~kh]
    if ppi_eval_rev.shape[1] == 0:
        ppi_eval_rev = torch.stack([ppi_eval_fwd[1], ppi_eval_fwd[0]], dim=0)
    dti_full_fwd = hetero["drug", "targets", "gene"].edge_index
    dti_eval_rev = torch.stack([dti_full_fwd[1], dti_full_fwd[0]], dim=0)

    with torch.no_grad():
        g_emb, d_emb = encoder.forward_single(hetero["gene"].x, hetero["drug"].x, omics_mean[0],
                                               ppi_eval_fwd, ppi_eval_rev, dti_full_fwd, dti_eval_rev)
        ppi_logits = ppi_pred(g_emb, ppi_pos, ppi_neg)
        dti_logits = dti_pred(g_emb, d_emb, omics_mean, dti_pos, dti_neg)
        ppi_prob = torch.sigmoid(ppi_logits).cpu().numpy()
        dti_prob = torch.sigmoid(dti_logits).cpu().numpy()
    y_ppi = np.concatenate([np.ones(ppi_pos.shape[1]), np.zeros(ppi_neg.shape[1])])
    y_dti = np.concatenate([np.ones(dti_pos.shape[1]), np.zeros(dti_neg.shape[1])])
    out = {}
    try: out["auc_ppi"] = float(roc_auc_score(y_ppi, ppi_prob))
    except ValueError: out["auc_ppi"] = float("nan")
    try: out["ap_ppi"] = float(average_precision_score(y_ppi, ppi_prob))
    except ValueError: out["ap_ppi"] = float("nan")
    try: out["auc_dti"] = float(roc_auc_score(y_dti, dti_prob))
    except ValueError: out["auc_dti"] = float("nan")
    try: out["ap_dti"] = float(average_precision_score(y_dti, dti_prob))
    except ValueError: out["ap_dti"] = float("nan")
    return out, g_emb, d_emb


def dti_hits_at_k(encoder, dti_pred, hetero, clf, heldout, cfg, device, k: int = 10) -> dict:
    encoder.eval(); dti_pred.eval()
    omics = inject_batch_omics(clf, torch.arange(min(cfg.eval_cell_batch_size, clf["expression"].shape[0])))
    omics_4 = torch.stack([omics["expr"], omics["mut"], omics["cnv"], omics["meth"]], dim=-1).to(device)
    omics_mean = omics_4.mean(dim=0, keepdim=True)
    ppi_full_fwd = hetero["gene", "ppi", "gene"].edge_index
    kh = ppi_full_fwd[0] <= ppi_full_fwd[1]
    ppi_eval_fwd = ppi_full_fwd[:, kh]
    ppi_eval_rev = ppi_full_fwd[:, ~kh]
    if ppi_eval_rev.shape[1] == 0:
        ppi_eval_rev = torch.stack([ppi_eval_fwd[1], ppi_eval_fwd[0]], dim=0)
    dti_full_fwd = hetero["drug", "targets", "gene"].edge_index
    dti_eval_rev = torch.stack([dti_full_fwd[1], dti_full_fwd[0]], dim=0)
    with torch.no_grad():
        g_emb, d_emb = encoder.forward_single(hetero["gene"].x, hetero["drug"].x, omics_mean[0],
                                               ppi_eval_fwd, ppi_eval_rev, dti_full_fwd, dti_eval_rev)
    heldout_pos = heldout["dti_pos"]
    per_drug_rows = []
    unique_drugs = heldout_pos[0].unique().tolist()
    n_eval_total = 0; hits_total = 0
    with torch.no_grad():
        for d in unique_drugs:
            mask = (heldout_pos[0] == d)
            true_genes = heldout_pos[1][mask].tolist()
            n_true = len(true_genes)
            if n_true == 0: continue
            cand_gene = torch.arange(g_emb.shape[0], device=device)
            drug_t = torch.full((g_emb.shape[0],), d, device=device, dtype=torch.long)
            edges = torch.stack([drug_t, cand_gene], dim=0)
            logits = dti_pred(g_emb, d_emb, omics_mean, edges, torch.zeros((2, 0), dtype=torch.long, device=device))
            ranks = torch.argsort(logits, descending=True).cpu().numpy()
            top_k_idx = set(ranks[:k].tolist())
            hits = sum(1 for tg in true_genes if tg in top_k_idx)
            hits_total += hits; n_eval_total += n_true
            per_drug_rows.append({"drug_idx": int(d), "n_true": n_true, "hits_in_k": hits})
    overall = hits_total / max(1, n_eval_total)
    return {"hits_at_k": overall, "n_drugs_eval": len(unique_drugs), "n_eval_total": n_eval_total, "k": k, "rows": per_drug_rows}


def recovered_dti_candidates(encoder, dti_pred, hetero, clf, sampler, cfg, device, k: int = 10) -> list[dict]:
    encoder.eval(); dti_pred.eval()
    omics = inject_batch_omics(clf, torch.arange(min(cfg.eval_cell_batch_size, clf["expression"].shape[0])))
    omics_4 = torch.stack([omics["expr"], omics["mut"], omics["cnv"], omics["meth"]], dim=-1).to(device)
    omics_mean = omics_4.mean(dim=0, keepdim=True)
    ppi_full_fwd = hetero["gene", "ppi", "gene"].edge_index
    kh = ppi_full_fwd[0] <= ppi_full_fwd[1]
    ppi_eval_fwd = ppi_full_fwd[:, kh]
    ppi_eval_rev = ppi_full_fwd[:, ~kh]
    if ppi_eval_rev.shape[1] == 0:
        ppi_eval_rev = torch.stack([ppi_eval_fwd[1], ppi_eval_fwd[0]], dim=0)
    dti_full_fwd = hetero["drug", "targets", "gene"].edge_index
    dti_eval_rev = torch.stack([dti_full_fwd[1], dti_full_fwd[0]], dim=0)
    with torch.no_grad():
        g_emb, d_emb = encoder.forward_single(hetero["gene"].x, hetero["drug"].x, omics_mean[0],
                                               ppi_eval_fwd, ppi_eval_rev, dti_full_fwd, dti_eval_rev)

    drug_idx_all = set(range(int(hetero["drug"].num_nodes)))
    no_dti_drugs = sorted(drug_idx_all - set(int(x) for x in sampler.drug_idx_with_dti.tolist()))

    core_gene_order = CORE_GENE_ORDER_TXT.read_text().splitlines() if CORE_GENE_ORDER_TXT.exists() else []
    drug_idx_to_cid: dict = {}
    if DRUG_CID_TO_IDX_JSON.exists():
        drug_idx_to_cid = {int(v): k for k, v in json.loads(DRUG_CID_TO_IDX_JSON.read_text()).items()}

    out_rows = []
    with torch.no_grad():
        for d in no_dti_drugs:
            cand_gene = torch.arange(g_emb.shape[0], device=device)
            drug_t = torch.full((g_emb.shape[0],), d, device=device, dtype=torch.long)
            edges = torch.stack([drug_t, cand_gene], dim=0)
            logits = dti_pred(g_emb, d_emb, omics_mean, edges, torch.zeros((2, 0), dtype=torch.long, device=device))
            ranks = torch.argsort(logits, descending=True).cpu().numpy()[:k]
            scores = torch.sigmoid(logits).cpu().numpy()
            cid = drug_idx_to_cid.get(d, "NA")
            top_syms = [core_gene_order[i] if i < len(core_gene_order) else f"gene_{i}" for i in ranks]
            top_scores = [float(scores[i]) for i in ranks]
            out_rows.append({
                "drug_idx": int(d),
                "drug_cid": cid,
                "top_gene_idx": "|".join(str(int(i)) for i in ranks),
                "top_gene_symbol": "|".join(top_syms),
                "top_scores": "|".join(f"{s:.4f}" for s in top_scores),
            })
    return out_rows


def embedding_stats(emb: torch.Tensor) -> dict:
    e = emb.detach().cpu().numpy()
    s = np.linalg.svd(e - e.mean(axis=0, keepdims=True), compute_uv=False)
    eff_rank = float(np.exp(np.sum(np.log(s / s.sum() + 1e-12) * (s / s.sum() + 1e-12)))) if e.shape[0] > 1 else float("nan")
    return {
        "shape": list(e.shape),
        "mean": float(e.mean()),
        "std": float(e.std()),
        "min": float(e.min()),
        "max": float(e.max()),
        "effective_rank": eff_rank,
    }


def tsne_visualization(emb: torch.Tensor, core_gene_order: list[str], out_png: Path):
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[verify] skip t-SNE: missing dependency ({e})")
        return
    is_cgc = [g in set(open(DRIVER_TXT).read().splitlines()) for g in core_gene_order]
    z = TSNE(n_components=2, init="pca", perplexity=30, random_state=42).fit_transform(emb.detach().cpu().numpy())
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(z[:, 0], z[:, 1], c=is_cgc, s=3, alpha=0.5, cmap="coolwarm")
    ax.set_title("Pretrained gene embeddings (red = CGC driver)")
    fig.colorbar(sc, ax=ax, ticks=[0, 1])
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[verify] t-SNE saved to {out_png}")


def write_report(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def run(smoke: bool, ckpt: Path | None, random_only: bool, k_hits: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[verify] device={device} smoke={smoke} ckpt={ckpt}")

    hetero = load_hetero_graph(device=device)
    clf = load_cell_line_features(device=device)
    n_genes = int(hetero["gene"].num_nodes)

    if ckpt is not None and (not ckpt.exists()) and not random_only:
        print(f"[verify] ckpt not found, falling back to --random_only")
        random_only = True

    if random_only:
        cfg = SimpleNamespace(hidden_dim=64, n_layers=2, heads_ppi=4, heads_dti=2, dropout=0.1,
                              eval_cell_batch_size=2, lambda_dti=10.0)
    elif ckpt is not None and ckpt.exists():
        obj = torch.load(ckpt, weights_only=False, map_location=device)
        cfg = SimpleNamespace(**obj["cfg"])
        if not hasattr(cfg, "eval_cell_batch_size"):
            cfg.eval_cell_batch_size = 8
    else:
        cfg = SimpleNamespace(hidden_dim=64, n_layers=2, heads_ppi=4, heads_dti=2, dropout=0.1,
                              eval_cell_batch_size=2, lambda_dti=10.0)

    sampler = EdgeMaskSampler(hetero, cfg)
    heldout = sampler.load_or_build_heldout(HELDOUT_PT)

    enc_target, ppi_pred, dti_pred, target_name = build_encoders(ckpt if not random_only else None, hetero, cfg, device, random_only=random_only)
    print(f"[verify] target encoder = {target_name}")

    metrics_target, g_emb, d_emb = eval_aucs(enc_target, ppi_pred, dti_pred, hetero, clf, heldout, cfg, device)
    print(f"[verify] {target_name} AUC/AP PPI: {metrics_target['auc_ppi']:.4f} / {metrics_target['ap_ppi']:.4f}  "
          f"DTI: {metrics_target['auc_dti']:.4f} / {metrics_target['ap_dti']:.4f}")

    if not smoke and not random_only:
        enc_random = PretrainGNNEncoder(
            gene_static_dim=int(hetero["gene"].x.shape[1]),
            drug_dim=int(hetero["drug"].x.shape[1]),
            omics_per_gene_dim=4,
            hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers,
            heads_ppi=cfg.heads_ppi, heads_dti=cfg.heads_dti, dropout=cfg.dropout,
        ).to(device)
        _e2 = PPIEdgePredictor(cfg.hidden_dim, inner_dim=min(128, cfg.hidden_dim * 2)).to(device)
        _d2 = DTIConditionedPredictor(cfg.hidden_dim, 4, inner_dim=min(128, cfg.hidden_dim * 2)).to(device)
        m_random, _, _ = eval_aucs(enc_random, _e2, _d2, hetero, clf, heldout, cfg, device)
        print(f"[verify] random_init AUC/AP PPI: {m_random['auc_ppi']:.4f} / {m_random['ap_ppi']:.4f}  "
              f"DTI: {m_random['auc_dti']:.4f} / {m_random['ap_dti']:.4f}")
    else:
        m_random = None

    stat_target = embedding_stats(g_emb)

    hits_summary = None
    hits_rows_csv = []
    if not smoke:
        hits_summary = dti_hits_at_k(enc_target, dti_pred, hetero, clf, heldout, cfg, device, k=k_hits)
        print(f"[verify] DTI Hits@{k_hits}: {hits_summary['hits_at_k']:.4f} "
              f"n_drugs={hits_summary['n_drugs_eval']} n_eval={hits_summary['n_eval_total']}")
        for row in hits_summary["rows"]:
            hits_rows_csv.append([row["drug_idx"], row["n_true"], row["hits_in_k"], row["hits_in_k"] / max(1, row["n_true"])])
        with open(HITS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["drug_idx", "n_true", "hits", "hit_ratio"])
            w.writerows(hits_rows_csv)
        print(f"[verify] hits saved to {HITS_CSV}")

    recovery_rows = None
    if not smoke and not random_only:
        recovery_rows = recovered_dti_candidates(enc_target, dti_pred, hetero, clf, sampler, cfg, device, k=k_hits)
        with open(RECOVERED_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["drug_idx", "drug_cid", "top_gene_idx", "top_gene_symbol", "top_scores"])
            for r in recovery_rows:
                w.writerow([r["drug_idx"], r["drug_cid"], r["top_gene_idx"], r["top_gene_symbol"], r["top_scores"]])
        print(f"[verify] recovery candidates ({len(recovery_rows)} drugs) saved to {RECOVERED_CSV}")

    tsne_path = None
    if not smoke:
        core_gene_order = CORE_GENE_ORDER_TXT.read_text().splitlines() if CORE_GENE_ORDER_TXT.exists() else []
        tsne_path = TSNE_PNG
        tsne_visualization(g_emb, core_gene_order, tsne_path)

    lines = []
    lines.append("====== Phase 2 预训练验证报告 ======")
    lines.append(f"node gene.x={tuple(hetero['gene'].x.shape)} drug.x={tuple(hetero['drug'].x.shape)}")
    lines.append(f"target_encoder: {target_name}")
    lines.append(f"ckpt: {ckpt}")
    lines.append(f"smoke: {smoke}")
    lines.append("")
    lines.append("== 边重构指标 (held-out) ==")
    lines.append(f"  PPI:  AUC={metrics_target['auc_ppi']:.4f}  AP={metrics_target['ap_ppi']:.4f}")
    lines.append(f"  DTI:  AUC={metrics_target['auc_dti']:.4f}  AP={metrics_target['ap_dti']:.4f}")
    if m_random is not None:
        lines.append("== 对比 random_init baseline ==")
        lines.append(f"  PPI:  random AUC={m_random['auc_ppi']:.4f}  pretrained AUC={metrics_target['auc_ppi']:.4f}  delta={metrics_target['auc_ppi']-m_random['auc_ppi']:+.4f}")
        lines.append(f"  DTI:  random AUC={m_random['auc_dti']:.4f}  pretrained AUC={metrics_target['auc_dti']:.4f}  delta={metrics_target['auc_dti']-m_random['auc_dti']:+.4f}")
    if hits_summary:
        lines.append(f"== DTI Hits@{k_hits} ==")
        lines.append(f"  hits_at_k={hits_summary['hits_at_k']:.4f}  n_drugs={hits_summary['n_drugs_eval']}  n_eval_total={hits_summary['n_eval_total']}")
    lines.append("== 嵌入统计 ==")
    for kk, vv in stat_target.items():
        lines.append(f"  {kk}: {vv}")
    if recovery_rows:
        lines.append(f"== DTI 候选恢复 ==")
        lines.append(f"  n_no_dti_drugs={len(recovery_rows)} top_k={k_hits} saved to {RECOVERED_CSV}")
    lines.append("")
    lines.append("== 验收标准 (计划 §7.8) ==")
    accept_lines = []
    accept_lines.append(f"  [{'OK' if (not np.isnan(metrics_target['auc_ppi']) and metrics_target['auc_ppi']>0.85) else 'NA'}] PPI AUC > 0.85 (actual {metrics_target['auc_ppi']:.4f})")
    accept_lines.append(f"  [{'OK' if (not np.isnan(metrics_target['ap_ppi']) and metrics_target['ap_ppi']>0.85) else 'NA'}] PPI AP  > 0.85 (actual {metrics_target['ap_ppi']:.4f})")
    accept_lines.append(f"  [{'OK' if (not np.isnan(metrics_target['auc_dti']) and metrics_target['auc_dti']>0.75) else 'NA'}] DTI AUC > 0.75 (actual {metrics_target['auc_dti']:.4f})")
    if hits_summary:
        accept_lines.append(f"  [{'OK' if hits_summary['hits_at_k']>0.5 else 'NA'}] DTI Hits@10 > 0.5 (actual {hits_summary['hits_at_k']:.4f})")
    accept_lines.append(f"  [{'OK' if stat_target['std']>0 and stat_target['std']<10 else 'NA'}] 嵌入 std 不为 0 / 不爆炸 (std={stat_target['std']:.4f})")
    lines.extend(accept_lines)
    write_report(REPORT_TXT, "\n".join(lines))
    print(f"[verify] report saved to {REPORT_TXT}")
    print("[verify] DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=str(BEST_ENCODER_PT), help="path to best_encoder.pt")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--random_only", action="store_true")
    args = ap.parse_args()
    ckpt = Path(args.ckpt) if args.ckpt else None
    run(smoke=args.smoke, ckpt=ckpt, random_only=args.random_only)


if __name__ == "__main__":
    main()