"""
Step 4: 预训练主循环 (Edge Perturbation Pretrain)
===================================================

策略 B (计划 §0.2.1): GNN 输入即注入 omics, 每个 batch 内 B 个条件 → B 次 forward.
- 每 epoch: 打乱 404 细胞系 -> 按 batch_condition 分批 -> 每 step 动态 mask + per-cell forward
- 每 step 内: 采样 mask 一次 (B 个 cell 共享 visible edges), per-cell omics 注入,
  loss_ci / B 累积梯度后 opt.step() 一次
- 每 epoch 末: 在冻结 held-out 边集上算 AUC / AP / 法则监控 (单一参考 omics = mean over cells)
- 早停: 连续 patience epoch val_auc 无提升即停 (§0.3.4)
- Checkpoint: best_encoder.pt (encoder), best_predictors.pt (预测器同存, 供 verify 与下游)
- 日志: pretrain_log.json 每 epoch 一条

Usage
-----
全量训练:    uv run python program/model/pretrain.py --config data/model/pretrain/pretrain_config.json
小规模 smoke: uv run python program/model/pretrain.py --smoke
显式指定 cfg: uv run python program/model/pretrain.py --config <path> --epochs 5 --batch_condition 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from program.model.dataset import load_hetero_graph, load_cell_line_features, inject_batch_omics
from program.model.edge_mask import EdgeMaskSampler, HELDOUT_PT, PRETRAIN_DIR
from program.model.pretrain_dataloader import PretrainCellSampler, get_n_cells
from program.model.pretrain_encoder import PretrainGNNEncoder
from program.model.edge_predictor import PPIEdgePredictor, DTIConditionedPredictor
from program.model.pretrain_loss import pretrain_loss, make_labels

DEFAULT_CONFIG_PT = PRETRAIN_DIR / "pretrain_config.json"
SMOKE_CONFIG_PT = PRETRAIN_DIR / "pretrain_config_smoke.json"
BEST_ENCODER_PT = PRETRAIN_DIR / "best_encoder.pt"
BEST_PREDICTORS_PT = PRETRAIN_DIR / "best_predictors.pt"
LOG_JSON = PRETRAIN_DIR / "pretrain_log.json"
CFG_JSON = PRETRAIN_DIR / "pretrain_used_config.json"


def _default_full_cfg() -> dict:
    return dict(
        hidden_dim=128, n_layers=2, heads_ppi=4, heads_dti=2, dropout=0.3,
        lr=1e-4, weight_decay=1e-5, lambda_dti=10.0, pos_weight_ppi=1.0, pos_weight_dti=1.0,
        mask_ppi_ratio=0.20, mask_dti_ratio=0.40, neg_ratio=1.0,
        heldout_ppi_ratio=0.01, heldout_dti_ratio=0.20,
        batch_condition=4, max_epochs=20, patience=10, lr_patience=5, lr_factor=0.5,
        clip_grad=5.0, seed=42, eval_cell_batch_size=8, metrics_combine="hits_aware",
        dti_margin_alpha=0.5, dti_margin_gamma=2.0,
    )


def _default_smoke_cfg() -> dict:
    return dict(
        hidden_dim=32, n_layers=2, heads_ppi=4, heads_dti=2, dropout=0.1,
        lr=1e-3, weight_decay=0.0, lambda_dti=10.0, pos_weight_ppi=1.0, pos_weight_dti=1.0,
        mask_ppi_ratio=0.20, mask_dti_ratio=0.40, neg_ratio=1.0,
        heldout_ppi_ratio=0.01, heldout_dti_ratio=0.20,
        batch_condition=2, max_epochs=1, patience=10, lr_patience=5, lr_factor=0.5,
        clip_grad=5.0, seed=42, eval_cell_batch_size=2, max_steps_per_epoch=2,
        metrics_combine="mean",
        dti_margin_alpha=0.5, dti_margin_gamma=2.0,
    )


def cfg_to_namespace(d: dict) -> SimpleNamespace:
    return SimpleNamespace(**d)


def load_config(path: Path | None, smoke: bool, cli_overrides: dict) -> SimpleNamespace:
    if smoke:
        if path is None and SMOKE_CONFIG_PT.exists():
            d = json.loads(SMOKE_CONFIG_PT.read_text())
        else:
            d = _default_smoke_cfg()
    elif path is not None and path.exists():
        d = json.loads(path.read_text())
    elif DEFAULT_CONFIG_PT.exists():
        d = json.loads(DEFAULT_CONFIG_PT.read_text())
    else:
        d = _default_full_cfg()
    for k, v in cli_overrides.items():
        if v is not None and k in d:
            d[k] = v
    return cfg_to_namespace(d)


def omics_to_per_gene(omics: dict) -> torch.Tensor:
    return torch.stack([omics["expr"], omics["mut"], omics["cnv"], omics["meth"]], dim=-1)


@torch.no_grad()
def evaluate_on_heldout(
    encoder: PretrainGNNEncoder,
    ppi_pred: PPIEdgePredictor,
    dti_pred: DTIConditionedPredictor,
    hetero,
    clf: dict,
    heldout: dict,
    cfg: SimpleNamespace,
    device: torch.device,
) -> dict:
    """用 mean-over-cells 的 omics 作单一参考条件 (条件无关 vs BatchNorm 部分), 单次前向 -> 边预测 -> AUC/AP."""
    encoder.eval()
    ppi_pred.eval()
    dti_pred.eval()

    n_genes = int(hetero["gene"].num_nodes)
    eval_b = max(1, min(cfg.eval_cell_batch_size, clf["expression"].shape[0]))
    eval_cells = torch.arange(eval_b, device="cpu")
    omics = inject_batch_omics(clf, eval_cells)
    omics_4 = omics_to_per_gene(omics).to(device)
    omics_mean = omics_4.mean(dim=0, keepdim=True)

    ppi_pos = heldout["ppi_pos"].to(device)
    ppi_neg = heldout["ppi_neg"].to(device)
    dti_pos = heldout["dti_pos"].to(device)
    dti_neg = heldout["dti_neg"].to(device)

    # 在评估时不再 mask 可见边 (用全 PPI/全 DTI 作 GNN 消息传递)
    ppi_full_fwd = hetero["gene", "ppi", "gene"].edge_index
    kh = ppi_full_fwd[0] <= ppi_full_fwd[1]
    ppi_eval_fwd = ppi_full_fwd[:, kh]
    ppi_eval_rev = ppi_full_fwd[:, ~kh]
    if ppi_eval_rev.shape[1] == 0:
        ppi_eval_rev = torch.stack([ppi_eval_fwd[1], ppi_eval_fwd[0]], dim=0)
    dti_full_fwd = hetero["drug", "targets", "gene"].edge_index
    dti_eval_rev = torch.stack([dti_full_fwd[1], dti_full_fwd[0]], dim=0)

    gene_static = hetero["gene"].x
    drug_x = hetero["drug"].x
    g_emb, d_emb = encoder.forward_single(gene_static, drug_x, omics_mean[0],
                                          ppi_eval_fwd, ppi_eval_rev, dti_full_fwd, dti_eval_rev)

    ppi_logits = ppi_pred(g_emb, ppi_pos, ppi_neg)
    dti_logits = dti_pred(g_emb, d_emb, omics_mean, dti_pos, dti_neg)

    y_ppi = torch.cat([torch.ones(ppi_pos.shape[1], device=device),
                       torch.zeros(ppi_neg.shape[1], device=device)]).cpu().numpy()
    y_dti = torch.cat([torch.ones(dti_pos.shape[1], device=device),
                       torch.zeros(dti_neg.shape[1], device=device)]).cpu().numpy()

    ppi_prob = torch.sigmoid(ppi_logits).cpu().numpy()
    dti_prob = torch.sigmoid(dti_logits).cpu().numpy()

    out: dict = {}
    try:
        out["auc_ppi"] = float(roc_auc_score(y_ppi, ppi_prob))
    except ValueError:
        out["auc_ppi"] = float("nan")
    try:
        out["ap_ppi"] = float(average_precision_score(y_ppi, ppi_prob))
    except ValueError:
        out["ap_ppi"] = float("nan")
    try:
        out["auc_dti"] = float(roc_auc_score(y_dti, dti_prob))
    except ValueError:
        out["auc_dti"] = float("nan")
    try:
        out["ap_dti"] = float(average_precision_score(y_dti, dti_prob))
    except ValueError:
        out["ap_dti"] = float("nan")

    metrics_combine = getattr(cfg, "metrics_combine", "hits_aware")
    if metrics_combine == "weighted":
        weight = cfg.lambda_dti / (1.0 + cfg.lambda_dti)
        out["auc"] = (1.0 - weight) * out["auc_ppi"] + weight * out["auc_dti"]
    elif metrics_combine == "mean":
        out["auc"] = 0.5 * out["auc_ppi"] + 0.5 * out["auc_dti"]
    elif metrics_combine == "hits_aware":
        hits10 = _dti_hits_at_k(d_emb, g_emb, dti_pos[0], dti_pos[1], dti_pred,
                                omics_mean, dti_pos, device, k=10)
        out["hits10"] = hits10
        out["auc"] = 0.3 * out["auc_ppi"] + 0.3 * out["auc_dti"] + 0.4 * hits10
    else:
        out["auc"] = out["auc_ppi"]
    return out


def _dti_hits_at_k(d_emb_all: torch.Tensor, g_emb_all: torch.Tensor,
                   drug_idx_held: torch.Tensor, gene_idx_held: torch.Tensor,
                   dti_pred: DTIConditionedPredictor, omics_per_gene: torch.Tensor,
                   heldout_dti_pos: torch.Tensor, device: torch.device, k: int = 10) -> float:
    """Per-drug Hits@K on held-out DTI edges (vectorized).

    For each unique drug in held-out pos, rank candidate gene scores and see if true gene falls in top-K.
    Batches all (D_unique * N_gene) edges through dti_pred in one forward for speed.
    """
    unique_drugs = heldout_dti_pos[0].unique()
    if unique_drugs.numel() == 0:
        return float("nan")
    n_genes = g_emb_all.shape[0]
    D = unique_drugs.numel()
    drug_part = unique_drugs.repeat_interleave(n_genes)
    gene_part = torch.arange(n_genes, device=device).repeat(D)
    all_edges = torch.stack([drug_part, gene_part], dim=0)
    with torch.no_grad():
        logits = dti_pred(g_emb_all, d_emb_all, omics_per_gene, all_edges,
                          torch.zeros((2, 0), dtype=torch.long, device=device))
    logits = logits.view(D, n_genes)
    topk_idx = logits.topk(k, dim=1).indices
    drug_to_row = {int(d): i for i, d in enumerate(unique_drugs.tolist())}
    drug_pos = heldout_dti_pos[0]
    gene_pos = heldout_dti_pos[1]
    hits = 0
    n_eval = 0
    for d in unique_drugs.tolist():
        mask = (drug_pos == d)
        true_genes = gene_pos[mask].tolist()
        row = drug_to_row[d]
        top_set = set(topk_idx[row].tolist())
        for tg in true_genes:
            if tg in top_set:
                hits += 1
            n_eval += 1
    return hits / max(1, n_eval)


def save_checkpoint(path: Path, encoder: torch.nn.Module, ppi_pred: torch.nn.Module,
                    dti_pred: torch.nn.Module, optimizer, epoch: int, val_metrics: dict, cfg: SimpleNamespace):
    obj = {
        "encoder_state_dict": encoder.state_dict(),
        "encoder_kwargs": dict(
            gene_static_dim=getattr(encoder, "gene_static_dim"),
            drug_dim=getattr(encoder, "drug_dim"),
            omics_per_gene_dim=getattr(encoder, "omics_per_gene_dim"),
            hidden_dim=getattr(encoder, "hidden_dim"),
            n_layers=getattr(encoder, "n_layers"),
            heads_ppi=getattr(encoder, "heads_ppi"),
            heads_dti=getattr(encoder, "heads_dti"),
            dropout=float(getattr(encoder.dropout, "p", cfg.dropout)),
        ),
        "ppi_pred_state_dict": ppi_pred.state_dict(),
        "dti_pred_state_dict": dti_pred.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "val_metrics": val_metrics,
        "cfg": vars(cfg),
    }
    torch.save(obj, path)


def run_train(cfg: SimpleNamespace, smoke: bool, wall_log: dict | None = None) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pretrain] device={device} smoke={smoke}")
    print(f"[pretrain] cfg = {json.dumps(vars(cfg), ensure_ascii=False)}")

    hetero = load_hetero_graph(device)
    clf = load_cell_line_features(device)
    gene_static = hetero["gene"].x
    drug_x = hetero["drug"].x
    n_genes = int(hetero["gene"].num_nodes)
    n_drugs = int(hetero["drug"].num_nodes)
    n_cells = int(clf["expression"].shape[0])

    if not hasattr(cfg, "metrics_combine"):
        cfg.metrics_combine = "weighted"

    sampler = EdgeMaskSampler(hetero, cfg)
    heldout = sampler.load_or_build_heldout(HELDOUT_PT)

    cell_sampler = PretrainCellSampler(n_cells=n_cells, batch_condition=cfg.batch_condition, seed=cfg.seed)

    encoder = PretrainGNNEncoder(
        gene_static_dim=gene_static.shape[1],
        drug_dim=drug_x.shape[1],
        omics_per_gene_dim=4,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        heads_ppi=cfg.heads_ppi,
        heads_dti=cfg.heads_dti,
        dropout=cfg.dropout,
    ).to(device)
    ppi_pred = PPIEdgePredictor(cfg.hidden_dim, inner_dim=min(128, cfg.hidden_dim * 2), dropout=cfg.dropout).to(device)
    dti_pred = DTIConditionedPredictor(cfg.hidden_dim, 4, inner_dim=min(128, cfg.hidden_dim * 2), dropout=cfg.dropout).to(device)

    params = list(encoder.parameters()) + list(ppi_pred.parameters()) + list(dti_pred.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=cfg.lr_factor, patience=cfg.lr_patience)

    best_auc = -1.0
    patience_left = cfg.patience
    log_records = []

    print(f"[pretrain] encoder params={sum(p.numel() for p in encoder.parameters()):,} "
          f"| ppi_pred params={sum(p.numel() for p in ppi_pred.parameters()):,} "
          f"| dti_pred params={sum(p.numel() for p in dti_pred.parameters()):,}")
    print(f"[pretrain] hetero: gene.x={tuple(gene_static.shape)} drug.x={tuple(drug_x.shape)} n_cells={n_cells}")

    max_steps = getattr(cfg, "max_steps_per_epoch", None)
    for epoch in range(cfg.max_epochs):
        encoder.train(); ppi_pred.train(); dti_pred.train()
        epoch_t0 = time.time()
        step_stats = []
        for step, cell_batch in enumerate(cell_sampler):
            t0 = time.time()
            ppi = sampler.sample_ppi()
            dti = sampler.sample_dti()
            ppi_pos, ppi_neg = ppi["pos"].to(device), ppi["neg"].to(device)
            ppi_fwd, ppi_rev = ppi["visible_fwd"].to(device), ppi["visible_rev"].to(device)
            dti_pos, dti_neg = dti["pos"].to(device), dti["neg"].to(device)
            dti_fwd, dti_rev = dti["visible_fwd"].to(device), dti["visible_rev"].to(device)

            omics = inject_batch_omics(clf, cell_batch)
            omics_4 = omics_to_per_gene(omics).to(device)
            B = omics_4.shape[0]
            n_pos_ppi, n_neg_ppi = ppi_pos.shape[1], ppi_neg.shape[1]
            n_pos_dti, n_neg_dti = dti_pos.shape[1], dti_neg.shape[1]
            y_ppi = make_labels(n_pos_ppi, n_neg_ppi, device)
            y_dti = make_labels(n_pos_dti, n_neg_dti, device)

            opt.zero_grad()
            for ci in range(B):
                g_emb, d_emb = encoder.forward_single(gene_static, drug_x, omics_4[ci],
                                                     ppi_fwd, ppi_rev, dti_fwd, dti_rev)
                ppi_logits = ppi_pred(g_emb, ppi_pos, ppi_neg)
                dti_logits = dti_pred(g_emb, d_emb, omics_4[ci:ci+1], dti_pos, dti_neg)
                loss_ci, stats = pretrain_loss(ppi_logits, y_ppi, dti_logits, y_dti,
                                                lambda_dti=cfg.lambda_dti,
                                                pos_weight_ppi=cfg.pos_weight_ppi,
                                                pos_weight_dti=cfg.pos_weight_dti,
                                                dti_margin_alpha=getattr(cfg, "dti_margin_alpha", 0.0),
                                                dti_margin_gamma=getattr(cfg, "dti_margin_gamma", 2.0))
                (loss_ci / B).backward()
            if cfg.clip_grad and cfg.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.clip_grad)
            opt.step()
            step_stats.append(stats)
            elapsed = time.time() - t0
            if step == 0 or (step + 1) % max(1, (len(cell_sampler) // 4)) == 0 or smoke:
                print(f"[epoch {epoch}] step {step+1}/{len(cell_sampler)} B={B} "
                      f"loss={stats['loss_total']:.4f} ppi={stats['loss_ppi']:.4f} dti={stats['loss_dti']:.4f} dt={elapsed:.1f}s")
            if max_steps is not None and step + 1 >= max_steps:
                print(f"[.epoch {epoch}] smoke mode: hit max_steps_per_epoch={max_steps}, breaking step loop")
                break

        train_mean = {k: float(np.mean([s[k] for s in step_stats])) for k in ("loss_total", "loss_ppi", "loss_dti")}
        val_metrics = evaluate_on_heldout(encoder, ppi_pred, dti_pred, hetero, clf, heldout, cfg, device)
        sched.step(val_metrics["auc"])
        log_records.append({
            "epoch": epoch,
            "train": train_mean,
            "val": val_metrics,
            "lr": float(opt.param_groups[0]["lr"]),
            "epoch_time": time.time() - epoch_t0,
        })
        print(f"[epoch {epoch}] train={train_mean} val={val_metrics} lr={opt.param_groups[0]['lr']:.2e}")

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            save_checkpoint(BEST_ENCODER_PT, encoder, ppi_pred, dti_pred, opt, epoch, val_metrics, cfg)
            torch.save({"ppi": ppi_pred.state_dict(), "dti": dti_pred.state_dict()}, BEST_PREDICTORS_PT)
            patience_left = cfg.patience
            print(f"[checkpoint] new best auc={best_auc:.4f} saved to {BEST_ENCODER_PT}")
        else:
            patience_left -= 1
            print(f"[early-stop] patience left = {patience_left}")
            if patience_left <= 0:
                print("[early-stop] triggered; stopping")
                break

    final_log_path = Path(LOG_JSON)
    final_log_path.write_text(json.dumps({"records": log_records, "cfg": vars(cfg)}, ensure_ascii=False, indent=2))
    Path(CFG_JSON).write_text(json.dumps(vars(cfg), ensure_ascii=False, indent=2))
    print(f"[pretrain] DONE best_auc={best_auc:.4f} log={final_log_path}")
    return {"best_auc": best_auc, "n_epochs_logged": len(log_records), "record_last": (log_records[-1] if log_records else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="path to JSON config")
    ap.add_argument("--smoke", action="store_true", help="use smoke default config")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_condition", type=int, default=None)
    ap.add_argument("--hidden_dim", type=int, default=None)
    ap.add_argument("--lambda_dti", type=float, default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config) if args.config else None
    cli_overrides = {
        "max_epochs": args.epochs,
        "batch_condition": args.batch_condition,
        "hidden_dim": args.hidden_dim,
        "lambda_dti": args.lambda_dti,
    }
    cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}
    cfg = load_config(cfg_path, args.smoke, cli_overrides)
    if not hasattr(cfg, "metrics_combine"):
        cfg.metrics_combine = "weighted"
    run_train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()